import gc
import json
import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import time
import logging
from accelerate.utils import find_executable_batch_size
from attacks.utils import sample_control, get_nonascii_toks
from attacks.base_gcg import BaseGCG
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

class GCGFinetunedDetector():
    def __init__(self, device, threshold=0.8):
        self.model = AutoModelForSequenceClassification.from_pretrained("../deberta-v3-base-prompt-injection-v2").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("../deberta-v3-base-prompt-injection-v2")
        self.device = device
        self.embed_layer = self.model.get_input_embeddings()
        self.vocab_size = self.embed_layer.weight.shape[0]
        self.vocab_embeds = self.embed_layer(torch.arange(0, self.vocab_size).long().to(device)).to(device)
        self.not_allowed_tokens = get_nonascii_toks(self.tokenizer)
        self.threshold = threshold

    def set_vals(self, tools_before, tools_after, search_batch_size):
        self.tools_before = tools_before
        self.tools_after = tools_after
        
        self.search_batch_size = search_batch_size
        
        
        cache_input_ids = self.tokenizer([tools_before, tools_after], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids[0].insert(0, 1)
        cache_input_ids[-1].append(2)
        
        cache_input_ids = [torch.tensor(input_ids, device=self.device).unsqueeze(0) for input_ids in cache_input_ids]
        self.before_ids, self.after_ids = cache_input_ids
        
        self.before_embed, self.after_embed = [self.embed_layer(input_ids) for input_ids in cache_input_ids]
            

    def optimize_string_step(self, start_adv_string, gcg_obj): 
        logging.info(f"Optimizing string for finetuned detector defense from {start_adv_string}")
        model = self.model
        tokenizer = self.tokenizer
        device = model.device
        embed_layer = self.embed_layer

        optim_ids = tokenizer(start_adv_string, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
                
        decoded_adv_string = tokenizer.decode(optim_ids[0], clean_up_tokenization_spaces=False)
        
        differ_prefix = ''
        if ' ' + decoded_adv_string == start_adv_string:
            differ_prefix = ' '
        
        num_optim_tokens = len(optim_ids[0])

        optim_ids_onehot = torch.zeros((1, num_optim_tokens, self.vocab_size), device=device, dtype=model.dtype).to(device)
        optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()


        before_embeds = self.before_embed
        after_embeds = self.after_embed

        model.zero_grad()
        if optim_ids_onehot.grad is not None:
            optim_ids_onehot.grad.zero_()
        
        optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), self.vocab_embeds).unsqueeze(0)

        input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds], dim=1)
        outputs = model(inputs_embeds=input_embeds)


        # compute loss
        loss_fct = BCEWithLogitsLoss()
        target = torch.tensor([[1, 0]], dtype=torch.float16).to(device)
        loss = loss_fct(outputs.logits, target)

        probabilities = F.softmax(outputs.logits, dim=-1)
        safe_score = probabilities[0, 0].item()

        if safe_score > self.threshold:
            del probabilities, outputs, optim_embeds, optim_ids_onehot, optim_ids, target, input_embeds
            torch.cuda.empty_cache()
            gc.collect()
            return start_adv_string, loss.item()

        current_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
        current_loss = loss
        

        # ============ copy GCG method ============
        # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
        sampled_top_indices = sample_control(optim_ids.squeeze(0), current_grad.squeeze(0), gcg_obj.search_width, topk=256, temp=1, not_allowed_tokens=self.not_allowed_tokens)

        # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
        new_sampled_top_indices = []
        count = 0
        
        # don't change the first token?
        for j in range(len(sampled_top_indices)):
            input_ids = torch.cat([self.before_ids[0], sampled_top_indices[j], self.after_ids[0]])
            decoded_text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
            retokenized_ids = tokenizer(decoded_text, return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
            if not torch.equal(input_ids, retokenized_ids):
                count += 1
                continue
            else:
                new_sampled_top_indices.append(sampled_top_indices[j])
        logging.info(f'Filtered {count} tokens from {len(sampled_top_indices)} in defense')
        search_width = gcg_obj.search_width
        
        if len(new_sampled_top_indices) == 0:
            logging.info('All removals; defaulting to keeping all')
            count = 0
        else:
            sampled_top_indices = torch.stack(new_sampled_top_indices)
        
        if count >= search_width // 2:
            logging.info(f"Lots of removals: {count}")
            
        new_search_width = search_width - count


        # ========== Compute loss on these candidates and take the argmin. ========== #
        sampled_top_embeds = embed_layer(sampled_top_indices)

        input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1), sampled_top_embeds, after_embeds.repeat(new_search_width, 1, 1)], dim=1)
        
        current_loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds)

        selected_id = current_loss.argmin().item()
        
        optim_ids = sampled_top_indices[selected_id].unsqueeze(0)

        current_loss = current_loss.min().item()

        optim_str = tokenizer.decode(optim_ids[0], clean_up_tokenization_spaces=False)

        optim_str = differ_prefix + optim_str

        logging.info(f"done optimizing string defense")

        del sampled_top_embeds, loss
        del probabilities, outputs, optim_embeds, optim_ids_onehot, optim_ids, target, input_embeds, sampled_top_indices

        torch.cuda.empty_cache()
        gc.collect()

        return optim_str, current_loss
    
    def compute_candidates_loss(self, search_batch_size, input_embeds):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()
                
        all_loss = []
        for index in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[index:index+search_batch_size]
                outputs = self.model(inputs_embeds=input_embeds_batch)
            logits = outputs.logits

            loss_fct = BCEWithLogitsLoss(reduction='none')
            target = torch.tensor([[1, 0]], dtype=torch.float16).repeat(outputs.logits.size()[0], 1).to(self.device)
            loss = loss_fct(outputs.logits, target)
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)
        
            del outputs, logits, loss, target
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(all_loss, dim=0)
        


class MGCG_DT(BaseGCG):
    def __init__(self,
                 model,
                 tokenizer,
                 num_steps=500,
                 adv_string_init=" ! ! ! ! !",
                 search_width=512,
                 use_prefix_cache=True,
                 **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.search_width = search_width
        self.search_batch_size = search_width
        self.not_allowed_tokens = get_nonascii_toks(tokenizer, device=self.device)
        self.use_prefix_cache = use_prefix_cache
        self.finetuned_detector_optimization = GCGFinetunedDetector(self.device)
        
    def train_adv_string(self, data_dict, result_file, with_eos=False):
        case_id = data_dict['CaseID']
        target = data_dict['Target']
        before_adv_string = data_dict['before_adv_string'] + ' '
        after_adv_string = data_dict['after_adv_string']
        tool_before = data_dict['tool_output_before'] + ' '
        tool_after = data_dict['tool_output_after']
        
        logging.info(f'GCG || Case: {case_id} || Target: {target}')

        model = self.model
        device = self.device
        tokenizer = self.tokenizer

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        search_width = self.search_width

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))


        # ========== setup optim_ids and cached components ========== #        
        cache_input_ids = tokenizer([tokenizer.bos_token + before_adv_string, after_adv_string, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_adv_string_ids, after_adv_string_ids, target_ids = cache_input_ids
        before_adv_string_embeds, after_adv_string_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            with torch.no_grad():
                outputs = model(inputs_embeds=before_adv_string_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values
        
        self.finetuned_detector_optimization.set_vals(tool_before, tool_after, self.search_batch_size)
        
        visited_optim_ids = []
        cur_optim_str = adv_string_init
        for i in tqdm(range(num_steps)):
            optim_ids = tokenizer(cur_optim_str, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
            num_optim_tokens = len(optim_ids[0])
            logging.info(f'Step {i} || Adv String: {cur_optim_str} || main GCG number of tokens: {num_optim_tokens}')
            
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            if self.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, after_adv_string_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
            else:
                input_embeds = torch.cat([before_adv_string_embeds, optim_embeds, after_adv_string_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds)
                    
            logits = outputs.logits

            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=self.not_allowed_tokens)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices)):
                input_ids = torch.cat([before_adv_string_ids[0], sampled_top_indices[j], after_adv_string_ids[0]])
                decoded_text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
                retokenized_ids = tokenizer(decoded_text, return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                if not torch.equal(input_ids, retokenized_ids) or any(torch.equal(sampled_top_indices[j], v) for v in visited_optim_ids):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
            logging.info(f'Filtered {count} tokens from {len(sampled_top_indices)}')
            if len(new_sampled_top_indices) == 0:
                logging.info('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
            
            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            sampled_top_embeds = embed_layer(sampled_top_indices)
            if self.use_prefix_cache:
                input_embeds = torch.cat([sampled_top_embeds,
                                        after_adv_string_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            else:
                input_embeds = torch.cat([before_adv_string_embeds.repeat(new_search_width, 1, 1),
                                    sampled_top_embeds,
                                    after_adv_string_embeds.repeat(new_search_width, 1, 1),
                                    target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            
            loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds, target_ids)
            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss.argmin()]
            visited_optim_ids.append(optim_ids)
            optim_ids = optim_ids.unsqueeze(0)
            current_loss = loss.min().item()
            
            
            
            cur_optim_str = tokenizer.decode(optim_ids[0], clean_up_tokenization_spaces=False)
            fd_string, fd_loss = self.finetuned_detector_optimization.optimize_string_step(cur_optim_str, self)
            logging.info(f'Finetuned Detector Loss: {fd_loss} || Adv String: {fd_string}')
            cur_optim_str = fd_string 
                        
            with open(result_file, 'a') as f:
                f.write(json.dumps({'step': i, 'loss': current_loss + fd_loss, 'main loss': current_loss, 'fd_loss': fd_loss, 'adv_string': cur_optim_str}) + '\n')
            
            del input_embeds, sampled_top_embeds, logits, shift_logits, loss
            torch.cuda.empty_cache()
            gc.collect()
