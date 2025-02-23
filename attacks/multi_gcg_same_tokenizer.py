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

class MGCG_ST(BaseGCG):
    def __init__(self,
                 model,
                 tokenizer,
                 num_steps=500,
                 adv_string_init=" ! ! ! ! !",
                 search_width=512,
                 use_prefix_cache=True,
                 alpha=0.5,
                 **kwargs):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.tokenizer = tokenizer
        self.device = model.device
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.search_width = search_width
        self.search_batch_size = search_width
        self.not_allowed_tokens = get_nonascii_toks(tokenizer, device=self.device)
        self.use_prefix_cache = use_prefix_cache
        
    def train_adv_string(self, data_dict, result_file, with_eos=False):
        case_id = data_dict['CaseID']
        target_1 = data_dict['Target_1']
        target_2 = data_dict['Target_2']
        before_adv_string_1 = data_dict['before_adv_string_1'] + ' '
        before_adv_string_2 = data_dict['before_adv_string_2'] + ' '
        after_adv_string_1 = data_dict['after_adv_string_1']
        after_adv_string_2 = data_dict['after_adv_string_2']
        
        logging.info(f'MGCG_ST || Case: {case_id} || Target 1: {target_1} || Target 2: {target_2}')

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
        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])
        
        cache_input_ids_1 = tokenizer([tokenizer.bos_token + before_adv_string_1, after_adv_string_1, target_1], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids_1 = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids_1] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_adv_string_ids_1, after_adv_string_ids_1, target_ids_1 = cache_input_ids_1
        before_adv_string_embeds_1, after_adv_string_embeds_1, target_embeds_1 = [embed_layer(input_ids) for input_ids in cache_input_ids_1]
        
        cache_input_ids_2 = tokenizer([tokenizer.bos_token + before_adv_string_2, after_adv_string_2, target_2], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids_2 = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids_2] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_adv_string_ids_2, after_adv_string_ids_2, target_ids_2 = cache_input_ids_2
        before_adv_string_embeds_2, after_adv_string_embeds_2, target_embeds_2 = [embed_layer(input_ids) for input_ids in cache_input_ids_2]
        
        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            with torch.no_grad():
                outputs_1 = model(inputs_embeds=before_adv_string_embeds_1, use_cache=True)
                self.prefix_cache_1 = outputs_1.past_key_values
                
                outputs_2 = model(inputs_embeds=before_adv_string_embeds_2, use_cache=True)
                self.prefix_cache_2 = outputs_2.past_key_values
        
        
        visited_optim_ids = []
        for i in tqdm(range(num_steps)):
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            if self.use_prefix_cache:
                input_embeds_1 = torch.cat([optim_embeds, after_adv_string_embeds_1, target_embeds_1], dim=1)
                outputs_1 = model(inputs_embeds=input_embeds_1, past_key_values=self.prefix_cache_1)
                
                input_embeds_2 = torch.cat([optim_embeds, after_adv_string_embeds_2, target_embeds_2], dim=1)
                outputs_2 = model(inputs_embeds=input_embeds_2, past_key_values=self.prefix_cache_2)
            else:
                input_embeds_1 = torch.cat([before_adv_string_embeds_1, optim_embeds, after_adv_string_embeds_1, target_embeds_1], dim=1)
                outputs_1 = model(inputs_embeds=input_embeds_1)
                
                input_embeds_2 = torch.cat([before_adv_string_embeds_2, optim_embeds, after_adv_string_embeds_2, target_embeds_2], dim=1)
                outputs_2 = model(inputs_embeds=input_embeds_2)
                    
            logits_1 = outputs_1.logits
            logits_2 = outputs_2.logits

            tmp_1 = input_embeds_1.shape[1] - target_embeds_1.shape[1]
            shift_logits_1 = logits_1[..., tmp_1-1:-1, :].contiguous()
            shift_labels_1 = target_ids_1
            loss_fct = CrossEntropyLoss()
            loss_1 = loss_fct(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))
            
            tmp_2 = input_embeds_2.shape[1] - target_embeds_2.shape[1]
            shift_logits_2 = logits_2[..., tmp_2-1:-1, :].contiguous()
            shift_labels_2 = target_ids_2
            loss_fct = CrossEntropyLoss()
            loss_2 = loss_fct(shift_logits_2.view(-1, shift_logits_2.size(-1)), shift_labels_2.view(-1))
            
            loss = self.alpha*loss_1 + (1-self.alpha)*loss_2
            logging.info(f"Loss 1: {loss_1.item()} Loss 2: {loss_2.item()} Combined Loss: {loss.item()}")

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=self.not_allowed_tokens)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices)):
                input_ids_1 = torch.cat([before_adv_string_ids_1[0], sampled_top_indices[j], after_adv_string_ids_1[0]])
                decoded_text_1 = tokenizer.decode(input_ids_1, clean_up_tokenization_spaces=False)
                retokenized_ids_1 = tokenizer(decoded_text_1, return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                
                input_ids_2 = torch.cat([before_adv_string_ids_2[0], sampled_top_indices[j], after_adv_string_ids_2[0]])
                decoded_text_2 = tokenizer.decode(input_ids_2, clean_up_tokenization_spaces=False)
                retokenized_ids_2 = tokenizer(decoded_text_2, return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                
                
                if not torch.equal(input_ids_1, retokenized_ids_1) or not torch.equal(input_ids_2, retokenized_ids_2) or any(torch.equal(sampled_top_indices[j], v) for v in visited_optim_ids):
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
                input_embeds_1 = torch.cat([sampled_top_embeds,
                                        after_adv_string_embeds_1.repeat(new_search_width, 1, 1),
                                        target_embeds_1.repeat(new_search_width, 1, 1)], dim=1)
                
                input_embeds_2 = torch.cat([sampled_top_embeds,
                                        after_adv_string_embeds_2.repeat(new_search_width, 1, 1),
                                        target_embeds_2.repeat(new_search_width, 1, 1)], dim=1)
            else:
                input_embeds_1 = torch.cat([before_adv_string_embeds_1.repeat(new_search_width, 1, 1),
                                    sampled_top_embeds,
                                    after_adv_string_embeds_1.repeat(new_search_width, 1, 1),
                                    target_embeds_1.repeat(new_search_width, 1, 1)], dim=1)
                
                input_embeds_2 = torch.cat([before_adv_string_embeds_2.repeat(new_search_width, 1, 1),
                                    sampled_top_embeds,
                                    after_adv_string_embeds_2.repeat(new_search_width, 1, 1),
                                    target_embeds_2.repeat(new_search_width, 1, 1)], dim=1)
            
            loss_1 = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds_1, target_ids_1, self.prefix_cache_1)
            loss_2 = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds_2, target_ids_2, self.prefix_cache_2)
            loss = self.alpha*loss_1 + (1-self.alpha)*loss_2

            
            
            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss.argmin()]
            visited_optim_ids.append(optim_ids)
            optim_ids = optim_ids.unsqueeze(0)
            
            input_ids_1 = torch.cat([before_adv_string_ids_1, optim_ids, after_adv_string_ids_1], dim=1)
            decoded_text_1 = tokenizer.decode(input_ids_1[0], clean_up_tokenization_spaces=False)
            
            input_ids_2 = torch.cat([before_adv_string_ids_2, optim_ids, after_adv_string_ids_2], dim=1)
            decoded_text_2 = tokenizer.decode(input_ids_2[0], clean_up_tokenization_spaces=False)
            
            
            decoded_adv_string = tokenizer.decode(optim_ids[0], clean_up_tokenization_spaces=False)
            logging.info(f'Step {i} || Loss: {loss.min().item()} || Adv String: {decoded_adv_string}')
            
            with open(result_file, 'a') as f:
                f.write(json.dumps({'step': i, 'loss': loss.min().item(), 'adv_string': decoded_adv_string, 'input_1': decoded_text_1, 'input_2': decoded_text_2}) + '\n')
            
            del input_embeds_1, input_embeds_2, sampled_top_embeds, logits_1, logits_2, shift_logits_1, shift_logits_2, loss_1, loss_2, loss
            torch.cuda.empty_cache()
            gc.collect()

