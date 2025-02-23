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

class BaseGCG():
    def __init__(self):
        pass
   
    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids, prefix_cache=None):
        if self.search_batch_size != search_batch_size:
            logging.info(f"Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()
                
        all_loss = []
        for index in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[index:index+search_batch_size]

                if self.use_prefix_cache:
                    # Expand prefix cache to match batch size
                    if prefix_cache is None:
                        prefix_cache = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(prefix_cache[i])):
                            prefix_cache_batch[i].append(prefix_cache[i][j].expand(current_batch_size, -1, -1, -1))
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch) # broken step
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)
            logits = outputs.logits

            
            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)
        
            del outputs, logits, loss
            # Used GPU Memory: 16747 MB
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(all_loss, dim=0)
        