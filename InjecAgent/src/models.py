import os
from InjecAgent.src.utils import get_response_text
import time
import transformers
import torch
transformers.set_seed(42)
class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt,  user_prompt_filled):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def call_model(self, model_input):
        raise NotImplementedError("This method should be overridden by subclasses.")

class LlamaToolUseModel(BaseModel):
    def __init__(self, params):
        super().__init__()  
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        self.tokenizer = tokenizer  
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            # max_length=4096,
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
        )
        if params['defense'] == 'PerplexityFiltering':
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                params['model_name'], 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
        self.params = params
        self.template = """<|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def call_model(self, model_input):
        output = self.pipeline(model_input)
        return get_response_text(output, model_input)
    

class VicunaModel(BaseModel):     
    def __init__(self, params):
        super().__init__()          
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'], use_fast=False)
        self.dropout = False
        if params['defense'] == 'ReTokenization':
            self.dropout = True
            print("Using BPE-dropout tokenizer")
            self.tokenizer_dropout = transformers.AutoTokenizer.from_pretrained(params['model_name'], use_fast=False, sp_model_kwargs={"enable_sampling": True, "alpha": 0.4})
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                params['model_name'], 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
            self.max_length = 4096  
            
        if params['defense'] == 'PerplexityFiltering':
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                params['model_name'], 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
            
        if params['defense'] != 'ReTokenization':
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=params['model_name'],
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=256,
                truncation=True,
                do_sample=False,
            )
        self.template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {system_prompt}\n{user_prompt} ASSISTANT:"

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = self.template.format(system_prompt=sys_prompt, user_prompt=user_prompt_filled)
        return model_input

    def call_model(self, model_input, tool_response=None):
        if self.dropout:
            parts = model_input.split(tool_response)
            tokenized_parts = []
            
            for i, part in enumerate(parts):
                tokenized_part = self.tokenizer(part, return_tensors="pt", add_special_tokens=(i == 0))
                tokenized_parts.append(tokenized_part["input_ids"])
                
                if i < len(parts) - 1:
                    tool_response_tokenized = self.tokenizer_dropout(tool_response, return_tensors="pt", add_special_tokens=False)
                    tokenized_parts.append(tool_response_tokenized["input_ids"])
            
            # Concatenate all tokenized parts
            concatenated_input_ids = torch.cat(tokenized_parts, dim=-1)
            
            attention_mask = torch.ones_like(concatenated_input_ids)
            
            # Move input to the correct device
            concatenated_input_ids = concatenated_input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            # Generate output with no sampling (equivalent to `do_sample=False`)
            output_tokens = self.model.generate(
                input_ids=concatenated_input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                do_sample=False
            )
            
            # Decode the output tokens
            output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                
            return get_response_text([{"generated_text": output_text}], "ASSISTANT:")
        else:
            output = self.pipeline(model_input)
            return get_response_text(output, "ASSISTANT:")
           

MODELS = {
    "Vicuna": VicunaModel,
    "Llama3": LlamaToolUseModel,
}   