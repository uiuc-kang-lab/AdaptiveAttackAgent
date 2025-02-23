import argparse
import logging
import datetime
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attacks import get_method_class
from data_processing.data_preparation_llama import get_training_data_llama3
from data_processing.data_preparation_vicuna import get_training_data_vicuna
from data_processing.results_processing import extract_adv_string
from tqdm import tqdm
from InjecAgent.src.evaluate_finetuned_agent_llama import evaluate_llama
from InjecAgent.src.evaluate_prompted_agent_vicuna import evaluate_vicuna

def init_configs():
    config_dict = {}
    with open("configs.json", 'r') as f:
        for line in f:
            tmp = json.loads(line)
            config_dict[(tmp['base_model'], tmp['defense'])] = tmp
    return config_dict   

def evaluate(base_model, file_prefix, args, config=None, adaptive_attack_files=None):
    logging.info(f"Evaluating")
    print(f"Evaluating")
    if adaptive_attack_files is None:
        test_case_files = {
            "dh": f"./InjecAgent/data/test_cases_dh_{args.data_setting}.json",
            "ds": f"./InjecAgent/data/test_cases_ds_{args.data_setting}.json",
        }
    else:
        test_case_files = adaptive_attack_files
    if base_model == 'Llama-3.1-8B-Instruct': 
        params = {
            "model_type": "Llama3",
            "file_prefix": file_prefix,
            "defense": args.defense,
            "test_case_files": test_case_files,
            "model_name": args.model,
        } 
        evaluate_llama(params)
    elif base_model == "vicuna-7b-v1.5":
        params = {
            "model_type": "Vicuna",
            "file_prefix": file_prefix,
            "defense": args.defense,
            "test_case_files": test_case_files,
            "model_name": args.model,
            'prompt_type': "hwchase17_react"
        } 
        evaluate_vicuna(params)
    else:
        raise ValueError(f"Unknown base model: {base_model}")

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def data_preparation(file_prefix, config, data_setting=None, input_files=None):
    logging.info(f"Data Preparation")
    print(f"Data Preparation")
    data_output_file = f"./data/{file_prefix}_training_data.json"
    
    if input_files is None:
        input_files = {
            "dh": f"./InjecAgent/data/test_cases_dh_{data_setting}.json",
            "ds": f"./InjecAgent/data/test_cases_ds_{data_setting}.json",
        }
    
    if config['base_model'] == 'Llama-3.1-8B-Instruct':
        data = get_training_data_llama3(config, input_files, data_output_file)
    elif config['base_model'] == "vicuna-7b-v1.5":
        data = get_training_data_vicuna(config, input_files, data_output_file)
    else:
        raise ValueError(f"Unknown base model: {config['base_model']}")
    return data

def adaptive_attack(args, config, method_class, file_prefix, data, with_eos=False):
    logging.info(f"Training adv string")
    print(f"Training adv string")
    model, tokenizer = load_model(args.model)    
    adaptive_attack_method = method_class(model, tokenizer, **config)

    result_dir = f"./results/{file_prefix}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    for item in tqdm(data):
        logging.info(f"Training adv string for case {item['CaseID']}")
        
        result_file = f"{result_dir}/{item['CaseID']}.json"
        adaptive_attack_method.train_adv_string(item, result_file, with_eos=with_eos)
    return result_dir


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--model', type=str, default='../llm/Llama-3.1-8B-Instruct')
    arguments.add_argument('--defense', type=str, default='InstructionalPrevention')
    arguments.add_argument('--data_setting', type=str, default='base_subset')

    args = arguments.parse_args()
    
    config_dict = init_configs()
    
    # get current time in yymmddhhmmss format
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Configure logging once in a.py
    logging.basicConfig(
        filename=f"./logs/{current_time}.log",  # Log to a file
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print(f"Logging to {logging.getLogger().handlers[0].baseFilename}")
    logging.info(f"Arguments: {args}")
    
    base_model = args.model.split('/')[-1]
    evaluate(base_model, f"{base_model}_{args.defense}_{args.data_setting}", args)

    
    config = config_dict[(base_model, args.defense)]
    logging.info(f"Config: {config}")
    if config['adaptive_attack'] == 'TGCG':
        method_class = get_method_class("GCG")
    else:
        method_class = get_method_class(config['adaptive_attack'])
    
    
   
    
    if config['defense'] == 'Paraphrasing':
        file_prefix = f"{current_time}_{config['adaptive_attack']}_{config['base_model']}_{config['defense']}_{config['adv_string_position_1']}_{config['adv_string_position_2']}_{args.data_setting}_{config['num_steps_1']}_{config['num_steps_2']}"
        logging.info("First step of paraphrasing")
        print("First step of paraphrasing")
        file_prefix_step_1 = file_prefix + '_step_1'
        config_1 = config.copy()
        config_1['adv_string_init'] = config_1['adv_string_init_1']
        config_1['num_steps'] = config_1['num_steps_1']
        config_1['adv_string_position'] = config_1['adv_string_position_1']
        data = data_preparation(file_prefix_step_1, config_1, data_setting=args.data_setting)
        adv_string_result_dir = adaptive_attack(args, config_1, method_class, file_prefix_step_1, data)
        adaptive_attack_files = extract_adv_string(file_prefix_step_1, config_1, adv_string_result_dir, data_setting=args.data_setting)
        
        logging.info("Second step of paraphrasing")
        print("Second step of paraphrasing")
        file_prefix_step_2 = file_prefix + '_step_2'
        config_2 = config.copy()
        config_2['adv_string_init'] = config_2['adv_string_init_2']
        config_2['num_steps'] = config_2['num_steps_2']
        config_2['adv_string_position'] = config_2['adv_string_position_2']
        
        data = data_preparation(file_prefix_step_2, config_2, input_files=adaptive_attack_files)

        adv_string_result_dir = adaptive_attack(args, config_2, method_class, file_prefix_step_2, data, with_eos=True)
        
        adaptive_attack_files = extract_adv_string(file_prefix_step_2, config_2, adv_string_result_dir, input_files=adaptive_attack_files)
        
    else:
        file_prefix = f"{current_time}_{config['adaptive_attack']}_{config['base_model']}_{config['defense']}_{config['adv_string_position']}_{args.data_setting}_{config['num_steps']}"
        data = data_preparation(file_prefix, config, args.data_setting)
        adv_string_result_dir = adaptive_attack(args, config, method_class, file_prefix, data)
        adaptive_attack_files = extract_adv_string(file_prefix, config, adv_string_result_dir, data_setting=args.data_setting)
    
    evaluate(base_model, file_prefix, args, config=config,  adaptive_attack_files=adaptive_attack_files)
    
    
    
    


