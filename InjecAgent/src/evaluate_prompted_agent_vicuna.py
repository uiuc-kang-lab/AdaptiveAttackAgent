import os
import json
import logging
from tqdm import tqdm 
from InjecAgent.src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from InjecAgent.src.params import parse_arguments
from InjecAgent.src.prompts.agent_prompts import PROMPT_DICT
from InjecAgent.src.prompts.llm_detector import LLM_DETECTOR_PROMPT, LLM_DETECTOR_EXAMPLE
from InjecAgent.src.prompts.llm_paraphrasing import LLM_PARAPHRASING_PROMPT
from InjecAgent.src.output_parsing import evaluate_output_prompted
from InjecAgent.src.models import MODELS
import torch
from transformers import pipeline
import openai
openai_client = openai.Client(api_key = os.environ.get("OPENAI_API_KEY"))

def predict_one_case_no_defense(params, item, model_class, tool_dict, step_2=False):
    if params['use_output']:
        if step_2:
            return {"output": item['output Step 2']}
        else:
            return {"output": item['output']}
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {"output": output}

def predict_one_case_data_prompt_isolation(params, item, model_class, tool_dict, step_2=False):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{item['Tool Response']}\'\'\'\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: \'\'\'{item['Attacker Tool Response']}\'\'\'\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{item['Tool Response']}\'\'\'\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {"output": output}


def predict_one_case_llm_detector(params, item, model_class, tool_dict, step_2=False):
    # check if the attack is detected
    filled_detection_prompt = LLM_DETECTOR_PROMPT.format(llm_detector_example = LLM_DETECTOR_EXAMPLE, user_instruction = item['User Instruction'], tool_description = json.dumps(tool_dict[item['User Tool']], indent=True), tool_response = item['Tool Response'])
    detection_model_input = model_class.prepare_input('', filled_detection_prompt)
    detection_output = model_class.call_model(detection_model_input)
    if 'Yes' in detection_output or 'yes' in detection_output:
        return {"output":'unsuccessful attack'}

    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{item['Tool Response']}\'\'\'\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: \'\'\'{item['Attacker Tool Response']}\'\'\'\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{item['Tool Response']}\'\'\'\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {"output": output}

def predict_one_case_instructional_prevention(params, item, model_class, tool_dict, step_2=False):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type'] + '_instructional_prevention']
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {"output": output}

def predict_one_case_sandwich_prevention(params, item, model_class, tool_dict, step_2=False):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type'] + '_sandwich_prevention']
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {"output": output}

def predict_one_case_paraphrasing(params, item, model_class, tool_dict, step_2=False):
    if 'Attacker Input' in item:
        filled_paraphrasing_prompt = LLM_PARAPHRASING_PROMPT.format(text = item['Attacker Input'])
    else:
        filled_paraphrasing_prompt = LLM_PARAPHRASING_PROMPT.format(text = item['Attacker Instruction'])
    paraphrasing_model_input = model_class.prepare_input('', filled_paraphrasing_prompt)
    paraphrased_instruction = model_class.call_model(paraphrasing_model_input)
    paraphrased_tool_response = item['Tool Response Template'].replace('<Attacker Instruction>', paraphrased_instruction)
    
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {paraphrased_tool_response}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {paraphrased_tool_response}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {
        "output": output,
        "Paraphrased Tool Response": paraphrased_tool_response
    }



def predict_one_case_perplexity_filtering(params, item, model_class, tool_dict, step_2=False, perplexity_threshold=68, cache_output=None):
    # get the perplexity of the tool response
    input_ids = model_class.tokenizer.encode(item['Tool Response'], return_tensors='pt').to(model_class.model.device)
    with torch.no_grad():
        loss = model_class.model(input_ids, labels=input_ids).loss
        perplexity = torch.exp(loss)
    

    
    log_perplexity = loss.item()
    perplexity = perplexity.item()
    if perplexity > perplexity_threshold:
        return {'output': 'unsuccessful attack', 'LogPerplexity': log_perplexity, 'Perplexity': perplexity}
    
    if params['use_output']:
        if step_2:
            return {"output": item['output Step 2'],"LogPerplexity": log_perplexity, "Perplexity": perplexity}
        else:
            return {"output": item['output'],"LogPerplexity": log_perplexity, "Perplexity": perplexity}
    
    if cache_output and cache_output != 'unsuccessful attack':
        if step_2:
            return {"output": cache_output,"LogPerplexity": log_perplexity, "Perplexity": perplexity}
        else:
            return {"output": cache_output,"LogPerplexity": log_perplexity, "Perplexity": perplexity}
    
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {
        "output": output,
        "LogPerplexity": log_perplexity, 
        "Perplexity": perplexity
    }

def predict_one_case_finetuned_detector(params, item, model_class, tool_dict, detector, step_2=False):
    results = detector(item['Tool Response'])[0]
    if results['label'] == 'INJECTION':
        return {"output": 'unsuccessful attack', "Detector Results": results}

    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{item['Tool Response']}\'\'\'\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: \'\'\'{item['Attacker Tool Response']}\'\'\'\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{item['Tool Response']}\'\'\'\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return {"output": output, "Detector Results": results}

DEFENSE_METHODS = {
    "None": predict_one_case_no_defense,
    "DataPromptIsolation": predict_one_case_data_prompt_isolation,
    "InstructionalPrevention": predict_one_case_instructional_prevention,
    "SandwichPrevention": predict_one_case_sandwich_prevention,
    "LLMDetector": predict_one_case_llm_detector,
    "PerplexityFiltering": predict_one_case_perplexity_filtering,
    "FinetunedDetector": predict_one_case_finetuned_detector,
    'Paraphrasing': predict_one_case_paraphrasing
}

def evaluate_vicuna(params):
    logging.info(f"Start evaluating vicuna with params: {params}")
    print(f"Start evaluating vicuna with params: {params}")
    
    output_dir = f"./InjecAgent/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_files = {}
    for attack in ['dh', 'ds']:
        output_file = f"{output_dir}/{params['file_prefix']}_{attack}_eval_results.json"
        output_files[attack] = output_file
        
        
    if all([os.path.exists(output_files[attack]) for attack in ['dh', 'ds']]):
        logging.info("All output files exist. Skip evaluation.")
        scores = get_score(output_files)
        logging.info(f"Scores: {scores}")
        return
    
    model_class = MODELS[params['model_type']](params)
    predict_one_case = DEFENSE_METHODS[params['defense']]
    
    if params['defense'] == 'FinetunedDetector':
        pipe = pipeline("text-classification", model="../deberta-v3-base-prompt-injection-v2")
    
    tool_dict = get_tool_dict()
    for attack in ['dh', 'ds']:
        test_case_file = params['test_case_files'][attack]
        with open(test_case_file, 'r') as f:
            data = json.load(f)
            
        with open(output_files[attack], 'w') as f:
            for item in tqdm(data):
                if params['defense'] == 'FinetunedDetector':
                    output_dict = predict_one_case(params, item, model_class, tool_dict, pipe)
                else:
                    output_dict = predict_one_case(params, item, model_class, tool_dict)
                for key in output_dict:
                    item[key] = output_dict[key]            
                parsed_output = evaluate_output_prompted(item['output'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                item['eval'] = parsed_output['eval']
                if item['eval'] == 'invalid':
                    item['invalid_reason'] = parsed_output['invalid_reason']
                
                if item['eval'] == 'succ' and attack == 'ds':
                    item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
                    item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
                    item["Attacker Tool Response"] = get_simulated_attacker_tool_response(item['Attacker Tools'][0], item["Attacker Tool Parameters"], tool_dict)
                    
                    
                    if params['defense'] == 'FinetunedDetector':
                        output_dict = predict_one_case(params, item, model_class, tool_dict, pipe, step_2=True)
                    else:
                        output_dict = predict_one_case(params, item, model_class, tool_dict, step_2=True)
                    for key in output_dict:
                        item[key + ' Step 2'] = output_dict[key]
                    
                    parsed_output = evaluate_output_prompted(item['output Step 2'], item['Attacker Tools'][1], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                    item['eval Step 2'] = parsed_output['eval']
                    if item['eval Step 2'] == 'invalid':
                        item['invalid_reason  Step 2'] = parsed_output['invalid_reason']
                
                f.write(json.dumps(item)+'\n')

    
    scores = get_score(output_files)
    print(f"Scores: {scores}")
    logging.info(f"Scores: {scores}")

if __name__ == "__main__":

    params = parse_arguments(agent_type="prompted")

    evaluate_vicuna(params)
    
    
    
    
