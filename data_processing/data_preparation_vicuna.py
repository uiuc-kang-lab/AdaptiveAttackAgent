import json
from tqdm import tqdm
import pandas as pd
from InjecAgent.src.params import parse_arguments
from InjecAgent.src.prompts.agent_prompts import PROMPT_DICT
from InjecAgent.src.prompts.llm_detector import LLM_DETECTOR_PROMPT, LLM_DETECTOR_EXAMPLE
from InjecAgent.src.prompts.llm_paraphrasing import LLM_PARAPHRASING_PROMPT
from InjecAgent.src.utils import get_tool_dict
from data_processing.utils import get_tool_response_with_placeholder, get_case, get_case_multi_objective, get_case_finetuned_detector
import logging


vicuna_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {system_prompt}\n{user_prompt} ASSISTANT:"


def get_one_case_no_defense(params, item, tool_dict):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {tool_response}\n"
    
    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    
    model_input = vicuna_template.format(system_prompt = system_prompt, user_prompt = user_prompt_filled)
    
    return get_case(item, model_input, f" Thought: I will use the {item['Attacker Tools'][0]} tool to")

def get_one_case_data_prompt_isolation(params, item, tool_dict):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: \'\'\'{tool_response}\'\'\'\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    
    model_input = vicuna_template.format(system_prompt = system_prompt, user_prompt = user_prompt_filled)
    
    return get_case(item, model_input, f" Thought: I will use the {item['Attacker Tools'][0]} tool to")


def get_one_case_sandwich_prevention(params, item, tool_dict):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type'] + '_sandwich_prevention']
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {tool_response}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    
    model_input = vicuna_template.format(system_prompt = system_prompt, user_prompt = user_prompt_filled)
    
    return get_case(item, model_input, f" Thought: I will use the {item['Attacker Tools'][0]} tool to")

def get_one_case_instructional_prevention(params, item, tool_dict):
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type'] + '_instructional_prevention']
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {tool_response}\n"
    
    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    
    model_input = vicuna_template.format(system_prompt = system_prompt, user_prompt = user_prompt_filled)
    
    return get_case(item, model_input, f" Thought: I will use the {item['Attacker Tools'][0]} tool to")

def predict_one_case_llm_detector(params, item, tool_dict):
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    filled_detection_prompt = LLM_DETECTOR_PROMPT.format(llm_detector_example = LLM_DETECTOR_EXAMPLE, user_instruction = item['User Instruction'], tool_description = json.dumps(tool_dict[item['User Tool']], indent=True), tool_response = tool_response)
    
    detection_model_input = vicuna_template.format(system_prompt = "", user_prompt = filled_detection_prompt)
    
    
    
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    
    input = item['User Instruction']
    
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {tool_response}\n"
    
    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    
    model_input = vicuna_template.format(system_prompt = system_prompt, user_prompt = user_prompt_filled)
    
    
    return get_case_multi_objective(item, model_input, detection_model_input, f" Thought: I will use the {item['Attacker Tools'][0]} tool to", " No")    

def predict_one_case_finetuned_detector(params, item, tool_dict):
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    
    input = item['User Instruction']
    
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {tool_response}\n"
    
    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    
    model_input = vicuna_template.format(system_prompt = system_prompt, user_prompt = user_prompt_filled)
    
    return get_case_finetuned_detector(item, model_input, tool_response, f" Thought: I will use the {item['Attacker Tools'][0]} tool to")

def get_one_case_paraphrase_step_2(params, item, tool_dict):
    filled_paraphrasing_prompt = LLM_PARAPHRASING_PROMPT.format(text = '<advstring> ' + item['Attacker Input'] if params['adv_string_position'] == 'prefix' else item['Attacker Input'] + ' <advstring>')
    paraphrasing_model_input = vicuna_template.format(system_prompt = '', user_prompt = filled_paraphrasing_prompt)
    return get_case(item, paraphrasing_model_input, target = ' ' + item['Attacker Input'])

DEFENSE_METHODS = {
    "Paraphrasing": get_one_case_no_defense,
    "Paraphrasing_step_2": get_one_case_paraphrase_step_2,
    "InstructionalPrevention": get_one_case_instructional_prevention,
    "DataPromptIsolation": get_one_case_data_prompt_isolation,
    "SandwichPrevention": get_one_case_sandwich_prevention,
    "LLMDetector": predict_one_case_llm_detector,
    "FinetunedDetector": predict_one_case_finetuned_detector
}
def get_training_data_vicuna(params, input_files, data_output_file = None):
    params['prompt_type'] = "hwchase17_react"
    if params['defense'] == 'Paraphrasing' and 'step_2' in data_output_file:
        get_one_case = DEFENSE_METHODS['Paraphrasing_step_2']
    else:
        get_one_case = DEFENSE_METHODS[params['defense']]
    tool_dict = get_tool_dict()
    overall_output = []
    for attack in ['dh', 'ds']:
        input_file = input_files[attack]
        with open(input_file, 'r') as f:
            data = json.load(f)
        for item in tqdm(data):
            overall_output.append(get_one_case(params, item, tool_dict))
    # save overall_output in to output file
    with open(data_output_file, 'w') as f:
        json.dump(overall_output, f, indent=4)
    return overall_output
if __name__ == "__main__":
    params = parse_arguments(agent_type="get_gcg_data")

    get_training_data_vicuna(params)