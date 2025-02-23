import json
from tqdm import tqdm
import pandas as pd
from InjecAgent.src.params import parse_arguments
from InjecAgent.src.utils import get_tool_dict
from InjecAgent.src.prompts.llm_detector import LLM_DETECTOR_PROMPT, LLM_DETECTOR_EXAMPLE
from InjecAgent.src.prompts.llm_paraphrasing import LLM_PARAPHRASING_PROMPT
import jinja2
import logging
from data_processing.utils import get_tool_response_with_placeholder, get_case, get_case_multi_objective, get_case_finetuned_detector

# load template from InjecAgent.src.prompts.jinja.llama_tool_use.jinja
env = jinja2.Environment(loader=jinja2.FileSystemLoader('./InjecAgent/src/prompts/jinja'))
agent_template = env.get_template('llama_tool_use_new_prompt.jinja')
agent_template_with_instructional_prevention = env.get_template('llama_tool_use_instruction_prevention_new_prompt.jinja')
call_tool_template = env.get_template('call_tool.jinja')
tool_response_template = env.get_template('tool_response.jinja')

llama3_template = """<|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


    



def get_one_case_no_defense(params, item, tool_dict):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    user_instruction = item['User Instruction']    
    
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
        
    model_input = agent_template.render(
        available_tool_descriptions = available_tool_descriptions,
        user_instruction = user_instruction,
        call_of_user_tool = call_tool_template.render(tool_name = item['User Tool'], tool_parameters = item['Tool Parameters']),
        response_of_user_tool = tool_response,
    )
    return get_case(item, model_input, '\n\n{"name": "'+ item['Attacker Tools'][0] + '",')

def get_one_case_instructional_prevention(params, item, tool_dict):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    user_instruction = item['User Instruction']
    
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
        
    
    model_input = agent_template_with_instructional_prevention.render(
        available_tool_descriptions = available_tool_descriptions,
        user_instruction = user_instruction,
        call_of_user_tool = call_tool_template.render(tool_name = item['User Tool'], tool_parameters = item['Tool Parameters']),
        response_of_user_tool = tool_response,
    )
    return get_case(item, model_input, '\n\n{"name": "'+ item['Attacker Tools'][0] + '",')

def predict_one_case_llm_detector(params, item, tool_dict):
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
    
    filled_detection_prompt = LLM_DETECTOR_PROMPT.format(llm_detector_example = LLM_DETECTOR_EXAMPLE, user_instruction = item['User Instruction'], tool_description = json.dumps(tool_dict[item['User Tool']], indent=True), tool_response = tool_response)

    detection_model_input = llama3_template.format(user_prompt=filled_detection_prompt)
    
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    user_instruction = item['User Instruction']
        
    model_input = agent_template.render(
        available_tool_descriptions = available_tool_descriptions,
        user_instruction = user_instruction,
        call_of_user_tool = call_tool_template.render(tool_name = item['User Tool'], tool_parameters = item['Tool Parameters']),
        response_of_user_tool = tool_response,
    )
    training_case = get_case_multi_objective(item, model_input, detection_model_input, '\n\n{"name": "'+ item['Attacker Tools'][0] + '",', "\n\nNo")
    return training_case    

def predict_one_case_finetuned_detector(params, item, tool_dict):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    user_instruction = item['User Instruction']
        
    tool_response = get_tool_response_with_placeholder(item, params['adv_string_position'])
        
    model_input = agent_template.render(
        available_tool_descriptions = available_tool_descriptions,
        user_instruction = user_instruction,
        call_of_user_tool = call_tool_template.render(tool_name = item['User Tool'], tool_parameters = item['Tool Parameters']),
        response_of_user_tool = tool_response,
    )
    return get_case_finetuned_detector(item, model_input, tool_response, '\n\n{"name": "'+ item['Attacker Tools'][0] + '",')

def get_one_case_paraphrase_step_2(params, item, tool_dict):
    filled_paraphrasing_prompt = LLM_PARAPHRASING_PROMPT.format(text = '<advstring> ' + item['Attacker Input'] if params['adv_string_position'] == 'prefix' else item['Attacker Input'] + ' <advstring>')
    paraphrasing_model_input = llama3_template.format(user_prompt = filled_paraphrasing_prompt)
    return get_case(item, paraphrasing_model_input, '\n\n' + item['Attacker Input'])

DEFENSE_METHODS = {
    "Paraphrasing": get_one_case_no_defense,
    "Paraphrasing_step_2": get_one_case_paraphrase_step_2,
    "InstructionalPrevention": get_one_case_instructional_prevention,
    "LLMDetector": predict_one_case_llm_detector,
    "FinetunedDetector": predict_one_case_finetuned_detector
}
def get_training_data_llama3(params, input_files, data_output_file = None):
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
    params = parse_arguments(agent_type="get_gcg_data_finetuned_tool_use")

    get_training_data_llama3(params)