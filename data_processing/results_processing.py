import os
import json
import pandas as pd
from tqdm import tqdm
import logging
def get_training_data(input_files):
    test_cases_dh_file = input_files['dh']
    test_cases_ds_file = input_files['ds']
    
    dh_case_dict = {}
    ds_case_dict = {}
    dh_cases = json.load(open(test_cases_dh_file))
    ds_cases = json.load(open(test_cases_ds_file))
    
    for case_index, case in enumerate(dh_cases):
        index_name = f"{case['Attacker Tools']}_{case['User Tool']}"
        dh_case_dict[index_name] = case

    for case_index, case in enumerate(ds_cases):
        index_name = f"{case['Attacker Tools']}_{case['User Tool']}"
        ds_case_dict[index_name] = case
        
    return dh_case_dict, ds_case_dict


def extract_adv_string(file_prefix, config, adv_string_result_dir, data_setting=None, input_files=None):
    logging.info(f"Extracting adv string")
    print(f"Extracting adv string")
    if input_files is None:
        input_files = {
            "dh": f'InjecAgent/data/test_cases_dh_{data_setting}.json',
            "ds": f'InjecAgent/data/test_cases_ds_{data_setting}.json',
        }
    dh_case_dict, ds_case_dict = get_training_data(input_files)

    cnt_dh = 0
    cnt_ds = 0
    all_losses = []
    for filename in tqdm(os.listdir(adv_string_result_dir)):
        if 'GmailSendEmail' in filename:
            current_case_dict = ds_case_dict
            cnt_ds += 1
        else:
            current_case_dict = dh_case_dict
            cnt_dh += 1
        cur_index = filename[:-len('.json')]
        original_test_case = current_case_dict[cur_index]
        
        
        string_with_min_loss = None
        min_loss = 1000000
        with open(os.path.join(adv_string_result_dir, filename), 'r') as f:
            for line in f:
                tmp = json.loads(line)
                if tmp['loss'] < min_loss:
                    string_with_min_loss = tmp
                    min_loss = tmp['loss']
                    
        adv_string = string_with_min_loss['adv_string']
        
    

        if config["adv_string_position"] == "prefix":
            if 'Paraphrasing' in file_prefix and 'step_2' in file_prefix:
                attacker_input = ' ' + adv_string + ' ' + original_test_case['Attacker Input']
            else:
                attacker_input = ' ' + adv_string + ' ' + original_test_case['Attacker Instruction']
            tool_response = '"' + original_test_case['Tool Response Template'].replace('<Attacker Instruction>', attacker_input) + '"'
        elif config["adv_string_position"] == "suffix":
            if 'Paraphrasing' in file_prefix and 'step_2' in file_prefix:
                attacker_input = original_test_case['Attacker Input'] + ' ' + adv_string + ' '
            else:
                attacker_input = original_test_case['Attacker Instruction'] + '   ' + adv_string + ' '
            tool_response = '"' + original_test_case['Tool Response Template'].replace('<Attacker Instruction>', attacker_input) + '"'
            
        original_test_case['Tool Response'] = tool_response
        original_test_case['Adv String'] = adv_string
        original_test_case['Attacker Input'] = attacker_input
        original_test_case['Loss'] = min_loss
        all_losses.append(min_loss)
    output_file_name_dh = f"./InjecAgent/adaptive_attack_results/{file_prefix}_dh_data.json"
    output_file_name_ds = f"./InjecAgent/adaptive_attack_results/{file_prefix}_ds_data.json"
    json.dump(list(dh_case_dict.values()), open(output_file_name_dh, 'w'), indent=4)
    json.dump(list(ds_case_dict.values()), open(output_file_name_ds, 'w'), indent=4)

    logging.info(f"Write results to {output_file_name_dh} and {output_file_name_ds}")
    logging.info(f"Extracted {cnt_dh} dh cases and {cnt_ds} ds cases")
    logging.info(f"Average loss: {sum(all_losses) / len(all_losses)}")
    logging.info(f"Max loss: {max(all_losses)}")
    logging.info(f"Min loss: {min(all_losses)}")

    return {"dh": output_file_name_dh, "ds": output_file_name_ds}

if __name__ == '__main__':
    extract_adv_string("250221174626_TGCG_Llama-3.1-8B-Instruct_Paraphrasing_prefix_prefix_base_subset_1_10_50_step_2", {"adv_string_position": "prefix"}, "results/250221174626_TGCG_Llama-3.1-8B-Instruct_Paraphrasing_prefix_prefix_base_subset_1_10_50_step_2", input_files={"dh": "InjecAgent/adaptive_attack_results/250221174626_TGCG_Llama-3.1-8B-Instruct_Paraphrasing_prefix_prefix_base_subset_1_10_50_step_1_dh_data.json", "ds": "InjecAgent/adaptive_attack_results/250221174626_TGCG_Llama-3.1-8B-Instruct_Paraphrasing_prefix_prefix_base_subset_1_10_50_step_1_ds_data.json"})