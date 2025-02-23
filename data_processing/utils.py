
def get_tool_response_with_placeholder(item, position):
    if position == 'prefix':
        return '"' + item['Tool Response Template'].replace('<Attacker Instruction>', '<advstring> ' + item['Attacker Instruction']) + '"'
    elif position == 'suffix':
        return '"' + item['Tool Response Template'].replace('<Attacker Instruction>', item['Attacker Instruction'] + ' <advstring>') + '"'
    else:
        raise ValueError(f"Unknown position: {position}")


def get_case(item, model_input, target):
    return {
        "CaseID": f"{item['Attacker Tools']}_{item['User Tool']}",
        "Attacker Tools": item['Attacker Tools'],
        "User Tool": item['User Tool'],
        "Attacker Instruction": item['Attacker Instruction'],
        "before_adv_string": model_input.split("<advstring>")[0],
        "after_adv_string": model_input.split("<advstring>")[1],
        "Target": target
    } 
    
def get_case_multi_objective(item, model_input, detection_model_input, target_1, target_2):
    return {
        "CaseID": f"{item['Attacker Tools']}_{item['User Tool']}",
        "Attacker Tools": item['Attacker Tools'],
        "User Tool": item['User Tool'],
        "Attacker Instruction": item['Attacker Instruction'],
        "before_adv_string_1": model_input.split("<advstring>")[0],
        "after_adv_string_1": model_input.split("<advstring>")[1],
        "Target_1": target_1,
        "before_adv_string_2": detection_model_input.split("<advstring>")[0],
        "after_adv_string_2": detection_model_input.split("<advstring>")[1],
        "Target_2": target_2
    } 
    
def get_case_finetuned_detector(item, model_input, tool_response, target):
    return {
        "CaseID": f"{item['Attacker Tools']}_{item['User Tool']}",
        "Attacker Tools": item['Attacker Tools'],
        "User Tool": item['User Tool'],
        "Attacker Instruction": item['Attacker Instruction'],
        "before_adv_string": model_input.split("<advstring>")[0],
        "after_adv_string": model_input.split("<advstring>")[1],
        "Target": target,
        "tool_output_before": tool_response.split("<advstring>")[0],
        "tool_output_after": tool_response.split("<advstring>")[1]
    } 
    