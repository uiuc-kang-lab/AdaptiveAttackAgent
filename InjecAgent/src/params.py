import argparse

def define_arguments_prompted(parser):
    parser.add_argument("--model_type", help="Type of the model to evaluate")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default='', required=True)
    parser.add_argument("--setting", help="base or enhanced setting", required=True)
    parser.add_argument("--defense", help="Defense method", choices=['None', 'DataPromptIsolation', 'InstructionalPrevention', 'SandwichPrevention', 'LLMDetector', 'Paraphrasing', 'Paraphrasing_openai', 'ReTokenization', 'PerplexityFiltering', 'FinetunedDetector', 'Paraphraising_InstructionOnly'], default='None')
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react', "new_prompt"], help="prompt type")
    parser.add_argument("--only_first_step", default=False, action='store_true', help="Only predict the first step")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--use_output", default=False, action='store_true', help="Use output from the data file")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    
def define_arguments_get_data_finetuned_tool_use(parser):
    parser.add_argument("--defense", help="Defense method", choices=['None', 'LLMDetector', 'InstructionalPrevention', 'Paraphrasing_InstructionOnly', 'FinetunedDetector'], default='None')
    parser.add_argument("--setting", help="base or enhanced setting", default='advstring_subset')
    
def define_arguments_get_data(parser):
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--defense", help="Defense method", choices=['None', 'DataPromptIsolation', 'InstructionalPrevention', 'SandwichPrevention', 'Paraphrase', 'Paraphrasing_InstructionOnly'], default='None')
    parser.add_argument("--setting", help="base or enhanced setting", default='advstring_subset')

    
def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", action='store_true', help="Only predict the first step")
    
def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    elif agent_type == "get_gcg_data" or agent_type == "get_autodan_data":
        define_arguments_get_data(parser)
    elif agent_type == "get_gcg_data_finetuned_tool_use":
        define_arguments_get_data_finetuned_tool_use(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params