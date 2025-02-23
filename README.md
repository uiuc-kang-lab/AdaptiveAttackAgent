# Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents

<span>
   <a href='https://arxiv.org/abs/2403.02691'>
      <img src='https://img.shields.io/badge/Arxiv-2403.02691-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
   <a href='blog'>
      <img src='https://img.shields.io/badge/Blog-Medium-black'></a>
</span>


This repository contains the official code for the paper ["Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents,"](arxiv.org) accepted to NAACL 2025 Findings. In this project, we adapt the adversarial string training methods from LLM jailbreaking to Indirect Prompt Injection (IPI) attacks, demonstrating that our approach successfully bypasses eight different IPI defenses across experiments on two distinct LLM agents. Our research underscores the need for adaptive attack evaluation when designing defenses to ensure robustness and reliability.


<p align="center">
  <img src="asset/overview.png" alt="Overview" width="500px" />
</p>




## Set up

```sh
git clone https://github.com/uiuc-kang-lab/AdaptiveAttackAgent.git
cd AdaptiveAttackAgent
pip install -r requirements.txt
export OPENAI_API_KEY=Your_OpenAI_key
```

## Usage

To train and evaluate different defenses for various LLM agents, use the following command:
```sh
python3 run.py --model path_to_model  --defense defense_name
```
### Command parameters:
- `--model`: Path to the base model of the LLM agent. We currently support two base models: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5). The code can be easily adapted to other base models.
- `--defense`: Name of the defense to evaluate and attack adaptively. 
For [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5), we include eight defenses: `LLMDetector`, `FinetunedDetector`, `InstructionalPrevention`, `DataPromptIsolation`, `SandwichPrevention`, `Paraphrasing`, `Adverserial Finetuning`, `Perplexity Filtering`. 
For [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), we exclude `DataPromptIsolation` and `SandwichPrevention`.

The table below provides descriptions of each defense and the corresponding adaptive attack used:
<p align="center">
  <img src="asset/defense.png" alt="Defense" width="700px" />
</p>


## Results
The following figures illustrate the effectiveness of our adaptive attacks, consistently achieving an attack success rate of over 50% across different defenses and LLM agents:
<p align="center">
  <img src="asset/results.png" alt="Results" width="700px" />
</p>

## Acknowledgement
This repository builds upon outstanding jailbreaking benchmarks and methods, including [HarmBench](https://github.com/centerforaisafety/HarmBench), [GGC](https://github.com/llm-attacks/llm-attacks), [AutoDan](https://github.com/rotaryhammer/code-autodan). It also leverages the IPI benchmark [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent).



