<p align="center">
<img src=images/claude-alpaca.png  width="80%" height="60%">
</p>

# Claude2-Alpaca: Instruction tuning datasets distilled from claude

This is the repo for the Claude2-Alpaca project, which aims to build and share an instruction-following LLaMA model. The repo contains 52k prompts and responses. The repo contains:
- The 52k claude-2 data used for finetuning
- The code for generating the data 
- The code for finetuning 7B and 13B models

## Overview
The current open-sourced instruction tuning datasets usually distilled the data from the GPT-families, e.g., WizardLM distills the data from GPT-3.5-turbo; GPT4LLM distills the data from GPT-4 and Alpaca distills the data from the Text-Davinci-003. We would like to increase the diversity of the instruction tuning dataset and provide the community with more options!

In this repo, we use the same Alpaca 52k prompts to query the [Claude-2](https://www.anthropic.com/index/claude-2) and obtain the claude2-alpaca dataset. We also include the instruction-tuned LLaMA-2 models, training code for re-implementation, and the results.


## Data Generation
```
export ANTHROPIC_API_KEY=xxx # export your claude key here
python generate_data.py
```

TODO: 
- Follow the alpaca repo and write the guidance for using our repo.
- Release the data and make it public on Huggingface.
- Evaluation of the model. Alpaca-Eval as well as some benchmarks. (huggingface OPEN-LLM leaderboard or llm-harness repos)
- Investigate the bias of the current model-based evaluations. GPT-4 and Claude-2 may have preference to its distilled models.
- Transfer Attack
- The synergy of different models distilled from different sources.
- Design the logo for our project.
- Project Page
- Could have more .....
