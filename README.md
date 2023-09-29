# claude2-alpaca

This is the repo for the Claude2-Alpaca project, which aims to build and share an instruction-following LLaMA model. The repo contains 52k prompts and responses. The repo contains:
- The 52k claude-2 data used for finetuning
- The code for generating the data 
- The code for finetuning 7B and 13B models

## Overview
The current Alpaca model distills the data from the Text-Davinci-003. We would like to have the 

TODO: 
- Follow the alpaca repo and write the guidance for using our repo.
- Release the data and make it public on Huggingface.
- Evaluation of the model. Alpaca-Eval as well as some benchmarks. (huggingface OPEN-LLM leaderboard or llm-harness repos)
- Investigate the bias of the current model-based evaluations. GPT-4 and Claude-2 may have preference to its distilled models.
- Transfer Attack
- Project Page
- The synergy of different models distilled from different sources.
- Design the logo for our project.
- Could have more .....
