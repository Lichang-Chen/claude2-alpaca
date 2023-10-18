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


## Training
We incldue the training script for 7B and 13B models:
```
# make sure the path in the train.sh is correct (use your own path to llama-2's weight and the output path.)
bash train.sh
```


## Data Generation
```
export ANTHROPIC_API_KEY=xxx # export your claude key here
python generate_data.py
```
## Results
We use the generated data to fine-tune 7B/13B LLaMA-2 models and show the results here:

|                    | Average | ARC  | HellaSwag | MMLU  | TruthfulQA | Alpaca_Eval | Avg Length |
|---|---|---|---|---|---|---|---|
| Llama-2-7b-chat | 56.335  | 52.9 | 78.55     | 48.32 | 45.57      | 71.37       | 1479       |
| Llama-2-13b-chat   | 59.935  | 59.04| 81.94     | 54.64 | 44.12      | 81.09       | 1513       |
| claude_alpaca-7b  | 57.78   | 56.66 | 81.17     | 46.58 | 46.71      | 71.23       | 1066       |
| claude_alpaca-13b | 61.29   | 61.18 | 84.08     | 55.74 | 44.18      | 78.93       | 1127       |

Compared to the llama2-chat, our models can have better averaged performance.


## Authors
All grad students below contributed equally.
- [Lichang Chen](https://lichang-chen.github.io/)
- [Khalid Saifullah](https://khalidsaifullaah.github.io/)
- [Ming Li](https://github.com/MingLiiii)

Special thanks to [Ping-yeh Chiang](https://github.com/Ping-C) for sharing their FSDP model fine-tuning script, which we utilized in this project.

## Citation
Please cite the repo if you use the data or code in this repo.
```
@misc{claude2-alpaca,
  author = {Lichang Chen and Khalid Saifullah and Ming Li and Tianyi Zhou and Heng Huang},
  title = {Claude2-Alpaca: Instruction tuning datasets distilled from claude},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Lichang-Chen/claude2-alpaca}},
}
```



## TODO: 
We include the TODO list for our project. If you also are interested in the following research project, do not hesitate to contact with us (we are open to any kinds of collaborations)!
- Investigate the bias of the current model-based evaluations. GPT-4 and Claude-2 may have preference to its distilled models.
- Transfer Attack
- The synergy of different models distilled from different sources.
- Project Page
