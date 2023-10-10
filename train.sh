python train.py \
--init_checkpoint_path xxx \
--model_config_path xxx \
--checkpoint_path xxx \
--wrapped_class_name LlamaDecoderLayer \
--data_path ./claude2-alpaca-52k.json \
--hack --filtering_method random \
--dont_save_opt --num_epochs 3 \
--lr 2e-5 --data_fraction 1.00 \ 
--batch_size 1 --accumulation_steps 8 \ 
--wandb --wb_name llama2_13B_claude_alpaca

alpaca_eval --model_outputs llama2_7B_claude_alpaca.json --annotators_config 'claude'
alpaca_eval --model_outputs llama2_13B_claude_alpaca.json --annotators_config 'claude'