# python train.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ../llama2_7B_claude_alpaca \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path ./claude2-alpaca-52k.json \
# --hack --filtering_method random --dont_save_opt --num_epochs 3 --lr 2e-5 --data_fraction 1.00 --batch_size 1 --accumulation_steps 8 --wandb --wb_name llama2_7B_claude_alpaca
# python eval_generate.py --batch_size 8 --sharded_model ../llama2_7B_claude_alpaca --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name llama2_7B_claude_alpaca.json

python train.py \
--init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded \
--model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf \
--checkpoint_path ../llama2_13B_claude_alpaca \
--wrapped_class_name LlamaDecoderLayer \
--data_path ./claude2-alpaca-52k.json \
--hack --filtering_method random --dont_save_opt --num_epochs 3 --lr 2e-5 --data_fraction 1.00 --batch_size 1 --accumulation_steps 8 --wandb --wb_name llama2_13B_claude_alpaca
python eval_generate.py --batch_size 8 --sharded_model ../llama2_13B_claude_alpaca --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/  --file_path alpaca_eval --save_file_name llama2_13B_claude_alpaca.json

alpaca_eval --model_outputs llama2_7B_claude_alpaca.json --annotators_config 'claude'
alpaca_eval --model_outputs llama2_13B_claude_alpaca.json --annotators_config 'claude'

cp -r ../llama2_13B_claude_alpaca/ /sensei-fs/users/ksaifullah/