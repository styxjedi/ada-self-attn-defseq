#! /usr/bin/env bash
model_path="models/$1"
mkdir -p $model_path
CUDA_VISIBLE_DEVICES=0 python -u src/train.py \
    --model_path $model_path \
    --alpha 0.9 \
    --beta 0.999 \
    --learning_rate 1e-3 \
    --embed_size 300 \
    --encoder_attention_heads 5 \
    --encoder_layers 6 \
    --encoder_ffn_embed_dim 512 \
    --decoder_attn_embed_size 100 \
    --hidden_size 300 \
    --test_epoch 20 \
    --warmup_epochs 10 \
    --warmup_lr 1e-7 \
    --batch_size 128 \
    --eval_size 128 \
    --vocab_size 43648 \
    --max_source_positions 20 \
    --max_target_positions 44 \
    --clip 0.1 \
    --l2_rate 1e-4 \
    --dropout 0.2 \
    --attention_dropout 0.2 \
    --relu_dropout 0.2 \
    --num_epochs 100 \
    --log_step 10 \
    --encoder_learned_pos \
    --model_name $1 \
    --output_path $model_path/greedy_output.txt \
    --beam_size 1 \
    --test_path data/processed/test.npz \
    --test_size 50 \
    --temperature 1.0 \
    --seed 1024 2>&1 | tee $model_path/training_log.txt
    #--pretrained $model_path/adaptive-19.pkl \
    
