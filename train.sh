#! /usr/bin/env bash
model_path="models/$1"
mkdir -p $model_path
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u src/train.py \
    --model_path $model_path \
    --alpha 0.9 \
    --beta 0.999 \
    --learning_rate 1e-4 \
    --embed_size 300 \
    --encoder_attention_heads 6 \
    --encoder_layers 6 \
    --encoder_ffn_embed_dim 512 \
    --decoder_attn_embed_size 100 \
    --hidden_size 512 \
    --batch_size 64 \
    --eval_size 64 \
    --vocab_size 43648 \
    --max_source_positions 20 \
    --max_target_positions 44 \
    --clip 5 \
    --l2_rate 1e-5 \
    --dropout 0.2 \
    --attention_dropout 0.2 \
    --relu_dropout 0.2 \
    --num_epochs 150 \
    --log_step 10 \
    --encoder_learned_pos \
    --seed 1024 2>&1 | tee $model_path/training_log.txt
    #--pretrained $model_path/adaptive-best.pkl \
