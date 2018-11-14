#! /usr/bin/env bash
model_path="./models/$1"
mkdir -p $model_path
epoch=$2
CUDA_VISIBLE_DEVICES=1 python src/inference.py \
    --pretrained $model_path/adaptive-$epoch.pkl \
    --output_path $model_path/greedy_output.txt \
    --embed_size 300 \
    --encoder_attention_heads 5 \
    --encoder_layers 6 \
    --encoder_ffn_embed_dim 512 \
    --decoder_attn_embed_size 100 \
    --vocab_size 43648 \
    --max_source_positions 20 \
    --max_target_positions 44 \
    --encoder_learned_pos \
    --hidden_size 300 \
    --dropout 0.2 \
    --seed 1024 \
    --beam_size 1 \
    --test_path data/processed/test.npz \
    --test_size 50 \
    --temperature 1.0 \
    --max_len 50
