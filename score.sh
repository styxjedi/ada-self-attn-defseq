#! /usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
model_path=./models/$1
epoch=$2

python ./metrics/rerank.py\
    $model_path/greedy_output.txt\
    ./data/lm.bin\
    ./data/function_words.txt\
    $model_path/greedy_output_rank.txt

# python src/score.py\
#     --gen_file_path $model_path/greedy_output_rank.txt\
#     --pretrained $model_path/adaptive-$2.pkl\
#     --embed_size 300\
#     --encoder_attention_heads 5 \
#     --encoder_layers 6 \
#     --encoder_ffn_embed_dim 512 \
#     --decoder_attn_embed_size 100 \
#     --vocab_size 43648 \
#     --max_source_positions 20 \
#     --max_target_positions 44 \
#     --encoder_learned_pos \
#     --hidden_size 300\
#     --seed 1024\
#     --output_path $model_path/score_greedy_output_rank.txt
# 
# python ./metrics/rerank2.py\
#     $model_path/greedy_output_rank.txt\
#     $model_path/score_greedy_output_rank.txt\
#     ./data/function_words.txt\
#     $model_path/greedy_output_rank2.txt.top

python ./metrics/bleu.py\
    ./data/test.txt\
    $model_path/greedy_output_rank.txt.top
