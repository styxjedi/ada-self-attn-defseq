#! /usr/bin/env bash
python preprocess/prep_datasets.py
python preprocess/prep_w2v.py\
    data/processed/sememe2idx.json\
    data/gigaword_300d_jieba_unk.bin\
    data/processed/sememe_matrix.npy
python preprocess/prep_w2v.py\
    data/processed/word2idx.json\
    data/gigaword_300d_jieba_unk.bin\
    data/processed/word_matrix.npy
