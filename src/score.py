# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils import data
from modules.utils import to_var
from options import add_args
from model_wrapper import ModelWrapper
import codecs


class ResDataset(data.Dataset):
    def __init__(self, file_path, word2idx, sememe2idx):
        word_sememes = []
        definitions = []
        with codecs.open(file_path, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip().split(' ||| ')
                cur_w = [word2idx[line[0]]]
                cur_s = [sememe2idx[l] for l in line[1].split(' ')]
                cur_d = [word2idx[l] for l in line[2].split(' ')]
                cur_w.extend(cur_s)
                word_sememes.append(cur_w)
                definitions.append(cur_d)
        word_sememes_padded = self.padding(word_sememes, args.max_source_positions)
        definitions_padded = self.padding(definitions, args.max_target_positions)
        self.word_sememes = np.array(word_sememes_padded)
        self.definitions = np.array(definitions_padded)

    def padding(self, some_list, some_len):
        new_list = []
        for l in some_list:
            l.extend([0] * (some_len - len(l)))
            if len(l) != some_len:
                print(l)
                print(len(l))
            new_list.append(l)
        return new_list

    def __getitem__(self, index):
        word_sememes = torch.LongTensor(self.word_sememes[index])
        definition = torch.LongTensor(self.definitions[index])
        return word_sememes, definition

    def __len__(self):
        return len(self.word_sememes)


def main(args):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    adaptive = ModelWrapper(args, loss_fn)

    with open(args.word2idx_path, 'r') as fr:
        word2idx = json.loads(fr.read())
    with open(args.sememe2idx_path, 'r') as fr:
        sememe2idx = json.loads(fr.read())
    results = ResDataset(args.gen_file_path, word2idx, sememe2idx)
    res_loader = data.DataLoader(dataset=results, batch_size=1, shuffle=False)

    scores = adaptive.score(res_loader)

    with codecs.open(args.output_path, 'w', 'utf-8') as fw:
        fw.write('\n'.join(scores))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser, mode='score')
    print(args)
    sys.exit(main(args))
