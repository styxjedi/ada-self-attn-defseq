# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils import data
from utils import to_var
from ada_self_attn import Encoder2Decoder
from options import add_args
import codecs
import os


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
                cur_w.append(word2idx['<s>'])
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


def gen_score(adaptive, res_loader):
    LMcriterion = nn.CrossEntropyLoss(ignore_index=0)
    if torch.cuda.is_available():
        LMcriterion.cuda()

    adaptive.eval()
    total_scores = []
    print('--------------Start Scoring on Generated dataset---------------')
    for i, (word_sememes, definition) in enumerate(res_loader):
        word_sememes = to_var(word_sememes)
        definition = to_var(definition)
        targets = definition[:, 1:]

        scores, _ = adaptive(word_sememes, definition)
        scores = scores[:, :-1, :].transpose(1, 2)
        loss = LMcriterion(scores, targets)
        total_scores.append(str(np.exp(loss.item())))
        if (i + 1) % 10 == 0:
            print('[%s/%s]' % ((i + 1), len(res_loader)))
    return total_scores


def main(args):
    with open(args.word2idx_path, 'r') as fr:
        word2idx = json.loads(fr.read())
    with open(args.sememe2idx_path, 'r') as fr:
        sememe2idx = json.loads(fr.read())
    results = ResDataset(args.gen_file_path, word2idx, sememe2idx)
    res_loader = data.DataLoader(dataset=results, batch_size=1, shuffle=False)

    if torch.cuda.is_available():
        pretrained_word_emb = torch.Tensor(
            np.load(args.pretrained_word_emb_path)).cuda()
        pretrained_sememe_emb = torch.Tensor(
            np.load(args.pretrained_sememe_emb_path)).cuda()
    else:
        pretrained_word_emb = torch.Tensor(
            np.load(args.pretrained_word_emb_path))
        pretrained_sememe_emb = torch.Tensor(
            np.load(args.pretrained_sememe_emb_path))

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args, pretrained_word_emb, pretrained_sememe_emb)
    if torch.cuda.is_available():
        adaptive = adaptive.cuda()

    if torch.cuda.device_count() > 1:
        device_ids = range(torch.cuda.device_count())
        adaptive = nn.DataParallel(adaptive, device_ids=device_ids)
        print(list(adaptive.children())[0])
    else:
        print(adaptive)

    if args.pretrained:
        pretrained = args.pretrained
        if os.path.islink(pretrained):
            pretrained = os.readlink(pretrained)
        if torch.cuda.device_count() > 1:
            adaptive.module.load_state_dict(torch.load(pretrained))
        else:
            adaptive.load_state_dict(torch.load(pretrained))

    scores = gen_score(adaptive, res_loader)
    with codecs.open(args.output_path, 'w', 'utf-8') as fw:
        fw.write('\n'.join(scores))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser, mode='score')
    sys.exit(main(args))
