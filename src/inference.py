# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import argparse
from options import add_args
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
from utils import to_var
from data_loader import get_loader
from ada_self_attn import Encoder2Decoder
import codecs
from beamsearch import BeamSearch
import os


def greedy_sampler(model, data_loader, idx2word, idx2sememe):
    model.eval()
    results = []
    print('---------Start evaluation on test dataset----------')
    for i, (word_sememes) in enumerate(data_loader):

        word_sememes = to_var(word_sememes)
        pred, _, _ = model.module.greedy_sampler(word_sememes)

        if torch.cuda.is_available():
            pred = pred.cpu().data.numpy()
            word_sememes = word_sememes.cpu().data.numpy()
        else:
            pred = pred.data.numpy()
            word_sememes = word_sememes.data.numpy()

        # Build caption based on Vocabulary and the '<end>' token
        for idx in range(pred.shape[0]):
            sampled_ids = pred[idx]
            cur_word = idx2word[str(word_sememes[idx][0])]
            cur_sememes = [idx2sememe[str(s)] for s in word_sememes[idx][2:] if s != 0]
            cur_sememes = ' '.join(cur_sememes)
            sampled_caption = []
            for word_id in sampled_ids:
                try:
                    w = idx2word[str(word_id)]
                except KeyError:
                    w = idx2word[str(args.unk)]
                if w == idx2word[str(args.eos)]:
                    break
                else:
                    sampled_caption.append(w)
            sentence = ' '.join(sampled_caption)
            results.append((cur_word, cur_sememes, sentence))
        # Disp evaluation process
        if (i + 1) % 10 == 0:
            print('[%d/%d]' % ((i + 1), len(data_loader)))
    return results


def beam_sampler(model, data_loader, idx2word, idx2sememe):
    model.eval()
    assert args.test_size == 1,\
        "batch size(test_size) must be 1 when beam searching"
    results = []
    beam = BeamSearch(args.max_len, args.unk, args.bos, args.eos,
                      args.beam_size)
    print('---------Start evaluation on test dataset----------')
    for i, (word, sememes) in enumerate(data_loader):
        print('[%d/%d]' % ((i + 1), len(data_loader)))
        word = to_var(word)
        sememes = to_var(sememes)
        definition = torch.LongTensor(word.size(0), 1).fill_(1)
        definition = to_var(definition)
        beam.reset()
        scores, states = model(word, sememes, definition)
        scores = F.log_softmax(scores, dim=-1)
        if torch.cuda.is_available():
            scores = scores.cpu().data.numpy().squeeze(1)
            cur_word = word.cpu().data.numpy().squeeze()
            cur_sememes = sememes.cpu().data.numpy().squeeze()
        else:
            scores = scores.data.numpy().squeeze(1)
            cur_word = word.data.numpy().squeeze()
            cur_sememes = sememes.data.numpy().squeeze()
        cur_word = idx2word[str(cur_word)]
        cur_sememes = [idx2sememe[str(s)] for s in cur_sememes if s != 0]
        cur_sememes = ' '.join(cur_sememes)

        while beam.beam(scores):
            definition = to_var(torch.LongTensor(beam.live_samples))
            definition = definition[:, -1].unsqueeze(1)
            word = word[0].unsqueeze(0).repeat(definition.size(0), 1)
            sememes = sememes[0].unsqueeze(0).repeat(definition.size(0), 1)
            scores, states = model(word, sememes, definition, states)
            scores = F.log_softmax(scores, dim=-1)
            if torch.cuda.is_available():
                scores = scores.cpu().data.numpy().squeeze(1)
            else:
                scores = scores.data.numpy().squeeze(1)

        cur_definition = []
        for line in beam.output:
            tmp = []
            for i in line:
                if i in [0, 1, 2]:
                    continue
                try:
                    w = idx2word[str(i)]
                except KeyError:
                    w = 'UNK'
                tmp.append(w)
            cur_definition.append(tmp)

        for d in cur_definition:
            results.append((cur_word, cur_sememes, cur_definition))
    return results


def main(args):
    # To reproduce training results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load word2idx
    with open(args.word2idx_path, 'r') as fr:
        word2idx = json.loads(fr.read())
    with open(args.idx2word_path, 'r') as fr:
        idx2word = json.loads(fr.read())
    with open(args.idx2sememe_path, 'r') as fr:
        idx2sememe = json.loads(fr.read())

    # Build training data loader
    test_loader = get_loader(
        args.test_path,
        args.test_size,
        shuffle=False,
        num_workers=args.num_workers,
        mode='test')

    # Load pretrained embeddings
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

    if args.beam_size == 1:
        results = greedy_sampler(adaptive, test_loader, idx2word, idx2sememe)
    else:
        results = beam_sampler(adaptive, test_loader, idx2word, idx2sememe)

    with codecs.open(args.output_path, 'w', 'utf-8') as fw:
        for word, sememes, definition in results:
            fw.write('%s ||| %s ||| %s\n' % (word, sememes, definition))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser, mode='inference')
    print('----------------Model and Test Args----------------')
    print(args)
    sys.exit(main(args))
