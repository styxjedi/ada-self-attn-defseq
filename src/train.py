from __future__ import print_function
# import math
import json
import torch
import torch.nn as nn
import argparse
from options import add_args
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
import sys
from utils import defseq_eval, to_var
from data_loader import get_loader
from ada_self_attn import Encoder2Decoder

# from torch.nn.utils.rnn import pack_padded_sequence

# from torch.autograd import Variable


def main(args):

    # To reproduce training results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load word2idx
    with open(args.word2idx_path, 'r') as fr:
        word2idx = json.loads(fr.read())

    # Build training data loader
    data_loader = get_loader(
        args.train_path,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        mode='train')

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

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args, pretrained_word_emb, pretrained_sememe_emb)
    if torch.cuda.device_count() > 1:
        device_ids = range(torch.cuda.device_count())
        adaptive = nn.DataParallel(adaptive, device_ids=device_ids)
        print(list(adaptive.children())[0])
    else:
        print(adaptive)

    if args.pretrained:
        pretrained = args.pretrained
        if os.path.islink(pretrained):
            pretrained = os.readlink(args.pretrained)
        # Get starting epoch #,
        # note that model is named as
        # '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(
            pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1
        if torch.cuda.device_count() > 1:
            adaptive.module.load_state_dict(torch.load(pretrained))
        else:
            adaptive.load_state_dict(torch.load(pretrained))
    else:
        start_epoch = 1

    # Will decay later
    # learning_rate = args.learning_rate

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss(ignore_index=0)

    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    ppl_scores = []
    best_ppl = 0.0
    best_epoch = 0

    # Start Learning Rate Decay
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adaptive.parameters()),
        lr=args.learning_rate,
        betas=(args.alpha, args.beta),
        weight_decay=args.l2_rate)
    if torch.cuda.device_count() > 1:
        device_ids = range(torch.cuda.device_count())
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    # Start Training
    for epoch in range(start_epoch, args.num_epochs + 1):

        epoch_loss = []

        # Language Modeling Training
        print('------------------Training for Epoch %d----------------' %
              (epoch))
        for i, (word_sememes, definition) in enumerate(data_loader):
            # Set mini-batch dataset
            word_sememes = to_var(word_sememes)
            definition = to_var(definition)
            targets = definition[:, 1:]

            # Forward, Backward and Optimize
            adaptive.train()
            adaptive.zero_grad()

            scores, _ = adaptive(word_sememes, definition)
            scores = scores[:, :-1, :].transpose(1, 2)

            # Compute loss and backprop
            loss = LMcriterion(scores, targets)
            epoch_loss.append(loss.item())
            loss.backward()

            # Gradient clipping for gradient exploding problem in LSTM
            # for p in adaptive.decoder.LSTM.parameters():
            #     p.data.clamp_(-args.clip, args.clip)

            clip_grad_norm_(
                filter(lambda p: p.requires_grad, adaptive.parameters()),
                args.clip)
            # print(args.clip)
            if torch.cuda.device_count() > 1:
                optimizer.module.step()
            else:
                optimizer.step()

            # Print log info
            if (i + 1) % args.log_step == 0:
                print(
                    'Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f'
                    % (epoch, args.num_epochs, i + 1, total_step, loss.item(),
                       np.exp(loss.item())))
        train_loss = np.mean(epoch_loss)
        train_ppl = np.exp(train_loss)
        # Save the Adaptive Attention model after each epoch
        try:
            state_dict = adaptive.module.state_dict()
        except AttributeError:
            state_dict = adaptive.state_dict()
        epoch_save_path = os.path.join(args.model_path, 'adaptive-{}.pkl'.format(epoch))
        best_save_path = os.path.join(args.model_path, 'adaptive-best.pkl')
        torch.save(state_dict, epoch_save_path)

        # Evaluation on validation set
        valid_ppl = defseq_eval(adaptive, args, epoch)
        ppl_scores.append(valid_ppl)

        print(
            'Epoch [%d/%d], Train Loss: %.4f, Train PPL: %5.4f, Valid PPL: %5.4f'
            % (epoch, args.num_epochs, train_loss, train_ppl, valid_ppl))

        if valid_ppl < best_ppl or best_ppl == 0.0:
            best_ppl = valid_ppl
            best_epoch = epoch
            if os.path.islink(best_save_path):
                os.remove(best_save_path)
            os.symlink(epoch_save_path, best_save_path)

        if len(ppl_scores) > 5:
            last_6 = ppl_scores[-6:]
            last_6_min = min(last_6)

            # Test if there is improvement, if not do early stopping
            if last_6_min != best_ppl:

                print(
                    'No improvement with ppl in the last 6 epochs...Early stopping triggered.'
                )
                print('Model of best epoch #: %d with ppl score %.2f' %
                      (best_epoch, best_ppl))
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser, 'train')
    print('-----------Model and Training Details------------')
    print(args)

    # Start training
    sys.exit(main(args))
