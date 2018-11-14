#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import argparse
from options import add_args
import sys
from data_loader import get_loader
from model_wrapper import ModelWrapper


def train(args):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    adaptive = ModelWrapper(args, loss_fn, get_loader)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adaptive.model.parameters()),
        lr=args.learning_rate,
        betas=(args.alpha, args.beta),
        weight_decay=args.l2_rate)
    if adaptive.train(optimizer, args):
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser, 'train')
    print('{0}Model and Training Details{0}'.format('-' * 20))
    print(args)
    # Start training
    sys.exit(train(args))
