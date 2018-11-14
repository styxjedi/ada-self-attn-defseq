#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from options import add_args
from torch import nn
import sys
from data_loader import get_loader
from model_wrapper import ModelWrapper


def generate(args):
    # Load word2idx
    adaptive = ModelWrapper(args, data_loader=get_loader)
    if adaptive.generate(args):
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser, mode='inference')
    print('{0}Model and Test Args{0}'.format('-' * 20))
    print(args)
    sys.exit(generate(args))
