# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from modules.utils import to_var
from self_attn_encoder import TransformerEncoder
from adaptive_decoder import AdaptiveDecoder

# Whole Architecture with Image Encoder and Caption decoder
class Encoder2Decoder(nn.Module):
    def __init__(self, args, word_emb, sememe_emb, left_pad=False):
        super(Encoder2Decoder, self).__init__()

        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = TransformerEncoder(args, word_emb, sememe_emb, left_pad)
        self.decoder = AdaptiveDecoder(args, word_emb)

    def forward(self, word_sememes, definition, states=None):

        V = self.encoder(word_sememes)
        scores, states, _, _ = self.decoder(V, word_sememes, definition)

        return scores, states

    # Caption generator
    def greedy_sampler(self, word_sememes, max_len=20):
        """
        Samples captions for given image features (Greedy search).
        """

        V = self.encoder(word_sememes)

        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            pred_definition = Variable(
                torch.LongTensor(word_sememes.size(0), 1).fill_(1).cuda())
        else:
            pred_definition = Variable(
                torch.LongTensor(word_sememes.size(0), 1).fill_(1))

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []
        Beta = []

        # Initial hidden states
        states = None

        for i in range(max_len):

            scores, states, atten_weights, beta = self.decoder(
                V, word_sememes, pred_definition, states)
            predicted = scores.max(2)[1]  # argmax
            pred_definition = predicted

            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append(pred_definition)
            attention.append(atten_weights)
            Beta.append(beta)

        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat(sampled_ids, dim=1)
        attention = torch.cat(attention, dim=1)
        Beta = torch.cat(Beta, dim=1)

        return sampled_ids, attention, Beta
