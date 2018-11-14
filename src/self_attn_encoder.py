#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
from modules import utils
import torch.nn as nn
import torch.nn.functional as F

from modules import (
        LearnedPositionalEmbedding, SinusoidalPositionalEmbedding,
        MultiheadAttention
        )


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
         m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


class TransformerEncoder(nn.Module):
    def __init__(self, args, word_emb, sememe_emb, left_pad=False):
        super(TransformerEncoder, self).__init__()

        self.padding_idx = args.padding_idx
        self.dropout = args.dropout
        self.max_source_positions = args.max_source_positions
        self.word_emb = self.from_pretrained(word_emb, freeze=True)
        self.sememe_emb = self.from_pretrained(sememe_emb, freeze=True)
        self.embed_positions = PositionalEmbedding(self.max_source_positions,
                args.embed_size, self.padding_idx, left_pad=left_pad,
                learned=args.encoder_learned_pos) if not args.no_token_positional_embeddings else None
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
            ])
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = nn.LayerNorm(args.embed_size)
        self.embed_scale = math.sqrt(args.embed_size)

    def from_pretrained(self, embeddings, freeze=True):
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = torch.nn.Embedding(
            num_embeddings=rows, embedding_dim=cols, padding_idx=0)
        embedding.weight = torch.nn.Parameter(embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, src_tokens):
        # embed tokens and positions
        x1 = self.embed_scale * self.word_emb(src_tokens[:, :1])
        x2 = self.embed_scale * self.sememe_emb(src_tokens[:, 1:])
        x = torch.cat([x1, x2], dim=1)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return x.transpose(0, 1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out.index_select(1, new_order)

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = args.embed_size
        self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout
                )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        utils.init_weights(self.fc1, 'relu')
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        utils.init_weights(self.fc2, 'relu')
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for i in range(2)]) 

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
