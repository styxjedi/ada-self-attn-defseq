# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence
# import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from utils import to_var


# =================Knowing When to Look==================
# Attention Block for C_hat calculation
class Atten(nn.Module):
    def __init__(self, hidden_size, decoder_attn_embed_size, dropout):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, decoder_attn_embed_size, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, decoder_attn_embed_size, bias=False)  # W_g
        self.affine_s = nn.Linear(hidden_size, decoder_attn_embed_size, bias=False)  # W_s
        self.affine_h = nn.Linear(decoder_attn_embed_size, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        init.xavier_uniform_(self.affine_s.weight)

    def forward(self, V, h_t, s_t):
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''

        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v(
            self.dropout(V)).unsqueeze(1) + self.affine_g(
                self.dropout(h_t)).unsqueeze(2)

        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h(self.dropout(torch.tanh(content_v))).squeeze(3)
        alpha_t = F.softmax(
            z_t.view(-1, z_t.size(2)), dim=-1).view(
                z_t.size(0), z_t.size(1), -1)

        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm(alpha_t, V).squeeze(2)

        # W_s * s_t + W_g * h_t
        content_s = self.affine_s(self.dropout(s_t)) + self.affine_g(
            self.dropout(h_t))
        # w_t * tanh( content_s )
        z_t_extended = self.affine_h(self.dropout(torch.tanh(content_s)))

        # Attention score between sentinel and image content
        extended = torch.cat((z_t, z_t_extended), dim=2)
        alpha_hat_t = F.softmax(
            extended.view(-1, extended.size(2)), dim=-1).view(
                extended.size(0), extended.size(1), -1)
        beta_t = alpha_hat_t[:, :, -1]

        # c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        beta_t = beta_t.unsqueeze(2)
        c_hat_t = beta_t * s_t + (1 - beta_t) * c_t

        return c_hat_t, alpha_t, beta_t


# Sentinel BLock
class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(input_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout applied before affine transformation
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.affine_x.weight)
        init.xavier_uniform_(self.affine_h.weight)

    def forward(self, x_t, h_t_1, cell_t):

        # g_t = sigmoid( W_x * x_t + W_h * h_(t-1) )
        gate_t = self.affine_x(self.dropout(x_t)) + self.affine_h(
            self.dropout(h_t_1))
        gate_t = torch.sigmoid(gate_t)

        # Sentinel embedding
        s_t = gate_t * torch.tanh(cell_t)

        return s_t


# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding
class AdaptiveBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, decoder_attn_embed_size, vocab_size, dropout):
        super(AdaptiveBlock, self).__init__()

        # Sentinel block
        self.sentinel = Sentinel(embed_size, hidden_size, dropout)

        # Image Spatial Attention Block
        self.atten = Atten(hidden_size, decoder_attn_embed_size, dropout)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size, vocab_size, bias=False)

        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout(dropout)

        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        '''
        Initialize final classifier weights
        '''
        init.xavier_uniform_(self.mlp.weight)

    def forward(self, x, hiddens, cells, V):

        # hidden for sentinel should be h0-ht-1
        h0 = self.init_hidden(x.size(0))[0].transpose(0, 1)

        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        if hiddens.size(1) > 1:
            hiddens_t_1 = torch.cat((h0, hiddens[:, :-1, :]), dim=1)
        else:
            hiddens_t_1 = h0

        # Get Sentinel embedding, it's calculated blockly
        sentinel = self.sentinel(x, hiddens_t_1, cells)

        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights, beta = self.atten(V, hiddens, sentinel)

        # Final score along vocabulary
        scores = self.mlp(self.dropout(c_hat + hiddens))

        return scores, atten_weights, beta

    def init_hidden(self, bsz):
        '''
        Hidden_0 & Cell_0 initialization
        '''
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            return (Variable(
                weight.new(1, bsz, self.hidden_size).zero_().cuda()),
                    Variable(
                        weight.new(1, bsz, self.hidden_size).zero_().cuda()))
        else:
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))


# Caption Decoder
class AdaptiveDecoder(nn.Module):
    def __init__(self, args, word_emb):
        super(AdaptiveDecoder, self).__init__()

        # word embedding
        self.embed = self.from_pretrained(word_emb, freeze=True)

        # V affine
        self.affine_v = nn.Linear(args.embed_size, args.hidden_size, bias=False)
        self.init_weights()

        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM(args.embed_size, args.hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden and cell variable
        self.hidden_size = args.hidden_size

        # Adaptive Attention Block:
        # Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock(args.embed_size, args.hidden_size, args.decoder_attn_embed_size, args.vocab_size, args.dropout)

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform_(self.affine_v.weight)

    def from_pretrained(self, embeddings, freeze=True):
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = torch.nn.Embedding(
            num_embeddings=rows, embedding_dim=cols, padding_idx=0)
        embedding.weight = torch.nn.Parameter(embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, V, definition, states=None):
        # Sort def seq lengths
        # sorted_def_lens, indices = torch.sort(def_len, desending=True)
        # _, desorted_indices = torch.sort(indices, desending=False)
        # difinition = pack_padded_sequence(definition[indices], def_len, batch_first=True)

        # Word Embedding
        x = self.embed(definition)
        V = self.affine_v(V)
        # word_emb = self.embed(word).squeeze(1)
        # word_emb = self.affine_word(word_emb)

        # x_t = [w_t;v_g]
        # x = torch.cat(
            # (embeddings, v_g.unsqueeze(1).expand_as(embeddings)), dim=2)

        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = Variable(
                torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda())
            cells = Variable(
                torch.zeros(x.size(1), x.size(0), self.hidden_size).cuda())
            # if states is None:
                # states = [
                    # word_emb.unsqueeze(0),
                    # Variable(
                        # torch.zeros(1, x.size(0), self.hidden_size).cuda())
                # ]
        else:
            hiddens = Variable(
                torch.zeros(x.size(0), x.size(1), self.hidden_size))
            cells = Variable(
                torch.zeros(x.size(1), x.size(0), self.hidden_size))
            # if states is None:
                # states = [
                    # word_emb.unsqueeze(0),
                    # Variable(torch.zeros(1, x.size(0), self.hidden_size))
                # ]

        # Recurrent Block
        # Retrieve hidden & cell for Sentinel simulation
        for time_step in range(x.size(1)):
            # Feed in x_t one at a time
            x_t = x[:, time_step, :]
            x_t = x_t.unsqueeze(1)

            self.LSTM.flatten_parameters()
            h_t, states = self.LSTM(x_t, states)

            # Save hidden and cell
            hiddens[:, time_step, :] = h_t.squeeze()  # Batch_first
            cells[time_step, :, :] = states[1]

        # cell: Batch x seq_len x hidden_size
        cells = cells.transpose(0, 1)

        # Data parallelism for adaptive attention block
        # if torch.cuda.device_count() > 1:
            # device_ids = range(torch.cuda.device_count())
            # adaptive_block_parallel = nn.DataParallel(
                # self.adaptive, device_ids=device_ids)

            # scores, atten_weights, beta = adaptive_block_parallel(
                # x, hiddens, cells, V)
        # else:
        scores, atten_weights, beta = self.adaptive(x, hiddens, cells, V)

        # Return states for Caption Sampling purpose
        return scores, states, atten_weights, beta