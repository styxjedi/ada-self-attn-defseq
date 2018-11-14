#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import json
from modules.utils import to_var
from ada_self_attn import Encoder2Decoder
import visdom
import sys
sys.path.append('.')
from metrics import rerank
from metrics import bleu


class ModelWrapper(object):
    def __init__(self, args, loss_fn=None, data_loader=None):
        self.vis = visdom.Visdom(env='AdaSelfAttn')
        #if not self.vis.win_exists('valid_ppl'):
        #    self.vis.line(Y=None, win='valid_ppl')
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

        torch.manual_seed(args.seed)
        if self.cuda:
            torch.cuda.manual_seed(args.seed)
            pretrained_word_emb = torch.Tensor(
                np.load(args.pretrained_word_emb_path)).cuda()
            pretrained_sememe_emb = torch.Tensor(
                np.load(args.pretrained_sememe_emb_path)).cuda()
            if loss_fn is not None:
                self.loss_fn = loss_fn.cuda()
        else:
            pretrained_word_emb = torch.Tensor(
                np.load(args.pretrained_word_emb_path))
            pretrained_sememe_emb = torch.Tensor(
                np.load(args.pretrained_sememe_emb_path))
            if loss_fn is not None:
                self.loss_fn = loss_fn

        model = Encoder2Decoder(args, pretrained_word_emb, pretrained_sememe_emb)
        self.model = self.load_model(model)
        if data_loader:
            self.data_loader = data_loader

        if args.pretrained:
            self.start_epoch = self.load_pretrained_model(args.pretrained)
        else:
            self.start_epoch = 1

    def train(self, optimizer, args,):
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

        if args.warmup_lr is not None:
            warmup_optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=args.warmup_lr,
                    betas=(args.alpha, args.beta),
                    weight_decay=args.l2_rate
                    )
            if torch.cuda.device_count() > 1:
                device_ids = range(torch.cuda.device_count())
                warmup_optimizer = nn.DataParallel(warmup_optimizer,
                        device_ids=device_ids)

        train_data = self.data_loader(
                args.train_path,
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                mode='train'
                )
        total_step = len(train_data)
        ppl_scores = []
        best_ppl = 0.0
        best_epoch = 0
        for epoch in range(self.start_epoch, args.num_epochs + 1):
            epoch_loss = []
            print('{0}Training for Epoch {1}{0}'.format('-'*20, epoch))
            for i, (word_sememes, definition) in enumerate(train_data):
                word_sememes = to_var(word_sememes)
                definition = to_var(definition)
                targets = definition[:, 1:]

                self.model.train()
                self.model.zero_grad()

                scores, _ = self.model(word_sememes, definition)
                scores = scores[:, :-1, :].transpose(1, 2)

                loss = self.loss_fn(scores, targets)
                epoch_loss.append(loss.item())
                loss.backward()

                nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        args.clip
                        )

                if args.warmup_lr is not None and epoch <= args.warmup_epochs:
                    if torch.cuda.device_count() > 1:
                        warmup_optimizer.module.step()
                    else:
                        warmup_optimizer.step()
                else:
                    if torch.cuda.device_count() > 1:
                        optimizer.module.step()
                    else:
                        optimizer.step()

                if (i + 1) % args.log_step == 0:
                    print('Epoch [{}/{}], '.format(epoch, args.num_epochs), end='')
                    print('Step [{}/{}], '.format(i+1, total_step), end='')
                    print('CrossEntropyLoss: {:.4f}, '.format(loss.item()), end='')
                    print('Perplexity: {:.4f}'.format(np.exp(loss.item())))

            train_loss = np.mean(epoch_loss)
            train_ppl = np.exp(train_loss)
            valid_loss = self.valid(args)
            valid_ppl = np.exp(valid_loss)
            ppl_scores.append(valid_ppl)

            if epoch > args.warmup_epochs:
                self.vis.line(win='valid_ppl',
                    name=args.model_name,
                    X=torch.FloatTensor([epoch]),
                    Y=torch.FloatTensor([valid_ppl]),
                    opts=dict(title='Valid PPL', showlegend=True),
                    update='append')
            
            if valid_ppl < best_ppl or best_ppl == 0.0:
                best_ppl = valid_ppl
                best_epoch = epoch
                is_best_epoch = True
            else:
                is_best_epoch = False

            print('Epoch [{}/{}], '.format(epoch, args.num_epochs), end='')
            print('Train Loss: {:.4f}, '.format(train_loss), end='')
            print('Train PPL: {:.4f}, '.format(train_ppl), end='')
            print('Valid Loss: {:.4f}, '.format(valid_loss), end='')
            print('Valid PPL: {:.4f}, '.format(valid_ppl))
            print('Best Epoch: {}, Best PPL:{:.4f}'.format(best_epoch, best_ppl))

            if epoch >= args.test_epoch:
                bleu = self.test(args)
                self.vis.line(win='test_bleu',
                    name=args.model_name,
                    X=torch.FloatTensor([epoch]),
                    Y=torch.FloatTensor([bleu]),
                    opts=dict(title='Test BLEU', showlegend=True),
                    update='append')

            self.save_model(epoch, is_best_epoch, args.model_path, args.no_epoch_save)

            if args.early_stop and self.early_stop_fn(ppl_scores, best_ppl, args.patient):
                break
        return 1

    def valid(self, args):
        valid_data = self.data_loader(
                args.valid_path,
                args.eval_size,
                shuffle=False,
                num_workers=args.num_workers,
                mode='valid'
                )
        eval_loss = []
        print('{0}Evaluation on Validation Dataset{0}'.format('-'*20))
        for i, (word_sememes, definition) in enumerate(valid_data):
            self.model.eval()
            word_sememes = to_var(word_sememes)
            definition = to_var(definition)
            targets = definition[:, 1:]
            scores, _ = self.model(word_sememes, definition)
            scores = scores[:, :-1, :].transpose(1, 2)
            loss = self.loss_fn(scores, targets)
            eval_loss.append(loss.item())
        return np.mean(eval_loss)

    def test(self, args):
        if self.generate(args):
            print('Reranking...')
            rerank_args = [
                    None,
                    args.output_path,
                    './data/lm.bin',
                    './data/function_words.txt',
                    args.output_path + '.rank'
                    ]
            rerank.rerank(rerank_args)

            print('Computing BLEU...')
            score, _, _ = bleu.bleu('./data/test.txt',
                    args.output_path + '.rank.top')
            print('Test Bleu: ', score)
            return score
        else:
            return 0

    def generate(self, args):
        with open(args.idx2word_path, 'r') as fr:
            idx2word = json.loads(fr.read())
        with open(args.idx2sememe_path, 'r') as fr:
            idx2sememe = json.loads(fr.read())

        test_data = self.data_loader(
                args.test_path,
                args.test_size,
                shuffle=False,
                num_workers=args.num_workers,
                mode='test'
                )
    
        if args.beam_size == 1:
            results = self.greedy_sampler(test_data, idx2word, idx2sememe, args)
        else:
            results = self.beam_sampler(test_data, idx2word, idx2sememe, args)

        with open(args.output_path, 'w') as fw:
            for word, sememes, definition in results:
                fw.write('{} ||| {} ||| {}\n'.format(word, sememes, definition))
        return 1

    def greedy_sampler(self, test_data, idx2word, idx2sememe, args):
        results = []
        print('{0}Evaluation on Test Dataset{0}'.format('-'*20))
        for i, (word_sememes) in enumerate(test_data):
            self.model.eval()
            word_sememes = to_var(word_sememes)
            if torch.cuda.device_count() > 1:
                pred, _, _ = self.model.module.greedy_sampler(word_sememes, args.max_target_positions)
            else:
                pred, _, _ = self.model.greedy_sampler(word_sememes, args.max_target_positions)

            if torch.cuda.is_available():
                pred = pred.cpu().data.numpy()
                word_sememes = word_sememes.cpu().data.numpy()
            else:
                pred = pred.data.numpy()
                word_sememes = word_sememes.data.numpy()

            for idx in range(pred.shape[0]):
                sampled_ids = pred[idx]
                cur_word = idx2word[str(word_sememes[idx][0])]
                cur_sememes = [idx2sememe[str(s)] for s in word_sememes[idx][1:] if s != 0]
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

            # if (i + 1) % 10 == 0:
                # print('Generating: [{}/{}]'.format(i + 1, len(test_data)))
        return results

    def beam_sampler(self, test_data, idx2word, idx2sememe, args):
        raise NotImplementedError("Not Implemented Yet.")

    def score(self, result_data):
        total_scores = []
        print('{0}Scoring on Result Dataset{0}'.format('-'*20))
        for i, (word_sememes, definition) in enumerate(result_data):
            word_sememes = to_var(word_sememes)
            definition = to_var(definition)
            targets = definition[:, 1:]

            scores, _ = self.model(word_sememes, definition)
            scores = scores[:, :-1, :].transpose(1, 2)
            loss = self.loss_fn(scores, targets)
            total_scores.append(str(np.exp(loss.item())))
            if (i + 1) % 10 == 0:
                print('Scoring: [{}/{}]'.format(i + 1, len(result_data)))
        return total_scores

    def load_model(self, model):
        m = model
        if self.cuda:
            m = model.cuda()
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            m = nn.DataParallel(model, device_ids=device_ids)
            print(list(m.children())[0])
        else:
            print(m)
        return m

    def load_pretrained_model(self, pretrained):
        if os.path.islink(pretrained):
            pretrained = os.readlink(pretrained)
        if torch.cuda.device_count() > 1:
            self.model.module.load_state_dict(torch.load(pretrained))
        else:
            self.model.load_state_dict(torch.load(pretrained))

        start_epoch = int(pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1
        return start_epoch

    def save_model(self, epoch, is_best_epoch, model_path, no_epoch_save):
        epoch_save_path = os.path.join(model_path, 'adaptive-{}.pkl'.format(epoch))
        best_save_path = os.path.join(model_path, 'adaptive-best.pkl')
        try:
            state_dict = self.model.module.state_dict()
        except AttributeError:
            state_dict = self.model.state_dict()

        if no_epoch_save:
            if os.path.exists(best_save_path):
                os.remove(best_save_path)
            torch.save(state_dict, best_save_path)
        else:
            torch.save(state_dict, epoch_save_path)
            if is_best_epoch:
                if os.path.islink(best_save_path):
                    os.remove(best_save_path)
                os.symlink(epoch_save_path, best_save_path)
        return 1

    def early_stop_fn(self, ppl_scores, best_ppl, patient):
        if len(ppl_scores) >= patient:
            last = ppl_scores[-patient:]
            last_min = min(last)
            if last_min != best_ppl:
                print('No significant improvement with last {} epochs.'.format(patient))
                print('Early stopping triggered.')
                return True
        return False

