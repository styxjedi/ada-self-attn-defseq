import argparse


def add_training_args(parser):
    # Data Path
    parser.add_argument(
        '--model_path',
        type=str,
        default='./models',
        help='path for saving trained models')
    parser.add_argument(
        '--train_path',
        type=str,
        default='./data/processed/train.npz',
        help='path for train dataset')
    parser.add_argument(
        '--valid_path',
        type=str,
        default='./data/processed/valid.npz',
        help='path for valid dataset')

    # Optimizer Adam parameter
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='alpha in Adam')
    parser.add_argument(
        '--beta', type=float, default=0.999, help='beta in Adam')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=4e-4,
        help='learning rate for the whole model')
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--l2_rate', type=float, default=0.2)
    parser.add_argument(
        '--early_stop',
        action='store_true',
        default=True,
        help='stop if valid ppl not decrease')
    parser.add_argument(
        '--patient',
        type=int,
        default=5,
        help='epoches not decrease (for early stop)')
    parser.add_argument(
        '--no_epoch_save',
        action='store_true',
        default=False,
        help='only save the best model')
    parser.add_argument(
        '--eval_size', type=int,
        default=50)  # on cluster setup, 60 each x 4 for Huckle server
    parser.add_argument(
        '--test_epoch',
        type=int,
        default=10,
        help='test after n epoches')
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        help='warmup for n epoches')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-5,
        help='warmup learning rate')

    # Training details
    parser.add_argument('--num_epochs', type=int, default=50)

    return parser


def add_inference_args(parser):
    parser.add_argument(
        '--test_path', type=str, default='./data/processed/test.npz')
    parser.add_argument(
        '--output_path', type=str, help='path for output directory')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size')
    parser.add_argument(
        '--test_size', type=int,
        default=50)  # on cluster setup, 60 each x 4 for Huckle server
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='temperature argument for softmax')
    parser.add_argument(
        '--max_len', type=int, default=50, help='max length for generation')
    return parser


def add_score_args(parser):
    parser.add_argument(
        '--gen_file_path', type=str, help='path for generated file')
    parser.add_argument(
        '--output_path', type=str, help='path for generated file')
    return parser

def add_args(parser, mode):
    assert mode in ['train', 'inference', 'score'],\
        'mode must be train, inference or score.'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument(
        '--model_name',
        default='model',
        type=str,
        help='name of the model')
    parser.add_argument(
        '--word2idx_path',
        type=str,
        default='./data/processed/word2idx.json',
        help='path for word2idx file')
    parser.add_argument(
        '--idx2word_path',
        type=str,
        default='data/processed/idx2word.json',
        help='path for idx2word file')
    parser.add_argument(
        '--sememe2idx_path',
        type=str,
        default='./data/processed/sememe2idx.json',
        help='path for sememe2idx file')
    parser.add_argument(
        '--idx2sememe_path',
        type=str,
        default='data/processed/idx2sememe.json',
        help='path for idx2sememe file')
    parser.add_argument(
        '--pretrained_word_emb_path',
        type=str,
        default='data/processed/word_matrix.npy',
        help='path for pretrained word embedding path')
    parser.add_argument(
        '--pretrained_sememe_emb_path',
        type=str,
        default='data/processed/sememe_matrix.npy',
        help='path for pretrained sememe embedding path')
    parser.add_argument(
        '--embed_size',
        type=int,
        default=300,
        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument(
        '--encoder_attention_heads',
        type=int,
        default=6,
        help='number of encoder attention heads')
    parser.add_argument(
        '--encoder_layers',
        type=int,
        default=6,
        help='number of encoder layers')
    parser.add_argument(
        '--encoder_ffn_embed_dim',
        type=int,
        default=512,
        help='dimension of projection between layers')
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='dimension of lstm hidden states')
    parser.add_argument(
        '--decoder_attn_embed_size',
        type=int,
        default=100,
        help='dimension of decoder attention projections')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=0,
        help='number of words in vocabulary')
    parser.add_argument(
        '--max_source_positions',
        type=int,
        default=19,
        help='max number of source sentence length')
    parser.add_argument(
        '--max_target_positions',
        type=int,
        default=45,
        help='max number of source sentence length')
    parser.add_argument(
        '--no_token_positional_embeddings',
        action='store_true',
        default=False,
        help='train models without positional embeddings')
    parser.add_argument(
        '--encoder_learned_pos',
        action='store_true',
        default=False,
        help='learn positional embeddings')
    parser.add_argument(
        '--encoder_normalize_before',
        action='store_true',
        default=False,
        help='add layer normalization before each encoder layer')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument(
        '--attention_dropout',
        type=float,
        default=0.5,
        help='dropout for attention layers')
    parser.add_argument(
        '--relu_dropout',
        type=float,
        default=0.5,
        help='dropout for relu functions')
    parser.add_argument(
        '--seed',
        type=int,
        default=1024,
        help='random seed for model reproduction')
    parser.add_argument(
        '--batch_size', type=int,
        default=50)  # on cluster setup, 60 each x 4 for Huckle server
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--bos', type=int, default=1)
    parser.add_argument('--eos', type=int, default=2)
    parser.add_argument('--unk', type=int, default=3)
    parser.add_argument('--padding-idx', type=int, default=0)
    parser.add_argument(
        '--log_step',
        type=int,
        default=10)
    parser.add_argument(
        '--pretrained',
        type=str,
        default='',
        help='start from checkpoint or scratch')

    if mode == 'train':
        parser = add_training_args(parser)
        parser = add_inference_args(parser)
    elif mode == 'inference':
        parser = add_inference_args(parser)
    else:
        parser = add_score_args(parser)
    args = parser.parse_args()
    return args
