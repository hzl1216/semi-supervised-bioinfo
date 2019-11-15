import argparse
import logging
from util.utils import str2bool
def create_parser(dataset):
    if dataset is 'cifar10':
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument('--epochs', default=180, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--unlabeled-batch-size', default=256, type=int,
                            metavar='N', help='unlabeled-batch size (default: 128)')
        parser.add_argument('--labeled-batch-size', default=256, type=int,
                            metavar='N', help='labeled-batch size (default: 128)')
        parser.add_argument('--lr', '--learning-rate', default=0.002, type=float)
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--ema-decay', default=0.99, type=float, metavar='ALPHA',
                            help='ema variable decay rate (default: 0.999)')
        parser.add_argument('--consistency', default=10.0, type=float, metavar='WEIGHT',
                            help='use consistency loss with given weight (default: None)')
        parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                            choices=['mse', 'kl'],
                            help='consistency loss type to use')
        parser.add_argument('--consistency-rampup', default=10, type=int, metavar='EPOCHS',
                            help='length of the consistency loss ramp-up')
        parser.add_argument('--entropy-cost', default=0, type=float, metavar='WEIGHT')
        parser.add_argument('--checkpoint-epochs', default=1, type=int,
                            metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
        parser.add_argument('--evaluation-epochs', default=1, type=int,
                            metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
        parser.add_argument('--print-freq', '-p', default=20, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--out_path', default='result',
                            help='Directory to output the result')
        parser.add_argument('--n-labeled', type=int, default=4000,
                            help='Number of labeled data')
        parser.add_argument('-e', '--evaluate', type=bool,
                            help='evaluate model on evaluation set')
        parser.add_argument('--num-workers', type=int, default=12,
                            help='Number of workers')
        parser.add_argument('--epoch-iteration', type=int, default=256,
                            help='Number of workers')
        parser.add_argument('--warmup-step', type=int, default=0,
                            help='Number of workers')
        parser.add_argument('--alpha', default=0.75, type=float)
        parser.add_argument('--mixup', default=False, type=str2bool,
                            help='use mixup', metavar='BOOL')
    else:
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument('--epochs', default=180, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--unlabeled-batch-size', default=60, type=int,
                            metavar='N', help='unlabeled-batch size (default: 128)')
        parser.add_argument('--labeled-batch-size', default=60, type=int,
                            metavar='N', help='labeled-batch size (default: 128)')
        parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--ema-decay', default=0.995, type=float, metavar='ALPHA',
                            help='ema variable decay rate (default: 0.999)')
        parser.add_argument('--consistency', default=10.0, type=float, metavar='WEIGHT',
                            help='use consistency loss with given weight (default: None)')
        parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                            choices=['mse', 'kl'],
                            help='consistency loss type to use')
        parser.add_argument('--consistency-rampup', default=5, type=int, metavar='EPOCHS',
                            help='length of the consistency loss ramp-up')
        parser.add_argument('--entropy-cost', default=0, type=float, metavar='WEIGHT')
        parser.add_argument('--checkpoint-epochs', default=1, type=int,
                            metavar='EPOCHS',
                            help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
        parser.add_argument('--evaluation-epochs', default=1, type=int,
                            metavar='EPOCHS',
                            help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
        parser.add_argument('--print-freq', '-p', default=20, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--out_path', default='result',
                            help='Directory to output the result')
        parser.add_argument('--n-labeled', type=int, default=1000,
                            help='Number of labeled data')
        parser.add_argument('-e', '--evaluate', type=bool,
                            help='evaluate model on evaluation set')
        parser.add_argument('--num-workers', type=int, default=12,
                            help='Number of workers')
        parser.add_argument('--epoch-iteration', type=int, default=256,
                            help='Number of workers')
        parser.add_argument('--warmup-step', type=int, default=0,
                            help='Number of workers')
        parser.add_argument('--alpha', default=0.75, type=float)
        parser.add_argument('--mixup', default=False, type=str2bool,
                            help='use mixup', metavar='BOOL')
    return parser.parse_args()
