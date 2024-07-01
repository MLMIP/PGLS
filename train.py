import argparse
import importlib
from utils import *
import torch
MODEL_DIR=None

torch.set_num_threads(2)

DATA_DIR = '/media/s5/wyj/Datas_with_code/CUB200/data'


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # about dataset and network
    parser.add_argument('-project', type=str, default='base')
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=-1)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.01)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[20,40,80])
    parser.add_argument('-step', type=int, default=25)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.25)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                            choices=['ft_dot', 'ft_cos', 'avg_cos', 'train_model']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    #   about training
    parser.add_argument('-gpu', default='6')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=10)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-fantasy', type=str, default= 'None', help='Data Augmentation',
                        choices=['rotation', 'rotation2', 'color_perm', 'color_perm3','color_perm6','rot_color_perm12', 'None'])
    parser.add_argument('-dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('-std',type=float,default = 0.01, help = 'noise',choices=[0.01,0.1,1])
    parser.add_argument('-drop_rate', type=float,default=0.8, help='noise',choices=[0.3,0.5,0.8])
    parser.add_argument('-sample_nums', default=5, help='sample_nums')
    parser.add_argument('-alph',type=float,default = 0.5, help = 'Robust_loss')
    parser.add_argument('-dist_ways', default='cos', help='sample_nums',choices=['cos', 'norm_2', 'norm_1', 'chebyshev'])
    return parser

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()