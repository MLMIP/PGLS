import argparse
import importlib
from utils import *
import torch

MODEL_DIR=None

# node5 increase
# DATA_DIR = '/media/s5/wyj/pytorch/pytorch/Datas_with_code/CVPR22-Fact-main/data/'
# MODEL_DIR = '/media/s5/wyj/pytorch/pytorch/Datas_with_code/CVPR22-Fact-main/checkpoint/cub200/increase/5/ft_cos-avg_cos-data_init-start_0/session0_max_acc.pth/bestst.pth'
# logit_dir = '/media/s5/wyj/pytorch/pytorch/Datas_with_code/CVPR22-Fact-main/logits_Data/Model_1'

# MODEL_DIR = '/storage/student1/cxb/wyj/checkpoint/cub200/increase/ft_cos-avg_cos-data_init-start_0/Epo_200-Lr_0.0100-Step_20-Gam_0.25-Bs_128-Mom_0.90-T_16.00/session0_max_acc.pth'

GPU = '1'
BatchSize = 256
DATA_DIR = '/storage/student1/cxb/wyj/data'
logit_dir = '/storage/student1/cxb/wyj/logits_Data/CUB200/Model_' + GPU
PROJECT='fact'
Seed = 1  # 0 is random
epochs_base = 200
torch.set_num_threads(1)
step = 25
def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-n_components', default=3)
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=epochs_base)
    parser.add_argument('-lr_base', type=float, default=0.01)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[50,100 ,125, 150, 175])
    parser.add_argument('-step', type=int, default=step)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.25)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=BatchSize)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=50)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    parser.add_argument('-epoch',type = int ,default=0)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default=GPU)
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=Seed)
    parser.add_argument('-debug', action='store_true')


    parser.add_argument('-to_TSNE',default = False,
                        help = 'last epoch train_data or last session test_data save for STNE')

    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('-logits_Data',default = logit_dir,help='Data to TSNE')
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()