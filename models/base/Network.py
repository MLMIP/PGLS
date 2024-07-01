import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12 import *
import warnings

from utils import *
from torch.distributions.multivariate_normal import MultivariateNormal

# 存储计算分布
class Cov:
    def __init__(self, num_classes, num_features,sample_nums = 1):
        self.num_classes =  num_classes
        self.num_features = num_features
        self.distribution_functions = {}
        self.sample_nums = sample_nums
        self.pre_generated_samples = self._generate_pre_samples()

    def _generate_distribution_function(self, cov_matrix, mean_vector=None):
        # 如果没有传入均值矩阵，则使用零均值
        if mean_vector is None:
            mean_vector = torch.zeros(self.num_features, dtype=torch.double)
        else:
            mean_vector = torch.from_numpy(mean_vector).double()

        # 将协方差矩阵转换为 double 类型
        cov_matrix = cov_matrix.double()
        # 返回一个函数，该函数生成多元正态分布对象
        return lambda: MultivariateNormal(loc= mean_vector, covariance_matrix =cov_matrix)

    def _generate_pre_samples(self):
        pre_generated_samples = {}

        for class_idx in range(self.num_classes):
            if class_idx in self.distribution_functions:
                distribution = self.distribution_functions[class_idx]
                samples = distribution().sample((self.sample_nums,)).to(torch.float32)
                pre_generated_samples[class_idx] = samples
            else:
                # 处理未定义分布函数的情况
                # 发出警告提示
                # warnings.warn(
                #     f"No defined distribution function for class {class_idx}. Generating default normal distribution samples.")

                default_distribution = self._generate_distribution_function(torch.eye(self.num_features))
                samples = default_distribution().sample((self.sample_nums,)).to(torch.float32)
                pre_generated_samples[class_idx] = samples

        return pre_generated_samples

    def add_or_update_distribution(self, class_idx, cov_matrix, mean = None):

        self.distribution_functions[class_idx] = self._generate_distribution_function(cov_matrix.double(),mean)

    def generate_samples_batch(self, indices,i):
        generated_samples = []

        for class_idx in indices:
            if class_idx not in self.distribution_functions:
                generated_samples.append(torch.randn(self.num_features ))
                continue
            samples = self.pre_generated_samples[class_idx][i]
            generated_samples.append(samples)

        return torch.stack(generated_samples)

    # def generate_noise_rand(self, num):
    #     index = torch.randint(0, self.num_classes, (num,))
    #     data = [self.pre_generated_samples[int(idx)][0] for idx in index]  # 假设每个类别至少有一个样本
    #     return torch.stack(data)


class MYNET(nn.Module):

    def __init__(self, args, mode=None, trans =1):
        super().__init__()

        self.mode = mode
        self.args = args
        if args.dataset == 'cub200':
            self.K =3000
        else:
            self.K = 30000
        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            # self.encoder = resnet20()
            # self.num_features = 64
            self.encoder = ResNet12()
            self.num_features = 640


        if self.args.dataset in ['mini_imagenet', 'manyshotmini', 'imagenet100', 'imagenet1000',
                                 'mini_imagenet_withpath']:
            # self.encoder = resnet18(False, args)  # pretrained=False
            # self.num_features = 512
            self.encoder = ResNet12()  # pretrained=False
            self.num_features = 640

        if self.args.dataset in ['cub200', 'manyshotcub']:
            self.num_features = 512
            self.encoder = resnet18(True, args)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes*trans, bias=False)
        nn.init.normal_(self.fc.weight, std=0.01, mean=0)

        self.register_buffer("queue", torch.randn(self.num_features, self.K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("lab_que", torch.randint(0, self.args.num_classes, [self.K, ]))

        self.centroid =torch.randn(self.args.num_classes * trans, self.num_features)
        self.cov_matrix = Cov(self.args.base_class * trans, self.num_features ,sample_nums=args.sample_nums)
        self.base_noise = {}

    # 存储样本
    def _dequeue_and_enqueue(self, high_feature, train_label):
        # ptr 入队指针，queue 特征存储队列，lab_que 标签队列
        batch_size = high_feature.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            # 如果超出队列长度，需要从头开始存储
            remaining_size = self.K - ptr
            self.queue[:, ptr:] = high_feature[:remaining_size].T
            self.lab_que[ptr:] = train_label[:remaining_size]
            self.queue[:, :batch_size - remaining_size] = high_feature[remaining_size:].T
            self.lab_que[:batch_size - remaining_size] = train_label[remaining_size:]
            ptr = batch_size - remaining_size
        else:
            # 直接存储
            self.queue[:, ptr:ptr + batch_size] = high_feature.T
            self.lab_que[ptr:ptr + batch_size] = train_label
            ptr += batch_size
        self.queue_ptr[0] = ptr % self.K  # 更新指针位置

    def encode_fc(self,temp):
        if 'cos' in self.mode:
            temp = F.linear(F.normalize(temp, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))  # 16 * 200
            temp = self.args.temperature * temp
        elif 'dot' in self.mode:
            temp = self.fc(temp)
            temp = self.args.temperature * temp
        return temp

    def cov_loss(self, x, class_index):
        max_loss = torch.zeros(1).cuda()
        ret = None

        for i in range(self.args.sample_nums):
            cov = self.cov_matrix.generate_samples_batch(class_index, i).cuda()
            cov_feature1 = cov + x
            if self.args.dist_ways == 'cos':
                sim = F.cosine_similarity(cov_feature1, self.centroid[class_index].cuda())
            elif self.args.dist_ways == 'norm_2':
                sim = torch.norm(cov_feature1 - self.centroid[class_index].cuda(), p=2, dim=1)
            elif self.args.dist_ways == 'norm_1':
                sim = torch.norm(cov_feature1 - self.centroid[class_index].cuda(), p=1, dim=1)
            elif self.args.dist_ways == 'chebyshev':
                sim = torch.abs(cov_feature1 - self.centroid[class_index].cuda()).max(dim=1)[0]
            cos_loss = (1- sim.mean())

            if cos_loss > max_loss or ret == None:
                max_loss = cos_loss
                ret = cov_feature1
        # ret = F.normalize(ret, p=2, dim=-1)
        cov_logits1 = self.encode_fc(ret)
        # 更新存储样本用于计算分布
        self._dequeue_and_enqueue(x, torch.tensor(class_index).cuda())
        return cov_logits1

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    # def encode_new(self, args, x, session):
    #     x = F.normalize(x, p=2, dim=-1)
    #     base_protos = self.fc.weight.data[:args.base_class +(session - 1)*5 ].detach().cuda().data
    #     base_protos = F.normalize(base_protos, p=2, dim=-1)
    #     map1 = torch.matmul(x, base_protos.T)
    #     map1 = F.dropout(map1, p=0.8, training=True)
    #     map1 = torch.softmax(map1, dim=1)
    #
    #     x_base = torch.matmul(map1, base_protos)
    #     x_base =F.normalize(x_base, p=2, dim=-1)
    #     # x = torch.cat([x[:, random_index1, :],x_base[:, random_index2, :]],dim = 1)
    #     x = 0.5 * x + 0.5 *x_base
    #     return x


    def forward(self, x):
        feature = self.encode(x)
        logits = self.encode_fc(feature)
        return feature, logits


    @torch.no_grad()
    def update_fc(self, trainloader, class_list, session, m):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()
        # init  new_fc
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            self.update_fc_avg(data, label, class_list, m)

    @torch.no_grad()
    def update_fc_avg(self, data, labels, class_list, m):
        new_fc = []
        for class_index in class_list:
            for i in range(m):
                index = class_index * m + i
                data_index = (labels == index).nonzero().squeeze(-1)
                embedding = data[data_index]
                proto = embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))


    # def soft_calibration(self, args, session):
    #     base_protos = self.fc.weight.data[:args.base_class + (session - 1) * args.way ].detach().data
    #     base_protos = F.normalize(base_protos, p=2, dim=-1)
    #
    #     cur_protos = self.fc.weight.data[args.base_class + (
    #                 session - 1) * args.way: args.base_class + session * args.way].detach().data
    #     cur_protos = F.normalize(cur_protos, p=2, dim=-1)
    #
    #     weights = torch.mm(cur_protos, base_protos.T) * args.temperature
    #
    #     norm_weights = torch.softmax(weights, dim=1)
    #     delta_protos = torch.matmul(norm_weights, base_protos)
    #
    #     delta_protos = F.normalize(delta_protos, p=2, dim=-1)
    #
    #     updated_protos =  0.8 * cur_protos + 0.2 *delta_protos
    #
    #     self.fc.weight.data[
    #     args.base_class + (session - 1) * args.way: args.base_class + session * args.way] = updated_protos




