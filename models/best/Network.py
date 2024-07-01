import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12 import *
from tqdm import tqdm
from torch.nn import init
from collections import OrderedDict
import numpy as np
from utils import *

from torch.distributions.multivariate_normal import MultivariateNormal

class Cov:
    def __init__(self, num_classes, num_features,sample_nums = 1):
        self.args = {"num_classes": num_classes}
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

        for class_idx in range(self.args["num_classes"]):
            if class_idx in self.distribution_functions:
                distribution = self.distribution_functions[class_idx]
                samples = distribution().sample((self.sample_nums,)).to(torch.float32)
                pre_generated_samples[class_idx] = samples
            else:
                # 处理未定义分布函数的情况
                # 可以选择生成默认分布或者抛出异常，这里生成一个简单的单位方差正态分布
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


class MYNET(nn.Module):

    def __init__(self, args, mode=None, trans =1):
        super().__init__()

        self.mode = mode
        self.args = args
        if args.dataset == 'cub200':
            self.K =3000 * trans
        else:
            self.K = 30000 * trans

        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            self.encoder = ResNet12()
            self.num_features = 640

        if self.args.dataset in ['mini_imagenet', 'manyshotmini', 'imagenet100', 'imagenet1000',
                                 'mini_imagenet_withpath']:

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
        self.register_buffer("lab_que", torch.randint(0, self.args.num_classes * trans, [self.K, ]))

        self.centroid =torch.randn(self.args.num_classes * trans , self.num_features)
        self.cov_matrix = Cov(self.args.num_classes * trans, self.num_features,sample_nums=args.sample_nums)




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
            cosine_sim = F.cosine_similarity(cov_feature1, self.centroid[class_index].cuda())
            cos_loss = (1 - cosine_sim).mean()

            if cos_loss > max_loss:
                max_loss = cos_loss
                ret = cov_feature1
        cov_logits1 = self.encode_fc(ret)
        self._dequeue_and_enqueue(x, torch.tensor(class_index).cuda())
        return cov_logits1

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        feature = self.encode(x)
        logits = self.encode_fc(feature)
        return feature, logits


    @torch.no_grad()
    def update_fc(self, trainloader, class_list, session):
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
            self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(self.fc.weight.data[:self.args.base_class + session * self.args.way], data, label,
                              session)
        elif 'model' in self.args.new_mode:
            self.update_model_ft(self.fc.weight.data[:self.args.base_class + session * self.args.way], trainloader,
                              session)

    @torch.no_grad()
    def update_fc(self, dataloader, class_list, transform, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            data = self.encode(data)
            data.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list) * m, self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m)

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

    def update_fc_ft(self,new_fc,data,label,session):
        old_classes_nums = self.args.base_class + (session - 1) * self.args.way
        cur_nums = self.args.way * 5
        new_fc1 = new_fc[:self.args.base_class].clone().detach()
        Q, R = torch.linalg.qr(new_fc1.T)
        R_normal = F.normalize(R.abs(), p=2, dim=-1)
        R = R *torch.where(R_normal > 0.05, 1, 0)
        new_fc1 = torch.matmul(Q, R).T
        new_fc[:self.args.base_class] = new_fc1

        # 获取新数据在Q上面的激活，筛除部分不重要特征
        data_R = torch.matmul(data, Q)
        R_normal = F.normalize(data_R.abs(), p=2, dim=-1)
        data_R = data_R * torch.where(R_normal > 0.1, 1, 0)
        data = torch.matmul(data_R, Q.T)

        new_fc.requires_grad=True
        new_net = MLPFFNNeck(self.num_features, self.num_features).cuda()
        for param in new_net.parameters():
            param.requires_grad = True

        optimized_parameters = [{'params': [new_fc]}, {'params': new_net.parameters()}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)


        data = torch.cat([data,self.fc.weight.data[:old_classes_nums].cuda()], dim =0)
        label =torch.cat([label, torch.arange(old_classes_nums).cuda()], dim=0)


        for epoch in range(self.args.epochs_new):
            # data = new_net(data.detach())
            logits = self.get_logits(data,new_fc)
            # loss = F.cross_entropy(logits[:50], label[:50])
            loss =  F.cross_entropy(logits[:cur_nums], label[:cur_nums]) + F.cross_entropy(logits[cur_nums:], label[cur_nums:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        self.fc.weight.data[:self.args.base_class + self.args.way * session, :].copy_(new_fc.data)


    def update_model_ft(self,new_fc,trainloader,session):
        old_classes_nums = self.args.base_class + (session - 1) * self.args.way
        cur_nums = self.args.way * 5
        new_fc1 = new_fc[:self.args.base_class].clone().detach()

        Q, R = torch.linalg.qr(new_fc1.T)

        R_normal = F.normalize(R.abs(), p=2, dim=-1)
        R = R *torch.where(R_normal > 0.05, 1, 0)
        new_fc1 = torch.matmul(Q, R).T

        # 分类任务的优化器
        optimizer_classification = torch.optim.SGD([
            {'params': self.encoder.parameters()},
            {'params': self.fc.parameters()}
        ], lr=0.1, momentum=0.9, dampening=0.9, weight_decay=0)

        # 蒸馏任务的优化器
        optimizer_distillation = torch.optim.SGD(self.encoder.parameters(), lr=0.01, momentum=0.9,
                                                 dampening=0.9, weight_decay=0)

        for epoch in range(self.args.epochs_new):
            for batch in trainloader:
                data, label = [_.cuda() for _ in batch]
                data = self.encode(data)
                # 分类任务的前向传播和损失计算
                logits_classification = self.get_logits(data, self.fc.weight)
                loss_classification = F.cross_entropy(logits_classification, label)

                # 蒸馏任务的前向传播和损失计算
                loss_distillation = F.mse_loss(new_fc1, self.fc.weight[:self.args.base_class])

                # 分类任务的反向传播和优化
                optimizer_classification.zero_grad()
                loss_classification.backward()
                optimizer_classification.step()

                # 蒸馏任务的反向传播和优化
                optimizer_distillation.zero_grad()
                loss_distillation.backward()
                optimizer_distillation.step()



def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


