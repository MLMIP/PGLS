import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from tqdm import tqdm


def cos_f(a, b):
    cos = nn.CosineSimilarity()
    if len(a.shape) == 1 :
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    similiarity = cos(a,b)
    return similiarity





class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if args.dataset == 'cub200':
            self.K =3000

        else:
            self.K = 30000
        print('K:\n', self.K)
        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
            self.encoder_k = resnet50(False, args)

        if self.args.dataset in ['mini_imagenet', 'manyshotmini', 'imagenet100', 'imagenet1000',
                                 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
            self.encoder_k = resnet50(False, args)

        if self.args.dataset in ['cub200', 'manyshotcub']:
            self.num_features = 512
            self.encoder = resnet18(True, args)
            self.encoder_k = resnet50(True,args)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        nn.init.normal_(self.fc.weight, std=0.01, mean=0)

        self.to_num = nn.Linear(2048, self.num_features, bias=False)

        self.register_buffer("queue", torch.randn(self.num_features, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("lab_que", torch.randint(0, self.args.num_classes, [self.K, ]))


        self.centroid =torch.randn(self.args.num_classes, self.num_features)
        self.cov = torch.randn(self.args.num_classes, self.num_features, self.num_features)

        self.cat = nn.Sequential(
            nn.Linear(self.num_features * 2, self.num_features)
        )


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
            # temp = F.linear(F.normalize(temp, p=2, dim=-1), self.fc.weight)  # 16 * 200
            temp = self.args.temperature * temp

        elif 'dot' in self.mode:
            temp = self.fc(temp)
            temp = self.args.temperature * temp
        return temp


    def forward_metric(self, x):
        feature = self.encode(x)
        finial_feature = feature

        finial_feature = self.encode_fc(finial_feature)
        return feature,finial_feature



    def cov_loss(self,x,index):

        cov = self.cov[index, :, :].cuda()
        # x_normalized = F.normalize(x, p=2, dim=-1)
        temp = []
        for i in range(x.shape[0]):
            # temp.append(F.linear(x[i], F.normalize(cov[i], p=2, dim=-1)))
            temp.append(F.linear(x[i],cov[i] ))

        temp =  x + torch.stack(temp)/self.num_features
        return  temp

    def encode(self, x1):
        x2 = x1.detach()
        x1 = self.encoder(x1)  # 256 512 7 7
        x2 = self.encoder_k(x2)

        x1 = F.adaptive_avg_pool2d(x1, 1)
        x1 = x1.squeeze(-1).squeeze(-1)

        x2 = F.adaptive_avg_pool2d(x2, 1)
        x2 = x2.squeeze(-1).squeeze(-1)
        x2 = self.to_num(x2)
        x = self.cat(torch.cat((x1,x2),dim = 1))
        return x


    def forward(self, input):
        if self.mode != 'encoder':
            temp_concat,concat_feature = self.forward_metric(input)
            return temp_concat, concat_feature

        elif self.mode == 'encoder':
            concat_feature = self.encode(input)
            return concat_feature
        else:
            raise ValueError('Unknown mode')

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
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc, trainloader, session)

    @torch.no_grad()
    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            # print(class_index)
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
            self.centroid[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        # print(new_fc.mean())
        # 获取原型相关性：
        # max_cos = {}
        # index = 0
        # for class_index in class_list:
        #     max = 0
        #     for i in range(class_index):
        #         cos_sim = cos_f(self.centroid[i].unsqueeze(0), self.centroid[class_index].unsqueeze(0))
        #         if cos_sim > max:
        #             max = cos_sim
        #             index = i
        #     max_cos[index] = max
        # print(class_list,max_cos)
        return new_fc




