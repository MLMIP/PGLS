# import new Network name here and add in model_class args
from .Network import *
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

# class SupContrastive(nn.Module):
#     def __init__(self, temperature = 0.1):
#         super(SupContrastive, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, y_pred, y_true,fc, alph):
#         # y_pred 25*640 y_true 25*classess  fc classes*640
#         normalized_features = F.normalize(y_pred, p=2, dim=1)
#         normalized_fc = F.normalize(fc, p=2, dim=1)
#         similarity_matrix = torch.mm(normalized_features, normalized_fc.T)
#
#         exp_sim = torch.exp(similarity_matrix)
#         pos_sum = torch.sum(similarity_matrix * y_true, dim=1)
#
#         # 计算每个样本对应的阈值
#         thresholds = pos_sum
#         thresholds_expanded = thresholds.unsqueeze(1).expand_as(exp_sim)
#         masked_sim = torch.where(exp_sim > thresholds_expanded, exp_sim, torch.zeros_like(exp_sim))
#         all_sum = torch.sum(masked_sim,dim=1)
#
#         # 避免除以0
#         pos_sum = torch.clamp(pos_sum, min=1e-9)
#         all_sum = torch.clamp(all_sum, min=1e-9)
#         contrastive_loss = -torch.log(pos_sum / all_sum)
#         return contrastive_loss.mean()


# def prototype_calibration_loss(model, session, base_class, way,  alpha=0.8):
#     """
#     model: 模型对象，具有fc层的权重属性。
#     args: 参数对象，包含base_class, way等信息。
#     session: 当前训练会话。
#     base_class: 基础类别的数量。
#     way: 每个训练会话中新引入类别的数量。
#     temperature: 温度参数，用于调整权重。
#     alpha: 原型更新时新原型的权重。
#     """
#     base_protos = model.module.fc.weight.data[:base_class + (session - 1) * way]
#     base_protos = F.normalize(base_protos, p=2, dim=-1)
#
#     cur_protos = model.module.fc.weight.data[base_class + (session - 1) * way: base_class + session * way]
#     cur_protos = F.normalize(cur_protos, p=2, dim=-1)
#
#     weights = torch.mm(cur_protos, base_protos.T)
#     norm_weights = torch.softmax(weights, dim=1)
#     delta_protos = torch.matmul(norm_weights, base_protos)
#     delta_protos = F.normalize(delta_protos, p=2, dim=-1)
#
#     # 计算软校准后的原型与当前原型之间的距离作为损失
#     updated_protos = alpha * cur_protos + (1 - alpha) * delta_protos
#     loss = F.mse_loss(cur_protos, updated_protos)
#
#     return loss

def Robust_loss(logits, label):
    logsoftmax_output = F.log_softmax(logits, dim=1)
    score = logsoftmax_output.softmax(dim=1)
    max_score, max_label = score.max(dim=1)
    # Condition
    weights1 = torch.where(max_label == label, 0.0, torch.zeros_like(max_score, dtype = float))
    # weights2 = torch.where( (max_score <= 0.5) & (max_label != label),5.0, torch.zeros_like(max_score, dtype = float))
    weights2 = torch.where( (max_label != label), 1.0, torch.zeros_like(max_score, dtype=float))
    targe =  torch.diagonal(logsoftmax_output.transpose(0, 1)[label])
    WCE = -1 * targe * (weights1 + weights2)

    return WCE.mean()




def base_train(model, trainloader, optimizer, epoch, args,transform):

    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    args.epoch = epoch
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        # 采取数据增强后的额外复制数据标签代码
        b =data.size()[0]
        data = transform(data)
        # m为增强后样本倍数
        m = data.size()[0] // b
        joint_labels = torch.stack([train_label * m + ii for ii in range(m)], 1).view(-1)
        train_label_np = joint_labels.cpu().numpy()

        # 前馈网络
        high_feture, logits = model(data)
        # 虚拟类标签
        label_new = (logits[:, args.base_class * m:].argmax(dim=1) + args.base_class * m).cuda()

        # 增加的细腻度虚拟类数目
        alph = max(0.2,epoch / args.epochs_base)
        N = int(alph * high_feture.size(0))

        # 粗粒度虚拟特征
        high_feture_new =  F.dropout(high_feture, p=args.drop_rate, training=True)

        # 细腻度虚拟类
        noise = torch.randn_like(high_feture).cuda() * args.std
        increment_data = high_feture + noise

        # 细腻度虚拟类噪声排序
        increment_logits = model.module.encode_fc(increment_data)
        inc_loss = F.cross_entropy(increment_logits, label_new, reduction='none')
        indices = torch.topk(inc_loss, k=N, largest=False).indices

        Virtual_data = torch.cat((high_feture_new,increment_data[indices]),dim=0)
        Virtual_label = torch.cat((label_new,label_new[indices]),dim = 0)
        Virtual_logits = model.module.encode_fc(Virtual_data)

        virtual_loss = F.cross_entropy(Virtual_logits, Virtual_label)

        # 协方差扰动
        cov_logits = model.module.cov_loss(high_feture, train_label_np)
        cov_loss = Robust_loss(cov_logits[:, :args.base_class * m], joint_labels)


        joint_preds = logits[:, :args.base_class * m]
        loss = F.cross_entropy(joint_preds, joint_labels)

        # if epoch in [0,20,40,60,80,100,120,140,160,180,199]:
        #     root = '/media/s5/wyj/Datas_with_code/CUB3/TSNE_data'
        #     Data_path = os.path.join(root,str(epoch), args.dataset, 'Data')
        #     # cov_feture_path = os.path.join(root,epoch, 'train', 'cov_feture')
        #     lab_path = os.path.join(root,str(epoch),  args.dataset, 'label')
        #     Vitual1_path = os.path.join(root,str(epoch),  args.dataset, 'VC')
        #     Vitual1_label_path = os.path.join(root,str(epoch), args.dataset, 'VC_label')
        #     if not os.path.exists(Data_path):
        #         os.makedirs(Data_path)
        #     if not os.path.exists(Vitual1_path):
        #         os.makedirs(Vitual1_path)
        #     if not os.path.exists(lab_path):
        #         os.makedirs(lab_path)
        #     # if not os.path.exists(cov_feture_path):
        #     #     os.makedirs(cov_feture_path)
        #     if not os.path.exists(Vitual1_label_path):
        #         os.makedirs(Vitual1_label_path)
        #     np.save(Data_path + '/{}_batch'.format(i), high_feture.detach().cpu())
        #     np.save(lab_path + '/{}_batch'.format(i), train_label.detach().cpu())
        #     np.save(Vitual1_path + '/{}_batch'.format(i), Virtual_data.detach().cpu())
        #     np.save(Vitual1_label_path + '/{}_batch'.format(i), Virtual_label.detach().cpu())
        #     # np.save(cov_feture_path + '/{}_batch'.format(i), cov_feture.detach().cpu())
        # total_loss = loss + virtual_loss + cov_loss
        total_loss = loss +  virtual_loss + 0.5 * cov_loss
        # total_loss = loss + virtual_loss + 2.0 * cov_loss

        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m
        acc = count_acc(agg_preds, train_label)

        # 输出显示
        lrc = optimizer.state_dict()['param_groups'][0]['lr']
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f} '.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


# base阶段训练完之后，用均值重新赋值base分类器权重，保持和增量阶段的一致性
def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []

    # 获取所有样本的特征表达
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            embedding,_ = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    # 求均值
    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    # 分类器赋值
    proto_list = torch.stack(proto_list, dim=0)
    model.module.fc.weight.data[:args.base_class*m] = proto_list

    return model


def test(model, testloader, epoch, transform, args, session,validation = True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            _,joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class * m]

            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

            lgt = torch.cat([lgt, agg_preds.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    lgt = lgt.view(-1, test_class)
    lbs = lbs.view(-1)
    if validation is not True:
        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
        cm = confmatrix(lgt, lbs, save_model_dir)
        perclassacc = cm.diagonal()
        seenac = np.mean(perclassacc[:args.base_class])
        unseenac = np.mean(perclassacc[args.base_class:])
        print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)


    return vl, va


# 增量阶段，没有继续做了
# def update_fc_ft(trainloader, data_transform, model, m, session, args):
#     # incremental finetuning
#     old_class = args.base_class + args.way * (session - 1)
#     new_class = args.base_class + args.way * session
#
#     new_fc = nn.Parameter(
#         torch.rand(args.way*m, model.module.num_features, device="cuda"),
#         requires_grad=True)
#     new_fc.data.copy_(model.module.fc.weight[old_class*m : new_class*m, :].data)
#
#     optimizer = torch.optim.SGD([
#         {'params': model.module.parameters(), 'lr': args.lr_new},  # 使用默认学习率
#         {'params': new_fc, 'lr': args.lr_new }  # 为 new_fc 设置特定的学习率
#     ])
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_new, eta_min=0.01)
#
#     for i, batch in enumerate(trainloader):
#         data, label = [_.cuda() for _ in batch]
#         data_classify = data_transform(data)
#         x = model.module.encode(data_classify)
#         # # 确保x的第一个维度（样本数）可以被5整除
#         # remainder = x.size(0) % 5
#         # if remainder != 0:
#         #     x = x[:-remainder]
#         #
#         # x_reshaped = x.view(-1, 5, x.size(1))  # 形状变为 [N/5, 5, D]
#         # x = x_reshaped.mean(dim=1)  # 对每5个样本取均值，形状为 [N/5, D]
#         #
#         # # 对label进行切片，每5个样本选择一个标签
#         # label_reshaped = label.view(-1, 5)  # 形状变为 [N/5, 5]
#         # label = label_reshaped[:, 0]  # 选择每个5样本切片的第一个标签
#
#
#     for epoch in tqdm(range(args.epochs_new)):
#         features = model.module.encode_new(args, x.detach(), session)
#
#         old_fc = model.module.fc.weight[:old_class * m, :].clone().detach()
#         fc = torch.cat([old_fc, new_fc], dim=0)
#
#         # # 创建一个全零张量作为 one-hot 编码的标签
#         # one_hot_labels = torch.zeros(features.shape[0], new_class).cuda()
#         # one_hot_labels = one_hot_labels.scatter_(1, label.unsqueeze(1), 1)
#
#         # 分类损失
#         logits = model.module.get_logits(features , fc)
#
#         calibration_loss = prototype_calibration_loss(model, session, args.base_class, args.way, alpha=0.8)
#
#         loss = logits + calibration_loss
#         # 累计梯度
#         if (epoch + 1) % 1 == 0:
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()  # 清除累计的梯度
#             # print(optimizer.state_dict()['param_groups'][0]['lr'])
#             scheduler.step()
#
#     model.module.fc.weight.data[old_class * m: new_class * m, :].copy_(new_fc.data)
#     # model.module.fc.weight.data[old_class * m: new_class * m, :].copy_(model.module.soft_calibration(args, session).data)
#
#
#     # if args.dataset == 'mini_imagenet':
#     #     optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
#     #                                  {'params': model.module.fc.parameters(), 'lr': 0.05 * args.lr_new},
#     #                                  {'params': model.module.encoder.layer4.parameters(), 'lr': 0.001 * args.lr_new}, ],
#     #                                 momentum=0.9, dampening=0.9, weight_decay=0)
#     #
#     # if args.dataset == 'cub200':
#     #     optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
#     #                                 momentum=0.9, dampening=0.9, weight_decay=0)
#     #
#     # elif args.dataset == 'cifar100':
#     #     optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
#     #                                   {'params': model.module.fc.parameters(), 'lr': 0.01 * args.lr_new},
#     #                                   {'params': model.module.encoder.layer3.parameters(), 'lr': 0.02 * args.lr_new}],
#     #                                  weight_decay=0)
#
#     # optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}])
#     # criterion = SupContrastive().cuda()
#     #
#     # # 复现样本
#     # old_center = model.module.fc.weight[:old_class * m, :]
#     # # 存储数据和标签的列表
#     # all_samples = []
#     # # 遍历每个类别
#     # for i in range(args.base_class):
#     #     # 生成噪声样本
#     #     noise = model.module.cov_matrix.distribution_functions[i]().sample((args.way,)).to(torch.float32).cuda()
#     #     # 生成数据样本
#     #     samples = old_center[i].repeat(args.way, 1) + noise
#     #     # 生成标签
#     #     labels = torch.full((args.way,), i, dtype=torch.int64).cuda()
#     #     # 将数据和标签打包成一个数据集
#     #     dataset = TensorDataset(samples, labels)
#     #     # 将数据集添加到列表中
#     #     all_samples.append(dataset)
#     # for i, batch in enumerate(trainloader):
#     #     data, label = [_.cuda() for _ in batch]
#     #     data_classify = data_transform(data)
#     #     features, _ = model(data_classify)
#     #     dataset = TensorDataset(features.detach(), label)
#     #     all_samples.append(dataset)
#     # # 将所有数据集拼接成一个数据集
#     # concat_dataset = ConcatDataset(all_samples)
#     # dataloader = DataLoader(concat_dataset, batch_size=64, shuffle=True)