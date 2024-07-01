# import new Network name here and add in model_class args
from .Network import *
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import torch.nn as nn
import random
import math

criterion = torch.nn.CrossEntropyLoss().cuda()
softmax_func = nn.Softmax(dim = 1).cuda()
logsoftmax_func = nn.LogSoftmax(dim=1).cuda()


def loss_a(logits, label):
    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device)
    label = label.to(device)

    loss = torch.zeros(1, device=device)
    logsoftmax_output = torch.nn.functional.log_softmax(logits, dim=1)

    # Get the confidence scores
    softmax = torch.nn.functional.softmax(logsoftmax_output, dim=1)
    max_score, max_label = softmax.max(dim=1)

    # Sotmax1 = softmax_func(logsoftmax_output[:,:100])
    # Pseudo_score1, Pseudo_lab1 = Sotmax1.max(dim=1)
    # loss1 = torch.zeros(1).cuda()
    #
    # label_Score_index = [[i, u] for i, u in zip(range(logits.shape[0]), label)]
    # label_Score = [Sotmax1[i] for i in label_Score_index]
    # label_Score = torch.stack(label_Score)

    # logsoftmax_output_final = [logsoftmax_output[i] for i in label_Score_index]
    # logsoftmax_output_final = torch.stack(logsoftmax_output_final)
    batch_size = logits.shape[0]
    index = np.zeros(batch_size)

    for i in range(batch_size):
        if max_label[i] == label[i]:
            loss -= logsoftmax_output[i][label[i]] * (2 - max_score[i])
            index[i] = 0
        else:
            if max_score[i] < 0.5:
                loss -= logsoftmax_output[i][label[i]] * (2 - max_score[i])
                index[i] = 1
            else:
                loss -= logsoftmax_output[i][label[i]] * (1 - max_score[i])
                index[i] = 2

    return loss / batch_size


def cos_f(a, b):
    if len(a.shape)==1 and len(b.shape)==1:
        a = a.unsqueeze(dim = 0)
        b = b.unsqueeze(dim = 0)
    cos = nn.CosineSimilarity()
    similiarity = cos(a, b)
    return similiarity


def base_train(model, trainloader, optimizer, epoch, args):

    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    args.epoch = epoch

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        high_feture, logits = model(data)

        train_label_np = train_label.cpu().numpy()

        logits_base = logits[:, :args.base_class]
        logits_new = logits[:, args.base_class:]
        # label_new = (logits_new.argmax(dim=1) ).cuda()
        label_new = (logits_new.argmax(dim = 1) + args.base_class).cuda()

        cov_feture = model.module.cov_loss(high_feture, train_label_np)
        cov_feture = model.module.encode_fc(cov_feture)

        cov_loss = loss_a(cov_feture, train_label)
        loss = F.cross_entropy(logits_base, train_label)
        loss1 = F.cross_entropy(logits, label_new)

        if epoch < args.epochs_base * 0.05:
            total_loss = loss +  loss1
        else:
            total_loss = loss +  cov_loss + loss1
        # if args.to_TSNE:
        #     _, Pseudo_lab = softmax_func(logits).max(dim=1)
        #     Data_path = os.path.join(args.logits_Data, args.dataset, 'train', 'Data')
        #     lab_path = os.path.join(args.logits_Data, args.dataset, 'train', 'label')
        #     PsedoLab_path = os.path.join(args.logits_Data, args.dataset, 'train', 'PsedoLab')
        #     cov_feture_path = os.path.join(args.logits_Data, args.dataset, 'train', 'cov_feture')
        #     if not os.path.exists(Data_path):
        #         os.makedirs(Data_path)
        #     if not os.path.exists(lab_path):
        #         os.makedirs(lab_path)
        #     if not os.path.exists(PsedoLab_path):
        #         os.makedirs(PsedoLab_path)
        #     if not os.path.exists(cov_feture_path):
        #         os.makedirs(cov_feture_path)
        #     np.save(Data_path + '/{}_batch'.format(i), logits.detach().cpu())
        #     np.save(lab_path + '/{}_batch'.format(i), train_label.detach().cpu())
        #     np.save(PsedoLab_path + '/{}_batch'.format(i), Pseudo_lab.detach().cpu())
        #     np.save(cov_feture_path + '/{}_batch'.format(i), cov_feture.detach().cpu())

        acc = count_acc(logits, train_label)

        # 输出显示
        lrc = optimizer.state_dict()['param_groups'][0]['lr']
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f} '.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        model.module._dequeue_and_enqueue(high_feture, train_label)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()

    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []

    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)
            label_np = label.cpu().numpy()
            # 通过协方差向量进一步精确原型
            # embedding = model.module.cov_loss(embedding, label_np)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    # class_num = [0]*200

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            _, logits3 = model(data)

            # if args.to_TSNE:
            #     Data_path = os.path.join(args.logits_Data, args.dataset, 'test', 'Data')
            #     lab_path = os.path.join(args.logits_Data, args.dataset, 'test', 'label')
            #     _, Pseudo_lab = softmax_func(logits3).max(dim=1)
            #     PsedoLab_path = os.path.join(args.logits_Data, args.dataset, 'test', 'PsedoLab')
            #     if not os.path.exists(Data_path):
            #         os.makedirs(Data_path)
            #     if not os.path.exists(lab_path):
            #         os.makedirs(lab_path)
            #     if not os.path.exists(PsedoLab_path):
            #         os.makedirs(PsedoLab_path)
            #     np.save(Data_path + '/{}_batch'.format(i), logits3.detach().cpu())
            #     np.save(lab_path + '/{}_batch'.format(i), test_label.detach().cpu())
            #     np.save(PsedoLab_path + '/{}_batch'.format(i), Pseudo_lab.detach().cpu())
            logits3 = logits3[:, :test_class]
            loss = F.cross_entropy(logits3, test_label)
            acc = count_acc(logits3, test_label)
            top5acc = count_acc_topk(logits3, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits3.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f} '.format(epoch, vl, va, va5))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)

        unseenac = 0
        seenac = 0
        perclassacc =0.0
        format_acc = []
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)

        # 格式化存入每个类的acc
            for i in range(test_class):
                if i % 10 == 0:
                    format_acc.append('\n')
                    format_acc.append(round(perclassacc[i], 2))

                else:
                    format_acc.append(round(perclassacc[i],2))
    return vl, va,seenac,unseenac,format_acc


