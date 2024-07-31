# import new Network name here and add in model_class args
from .Network import *
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset


def Robust_loss(logits, label):
    logsoftmax_output = F.log_softmax(logits, dim=1)
    score = logsoftmax_output.softmax(dim=1)
    max_score, max_label = score.max(dim=1)
    # Condition
    weights1 = torch.where(max_label == label, 2.0, torch.zeros_like(max_score, dtype = float))
    weights2 = torch.where((max_score < 0.5) & (max_label != label), 1.0, torch.zeros_like(max_score, dtype = float))
    # weights3 = torch.where((max_score >= 0.5) & (max_label != label), 0.1, torch.zeros_like(max_score, dtype = float))
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


        b =data.size()[0]
        data = transform(data)

        m = data.size()[0] // b
        joint_labels = torch.stack([train_label * m + ii for ii in range(m)], 1).view(-1)
        train_label_np = joint_labels.cpu().numpy()


        high_feture, logits = model(data)
        label_new = (logits[:, args.base_class * m:].argmax(dim=1) + args.base_class * m).cuda()

        alph = max(0.2,epoch / args.epochs_base)
        N = int(alph * high_feture.size(0))

        high_feture_new =  F.dropout(high_feture, p=args.drop_rate, training=True)

        noise = torch.randn_like(high_feture).cuda() * args.std
        increment_data = high_feture + noise

        increment_logits = model.module.encode_fc(increment_data)
        inc_loss = F.cross_entropy(increment_logits, label_new, reduction='none')
        indices = torch.topk(inc_loss, k=N, largest=False).indices

        Virtual_data = torch.cat((high_feture_new,increment_data[indices]),dim=0)
        Virtual_label = torch.cat((label_new,label_new[indices]),dim = 0)
        Virtual_logits = model.module.encode_fc(Virtual_data)

        virtual_loss = F.cross_entropy(Virtual_logits, Virtual_label)


        cov_logits1 = model.module.cov_loss(high_feture, train_label_np)
        cov_loss = Robust_loss(cov_logits1[:, :args.base_class * m], joint_labels)


        joint_preds = logits[:, :args.base_class * m]
        loss = F.cross_entropy(joint_preds, joint_labels)


        total_loss = loss + virtual_loss + 0.5 * cov_loss


        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m
        acc = count_acc(agg_preds, train_label)


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



def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []

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

    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

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

