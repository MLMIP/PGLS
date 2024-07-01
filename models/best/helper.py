# import new Network name here and add in model_class args
from .Network import *
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch



def Robust_loss(logits, label, TH):
    logsoftmax_output = F.log_softmax(logits, dim=1)
    score = logsoftmax_output.softmax(dim=1)
    max_score, max_label = score.max(dim=1)

    # Condition
    weights1 = torch.where(max_label == label, 2.0, torch.zeros_like(max_score, dtype = float))
    weights2 = torch.where((max_score < TH) & (max_label != label), 1.0, torch.zeros_like(max_score, dtype = float))
    weights3 = torch.where((max_score >= TH) & (max_label != label), 0.5, torch.zeros_like(max_score, dtype = float))
    targe =  torch.diagonal(logsoftmax_output.transpose(0, 1)[label])
    WCE = -1 * targe * (weights1 + weights2 + weights3)
    return WCE.mean()


def base_train(model, trainloader, optimizer, scheduler, epoch, args,transform):

    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    args.epoch = epoch
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        train_label_np = train_label.cpu().numpy()
        # indices = np.random.permutation(data.size(0))
        b =data.size()[0]
        data = transform(data)
        m = data.size()[0] // b

        joint_labels = torch.stack([train_label + 0 for ii in range(m)], 1).view(-1)


        high_feture, logits = model(data)
        rand_feats = high_feture.view(data.size()[0] // m, -1, high_feture.size(-1))
        rand_index = torch.randint(0, m, size=(data.size()[0] // m,))
        rand_feats = rand_feats[torch.arange(rand_feats.size(0)), rand_index, :]


        label_new = (logits[:, args.base_class:].argmax(dim=1) + args.base_class).cuda()
        noise = torch.randn_like(rand_feats).cuda() * args.std
        high_feture_new =  F.dropout(rand_feats, p=0.8, training=True)
        num_samples_to_add = int(epoch/ args.epochs_base)
        increment_data = rand_feats[:num_samples_to_add] + noise[:num_samples_to_add]
        increment_label = label_new[:num_samples_to_add]
        increment_data = torch.cat((high_feture_new,increment_data),dim=0)
        increment_label = torch.cat((label_new,increment_label),dim = 0)
        logits_new = model.module.encode_fc(increment_data)
        Virtual_loss1 = F.cross_entropy(logits_new, increment_label)

        # j_candidates = [model.module.Virdict.get(i.item(), []) if isinstance(model.module.Virdict.get(i.item(), []), list)
        # else [model.module.Virdict.get(i.item(), [])] for i in train_label]
        # # Flatten the list of lists into a single list
        # j_indices = [idx for sublist in j_candidates for idx in sublist]
        # center_selected = center[j_indices].cuda()
        # similar_loss =  F.cosine_similarity(center_selected,rand_feats).mean()


        cov_logits1 = model.module.cov_loss(rand_feats, train_label_np)
        cov_loss = Robust_loss(cov_logits1[:, :args.base_class], train_label, 0.5)
        # cov_loss = F.cross_entropy(cov_logits1[:, :args.base_class], joint_labels)

        joint_preds = logits[:, :args.base_class]
        loss = F.cross_entropy(joint_preds, joint_labels)


        total_loss =   loss +  Virtual_loss1 +  0.5 * cov_loss

        acc = count_acc(joint_preds, joint_labels)

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


def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            embedding,_ = model(data)

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


def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            _,joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class]

            loss = F.cross_entropy(joint_preds, test_label)
            acc = count_acc(joint_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va
