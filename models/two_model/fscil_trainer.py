from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans
from .helper import *
from utils import *
from dataloader.data_utils import *
import  objgraph
from sklearn.decomposition import PCA

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            #self.best_model_dict = torch.load(self.args.model_dir)['state_dict']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load increase class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                for epoch in range(args.epochs_base):
                    start_time = time.time()

                    Data = self.model.module.queue
                    Data = Data.cpu().detach().numpy().T

                    for i in range(args.base_class):
                        data_index = (i == self.model.module.lab_que).nonzero().squeeze(-1)
                        data_index_np = data_index.cpu().numpy()
                        # 每个类的协方差矩阵
                        data_cov = torch.from_numpy(np.cov(Data[data_index_np], rowvar=False))
                        self.model.module.cov[i, :, :] = data_cov
                        embedding = np.mean(Data[data_index_np], axis=0)
                        self.model.module.centroid[i, :] = torch.from_numpy(embedding)
                        # 是否保存数据
                        # if epoch == args.epochs_base - 1 and epoch > 1:
                        #     # 保存最好一个epoch的训练数据用于TSNE
                        #     args.to_TSNE = True
                        #     self.model.load_state_dict(self.best_model_dict)
                    del Data,embedding,data_index_np
                    tl, ta = base_train(self.model, trainloader, optimizer, epoch, args)
                    args.to_TSNE = False
                    # test model with all seen class
                    tsl, tsa, _, _, _ = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        find_best = True
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))



                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa ,_,_,_= test(self.model, testloader, 0, args, session)
                    # if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of increase session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                if session == args.sessions - 1:
                    # 保存最后一个场景的测试数据用于TSNE
                    args.to_TSNE = True
                    # 初始化训练及测试代码
                tsl1, tsa1, seenac, unseenac, format_acc = test(self.model, testloader, 0, args, session,
                                                                validation=False)
                result_list.append(
                    '''seenac:%03d,unseenac:%.5f,test_acc:%.5f,session:%d 
                    all class acc:%s''' % (
                        seenac, unseenac, tsa1, session, format_acc))
                print("acc before train is : %.3f" % (tsa1 * 100))

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa1 * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None