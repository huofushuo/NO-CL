import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
import torch.nn as nn
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from loss import agmax_loss, cross_entropy_loss


class ExperienceReplay_DVC(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_DVC, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.agent = params.agent
        self.dl_weight = params.dl_weight

        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
           ColorJitter(0.4, 0.4, 0.4, 0.1),
           RandomGrayscale(p=0.2)

        )
        self.L2loss = torch.nn.MSELoss()

    def train_learner(self, x_train, y_train, session):
        self.before_train(x_train, y_train)
        # set up loader
        if session == 0:
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
            train_loader = data.DataLoader(train_dataset, batch_size=self.base_batch, shuffle=True, num_workers=0,
                                           drop_last=True)
        else:
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
            train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                           drop_last=True)
        # set up model
        self.model = self.model.train()
        self.transform = self.transform.cuda()

        # save base pretrain
        model_dir = './pretrained/dvc/base_{}classes_{}epoches_{}.pth'.format(self.base_class, self.base_epoch, self.data)

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()

        if session == 0:
            if self.resume:
                state_dict = torch.load(model_dir)
                self.model.load_state_dict(state_dict, strict=False)
                for i, batch_data in enumerate(train_loader):
                    # batch update
                    batch_x, batch_y = batch_data
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    # update mem
                    self.buffer.update(batch_x, batch_y)
                self.after_train()
            else:
                for ep in range(self.base_epoch):
                    for i, batch_data in enumerate(train_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_x_aug = self.transform(batch_x)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        for j in range(self.mem_iters):
                            y = self.model(batch_x, batch_x_aug)
                            z, zt, _,_ = y
                            ce = cross_entropy_loss(z, zt, batch_y, label_smoothing=0)


                            agreement_loss, dl = agmax_loss(y, batch_y, dl_weight=self.dl_weight)
                            loss  = ce + agreement_loss + dl

                            if self.params.trick['kd_trick']:
                                loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                           self.kd_manager.get_kd_loss(z, batch_x)
                            if self.params.trick['kd_trick_star']:
                                loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                                       (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z, batch_x)
                            _, pred_label = torch.max(z, 1)
                            correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                            # update tracker
                            acc_batch.update(correct_cnt, batch_y.size(0))
                            losses_batch.update(loss, batch_y.size(0))
                            # backward
                            self.opt.zero_grad()
                            loss.backward()

                            # mem update
                            if self.params.retrieve == 'MGI':
                                mem_x, mem_x_aug, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                            else:
                                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                                if mem_x.size(0) > 0:
                                    mem_x_aug = self.transform(mem_x)

                            if mem_x.size(0) > 0:
                                mem_x = maybe_cuda(mem_x, self.cuda)
                                mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                                mem_y = maybe_cuda(mem_y, self.cuda)
                                y = self.model(mem_x, mem_x_aug)
                                z, zt, _,_ = y
                                ce = cross_entropy_loss(z, zt, mem_y, label_smoothing=0)
                                agreement_loss, dl = agmax_loss(y, mem_y, dl_weight=self.dl_weight)
                                loss_mem = ce  + agreement_loss + dl

                                if self.params.trick['kd_trick']:
                                    loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                               self.kd_manager.get_kd_loss(z, mem_x)
                                if self.params.trick['kd_trick_star']:
                                    loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                           (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z,
                                                                                                                 mem_x)
                                # update tracker
                                losses_mem.update(loss_mem, mem_y.size(0))
                                _, pred_label = torch.max(z, 1)
                                correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                                acc_mem.update(correct_cnt, mem_y.size(0))

                                loss_mem.backward()
                            self.opt.step()

                        # update mem
                        self.buffer.update(batch_x, batch_y)

                        if (i+1) % 20 == 0 and self.verbose:
                            print('==>>> it: {}, avg. loss: {:.6f}, '
                                'running train acc: {:.3f}'.format(i, losses_batch.avg(), acc_batch.avg()))
                            print('==>>> it: {}, mem avg. loss: {:.6f}, '
                                'running mem acc: {:.3f}' .format(i, losses_mem.avg(), acc_mem.avg()))
                torch.save(self.model.state_dict(), model_dir)
                self.after_train()

        else:
            # for name, param in self.model.named_parameters():
            #     if 'head' not in name and 'projector' not in name and 'linear' not in name and 'QNet' not in name:
            #         param.requires_grad = False
            for ep in range(self.epoch):
                for i, batch_data in enumerate(train_loader):
                    # batch update
                    batch_x, batch_y = batch_data
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_x_aug = self.transform(batch_x)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    for j in range(self.mem_iters):
                        y = self.model(batch_x, batch_x_aug)
                        z, zt, _, _ = y
                        ce = cross_entropy_loss(z, zt, batch_y, label_smoothing=0)

                        agreement_loss, dl = agmax_loss(y, batch_y, dl_weight=self.dl_weight)
                        loss = ce + agreement_loss + dl

                        if self.params.trick['kd_trick']:
                            loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(z, batch_x)
                        if self.params.trick['kd_trick_star']:
                            loss = 1 / ((self.task_seen + 1) ** 0.5) * loss + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z, batch_x)
                        _, pred_label = torch.max(z, 1)
                        correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                        # update tracker
                        acc_batch.update(correct_cnt, batch_y.size(0))
                        losses_batch.update(loss, batch_y.size(0))
                        # backward
                        self.opt.zero_grad()
                        loss.backward()

                        # mem update
                        if self.params.retrieve == 'MGI':
                            mem_x, mem_x_aug, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                        else:
                            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                            if mem_x.size(0) > 0:
                                mem_x_aug = self.transform(mem_x)

                        if mem_x.size(0) > 0:
                            mem_x = maybe_cuda(mem_x, self.cuda)
                            mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                            mem_y = maybe_cuda(mem_y, self.cuda)
                            y = self.model(mem_x, mem_x_aug)
                            z, zt, _, _ = y
                            ce = cross_entropy_loss(z, zt, mem_y, label_smoothing=0)
                            agreement_loss, dl = agmax_loss(y, mem_y, dl_weight=self.dl_weight)
                            loss_mem = ce + agreement_loss + dl

                            if self.params.trick['kd_trick']:
                                loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                           self.kd_manager.get_kd_loss(z, mem_x)
                            if self.params.trick['kd_trick_star']:
                                loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                           (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z,
                                                                                                                 mem_x)
                            # update tracker
                            losses_mem.update(loss_mem, mem_y.size(0))
                            _, pred_label = torch.max(z, 1)
                            correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                            acc_mem.update(correct_cnt, mem_y.size(0))

                            loss_mem.backward()
                        self.opt.step()

                    # update mem
                    self.buffer.update(batch_x, batch_y)

                    if (i+1) % 20 == 0 and self.verbose:
                        print('==>>> it: {}, avg. loss: {:.6f}, '
                              'running train acc: {:.3f}'.format(i, losses_batch.avg(), acc_batch.avg()))
                        print('==>>> it: {}, mem avg. loss: {:.6f}, '
                              'running mem acc: {:.3f}'.format(i, losses_mem.avg(), acc_mem.avg()))
                torch.save(self.model.state_dict(), model_dir)
                self.after_train()








