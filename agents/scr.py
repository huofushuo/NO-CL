import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
import numpy as np
import os

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )

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

        # save base pretrain
        model_dir = './pretrained/scr/base_{}classes_{}epoches_{}.pth'.format(self.base_class, self.base_epoch, self.data)

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()

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
                        batch_y = maybe_cuda(batch_y, self.cuda)

                        for j in range(self.mem_iters):
                            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                            if mem_x.size(0) > 0:
                                mem_x = maybe_cuda(mem_x, self.cuda)
                                mem_y = maybe_cuda(mem_y, self.cuda)
                                combined_batch = torch.cat((mem_x, batch_x))
                                combined_labels = torch.cat((mem_y, batch_y))
                                # combined_batch = batch_x
                                # combined_labels = batch_y
                                combined_batch_aug = self.transform(combined_batch)
                                features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                                loss = self.criterion(features, combined_labels)
                                losses.update(loss, batch_y.size(0))
                                self.opt.zero_grad()
                                loss.backward()
                                self.opt.step()

                        # update mem
                        self.buffer.update(batch_x, batch_y)
                        if (i+1) % 20 == 0 and self.verbose:
                                print(
                                    '==>>> it: {}, avg. loss: {:.6f}, '
                                        .format(i, losses.avg(), acc_batch.avg())
                                )
                torch.save(self.model.state_dict(), model_dir)
                self.after_train()
        else:
            # for name, param in self.model.named_parameters():
            #     if 'head' not in name and 'projector' not in name and 'linear' not in name and 'hyper' not in name:
            #         param.requires_grad = False
            for ep in range(self.epoch):
                for i, batch_data in enumerate(train_loader):
                    # batch update
                    batch_x, batch_y = batch_data
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)

                    for j in range(self.mem_iters):
                        mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                        if mem_x.size(0) > 0:
                            mem_x = maybe_cuda(mem_x, self.cuda)
                            mem_y = maybe_cuda(mem_y, self.cuda)
                            combined_batch = torch.cat((mem_x, batch_x))
                            combined_labels = torch.cat((mem_y, batch_y))
                            # combined_batch = batch_x
                            # combined_labels = batch_y
                            combined_batch_aug = self.transform(combined_batch)
                            features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                                                  self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                            loss = self.criterion(features, combined_labels)
                            losses.update(loss, batch_y.size(0))
                            self.opt.zero_grad()
                            loss.backward()
                            self.opt.step()

                    # update mem
                    self.buffer.update(batch_x, batch_y)
                    if (i+1) % 20 == 0 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                            .format(i, losses.avg(), acc_batch.avg())
                        )
            self.after_train()



def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


