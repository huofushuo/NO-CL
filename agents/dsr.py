import torch
from torch.utils import data
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from utils.criterion import myCosineLoss
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class DSR(ContinualLearner):
    def __init__(self, model, opt, params):
        super(DSR, self).__init__(model, opt, params)
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
        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()
        cos_criterion = myCosineLoss(rep='real')
        cos_criterion = cos_criterion.cuda()
        # load base pretrain
        model_dir = './pretrained/dsr/base_{}classes_{}epoches_{}_w_heavy_2048_hd_0.1_w_tuckey.pth'.format(self.base_class, self.base_epoch, self.data)

        print(model_dir)
        if session == 0:
            # resume pretrained parameters
            if self.resume:
                state_dict = torch.load(model_dir)
                self.model.load_state_dict(state_dict, strict=False)
                # update prototype and save parameters
                prototype = compute_prototype(self.model, train_loader, session)
                torch.save(self.model.state_dict(), model_dir)
                self.after_train()
                return prototype
            else:
                for ep in range(self.base_epoch):
                    for i, batch_data in enumerate(train_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        batch_x_aug = self.transform(batch_x)
                        ######################sc_loss####################################
                        features = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                              self.model.forward(batch_x_aug).unsqueeze(1)], dim=1)
                        loss_sc = self.criterion(features, batch_y)
                        loss = loss_sc
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        if (i+1) % 10 == 0 and self.verbose:
                                print(
                                    '==>>> it: {}, avg. loss: {:.6f}, '
                                        .format(i, losses.avg(), acc_batch.avg()))

                # update prototype and save parameters
                prototype = compute_prototype(self.model, train_loader, session)
                torch.save(self.model.state_dict(), model_dir)
                self.after_train()
                return prototype
        else:
            for name, param in self.model.named_parameters():
                if 'head' not in name and 'projector' not in name and 'linear' not in name and 'hyper' not in name:
                    param.requires_grad = False

            prototype = np.load('./proto/prototypes.npy', allow_pickle=True).item()
            prototype = update_initial_prototype(self.model, train_loader, prototype)

            for ep in range(self.inner_epoch):
                prototype['class_mean'][:] = np.array((refine_hyper(torch.tensor(np.array(prototype['class_mean'][:])).cuda(), self.base_class)).cpu())
                ###########################pseudo_gaa_loss####################################
                feats, labels = sample_labeled_features(prototype['gaa_mean'][:], prototype['gaa_std'][:],
                                                        prototype['class_label'][:], num_per_base_class=20, num_per_novel_class=20, base=self.base_class)
                hd_features = self.model.hd_features(feats)
                class_mean = torch.as_tensor(np.array(prototype['class_mean'][:])).cuda()
                class_mean = torch.cat([class_mean[:self.base_class].repeat_interleave(20, 0), class_mean[self.base_class:].repeat_interleave(20, 0)], dim=0)
                loss_gaa_cos = cos_criterion(hd_features, class_mean)
                loss = loss_gaa_cos
                losses.update(loss, feats.size(0))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if (ep+1) % 10 == 0 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        .format(ep, losses.avg(), acc_batch.avg()))
                prototype = update_final_prototype(self.model, prototype, base=self.base_class, data_loader = train_loader)
                torch.cuda.empty_cache()
            self.after_train()
            return prototype

def compute_prototype(model, data_loader, session):
    model.eval()
    count = 0
    gaa_embeddings = []
    embeddings = []
    embeddings_labels = []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            count += 1
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            embed_gaa_feat = model.gaa_features(inputs)
            embed_feat = model.features(inputs)
            embeddings_labels.append(labels.cpu().numpy())
            gaa_embeddings.append(embed_gaa_feat.cpu().numpy())
            embeddings.append(embed_feat.cpu().numpy())
    gaa_embeddings = np.asarray(gaa_embeddings)
    gaa_embeddings = np.reshape(
        gaa_embeddings, (gaa_embeddings.shape[0] * gaa_embeddings.shape[1], gaa_embeddings.shape[2]))
    embeddings = np.asarray(embeddings)
    embeddings = np.reshape(
        embeddings, (embeddings.shape[0]*embeddings.shape[1], embeddings.shape[2]))
    embeddings_labels = np.asarray(embeddings_labels)
    embeddings_labels = np.reshape(
        embeddings_labels, embeddings_labels.shape[0]*embeddings_labels.shape[1])
    labels_set = np.unique(embeddings_labels)
    gaa_mean = []
    gaa_std = []
    class_mean = []
    class_mean_sign = []
    class_std = []
    class_label = []
    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        gaa_embeddings_tmp = gaa_embeddings[ind_cl]
        embeddings_tmp = embeddings[ind_cl]
        class_label.append(i)
        gaa_mean.append(np.mean(gaa_embeddings_tmp, axis=0))
        gaa_std.append(np.std(gaa_embeddings_tmp, axis=0))
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_mean_sign.append(np.sign(np.mean(embeddings_tmp, axis=0)))
        class_std.append(np.std(embeddings_tmp, axis=0))
    prototype_new = {'gaa_mean': gaa_mean, 'gaa_std': gaa_std, 'class_mean': class_mean, 'class_mean_sign': class_mean, 'class_std': class_std, 'class_label': class_label}
    if session != 0:
        prototype = np.load('./proto/prototypes.npy', allow_pickle=True).item()
        prototype['gaa_mean'].extend(prototype_new['gaa_mean'][:])
        prototype['gaa_std'].extend(prototype_new['gaa_std'][:])
        prototype['class_mean'].extend(prototype_new['class_mean'][:])
        prototype['class_std'].extend(prototype_new['class_std'][:])
        prototype['class_label'].extend(prototype_new['class_label'][:])
    else:
        prototype = prototype_new
    np.save('./proto/prototypes.npy', prototype)
    return prototype


def update_initial_prototype(model, data_loader, prototype):
    model.eval()
    gaa_embeddings = []
    embeddings = []
    embeddings_labels = []
    gaa_mean = []
    gaa_std = []
    class_mean = []
    class_std = []
    class_label = []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            embed_gaa_feat = model.gaa_features(inputs)
            embed_feat = model.features(inputs)
            embeddings_labels.append(labels.cpu().numpy())
            gaa_embeddings.append(embed_gaa_feat.cpu().numpy())
            embeddings.append(embed_feat.cpu().numpy())
        gaa_embeddings = np.asarray(gaa_embeddings)
        gaa_embeddings = np.reshape(gaa_embeddings, (gaa_embeddings.shape[0] * gaa_embeddings.shape[1], gaa_embeddings.shape[2]))
        embeddings = np.asarray(embeddings)
        embeddings = np.reshape(embeddings, (embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]))
        embeddings_labels = np.asarray(embeddings_labels)
        embeddings_labels = np.reshape(embeddings_labels, embeddings_labels.shape[0] * embeddings_labels.shape[1])
        labels_set = np.unique(embeddings_labels)
        print('initial prototypes:', labels_set)

        for i in labels_set:
            ind_cl = np.where(i == embeddings_labels)[0]
            embeddings_tmp = embeddings[ind_cl]
            class_mean.append(np.mean(embeddings_tmp, axis=0))
            class_std.append(np.std(embeddings_tmp, axis=0))

    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        gaa_embeddings_tmp = gaa_embeddings[ind_cl]
        class_label.append(i)
        gaa_mean.append(np.mean(gaa_embeddings_tmp, axis=0))
        gaa_std.append(np.std(gaa_embeddings_tmp, axis=0))

    prototype_new = {'gaa_mean': gaa_mean, 'gaa_std': gaa_std, 'class_mean': class_mean, 'class_std': class_std, 'class_label': class_label}
    prototype['gaa_mean'].extend(prototype_new['gaa_mean'][:])
    prototype['gaa_std'].extend(prototype_new['gaa_std'][:])
    prototype['class_mean'].extend(prototype_new['class_mean'][:])
    prototype['class_std'].extend(prototype_new['class_std'][:])
    prototype['class_label'].extend(prototype_new['class_label'][:])
    np.save('./proto/prototypes.npy', prototype)
    return prototype


def update_final_prototype(model, prototype, base=80):
    model.eval()
    gaa_mean = []
    gaa_std = []
    class_mean = []
    class_mean_sign = []
    class_std = []
    class_label = []
    labels_set = np.unique(prototype['class_label'][base:])
    feats, labels_hd = sample_labeled_features(prototype['gaa_mean'][base:], prototype['gaa_std'][base:], prototype['class_label'][base:], num_per_novel_class=1000, base=base)
    embeddings_hd = model.hd_features(feats)
    embeddings_hd = embeddings_hd.cpu().detach().numpy()

    labels_hd = labels_hd.cpu()
    for i in labels_set:
        ind_cl = np.where(i == labels_hd)[0]
        embeddings_hd_tmp = embeddings_hd[ind_cl]
        class_mean.append(np.mean(embeddings_hd_tmp, axis=0))
        class_mean_sign.append(np.sign(np.mean(embeddings_hd_tmp, axis=0)))
        class_std.append(np.std(embeddings_hd_tmp, axis=0))


    prototype_new = {'gaa_mean': gaa_mean, 'gaa_std': gaa_std, 'class_mean': class_mean, 'class_mean_sign': class_mean_sign, 'class_std': class_std, 'class_label': class_label}
    prototype['class_mean'][base:].extend(prototype_new['class_mean'][:])
    prototype['class_mean_sign'][base:].extend(prototype_new['class_mean_sign'][:])
    prototype['class_std'][base:].extend(prototype_new['class_std'][:])

    np.save('./proto/prototypes.npy', prototype)
    return prototype

def sample_labeled_features(class_mean, class_sig, label, num_per_base_class = 100, num_per_novel_class = 500, base = 80):
    feats = []
    labels = []
    class_mean = torch.tensor(np.array(class_mean))
    class_sig = torch.tensor(np.array(class_sig))
    label = torch.tensor(np.array(label))
    for i in range(len(label[:base])):
        dist = torch.distributions.Normal(class_mean[i], class_sig[i]+0.05)
        this_feat = dist.sample((num_per_base_class,)).cuda()
        this_label = torch.ones(this_feat.size(0)).cuda() * label[i]
        feats.append(this_feat)
        labels.append(this_label)
    for i in range(len(label[base:])):
        i = i+base
        dist = torch.distributions.Normal(class_mean[i], class_sig[i])
        this_feat = dist.sample((num_per_novel_class,)).cuda()
        this_label = torch.ones(this_feat.size(0)).cuda() * label[i]
        feats.append(this_feat)
        labels.append(this_label)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0).long()
    return feats, labels


class exp_loss(torch.nn.Module):
    def __init__(self, scale):
        super(exp_loss, self).__init__()
        self.scale = scale
    def forward(self, x):
        return torch.exp(self.scale * x) - 1


class refine_model(torch.nn.Module):
    def __init__(self, num_ways, base):
        super(refine_model, self).__init__()
        self.act_exp = 4
        self.tnhscaleP = torch.nn.Parameter(torch.ones([1], dtype=torch.float32) * 1, requires_grad=False)  # init to 1.5
        self.mask_novel = torch.nn.Parameter(torch.triu(torch.ones([num_ways-base, num_ways-base], dtype=torch.uint8), diagonal=1),
                                   requires_grad=False)  # .to(self._device)
        self.mask_sum_novel = torch.sum(self.mask_novel)
        self.mask_base = torch.nn.Parameter(torch.ones([base, num_ways-base], dtype=torch.uint8), requires_grad=False)  # .to(self._device)
        self.mask_sum_base = torch.sum(self.mask_base)
        self.cos = torch.nn.CosineSimilarity()
        self.act = exp_loss(4)

    def init_params(self, initial_prototypes, base):
        self.prod_vecs_base = torch.nn.Parameter(initial_prototypes[:base], requires_grad=False)
        self.prod_vecs_novel = torch.nn.Parameter(initial_prototypes[base:], requires_grad=True)

    def forward(self,):
        prod_vecs_novel = torch.tanh(self.tnhscaleP * self.prod_vecs_novel)
        norm_prod_vecs_novel = F.normalize(prod_vecs_novel, p=2, dim=1)
        sims_novel = torch.tensordot(norm_prod_vecs_novel, torch.transpose(norm_prod_vecs_novel, 0, 1),dims=1) * self.mask_novel
        sim_loss_novel = self.act(sims_novel)
        sim_loss_novel = torch.sum(sim_loss_novel) / self.mask_sum_novel

        prod_vecs_base = torch.tanh(self.tnhscaleP * self.prod_vecs_base)
        norm_prod_vecs_base = F.normalize(prod_vecs_base, p=2, dim=1)
        sims_base = torch.tensordot(norm_prod_vecs_base, torch.transpose(norm_prod_vecs_novel, 0, 1), dims=1) * self.mask_base
        sim_loss_base = self.act(sims_base)
        sim_loss_base = torch.sum(sim_loss_base) / self.mask_sum_base
        total_loss = sim_loss_base + sim_loss_novel
        return total_loss

def refine_hyper(h_p, base=60, gpu=0, num_epochs=5, learning_rate=0.01):

    num_ways, dim_features = h_p.shape
    model = refine_model(num_ways, base)
    model.init_params(h_p.detach().cpu(), base)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)
    model.cuda(gpu)
    h_p.cuda(gpu)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = model()
        total_loss.backward()
        optimizer.step()
    return 0.99*h_p + 0.01*torch.cat([model.prod_vecs_base.data, model.prod_vecs_novel.data], dim=0)

