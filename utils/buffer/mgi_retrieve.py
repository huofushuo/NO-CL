import torch
import torch.nn.functional as F
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector
import copy
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip,ColorJitter,RandomGrayscale
from utils.setup_elements import transforms_match, input_size_match
import torch.nn as nn
from utils.setup_elements import n_classes


class MGI_retrieve(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                              scale=(0.2, 1.)).cuda(),
            RandomHorizontalFlip().cuda(),
             ColorJitter(0.4, 0.4, 0.4, 0.1),
            RandomGrayscale(p=0.2)

        )
        self.out_dim = n_classes[params.data]

    def retrieve(self, buffer, **kwargs):
        sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in buffer.model.parameters():
             grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                sub_x_aug = self.transform(sub_x)
                logits_pre = buffer.model(sub_x,sub_x_aug)
                logits_post = model_temp(sub_x,sub_x_aug)

                z_pre, zt_pre, zzt_pre,fea_z_pre = logits_pre
                z_post, zt_post, zzt_post,fea_z_post = logits_post


                grads_pre_z= torch.sum(torch.abs(F.softmax(z_pre, dim=1) - F.one_hot(sub_y, self.out_dim)), 1)
                mgi_pre_z = grads_pre_z * fea_z_pre[0].reshape(-1)
                grads_post_z = torch.sum(torch.abs(F.softmax(z_post, dim=1) - F.one_hot(sub_y, self.out_dim)), 1)  # N * 1
                mgi_post_z = grads_post_z * fea_z_post[0].reshape(-1)


                scores = mgi_post_z - mgi_pre_z

                big_ind = scores.sort(descending=True)[1][:int(self.num_retrieve)]



            return sub_x[big_ind], sub_x_aug[big_ind],sub_y[big_ind]
        else:
            return sub_x, sub_x,sub_y

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1