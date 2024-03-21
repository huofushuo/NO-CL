import torch
import numpy as np

def fusion_aug_two_image(x_1, x_2, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x_1.size()[0]
    mix_data_1 = []
    mix_data_2 = []
    mix_target = []

    # print('fusion_aug_two_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data_1.append(lam * x_1[i] + (1 - lam) * x_1[index, :][i])
                mix_data_2.append(lam * x_2[i] + (1 - lam) * x_2[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data_1:
        x_1 = torch.cat((x_1, item.unsqueeze(0)), 0)
    for item in mix_data_2:
        x_2 = torch.cat((x_2, item.unsqueeze(0)), 0)
    # print('fusion_aug_two_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))

    return x_1, x_2, y


def fusion_aug_generate_label(y_a, y_b, session, args):
    current_total_cls_num = 10 + session * 10
    if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
        y_a = y_a - (current_total_cls_num - 10)
        y_b = y_b - (current_total_cls_num - 10)
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = int(((2 * 10 - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
    return label_index + current_total_cls_num