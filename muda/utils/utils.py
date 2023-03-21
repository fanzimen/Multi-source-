import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import itertools
import os
from sklearn.metrics import mean_squared_error
def set_seed(seed):
    """
    :param args:
    :return:
    """
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.enabled = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / (1. + 10 * p) ** 0.75

    return optimizer


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def save_model(model, training_name):
    print('Save models ...')

    save_folder = 'trained_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(model.state_dict(), 'log/' + str(training_name) + '.pt')

    print('Model is saved !!!')

def load_model(model):
    print('Loading model ...')

    model.load_state_dict(torch.load("trained_models/mfsan.pt"))

    print('Model is loaded !!!')


def plot_embedding(X, y, d, training_mode, save_name):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        else:
            colors = (1.0, 0.0, 0.0, 1.0)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if save_name is not None:
        plt.title(save_name)

    save_folder = 'saved_plot'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_name = 'saved_plot/' + str(training_mode) + '_' + str(save_name) + '.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))


def visualize(encoder, args, training_mode, save_name):
    # Draw 512 samples in test_data
    phase = 'test'
    source_test_loader = seq_creater.seq_Dataload(phase, args.source_path, args.sequence_length, args.sensor_drop)
    target_test_loader = seq_creater.seq_Dataload(phase, args.target_path, args.sequence_length, args.sensor_drop)

    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= 1:  # to get only 512 samples
            break
        img, label = test_data
        label = label.numpy()
        img = img.cuda()

        source_label_list.append(label)
        source_img_list.append(img)

    source_img_list = torch.stack(source_img_list)
    source_img_list = source_img_list.view(-1, args.l, 14)

    # Get target_test samples
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(target_test_loader):
        if i >= 1:
            break
        img, label = test_data
        label = label.numpy()
        img = img.cuda()
        target_label_list.append(label)
        target_img_list.append(img)

    target_img_list = torch.stack(target_img_list)
    target_img_list = target_img_list.view(-1, args.l, 14)

    # Stack source_list + target_list
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    combined_img_list = torch.cat((source_img_list, target_img_list), 0)

    source_domain_list = torch.zeros(args.batch_size).type(torch.FloatTensor)
    target_domain_list = torch.ones(args.batch_size).type(torch.FloatTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cuda()

    print("Extract features to draw T-SNE plot...")
    combined_feature = encoder(combined_img_list)  # combined_feature : 1024,2352

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())

    print('Draw plot ...')
    save_name = save_name + '_' + str(training_mode)
    plot_embedding(dann_tsne, combined_label_list, combined_domain_list, training_mode, save_name)


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()

def score_cal(y_hat, Y_test):
    d   = y_hat - Y_test
    tmp = np.zeros(len(d))
    for i in range(len(d)):
        if d[i] >= 0:
           tmp[i] = np.exp( d[i]*125/10) - 1
        else:
           tmp[i] = np.exp(-d[i]*125/13) - 1
    return np.sum(tmp)

def mse_cal(y_hat, Y_test):
    tmp = mean_squared_error(y_hat*125, Y_test*125)
    return tmp

def rmse_cal(y_hat, Y_test):
    tmp = np.sqrt(mean_squared_error(y_hat*125, Y_test*125))
    return tmp



def visualize_total_loss(epoch,scr1,scr2,scr3):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.plot(epoch, np.array(scr1), label='source1 Loss')
    plt.plot(epoch, np.array(scr2), label='source2 loss')
    plt.plot(epoch, np.array(scr3), label='source3 loss')

    plt.legend()

    plt.savefig("./log/total_loss.jpg")


### 网络参数初始化
def weight_init2(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def log_in_file(file_name):
    save_folder = 'log'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # print(os.path.abspath(os.path.join(os.getcwd(), "..")))
    log_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + '\\' +save_folder + file_name
    f = open(log_dir, "a")

    return f

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss



def coral(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss