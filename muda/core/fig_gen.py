# import math
# import torch.utils.model_zoo as model_zoo
# import muda.utils.utils as utils
from options import Options
import data_loader
# from muda.model import model_mfsan
from muda.model import model_mfsan2 as model_mfsan
# from train_mfsan import train,test
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
# import model_mfsan

from torch.autograd import Variable

opt = Options().parse()
seed = opt.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataset_dir =  os.path.abspath(os.path.join(os.getcwd(), ".."))
train_FD001_path = dataset_dir +'/data/cmapss/train_FD001.csv'
test_FD001_path = dataset_dir +'/data/cmapss/test_FD001.csv'
RUL_FD001_path = dataset_dir+'/data/cmapss/RUL_FD001.txt'
FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]

train_FD002_path = dataset_dir +'/data/cmapss/train_FD002.csv'
test_FD002_path = dataset_dir +'/data/cmapss/test_FD002.csv'
RUL_FD002_path = dataset_dir +'/data/cmapss/RUL_FD002.txt'
FD002_path = [train_FD002_path, test_FD002_path, RUL_FD002_path]

train_FD003_path = dataset_dir +'/data/cmapss/train_FD003.csv'
test_FD003_path = dataset_dir +'/data/cmapss/test_FD003.csv'
RUL_FD003_path = dataset_dir +'/data/cmapss/RUL_FD003.txt'
FD003_path = [train_FD003_path, test_FD003_path, RUL_FD003_path]

train_FD004_path =dataset_dir +'/data/cmapss/train_FD004.csv'
test_FD004_path = dataset_dir +'/data/cmapss/test_FD004.csv'
RUL_FD004_path = dataset_dir +'/data/cmapss/RUL_FD004.txt'
FD004_path = [train_FD004_path, test_FD004_path, RUL_FD004_path]

opt = Options().parse()
current_dir = os.path.dirname(os.path.abspath(__file__)) + "/log/"
cuda = True
opt.target_data_path = FD004_path
model_name = '2023-03-15_18_15_45mfsan.pt'
# def rul_fig(model_name):

if __name__ == '__main__':

    target_train_loader = data_loader.load_training(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                    1, suffle = False)

    model = model_mfsan.MFSAN()
    model.load_state_dict(torch.load(current_dir + model_name))
    model.cuda()
    model.eval()
    i = 0
    pred = []
    target_array = []
    cycle = []
    a,b = 150,400
    with torch.no_grad():
        for data, target in target_train_loader:
            # print(data, target)
            # break
            i+=1
            if i <= a:
                continue
            # print(i)
            data, target = data.cuda(), target.type(torch.FloatTensor).cuda()
            data, target = Variable(data), Variable(target)
            # pred1, pred2, pred3 = model(data)
            # pred1, pred2, pred3 =pred1.detach().cpu().numpy().squeeze(1), pred2.detach().cpu().numpy().squeeze(1), pred3.detach().cpu().numpy().squeeze(1)
            # target_numpy = target.detach().cpu().numpy()
            # pred_ave = (pred1 + pred2 + pred3)/3
            pred1, pred2 = model(data)
            pred1, pred2 = pred1.detach().cpu().numpy().squeeze(1), pred2.detach().cpu().numpy().squeeze(1)
            target_numpy = target.detach().cpu().numpy()
            pred_ave = (pred1 + pred2) / 2
            pred.append(pred_ave)
            target_array.append(target_numpy)
            cycle.append(i)

            if i >= b:
                break
    print('all finished')
    plt.figure()
    plt.xlabel('cycle')
    plt.ylabel('rul')
    plt.plot(cycle, pred, label='pred')
    plt.plot(cycle, target_array, label='target')
    plt.legend()
    plt.show()
