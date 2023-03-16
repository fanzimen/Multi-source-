# import math
# import torch.utils.model_zoo as model_zoo
# import muda.utils.utils as utils
from options import Options
import data_loader
# from muda.model import model_mfsan
from muda.model import model_mfsan as model_mfsan
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
# ## Read csv file to pandas dataframe
# FD_path = ["none", FD001_path, FD002_path, FD003_path, FD004_path]
# FD_name = ["none", "FD001", "FD002", "FD003", "FD004"]
#
# source_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]  #此处选择source 数据集 1 2 3 分别对应FD001-FD004
# target_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]
# datasetset_name = ["none", "FD001", "FD002", "FD003", "FD004"]
#
#
# source_chosen = ['None',1,3,4]
# target_chosen = ['None',2]
#
# target_name = datasetset_name[target_chosen[1]]
# source1_name = datasetset_name[source_chosen[1]]
# source2_name = datasetset_name[source_chosen[2]]
# source3_name = datasetset_name[source_chosen[3]]
# sensor_drop = ['sensor_01', 'sensor_05', 'sensor_06', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

opt = Options().parse()
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda = True
opt.target_data_path = FD001_path
if __name__ == '__main__':

    target_train_loader = data_loader.load_training(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                    1, suffle = False)

    model = model_mfsan.MFSAN()
    model.load_state_dict(torch.load(current_dir + "/log/2023-03-13_22_34_09mfsan.pt"))
    model.cuda()
    model.eval()
    i = 0
    pred = []
    target_array = []
    cycle = []
    with torch.no_grad():
        for data, target in target_train_loader:
            # print(data, target)
            # break
            data, target = data.cuda(), target.type(torch.FloatTensor).cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3 = model(data)
            pred1, pred2, pred3 =pred1.detach().cpu().numpy().squeeze(1), pred2.detach().cpu().numpy().squeeze(1), pred3.detach().cpu().numpy().squeeze(1)
            target_numpy = target.detach().cpu().numpy()
            pred_ave = (pred1 + pred2 + pred3)/3
            pred.append(pred_ave)
            target_array.append(target_numpy)
            cycle.append(i)
            i+=1
            if i >= 1000:
                break
    print('all finished')
    plt.figure()
    plt.xlabel('cycle')
    plt.ylabel('rul')
    plt.plot(cycle, pred, label='pred')
    plt.plot(cycle, target_array, label='target')
    plt.legend()
    plt.show()
