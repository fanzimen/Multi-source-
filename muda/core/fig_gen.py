import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

import itertools
import os
from sklearn.metrics import mean_squared_error

import torch.nn as nn
# import math
# import torch.utils.model_zoo as model_zoo
from muda.utils.utils import mmd, coral
import torch.nn.functional as F
import torch
from muda.utils.utils import ReverseLayerF
# import muda.utils.utils as utils
from options import Options
import random
import os
import numpy as np
from options import Options
import data_loader
# from muda.model import model_mfsan
from muda.model import model_mfsan_dann as model_mfsan
# from train_mfsan import train,test
from muda.utils.utils import weight_init, log_in_file, save_model, set_seed
import time
from muda.utils.EarlyStopper import EarlyStopper
from muda.utils.print_things import figure_generate,writing_settings,figure_generate_dann
import pandas as pd
import os
import torch
from torch.autograd import Variable
import numpy as np
import muda.utils.utils as utils
from muda.utils.utils import score_cal,rmse_cal
import math
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
# import model_mfsan

from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
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
import data_loader
# def load_testing(data_path_list, sequence_length, sensor_drop, batch_size,seed):
#     test_dataset = seq_Dataset("test", data_path_list, sequence_length, sensor_drop)
#     loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
#     return loader  # torch.Size([700, 14, 50])
# from muda.model import model_mfsan_dann as model_mfsan
opt = Options().parse()
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda = True
if __name__ == '__main__':

    target_train_loader = data_loader.load_training(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                    1,suffle = False)

    model = model_mfsan.MFSAN()
    model.load_state_dict(torch.load(current_dir + "/log/2023-03-11_19_38_57mfsan.pt"))
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
            if i >= 200:
                break
    print('all finished')
    plt.figure()
    plt.xlabel('cycle')
    plt.ylabel('rul')
    plt.plot(cycle, pred, label='pred')
    plt.plot(cycle, target_array, label='target')
    plt.legend()
    plt.show()
