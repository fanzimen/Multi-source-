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
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=199, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=10, out_features=2),
            # nn.ReLU()
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha) #梯度反转
        x = self.discriminator(reversed_input)
        return x   #二分类，因为Crossextropy最后一层有softmax


class CFE(nn.Module): #一维卷积？？

    def __init__(self, inplanes, planes,stride=1, downsample=None):
        super(CFE, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, 32, kernel_size=3, padding=1, bias=False) # torch.Size([700, 14, 50]) ——》 [700, 32, 50]
        self.bn1 = nn.BatchNorm1d(32)#归一化 加速模型的训练，把参数进行规范化的处理，让参数计算的梯度不会太小
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, stride=stride, padding=1, bias=False)#[700, 32, 50]——》 [700, 64, 50]
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, planes, kernel_size=2, bias=False)#[700, 64, 50]——》 [700, 128, 50]
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.flatten = nn.Flatten()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)          #700 32 30- 700 64 30

        out = out.permute(0, 2, 1)    #把700 64 30 变成 700 30 64   #难道不是700 50 128？
        # print(out.shape[-1])
        return out

class DFE(nn.Module): #BiLSTM

    def __init__(self,input_size, hidden_size):
        super(DFE, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        #700 30 64 nn.LSTM(64,100,2,True)
    def forward(self, x):
        x,_ = self.rnn(x)       #把700 30 64 变成 700 30 200 因为bi=True
        x = F.dropout(x,p=0.2)
        x = x[:,-1,:] #？？
        # print("xxxxx",x.shape)
        return x

class MFSAN(nn.Module):

    def __init__(self, num_classes=1, inplanes=14, planes=128, hidden_size=100,avgpool_size = 2):
        super(MFSAN, self).__init__()
        self.sharedNet = CFE(inplanes, planes)

        self.sonnet1 = DFE(planes, hidden_size)
        self.sonnet2 = DFE(planes, hidden_size)
        self.sonnet3 = DFE(planes, hidden_size)


        self.rul_fc_son1 = nn.Linear(hidden_size*2-avgpool_size+1, num_classes)
        self.rul_fc_son2 = nn.Linear(hidden_size*2-avgpool_size+1, num_classes)
        self.rul_fc_son3 = nn.Linear(hidden_size * 2 - avgpool_size + 1, num_classes)

        self.avgpool = nn.AvgPool1d(avgpool_size, stride=1)

        # self.dann = Discriminator()

    def forward(self, data_src, data_tgt = 0, label_src = 0, alpha = 0, mark = 1):


        mmd_loss, coral_loss= 0, 0
        if self.training == True:
            #目标域数据输入

            data_src = self.sharedNet(data_src)  ##700 29 4  #700 14，32-》700，64，128-》flatten
            data_tgt = self.sharedNet(data_tgt)  ##700 29 4

            data_tgt_son1 = self.sonnet1(data_tgt)  ##700 100 #700
            data_tgt_son1 = self.avgpool(data_tgt_son1.unsqueeze(1))  ### 700 1 99
            data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1) ### 700 99
            pred_tgt_son1 = self.rul_fc_son1(data_tgt_son1).squeeze(1) ### 700

            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2.unsqueeze(1))
            data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
            pred_tgt_son2 = self.rul_fc_son2(data_tgt_son2).squeeze(1)

            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3.unsqueeze(1))
            data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt_son3 = self.rul_fc_son3(data_tgt_son3).squeeze(1)


            if mark == 1:#源域数据1

                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src.unsqueeze(1))
                data_src = data_src.view(data_src.size(0), -1)
                pred_src = self.rul_fc_son1(data_src).squeeze(1)
                #pdata_tgt_son1:经过源域1专属子网络处理的目标域数据 data_src:经过源域1专属子网络处理的源域数据
                mmd_loss += mmd(data_src, data_tgt_son1)

                coral_loss += coral(data_src, data_tgt_son1)

                combined_seq = torch.cat((data_src, data_tgt_son1), 0)#按维数0（行）拼接

                #这一串代码不会没用吧
                # domain_pred = self.dann(combined_seq, alpha) #Domain Adversarial Training of NN（DANN）
                # domain_source_labels = torch.zeros(data_src.shape[0]).type(torch.LongTensor)
                # domain_target_labels = torch.ones(data_tgt_son1.shape[0]).type(torch.LongTensor)
                # domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
                # discriminator_criterion = nn.CrossEntropyLoss().cuda()  # 分类loss domain
                # domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

                rul_criterion = nn.MSELoss().cuda()  # 回归loss RUL
                rul_loss = rul_criterion(pred_src, label_src) #源域样本的回归误差

                #线性层输出
                mat = torch.cat((pred_tgt_son1.unsqueeze(1), pred_tgt_son2.unsqueeze(1), pred_tgt_son3.unsqueeze(1)), 1)
                # print("mat.shape",mat.shape)
                mat_sca = F.normalize(mat, p=1, dim=1) #除以tensor的L1范数
                pred_tgt_son1_sca, pred_tgt_son2_sca, pred_tgt_son3_sca = mat_sca[:, 0], mat_sca[:, 1], mat_sca[:, 2]
                #源域1回归与另外两个回归器的误差
                l1_loss = rul_criterion(pred_tgt_son1_sca, pred_tgt_son2_sca)
                l1_loss += rul_criterion(pred_tgt_son1_sca, pred_tgt_son3_sca)

                # com_loss=mmd_loss+5000*coral_loss

                return rul_loss, mmd_loss, l1_loss / 2


            if mark == 2:

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src.unsqueeze(1))
                data_src = data_src.view(data_src.size(0), -1)
                pred_src = self.rul_fc_son2(data_src).squeeze(1)

                # print(data_src.shape, data_tgt_son2.shape)
                mmd_loss += mmd(data_src, data_tgt_son2)

                coral_loss += coral(data_src, data_tgt_son2)

                # combined_seq = torch.cat((data_src, data_tgt_son2), 0)
                # domain_pred = self.dann(combined_seq, alpha)
                # domain_source_labels = torch.zeros(data_src.shape[0]).type(torch.LongTensor)
                # domain_target_labels = torch.ones(data_tgt_son2.shape[0]).type(torch.LongTensor)
                # domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
                # discriminator_criterion = nn.CrossEntropyLoss().cuda()  # 分类loss domain
                # domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

                rul_criterion = nn.MSELoss().cuda()  # 回归loss RUL
                rul_loss = rul_criterion(pred_src, label_src)

                mat = torch.cat((pred_tgt_son1.unsqueeze(1), pred_tgt_son2.unsqueeze(1), pred_tgt_son3.unsqueeze(1)), 1)
                # print("mat.shape",mat.shape)
                mat_sca = F.normalize(mat, p=1, dim=1)
                pred_tgt_son1_sca, pred_tgt_son2_sca, pred_tgt_son3_sca = mat_sca[:, 0], mat_sca[:, 1], mat_sca[:, 2]

                l1_loss = rul_criterion(pred_tgt_son2_sca, pred_tgt_son1_sca)
                l1_loss += rul_criterion(pred_tgt_son2_sca, pred_tgt_son3_sca)

                # com_loss = mmd_loss + 5000*coral_loss

                return rul_loss, mmd_loss, l1_loss / 2


            if mark == 3:

                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src.unsqueeze(1))
                data_src = data_src.view(data_src.size(0), -1)
                pred_src = self.rul_fc_son3(data_src).squeeze(1)

                # print(data_src.shape, data_tgt_son2.shape)
                mmd_loss += mmd(data_src, data_tgt_son3)

                coral_loss += coral(data_src, data_tgt_son3)

                # combined_seq = torch.cat((data_src, data_tgt_son3), 0)
                # domain_pred = self.dann(combined_seq, alpha)
                # domain_source_labels = torch.zeros(data_src.shape[0]).type(torch.LongTensor)
                # domain_target_labels = torch.ones(data_tgt_son3.shape[0]).type(torch.LongTensor)
                # domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
                # discriminator_criterion = nn.CrossEntropyLoss().cuda()  # 分类loss domain
                # domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

                rul_criterion = nn.MSELoss().cuda()  # 回归loss RUL
                rul_loss = rul_criterion(pred_src, label_src)

                mat = torch.cat((pred_tgt_son1.unsqueeze(1), pred_tgt_son2.unsqueeze(1), pred_tgt_son3.unsqueeze(1)), 1)
                # print("mat.shape",mat.shape)
                mat_sca = F.normalize(mat, p=1, dim=1)
                pred_tgt_son1_sca, pred_tgt_son2_sca, pred_tgt_son3_sca = mat_sca[:, 0], mat_sca[:, 1], mat_sca[:, 2]

                l1_loss = rul_criterion(pred_tgt_son3_sca, pred_tgt_son1_sca)
                l1_loss += rul_criterion(pred_tgt_son3_sca, pred_tgt_son2_sca)

                # com_loss = mmd_loss + 5000*coral_loss

                return rul_loss, mmd_loss, l1_loss / 2



        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = fea_son1.unsqueeze(1)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.rul_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = fea_son2.unsqueeze(1)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.rul_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data)
            fea_son3 = fea_son3.unsqueeze(1)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.rul_fc_son3(fea_son3)

            return pred1, pred2, pred3


