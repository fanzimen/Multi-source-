import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import model_mfsan
import data_loader_cluster
from utils import weight_init, log_in_file, visualize_total_loss, save_model,score_cal,rmse_cal
import math
from torch.autograd import Variable
import utils as utils
import pandas as pd

from utils.EarlyStopper import EarlyStopper
pd.set_option('mode.chained_assignment', None)



#path
dataset_dir = os.path.abspath(os.path.dirname(__file__))

# Dataset path
train_FD001_path = dataset_dir +'/cmapss/train_FD001.csv'
test_FD001_path = dataset_dir +'/cmapss/test_FD001.csv'
RUL_FD001_path = dataset_dir+'/cmapss/RUL_FD001.txt'
FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]

train_FD002_path = dataset_dir +'/cmapss/train_FD002.csv'
test_FD002_path = dataset_dir +'/cmapss/test_FD002.csv'
RUL_FD002_path = dataset_dir +'/cmapss/RUL_FD002.txt'
FD002_path = [train_FD002_path, test_FD002_path, RUL_FD002_path]

train_FD003_path = dataset_dir +'/cmapss/train_FD003.csv'
test_FD003_path = dataset_dir +'/cmapss/test_FD003.csv'
RUL_FD003_path = dataset_dir +'/cmapss/RUL_FD003.txt'
FD003_path = [train_FD003_path, test_FD003_path, RUL_FD003_path]

train_FD004_path =dataset_dir +'/cmapss/train_FD004.csv'
test_FD004_path = dataset_dir +'/cmapss/test_FD004.csv'
RUL_FD004_path = dataset_dir +'/cmapss/RUL_FD004.txt'
FD004_path = [train_FD004_path, test_FD004_path, RUL_FD004_path]


## Read csv file to pandas dataframe
FD_path = ["none", FD001_path, FD002_path, FD003_path, FD004_path]
FD_name = ["none", "FD001", "FD002", "FD003", "FD004"]

sensor_drop = ['sensor_01', 'sensor_05', 'sensor_06', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

source_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]  #此处选择source 数据集 1 2 3 分别对应FD001-FD004
target_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]
datasetset_name = ["none", "FD001", "FD002", "FD003", "FD004"]


# Training settings
sequence_length = 30
batch_size = 512
epochs = 10
lr = [0.001, 0.01,0.05]
momentum = 0.9
# cuda = True
cuda = False
seed = 8
log_interval = 5
l2_decay = 5e-4

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


source_chosen = ['None',1,3,4]
target_chosen = ['None',2]


source1_name = datasetset_name[source_chosen[1]]
source2_name = datasetset_name[source_chosen[2]]
source3_name = datasetset_name[source_chosen[3]]

#载入目标域数据
target_train_loader = data_loader_cluster.load_training(target_path[target_chosen[1]], sequence_length, sensor_drop,
                                                batch_size)
target_test_loader = data_loader_cluster.load_testing(target_path[target_chosen[1]], sequence_length, sensor_drop,
                                                  batch_size)
target_name = datasetset_name[target_chosen[1]]


def train(model):
    global mmd_loss1,mmd_loss2,mmd_loss3 #全局
    print("--------------------------MFSAN --------------------------------")

    f_mfsan_train = log_in_file("/mfsan_train_log.log")

    source1_loader = data_loader_cluster.load_training(source_path[source_chosen[1]], sequence_length, sensor_drop,
                                                batch_size)
    source2_loader = data_loader_cluster.load_training(source_path[source_chosen[2]], sequence_length, sensor_drop,
                                                batch_size)
    source3_loader = data_loader_cluster.load_training(source_path[source_chosen[3]], sequence_length, sensor_drop,
                                               batch_size)

    source1_len = len(source1_loader)
    source2_len = len(source2_loader)
    source3_len = len(source3_loader)

    target_len = len(target_train_loader)
    max_len = max(source1_len,source2_len,source3_len,target_len)
    # print("111111",source1_len,source2_len,target_len,max_len)

    batch_src1,batch_src2,batch_src3,batch_tar = 0,0,0,0
    running_loss_scr1, running_loss_scr2 ,running_loss_scr3= 0, 0, 0
    running_mmd_loss_scr1, running_mmd_loss_scr2, running_mmd_loss_scr3 = 0, 0, 0
    list_src1, list_src2, list_src3, list_tar = list(enumerate(source1_loader)), list(enumerate(source2_loader)), list(enumerate(source3_loader)),list(enumerate(target_train_loader))

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters(), 'lr': lr[1]},

        {'params': model.rul_fc_son1.parameters(), 'lr': lr[1]},
        {'params': model.rul_fc_son2.parameters(), 'lr': lr[2]},

        {'params': model.sonnet1.parameters(), 'lr': lr[1]},
        {'params': model.sonnet2.parameters(), 'lr': lr[2]},

    ], lr=lr[2], momentum=momentum, weight_decay=l2_decay)



    for batch in range(1, max_len):
        model.train()

        p = (batch - 1) / (max_len)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 #？？？
        optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p) #Adjust the learning rate of optimizer
        optimizer.zero_grad() #梯度初始化为0

        #####scr1 tgt
        _, (source_data1, source_label1) = list_src1[batch_src1]
        _, (target_data1, _) = list_tar[batch_tar]


        if cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.type(torch.FloatTensor).cuda()
            target_data1 = target_data1.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1) #tensor不能反向传播，variable可以反向传播
        target_data1 = Variable(target_data1)

        rul_loss1, mmd_loss1, l1_loss1 = model(source_data1, target_data1, source_label1, alpha, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (batch) / (max_len))) - 1

        loss1 = rul_loss1 + gamma * (mmd_loss1 + l1_loss1)
        loss1.backward()
        optimizer.step()
        running_loss_scr1 += loss1.item() #loss为什么要加item() 直接输出的话数据类型是Variable,
        running_mmd_loss_scr1 += mmd_loss1.item()
        batch_src1 += 1
        if batch_src1 >= len(list_src1)-1:
            batch_src1 = 0

        if batch_tar >= len(list_tar)-1:
            batch_tar = 0

        if batch % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tRUL_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                batch, 100. * batch/ max_len, loss1.item(), rul_loss1.item(), mmd_loss1.item(), l1_loss1.item()), file=f_mfsan_train, flush=True)

        # print("batch_src1,batch_tar",batch_src1,batch_tar)

        #####scr2 tgt
        _, (source_data2, source_label2) = list_src2[batch_src2]
        _, (target_data2, _) = list_tar[batch_tar]

        if cuda:
            source_data2, source_label2 = source_data2.cuda(), source_label2.type(torch.FloatTensor).cuda()
            target_data2 = target_data2.cuda()
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        target_data2 = Variable(target_data2)
        optimizer.zero_grad()

        rul_loss2, mmd_loss2, l1_loss2 = model(source_data2, target_data2, source_label2, alpha, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (batch) / (max_len))) - 1
        loss2 = rul_loss2 + gamma * (mmd_loss2 + l1_loss2)
        loss2.backward()
        optimizer.step()
        running_loss_scr2 += loss2.item()
        running_mmd_loss_scr2 += mmd_loss2.item()

        batch_src2 += 1
        if batch_src2 >= len(list_src2)-1:
            batch_src2 = 0


        if batch_tar >= len(list_tar)-1:
            batch_tar = 0
        # print("batch_src2,batch_tar", batch_src2, batch_tar)

        if batch % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tRUL_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    batch, 100. * batch / max_len, loss2.item(), rul_loss2.item(), mmd_loss2.item(), l1_loss2.item()), file=f_mfsan_train, flush=True)

        #####scr3 tgt
        _, (source_data3, source_label3) = list_src3[batch_src3]
        _, (target_data3, _) = list_tar[batch_tar]

        if cuda:
            source_data3, source_label3 = source_data3.cuda(), source_label3.type(torch.FloatTensor).cuda()
            target_data3 = target_data3.cuda()
        source_data3, source_label3 = Variable(source_data3), Variable(source_label3)
        target_data3 = Variable(target_data3)
        optimizer.zero_grad()

        rul_loss3, mmd_loss3, l1_loss3 = model(source_data3, target_data3, source_label3, alpha, mark=3)
        gamma = 2 / (1 + math.exp(-10 * (batch) / (max_len))) - 1
        loss3 = rul_loss3 + gamma * (mmd_loss3 + l1_loss3)
        loss3.backward()
        optimizer.step()
        running_loss_scr3 += loss3.item()
        running_mmd_loss_scr3 += mmd_loss3.item()
        batch_src3 += 1
        if batch_src3 >= len(list_src3) - 1:
            batch_src3 = 0

        batch_tar += 1
        if batch_tar >= len(list_tar) - 1:
            batch_tar = 0
        # print("batch_src2,batch_tar", batch_src2, batch_tar)

        if batch % log_interval == 0:
            print(
                'Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tRUL_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\n\n'.format(
                    batch, 100. * batch / max_len, loss3.item(), rul_loss3.item(), mmd_loss3.item(), l1_loss3.item()),
                file=f_mfsan_train, flush=True)

    f_mfsan_train.close()
    epoch_loss_scr1 = running_loss_scr1 / (max_len - 1)
    epoch_loss_scr2 = running_loss_scr2 / (max_len - 1)
    epoch_loss_scr3 = running_loss_scr3 / (max_len - 1)
    epoch_mmd_loss_scr1 = running_mmd_loss_scr1 / (max_len - 1)
    epoch_mmd_loss_scr2 = running_mmd_loss_scr2 / (max_len - 1)
    epoch_mmd_loss_scr3 = running_mmd_loss_scr3 / (max_len - 1)

    return epoch_loss_scr1, epoch_loss_scr2, epoch_loss_scr3, epoch_mmd_loss_scr1, epoch_mmd_loss_scr2, epoch_mmd_loss_scr3


def test(model):
    model.eval()

    f_mfsan_test = log_in_file("/mfsan_test_log.log")

    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.type(torch.FloatTensor).cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3 = model(data)

            pred1, pred2, pred3 =pred1.detach().cpu().numpy().squeeze(1), pred2.detach().cpu().numpy().squeeze(1), pred3.detach().cpu().numpy().squeeze(1)
            target = target.detach().cpu().numpy()

            rmse1 = rmse_cal(pred1, target)  # sum up batch loss
            score1 = score_cal(pred1, target)
            rmse2 = rmse_cal(pred2, target) # sum up batch loss
            score2 = score_cal(pred2, target)
            rmse3 = rmse_cal(pred3, target)  # sum up batch loss
            score3 = score_cal(pred3, target)
            #w1 = (mmd_loss1-0.5) ** (-2)
            #w2 = (mmd_loss2-0.5) ** (-2)
            #w3 = (mmd_loss3-0.5) ** (-2)
            w1 = mmd_loss1 ** (-1)
            w2 = mmd_loss2 ** (-1)
            w3 = mmd_loss3 ** (-1)
            #w1 = mmd_loss1
            #w2 = mmd_loss2
            #w3 = mmd_loss3
            w1 = w1.cpu().numpy()
            w2 = w2.cpu().numpy()
            w3 = w3.cpu().numpy()
            ws=w1+w2+w3

            #pred = (w1*pred1 + w2*pred2 + w3*pred3) / ws
            pred=(pred1+pred2+pred3)/3
            rmse = rmse_cal(pred, target) # sum up batch loss
            score = score_cal(pred, target)


        print(target_name, '\nTest set: Mutil_source rmse: {:.4f}, Mutil_source score: {:.4f}'.format(rmse, score), file=f_mfsan_test, flush=True)
        print('source1 rmse {:.4f}, source2 rmse {:.4f}, source3 rmse {:.4f}'.format(rmse1, rmse2,rmse3), file=f_mfsan_test, flush=True)
        print('source1 score {:.4f}, source2 score {:.4f}, source3 score {:.4f}\n\n'.format(score1, score2,score3), file=f_mfsan_test, flush=True)
    return rmse, score,rmse1,score1,rmse2,score2,rmse3,score3,pred1, pred2, pred3, pred,target


if __name__ == '__main__':
    early_stopper = EarlyStopper(patience=8, min_delta=0.01)
    rmse, score = 100.,0
    s1_rmse, s1_score, s2_rmse, s2_score, s3_rmse, s3_score = 0,0,0,0,0,0
    history = {}
    history['epoch'] = []
    history['epoch_loss_scr1'] = []
    history['epoch_loss_scr2'] = []
    history['epoch_loss_scr3'] = []
    history['test_rmse'] = []
    history['test_score'] = []
    history['s1_rmse'] = []
    history['s1_score'] = []
    history['s2_rmse'] = []
    history['s2_score'] = []
    history['s3_rmse'] = []
    history['s3_score'] = []
    history['pred1'] = []
    history['pred2'] = []
    history['pred3'] = []
    history['pred'] = []
    history['target'] = []

    model = model_mfsan.MFSAN().apply(weight_init)
    print(model)
    if cuda:
        model.cuda()

    for epoch in range(1,epochs+1):
        ### train phase
        epoch_loss_scr1,epoch_loss_scr2,epoch_loss_scr3,epoch_mmd_loss_scr1,epoch_mmd_loss_scr2,epoch_mmd_loss_scr3 = train(model)
        history['epoch'].append(epoch)
        history['epoch_loss_scr1'].append(epoch_loss_scr1)
        history['epoch_loss_scr2'].append(epoch_loss_scr2)
        history['epoch_loss_scr3'].append(epoch_loss_scr3)
        history['epoch_mmd_loss_scr1'].append(epoch_mmd_loss_scr1)
        history['epoch_mmd_loss_scr2'].append(epoch_mmd_loss_scr2)
        history['epoch_mmd_loss_scr3'].append(epoch_mmd_loss_scr3)
        print('Train result: Epoch: {}/{}\tepoch_loss_scr1: {:.4f}\tepoch_loss_scr2: {:.4f}\tepoch_loss_scr3: {:.4f}'.format(epoch, epochs, epoch_loss_scr1,epoch_loss_scr2,epoch_loss_scr3))

        ### test phase
        t_rmse, t_score, t_s1_rmse, t_s1_score, t_s2_rmse, t_s2_score, t_s3_rmse, t_s3_score, pred1, pred2, pred3, pred, target = test(model)
        history['test_rmse'].append(t_rmse)
        history['test_score'].append(t_score)
        history['s1_rmse'].append(t_s1_rmse)
        history['s1_score'].append(t_s1_score)
        history['s2_rmse'].append(t_s2_rmse)
        history['s2_score'].append(t_s2_score)
        history['s3_rmse'].append(t_s3_rmse)
        history['s3_score'].append(t_s3_score)
        history['pred1'].append(pred1)
        history['pred2'].append(pred2)
        history['pred3'].append(pred3)
        history['pred'].append(pred)
        history['target'].append(target)

        if t_rmse < rmse:
            rmse = t_rmse
            score = t_score
            s1_rmse, s1_score, s2_rmse, s2_score, s3_rmse, s3_score = t_s1_rmse, t_s1_score, t_s2_rmse, t_s2_score, t_s3_rmse, t_s3_score

            save_model(model, 'mfsan')

        print("Test result: ", source1_name, source2_name,source3_name, "to", target_name, "best rmse: %.6f best rmse's score: %.6f" % (rmse, score))
        print("best s1 mse: %.6f s1 score: %.6f s2 mse: %.6f s2 score: %.6f s3 mse: %.6f s3 score: %.6f" % (s1_rmse, s1_score,s2_rmse,s2_score, s3_rmse, s3_score), "\n")
        if early_stopper.early_stop(t_rmse):
            print('Early Stop！')
            break


    inf_dataframe = pd.DataFrame(history)
    inf_dataframe.to_csv("./log/process_inf.csv", index=False, sep=',')
    visualize_total_loss(history['epoch'], history['epoch_loss_scr1'], history['epoch_loss_scr2'], history['epoch_loss_scr3'])
    # visualize_total_loss(history['epoch'], history['epoch_mmd_loss_scr1'], history['epoch_mmd_loss_scr2'], history['epoch_mmd_loss_scr3'])
    # mmd loss visualize
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('mmd_loss')
    plt.plot(history['epoch'], np.array(history['epoch_mmd_loss_scr1']), label='source1 mmd Loss')
    plt.plot(history['epoch'], np.array(history['epoch_mmd_loss_scr2']), label='source2 mmd loss')
    plt.plot(history['epoch'], np.array(history['epoch_mmd_loss_scr3']), label='source3 mmd loss')
    plt.legend()

    print("ALL is finished!!!")

