from options import Options
import data_loader
from muda.model import model_mfsan2 as model_mfsan
# from train_mfsan import train,test
from muda.utils.utils import weight_init, log_in_file, save_model, set_seed
import time
from muda.utils.EarlyStopper import EarlyStopper
from muda.utils.print_things import figure_generate_2src,writing_settings_2src
import pandas as pd
import os
import torch
from torch.autograd import Variable
import numpy as np
import muda.utils.utils as utils
from muda.utils.utils import score_cal,rmse_cal
import math

pd.set_option('mode.chained_assignment', None)
current_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
def train(model,max_len,source1_loader,source2_loader,target_train_loader):
# def train(model, target_train_loader):
    global mmd_loss1,mmd_loss2,mmd_loss3 #全局
    print("--------------------------MFSAN --------------------------------")


    batch_src1,batch_src2,batch_src3,batch_tar = 0,0,0,0
    running_loss_scr1, running_loss_scr2 ,running_loss_scr3= 0, 0, 0
    running_mmd_loss_scr1, running_mmd_loss_scr2, running_mmd_loss_scr3 = 0, 0, 0
    running_l1_loss_scr1, running_l1_loss_scr2, running_l1_loss_scr3 = 0, 0, 0
    list_src1, list_src2, list_tar = list(enumerate(source1_loader)), list(enumerate(source2_loader)),list(enumerate(target_train_loader))
    #     {'params': model.sharedNet.parameters(), 'lr': lr[1]},

    #     {'params': model.rul_fc_son1.parameters(), 'lr': lr[1]},
    #     {'params': model.rul_fc_son2.parameters(), 'lr': lr[1]},

    #     {'params': model.sonnet1.parameters(), 'lr': lr[1]},
    #     {'params': model.sonnet2.parameters(), 'lr': lr[1]},

    # ], lr=lr[2], momentum=momentum, weight_decay=l2_decay)
    optimizer = torch.optim.Adam([
        {'params': model.sharedNet.parameters(), 'lr': opt.learning_rate},

        {'params': model.rul_fc_son1.parameters(), 'lr': opt.learning_rate},
        {'params': model.rul_fc_son2.parameters(), 'lr': opt.learning_rate},

        {'params': model.sonnet1.parameters(), 'lr': opt.learning_rate},
        {'params': model.sonnet2.parameters(), 'lr': opt.learning_rate},
        ], lr = opt.learning_rate, weight_decay = opt.l2_decay)


    for batch in range(1, max_len):
        model.train()

        p = (batch - 1) / (max_len)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 #？？？
        optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p) #Adjust the learning rate of optimizer
        optimizer.zero_grad() #梯度初始化为0

        #####scr1 tgt
        _, (source_data1, source_label1) = list_src1[batch_src1]
        _, (target_data1, _) = list_tar[batch_tar]


        if opt.cuda:
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
        running_l1_loss_scr1 += l1_loss1.item()
        batch_src1 += 1
        if batch_src1 >= len(list_src1)-1:
            batch_src1 = 0

        if batch_tar >= len(list_tar)-1:
            batch_tar = 0

        if batch % opt.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tRUL_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                batch, 100. * batch/ max_len, loss1.item(), rul_loss1.item(), mmd_loss1.item(), l1_loss1.item()), file=f_mfsan_train, flush=True)

        # print("batch_src1,batch_tar",batch_src1,batch_tar)

        #####scr2 tgt
        _, (source_data2, source_label2) = list_src2[batch_src2]
        _, (target_data2, _) = list_tar[batch_tar]

        if opt.cuda:
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
        running_l1_loss_scr2 += l1_loss2.item()
        batch_src2 += 1
        if batch_src2 >= len(list_src2)-1:
            batch_src2 = 0


        if batch_tar >= len(list_tar)-1:
            batch_tar = 0
        # print("batch_src2,batch_tar", batch_src2, batch_tar)

        if batch % opt.log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tRUL_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    batch, 100. * batch / max_len, loss2.item(), rul_loss2.item(), mmd_loss2.item(), l1_loss2.item()), file=f_mfsan_train, flush=True)

    epoch_loss_scr1 = running_loss_scr1 / (max_len - 1)
    epoch_loss_scr2 = running_loss_scr2 / (max_len - 1)
    # epoch_loss_scr3 = running_loss_scr3 / (max_len - 1)
    epoch_mmd_loss_scr1 = running_mmd_loss_scr1 / (max_len - 1)
    epoch_mmd_loss_scr2 = running_mmd_loss_scr2 / (max_len - 1)
    # epoch_mmd_loss_scr3 = running_mmd_loss_scr3 / (max_len - 1)
    epoch_l1_loss_scr1 = running_l1_loss_scr1 / (max_len - 1)
    epoch_l1_loss_scr2 = running_l1_loss_scr2 / (max_len - 1)
    # epoch_l1_loss_scr3 = running_l1_loss_scr3 / (max_len - 1)

    return epoch_loss_scr1, epoch_loss_scr2,epoch_mmd_loss_scr1, epoch_mmd_loss_scr2, epoch_l1_loss_scr1, epoch_l1_loss_scr2


def test(model,target_test_loader):

    model.eval()


    with torch.no_grad():
        for data, target in target_test_loader:
            if opt.cuda:
                data, target = data.cuda(), target.type(torch.FloatTensor).cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data)

            pred1, pred2 =pred1.detach().cpu().numpy().squeeze(1), pred2.detach().cpu().numpy().squeeze(1)
            target = target.detach().cpu().numpy()

            rmse1 = rmse_cal(pred1, target)  # sum up batch loss
            score1 = score_cal(pred1, target)
            rmse2 = rmse_cal(pred2, target) # sum up batch loss
            score2 = score_cal(pred2, target)
            # rmse3 = rmse_cal(pred3, target)  # sum up batch loss
            # score3 = score_cal(pred3, target)
            # w1 = (mmd_loss1-0.5) ** (-2)
            # w2 = (mmd_loss2-0.5) ** (-2)
            # w3 = (mmd_loss3-0.5) ** (-2)
            # w1 = mmd_loss1 ** (-1)
            # w2 = mmd_loss2 ** (-1)
            # w3 = mmd_loss3 ** (-1)
            #w1 = mmd_loss1
            #w2 = mmd_loss2
            #w3 = mmd_loss3
            # w1 = w1.cpu().numpy()
            # w2 = w2.cpu().numpy()
            # w3 = w3.cpu().numpy()
            # ws=w1+w2+w3

            # pred = (w1*pred1 + w2*pred2 + w3*pred3) / ws
            pred=(pred1+pred2)/2
            rmse = rmse_cal(pred, target) # sum up batch loss
            score = score_cal(pred, target)

       # print('source1 w {:.4f}, source2 w {:.4f}, source3 w {:.4f}\n\n'.format(w1,w2,w3))
    return rmse, score,rmse1,score1,rmse2,score2,pred1, pred2, pred,target

#path define
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
## Read csv file to pandas dataframe
FD_path = ["none", FD001_path, FD002_path, FD003_path, FD004_path]
FD_name = ["none", "FD001", "FD002", "FD003", "FD004"]

source_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]  #此处选择source 数据集 1 2 3 分别对应FD001-FD004
target_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]
datasetset_name = ["none", "FD001", "FD002", "FD003", "FD004"]


target_path = ["none", FD_path[1], FD_path[2], FD_path[3], FD_path[4]]
datasetset_name = ["none", "FD001", "FD002", "FD003", "FD004"]


source_chosen = ['None',1,3]
target_chosen = ['None',2]

target_name = datasetset_name[target_chosen[1]]
source1_name = datasetset_name[source_chosen[1]]
source2_name = datasetset_name[source_chosen[2]]

if __name__ == '__main__':
    # 读取参数
    opt = Options().parse()
    # opt.input_window = 15
    opt.batch_size = 512
    opt.learning_rate = 0.001
    opt.epochs = 500
    opt.seed = 6
    opt.source_data_path1 = source_path[source_chosen[1]]
    opt.source_data_path2 = source_path[source_chosen[2]]
    opt.target_data_path = target_path[target_chosen[1]]
    opt.source_data_name1 = source1_name
    opt.target_data_name = target_name
    opt.cuda = True
    opt.source_data_name2 = source2_name
    # opt.save_path = './outputs/model_files/muda.pkl'
    set_seed(opt.seed)
    # 读取源域训练数据
    target_train_loader = data_loader.load_training(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                    opt.batch_size, suffle= True)
    target_test_loader = data_loader.load_testing(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                  opt.batch_size, suffle= False)

    source1_loader = data_loader.load_training(opt.source_data_path1, opt.sequence_length, opt.sensor_drop,
                                               opt.batch_size, opt.seed)
    source2_loader = data_loader.load_training(opt.source_data_path2, opt.sequence_length, opt.sensor_drop,
                                               opt.batch_size, opt.seed)



    source1_len = len(source1_loader)
    source2_len = len(source2_loader)
    target_len = len(target_train_loader)
    min_len = min(source1_len, source2_len,  target_len)
    max_len = max(source1_len, source2_len,  target_len)


    early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    rmse, score = 1000., 0
    pred, target = [],[]
    s1_rmse, s1_score, s2_rmse, s2_score, s3_rmse, s3_score = 0, 0, 0, 0, 0, 0
    history = {}
    history['epoch'] = []
    history['total_rul_loss'] = []
    history['total_mmd_loss'] = []
    history['total_l1_loss'] = []
    history['epoch_rul_loss_scr1'] = []
    history['epoch_rul_loss_scr2'] = []
    # history['epoch_rul_loss_scr3'] = []
    history['epoch_mmd_loss_scr1'] = []
    history['epoch_mmd_loss_scr2'] = []
    # history['epoch_mmd_loss_scr3'] = []
    history['epoch_l1_loss_scr1'] = []
    history['epoch_l1_loss_scr2'] = []
    # history['epoch_l1_loss_scr3'] = []
    history['test_rmse'] = []
    history['test_score'] = []
    history['s1_rmse'] = []
    history['s1_score'] = []
    history['s2_rmse'] = []
    history['s2_score'] = []
    # history['s3_rmse'] = []
    # history['s3_score'] = []



    now = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    model = model_mfsan.MFSAN().apply(weight_init)
    print(model)
    if opt.cuda:
        model.cuda()


    #数据写入
    f_mfsan_train = log_in_file('\\' +opt.target_data_name+ '_2src_' + now + '_train.log')
    f_mfsan_test = log_in_file('\\' + opt.target_data_name+ '_2src_' + now + '_test.log')
    overall_log = log_in_file('\\' + 'overall_log.log')
    writing_settings_2src(now, opt, model, f_mfsan_train=f_mfsan_train)
    writing_settings_2src(now, opt, model, f_mfsan_train=f_mfsan_test)
    print("当前日期和时间：", now, file=overall_log, flush=True)
    print(opt.source_data_name1, opt.source_data_name2, "to", opt.target_data_name,
          file=overall_log, flush=True)
    print('training settings:\t', 'lr:', opt.learning_rate, 'l2_decay:',opt.l2_decay, 'optimizer:', opt.optimizer, 'seed:', opt.seed,
          file=overall_log, flush=True)

    #训练
    for epoch in range(1,opt.epochs+1):

        ### train phase

        epoch_loss_scr1, epoch_loss_scr2, epoch_mmd_loss_scr1, epoch_mmd_loss_scr2, epoch_l1_loss_scr1, epoch_l1_loss_scr2 = train(model,max_len,source1_loader,source2_loader,target_train_loader)
        print('Train epochs:{:.6f}\t'.format(epoch), file=f_mfsan_train, flush=True)
        print('Train result: Epoch: {}/{}\tepoch_loss_scr1: {:.4f}\tepoch_loss_scr2: {:.4f}'.format(epoch, opt.epochs, epoch_loss_scr1,epoch_loss_scr2))
        history['epoch'].append(epoch)
        history['total_rul_loss'].append((epoch_loss_scr1+epoch_loss_scr2) / 2)
        history['total_mmd_loss'].append((epoch_mmd_loss_scr1+epoch_mmd_loss_scr2) / 2)
        history['total_l1_loss'].append((epoch_l1_loss_scr1+epoch_l1_loss_scr2) / 2)
        history['epoch_rul_loss_scr1'].append(epoch_loss_scr1)
        history['epoch_rul_loss_scr2'].append(epoch_loss_scr2)
        # history['epoch_rul_loss_scr3'].append(epoch_loss_scr3)
        history['epoch_mmd_loss_scr1'].append(epoch_mmd_loss_scr1)
        history['epoch_mmd_loss_scr2'].append(epoch_mmd_loss_scr2)
        # history['epoch_mmd_loss_scr3'].append(epoch_mmd_loss_scr3)
        history['epoch_l1_loss_scr1'].append(epoch_l1_loss_scr1)
        history['epoch_l1_loss_scr2'].append(epoch_l1_loss_scr2)
        # history['epoch_l1_loss_scr3'].append(epoch_l1_loss_scr3)


        ### test phase

        t_rmse, t_score, t_s1_rmse, t_s1_score, t_s2_rmse, t_s2_score, pred1, pred2, t_pred, t_target = test(model,target_test_loader)

        print('multi_rmse: {:.4f}, multi_score: {:.4f}'.format(rmse, score), file=f_mfsan_test, flush=True)
        print('source1 rmse {:.4f}, source2 rmse {:.4f}'.format(t_s1_rmse, t_s2_rmse), file=f_mfsan_test, flush=True)
        print('source1 score {:.4f}, source2 score {:.4f}\n'.format(t_s1_score, t_s2_score), file=f_mfsan_test, flush=True)
        history['test_rmse'].append(t_rmse)
        history['test_score'].append(t_score)
        history['s1_rmse'].append(t_s1_rmse)
        history['s1_score'].append(t_s1_score)
        history['s2_rmse'].append(t_s2_rmse)
        history['s2_score'].append(t_s2_score)

        if t_rmse < rmse:
            rmse = t_rmse
            score = t_score
            pred = t_pred
            target = t_target
            s1_rmse, s1_score, s2_rmse, s2_score = t_s1_rmse, t_s1_score, t_s2_rmse, t_s2_score



            save_model(model, now + 'mfsan')

        print("Test result: ", "best rmse: %.6f best rmse's score: %.6f" % (rmse, score))
        print("best s1 mse: %.6f s1 score: %.6f s2 mse: %.6f s2 score: %.6f " % (s1_rmse, s1_score,s2_rmse,s2_score), "\n")

        if early_stopper.early_stop(t_rmse):
            print('Early Stop！')
            break

    print("Test result: ", "best rmse: %.6f best rmse's score: %.6f" % (rmse, score), file=overall_log, flush=True)
    print("best s1 mse: %.6f s1 score: %.6f s2 mse: %.6f s2 score: %.6f " % (s1_rmse, s1_score,s2_rmse,s2_score), "\n",
          file=overall_log, flush=True)
    overall_log.close()
    f_mfsan_train.close()
    f_mfsan_test.close()

    figure_generate_2src(current_dir, history, now, opt.target_data_name, pred, target)