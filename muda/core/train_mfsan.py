import numpy as np
import torch
from muda.utils.utils import score_cal,rmse_cal
import math
from torch.autograd import Variable
import muda.utils.utils as utils
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from options import Options
opt = Options().parse()
cuda = opt.cuda

def train(model,max_len,source1_loader,source2_loader,source3_loader,target_train_loader,f_mfsan_train):
    global mmd_loss1,mmd_loss2,mmd_loss3 #全局
    print("--------------------------MFSAN --------------------------------")

    batch_src1,batch_src2,batch_src3,batch_tar = 0,0,0,0
    running_loss_scr1, running_loss_scr2 ,running_loss_scr3= 0, 0, 0
    running_mmd_loss_scr1, running_mmd_loss_scr2, running_mmd_loss_scr3 = 0, 0, 0
    running_l1_loss_scr1, running_l1_loss_scr2, running_l1_loss_scr3 = 0, 0, 0
    list_src1, list_src2, list_src3, list_tar = list(enumerate(source1_loader)), list(enumerate(source2_loader)), list(enumerate(source3_loader)),list(enumerate(target_train_loader))

    # optimizer = torch.optim.SGD([
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

        #####scr3 tgt
        _, (source_data3, source_label3) = list_src3[batch_src3]
        _, (target_data3, _) = list_tar[batch_tar]

        if opt.cuda:
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
        running_l1_loss_scr3 += l1_loss3.item()
        batch_src3 += 1
        if batch_src3 >= len(list_src3) - 1:
            batch_src3 = 0

        batch_tar += 1
        if batch_tar >= len(list_tar) - 1:
            batch_tar = 0
        # print("batch_src2,batch_tar", batch_src2, batch_tar)

        if batch % opt.log_interval == 0:
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
    epoch_l1_loss_scr1 = running_l1_loss_scr1 / (max_len - 1)
    epoch_l1_loss_scr2 = running_l1_loss_scr2 / (max_len - 1)
    epoch_l1_loss_scr3 = running_l1_loss_scr3 / (max_len - 1)

    return epoch_loss_scr1, epoch_loss_scr2, epoch_loss_scr3, epoch_mmd_loss_scr1, epoch_mmd_loss_scr2, epoch_mmd_loss_scr3, epoch_l1_loss_scr1, epoch_l1_loss_scr2, epoch_l1_loss_scr3


def test(model,target_test_loader,f_mfsan_test):

    model.eval()

    with torch.no_grad():
        for data, target in target_test_loader:
            if opt.cuda:
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
            # w1 = (mmd_loss1-0.5) ** (-2)
            # w2 = (mmd_loss2-0.5) ** (-2)
            # w3 = (mmd_loss3-0.5) ** (-2)
            # w1 = mmd_loss1 ** (-1)
            # w2 = mmd_loss2 ** (-1)
            # w3 = mmd_loss3 ** (-1)
            # #w1 = mmd_loss1
            #w2 = mmd_loss2
            #w3 = mmd_loss3
            # w1 = w1.cpu().numpy()
            # w2 = w2.cpu().numpy()
            # w3 = w3.cpu().numpy()
            # ws=w1+w2+w3

            # pred = (w1*pred1 + w2*pred2 + w3*pred3) / ws
            pred=(pred1+pred2+pred3)/3
            rmse = rmse_cal(pred, target) # sum up batch loss
            score = score_cal(pred, target)


        print('multi_rmse: {:.4f}, multi_score: {:.4f}'.format(rmse, score), file=f_mfsan_test, flush=True)
        print('source1 rmse {:.4f}, source2 rmse {:.4f}, source3 rmse {:.4f}'.format(rmse1, rmse2,rmse3), file=f_mfsan_test, flush=True)
        print('source1 score {:.4f}, source2 score {:.4f}, source3 score {:.4f}\n\n'.format(score1, score2,score3), file=f_mfsan_test, flush=True)
        # print('source1 w {:.4f}, source2 w {:.4f}, source3 w {:.4f}\n\n'.format(w1,w2,w3))
    return rmse, score,rmse1,score1,rmse2,score2,rmse3,score3,pred1, pred2, pred3, pred,target