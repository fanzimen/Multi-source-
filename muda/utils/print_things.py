from muda.utils.utils import get_free_gpu, weight_init, weight_init2, log_in_file, visualize_total_loss, save_model,score_cal,rmse_cal,load_model
import sys
import matplotlib.pyplot as plt
import numpy as np
import muda.utils.utils as utils
def writing_settings(target_name,now,lr,momentum,l2_decay,optimizer,len_setting,source1_name,source2_name,source3_name,model):

    f_mfsan_train = log_in_file('/'+target_name+'_'+now+'_train.log')
    f_mfsan_test = log_in_file('/'+target_name+'_'+now+'_test.log')

    print("当前日期和时间：", now,file=f_mfsan_train, flush=True)
    print('training settings:\t','lr:',lr,'momentum:',momentum,'l2_decay:',l2_decay,'optimizer:',optimizer,'len_setting:',len_setting,file=f_mfsan_train, flush=True)
    print('model architecture:\n',model,file=f_mfsan_train, flush=True)
    print(source1_name, source2_name,source3_name, "to", target_name,file=f_mfsan_train, flush=True)
    print("当前日期和时间：", now,file=f_mfsan_test, flush=True)
    print('training settings:\t','lr:',lr,'momentum:',momentum,'l2_decay:',l2_decay,'optimizer:',optimizer,'len_setting:',len_setting,file=f_mfsan_test, flush=True)
    print('model architecture:\n',model,file=f_mfsan_test, flush=True)
    print(source1_name, source2_name,source3_name, "to", target_name,file=f_mfsan_test, flush=True)

def figure_generate(current_dir,history,now,target_name,rmse, score):
    # Generate the figure
    fig = plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('average loss')
    plt.plot(history['epoch'], np.array(history['total_rul_loss'])*125, label='rul loss')
    plt.plot(history['epoch'], np.array(history['total_mmd_loss'])*125, label='mmd loss')
    plt.plot(history['epoch'], np.array(history['total_l1_loss'])*125, label='l1 loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('rul_loss')
    plt.plot(history['epoch'], np.array(history['epoch_rul_loss_scr1'])*125, label='source1 rul loss')
    plt.plot(history['epoch'], np.array(history['epoch_rul_loss_scr2'])*125, label='source2 rul loss')
    plt.plot(history['epoch'], np.array(history['epoch_rul_loss_scr3'])*125, label='source3 rul loss')
    plt.legend()

    # mmd loss visualize
    plt.subplot(2, 2, 3)
    plt.xlabel('Epoch')
    plt.ylabel('mmd_loss')
    plt.plot(history['epoch'], np.array(history['epoch_mmd_loss_scr1'])*125, label='source1 mmd loss')
    plt.plot(history['epoch'], np.array(history['epoch_mmd_loss_scr2'])*125, label='source2 mmd loss')
    plt.plot(history['epoch'], np.array(history['epoch_mmd_loss_scr3'])*125, label='source3 mmd loss')
    plt.legend()

    # l1 loss visuliza
    plt.subplot(2, 2, 4)
    plt.xlabel('Epoch')
    plt.ylabel('l1_loss')
    plt.plot(history['epoch'], np.array(history['epoch_l1_loss_scr1'])*125, label='source1 l1 loss')
    plt.plot(history['epoch'], np.array(history['epoch_l1_loss_scr2'])*125, label='source2 l1 loss')
    plt.plot(history['epoch'], np.array(history['epoch_l1_loss_scr3'])*125, label='source3 l1 loss')
    plt.legend()
    print("ALL is finished!!!")
    # Save the plots to a file
    plt.savefig(current_dir+'/outputs/figures/'+now+"_target_"+target_name+'.png')