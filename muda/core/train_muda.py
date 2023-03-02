from options import Options
import data_loader
from model import model_mfsan
from train_mfsan import train,test
from utils.utils import weight_init, log_in_file, save_model
import time
from utils.EarlyStopper import EarlyStopper
from utils.print_things import figure_generate
import pandas as pd
import os
pd.set_option('mode.chained_assignment', None)
current_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

if __name__ == '__main__':
    # 读取参数
    opt = Options().parse()
    opt.input_window = 15
    opt.batch_size = 128
    opt.learning_rate = 0.001
    opt.epochs = 3000
    opt.device = 'cuda:0'
    opt.cuda = False
    opt.save_path = './outputs/model_files/muda.pkl'


    # 读取源域训练数据
    target_train_loader = data_loader.load_training(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                    opt.batch_size)
    target_test_loader = data_loader.load_testing(opt.target_data_path, opt.sequence_length, opt.sensor_drop,
                                                  opt.batch_size)

    source1_loader = data_loader.load_training(opt.source_data_path1, opt.sequence_length, opt.sensor_drop,
                     opt.batch_size)
    source2_loader = data_loader.load_training(opt.source_data_path2, opt.sequence_length, opt.sensor_drop,
                      opt.batch_size)
    source3_loader = data_loader.load_training(opt.source_data_path3, opt.sequence_length, opt.sensor_drop,
                      opt.batch_size)


    source1_len = len(source1_loader)
    source2_len = len(source2_loader)
    source3_len = len(source3_loader)

    target_len = len(target_train_loader)
    min_len = min(source1_len, source2_len, source3_len, target_len)
    max_len = max(source1_len, source2_len, source3_len, target_len)


    early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    rmse, score = 1000., 0
    s1_rmse, s1_score, s2_rmse, s2_score, s3_rmse, s3_score = 0, 0, 0, 0, 0, 0
    history = {}
    history['epoch'] = []
    history['total_rul_loss'] = []
    history['total_mmd_loss'] = []
    history['total_l1_loss'] = []
    history['epoch_rul_loss_scr1'] = []
    history['epoch_rul_loss_scr2'] = []
    history['epoch_rul_loss_scr3'] = []
    history['epoch_mmd_loss_scr1'] = []
    history['epoch_mmd_loss_scr2'] = []
    history['epoch_mmd_loss_scr3'] = []
    history['epoch_l1_loss_scr1'] = []
    history['epoch_l1_loss_scr2'] = []
    history['epoch_l1_loss_scr3'] = []
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


    now = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) #一定要注意这里的时间格式，不要用冒号
    model = model_mfsan.MFSAN().apply(weight_init)
    print(model)
    if opt.cuda:
        model.cuda()
    f_mfsan_train = log_in_file('\\' + opt.target_data_name+ '_' + now + '_train.log')
    f_mfsan_test = log_in_file('\\' + opt.target_data_name+ '_' + now + '_test.log')

    print("当前日期和时间：", now, file=f_mfsan_train, flush=True)
    print('training settings:\t', 'lr:', opt.learning_rate, 'momentum:', opt.momentum, 'l2_decay:',opt.l2_decay, 'optimizer:', opt.optimizer,
           file=f_mfsan_train, flush=True)
    print('model architecture:\n', model, file=f_mfsan_train, flush=True)
    print(opt.source_data_name1,opt.source_data_name2,opt.source_data_name3, "to", opt.target_data_name, file=f_mfsan_train, flush=True)

    for epoch in range(1,opt.epochs+1):

        ### train phase
        f_mfsan_train = log_in_file('/'+opt.target_data_name+'_'+now+'_train.log')
        print('Train epochs:{:.6f}\t'.format(epoch), file=f_mfsan_train, flush=True)
        epoch_loss_scr1,epoch_loss_scr2,epoch_loss_scr3,epoch_mmd_loss_scr1,epoch_mmd_loss_scr2,epoch_mmd_loss_scr3,epoch_l1_loss_scr1,epoch_l1_loss_scr2,epoch_l1_loss_scr3 = train(model,max_len,source1_loader,source2_loader,source3_loader,target_train_loader)
        history['epoch'].append(epoch)

        history['total_rul_loss'].append((epoch_loss_scr1+epoch_loss_scr2+epoch_loss_scr3) / 3)
        history['total_mmd_loss'].append((epoch_mmd_loss_scr1+epoch_mmd_loss_scr2+epoch_mmd_loss_scr3) / 3)
        history['total_l1_loss'].append((epoch_l1_loss_scr1+epoch_l1_loss_scr2+epoch_l1_loss_scr3) / 3)

        history['epoch_rul_loss_scr1'].append(epoch_loss_scr1)
        history['epoch_rul_loss_scr2'].append(epoch_loss_scr2)
        history['epoch_rul_loss_scr3'].append(epoch_loss_scr3)
        history['epoch_mmd_loss_scr1'].append(epoch_mmd_loss_scr1)
        history['epoch_mmd_loss_scr2'].append(epoch_mmd_loss_scr2)
        history['epoch_mmd_loss_scr3'].append(epoch_mmd_loss_scr3)
        history['epoch_l1_loss_scr1'].append(epoch_l1_loss_scr1)
        history['epoch_l1_loss_scr2'].append(epoch_l1_loss_scr2)
        history['epoch_l1_loss_scr3'].append(epoch_l1_loss_scr3)
        print('Train result: Epoch: {}/{}\tepoch_loss_scr1: {:.4f}\tepoch_loss_scr2: {:.4f}\tepoch_loss_scr3: {:.4f}'.format(epoch, opt.epochs, epoch_loss_scr1,epoch_loss_scr2,epoch_loss_scr3))

        ### test phase

        f_mfsan_test = log_in_file('/'+opt.target_data_name+'_'+now+'_test.log')
        t_rmse, t_score, t_s1_rmse, t_s1_score, t_s2_rmse, t_s2_score, t_s3_rmse, t_s3_score, pred1, pred2, pred3, pred, target = test(model,target_test_loader)
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

        print("Test result: ", "best rmse: %.6f best rmse's score: %.6f" % (rmse, score))
        print("best s1 mse: %.6f s1 score: %.6f s2 mse: %.6f s2 score: %.6f s3 mse: %.6f s3 score: %.6f" % (s1_rmse, s1_score,s2_rmse,s2_score, s3_rmse, s3_score), "\n")

        if early_stopper.early_stop(t_rmse):
            print('Early Stop！')
            break

    figure_generate(current_dir,history,now,opt.target_data_name)