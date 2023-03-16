from options import Options

# import model_mfsan
# from utils.utils import get_free_gpu, weight_init, weight_init2, log_in_file, visualize_total_loss, save_model,score_cal,rmse_cal,load_model


if __name__ == '__main__':
    opt = Options().parse()
    opt.input_window = 15
    opt.batch_size = 128
    opt.learning_rate = 0.001
    opt.epochs = 3000
    opt.device = 'cuda:0'
    opt.save_path = './outputs/model_files/muda.pkl'

    history = {}
    train_list = ['epoch','total_rul_loss']
    sigh = [[]*2]
    dic = dict(zip(train_list,sigh))
    print(dic)
    # history['total_rul_loss'] = []
    # history['total_mmd_loss'] = []
    # history['total_l1_loss'] = []
    # print(history)
    # history['epoch_rul_loss_scr1'] = []
    # history['epoch_rul_loss_scr2'] = []
    # history['epoch_rul_loss_scr3'] = []
    # history['epoch_mmd_loss_scr1'] = []
    # history['epoch_mmd_loss_scr2'] = []
    # history['epoch_mmd_loss_scr3'] = []
    # history['epoch_l1_loss_scr1'] = []
    # history['epoch_l1_loss_scr2'] = []
    # history['epoch_l1_loss_scr3'] = []
    # history['test_rmse'] = []
    # history['test_score'] = []
    # history['s1_rmse'] = []
    # history['s1_score'] = []
    # history['s2_rmse'] = []
    # history['s2_score'] = []
    # history['s3_rmse'] = []
    # history['s3_score'] = []