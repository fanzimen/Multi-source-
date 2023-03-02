from options import Options
import data_loader
import os
# import model_mfsan
from utils.utils import get_free_gpu, weight_init, weight_init2, log_in_file, visualize_total_loss, save_model,score_cal,rmse_cal,load_model


if __name__ == '__main__':
    opt = Options().parse()
    opt.input_window = 15
    opt.batch_size = 128
    opt.learning_rate = 0.001
    opt.epochs = 3000
    opt.device = 'cuda:0'
    opt.save_path = './outputs/model_files/muda.pkl'

    print(os.path.abspath(os.path.join(os.getcwd(), "..")))