"#_*_ coding:utf-8 _*_"
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import argparse
import os

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


source_chosen = ['None',1,2,3]
target_chosen = ['None',4]

target_name = datasetset_name[target_chosen[1]]
source1_name = datasetset_name[source_chosen[1]]
source2_name = datasetset_name[source_chosen[2]]
source3_name = datasetset_name[source_chosen[3]]
sensor_drop = ['sensor_01', 'sensor_05', 'sensor_06', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

lr = [0.001, 0.001,0.005]


class Options():
    """
    Options class
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--seed', default=6, type=int, help='set seed for model')
        self.parser.add_argument('--target_data_path', default=target_path[target_chosen[1]], help='目标数据集的路径')
        self.parser.add_argument('--source_data_path1', default=source_path[source_chosen[1]], help='源域数据集1的路径')
        self.parser.add_argument('--source_data_path2', default=source_path[source_chosen[2]], help='源域数据集2的路径')
        self.parser.add_argument('--source_data_path3', default=source_path[source_chosen[3]], help='源域数据集3的路径')
        self.parser.add_argument('--target_data_name', default=target_name, help='目标数据集的名称')
        self.parser.add_argument('--source_data_name1', default=source1_name, help='源域数据集1的名称')
        self.parser.add_argument('--source_data_name2', default=source2_name, help='源域数据集2的名称')
        self.parser.add_argument('--source_data_name3', default=source3_name, help='源域数据集3的名称')
        self.parser.add_argument('--sequence_length', type=int, default=30, help='序列长度')
        self.parser.add_argument('--sensor_drop',default=sensor_drop, help='去除的数据列')
        self.parser.add_argument('--l2_decay', type=float,default=1e-3, help='l2正则化')
        self.parser.add_argument('--cuda', type=bool, default=True, help ='是否使用GPU')
        self.parser.add_argument('--momentum', type=float, default=0.9, help ='动量设置')
        self.parser.add_argument('--log_interval', type=int, default=10, help ='日志batch间隔')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help ='优化器')


        self.parser.add_argument('--batch_size', type=int, default=128, help='每批数据的个数')
        self.parser.add_argument('--learning_rate', type=float, default=0.002, help='学习率')
        self.parser.add_argument('--epochs', type=int, default=30, help='轮数')
        self.parser.add_argument('--device', type=str, default='cuda:0', help='GPU')
        self.parser.add_argument('--save_path', default='/home/poac/lhr/Li_prediction/outputs/model_files/transfer_transformer.pkl', type=str, help='模型保存路径')
        self.parser.add_argument('--dis_save_path', type=str, help='判别器模型保存路径')
        self.parser.add_argument('--trans_save_path', type=str, help='transformer模型保存路径')
        self.parser.add_argument('--target_test_number', type=str, help='用于目标域测试的数据文件编号')


        # transformer_time_series_prediction
        self.parser.add_argument('--phase', type=str, default='train', help='程序运行方式')
        self.parser.add_argument('--enc_seq_len', default=10, help='输入encoder的序列的长度')
        self.parser.add_argument('--dec_seq_len', type=int, default=2, help='输入decoder的序列的长度')
        self.parser.add_argument('--output_sequence_len', type=int, default=1, help='输出序列的长度')
        self.parser.add_argument('--input_size', type=int, default=9, help='输入序列的维度')
        self.parser.add_argument('--dim_val', type=int, default=16, help=' ')
        self.parser.add_argument('--dim_attn', type=int, default=8, help=' ')
        self.parser.add_argument('--n_heads', type=int, default=9, help='多头个数')
        self.parser.add_argument('--n_decoder_layers', type=int, default=5, help='解码器的层数')
        self.parser.add_argument('--n_encoder_layers', type=int, default=5, help='编码器的层数')

        # JDOT
        self.parser.add_argument('--net', default='conv1d', help='选择JDOT中具体使用的网络类型')


        # 运行参数
        self.parser.add_argument('--r', type=str, help='运行方式：train/eval')

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        # self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)
        return self.opt
