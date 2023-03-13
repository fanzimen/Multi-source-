import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# from pyts.image import RecurrencePlot
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
import random
# from sklearn.decomposition import PCA
# from pyts.approximation import SymbolicFourierApproximation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from options import Options

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

def oc_history_cols(read_path, sub_dataset, train_data, test_data, save=False):
    
    if "operating_condition" not in train_data.columns or "operating_condition" not in test_data.columns:
        print("Column operating_condition is not found in the data frame")

    else:
        print("Adding History Columns in the Data Frame")
        train_data[["oc_0","oc_1","oc_2","oc_3","oc_4","oc_5"]]= pd.DataFrame([[0,0,0,0,0,0]], index=train_data.index)
        test_data[["oc_0","oc_1","oc_2","oc_3","oc_4","oc_5"]]= pd.DataFrame([[0,0,0,0,0,0]], index=test_data.index)
        
        for file in ["train", "test"]:
            if file == "train":
                groupby_traj = train_data.groupby('engine_id', sort=False)
            else:
                groupby_traj = test_data.groupby('engine_id', sort=False)
                
            additional_oc=[]
            for engine_id, data in groupby_traj:
                data=data.reset_index()
                for i in range(data.shape[0]):
                    check_oc=data.iloc[i]["operating_condition"]
                    if  i != data.shape[0]-1:
                        data.at[i+1:, "oc_"+str(int(check_oc))]=data.iloc[i+1]["oc_"+str(int(check_oc))]+1
                additional_oc.append(data)
            
            oc_cols=pd.concat(additional_oc,  sort=False, ignore_index=False)
            oc_cols=oc_cols.set_index('index', drop=True)
            
            if save:
                print("Saving the {} data with operating condition history columns".format(file))
                oc_cols.to_csv(read_path+file+"_FD"+sub_dataset+"_cluster.csv", index=False)
    return None

def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0] #矩阵行数

    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    return data_matrix[seq_length:num_elements, :]


### 对输入数据操作：
# 截取MAX 先有一段平台，
# 然后minmax（这里只针对了sensor 除去了3个os、和cycle、还有空、和const值、一些sensors）
# 最后分段利用gen_sequence和·gen_labels 对train和test（对于test还需要挑选长度大于我们的窗口长度的数据）
class input_trans(object):
    def __init__(self, data_path_list, sequence_length, sensor_drop, piecewise_lin_ref=125, preproc=True):


        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.data_path_list = data_path_list
        self.sequence_length = sequence_length
        self.sensor_drop = sensor_drop
        self.preproc = preproc   #预处理
        self.piecewise_lin_ref = piecewise_lin_ref #分段线性

        ## Assign columns name
        cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
        cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
        col_rul = ['RUL_truth']

        train_FD = pd.read_csv(self.data_path_list[0], sep=' ', header=None,
                              names=cols, index_col=False)
        test_FD = pd.read_csv(self.data_path_list[1], sep=' ', header=None,
                              names=cols, index_col=False)
        # train_FD, test_FD = cluster(self.data_path_list)
        RUL_FD = pd.read_csv(self.data_path_list[2], sep=' ', header=None,
                             names=col_rul, index_col=False)

        ## Calculate RUL and append to train data 并设置max值 砍掉过大的那些
        # get the time of the last available measurement for each unit  dict{1: 23, 2: 50, 3: 5} 得到每个id cycle最大值
        mapper = {}
        for unit_nr in train_FD['unit_nr'].unique():
            mapper[unit_nr] = train_FD['cycles'].loc[train_FD['unit_nr'] == unit_nr].max()

        # calculate RUL = time.max() - time_now for each unit
        train_FD['RUL'] = (train_FD['unit_nr'].apply(lambda nr: mapper[nr]) - train_FD['cycles'])/train_FD['unit_nr'].apply(lambda nr: mapper[nr])
        # piecewise linear for RUL labels
        train_FD['RUL'].loc[(train_FD['RUL'] > self.piecewise_lin_ref)] = self.piecewise_lin_ref
        #print(train_FD['RUL'])
        # Cut max RUL ground truth
        RUL_FD['RUL_truth'].loc[(RUL_FD['RUL_truth'] > self.piecewise_lin_ref)] = self.piecewise_lin_ref

        ## Excluse columns which only have NaN as value
        # nan_cols = ['sensor_{0:02d}'.format(s + 22) for s in range(5)]
        cols_nan = train_FD.columns[train_FD.isna().any()].tolist()
        # print('Columns with all nan: \n' + str(cols_nan) + '\n')
        cols_const = [col for col in train_FD.columns if len(train_FD[col].unique()) <= 2]  ###os3列在这个时候已经被去掉了
        # print('Columns with all const values*******: \n' + str(cols_const) + '\n')

        ## Drop exclusive columns
        train_FD = train_FD.drop(columns=cols_const + cols_nan + sensor_drop)
        test_FD = test_FD.drop(columns=cols_const + cols_nan + sensor_drop)


        # 对数据进行Min—Max变换
        if self.preproc == True:
            ## preprocessing(normailization for the neural networks)
            min_max_scaler = preprocessing.MinMaxScaler()
            # min_max_scaler = preprocessing.StandardScaler()
            # for the training set
            # train_FD['cycles_norm'] = train_FD['cycles']
            cols_normalize = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL']) #求差集保留train_FD.columns

            norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_FD[cols_normalize]),
                                         columns=cols_normalize,
                                         index=train_FD.index)
            join_df = train_FD[train_FD.columns.difference(cols_normalize)].join(norm_train_df)
            train_FD = join_df.reindex(columns=train_FD.columns)

            # for the test set
            # test_FD['cycles_norm'] = test_FD['cycles']
            cols_normalize_test = test_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2'])
            # print ("cols_normalize_test", cols_normalize_test)
            norm_test_df = pd.DataFrame(min_max_scaler.transform(test_FD[cols_normalize_test]),
                                        columns=cols_normalize_test,
                                        index=test_FD.index)
            test_join_df = test_FD[test_FD.columns.difference(cols_normalize_test)].join(norm_test_df)
            test_FD = test_join_df.reindex(columns=test_FD.columns)
            test_FD = test_FD.reset_index(drop=True)
        else:
            print ("No preprocessing")


        # Specify the columns to be used
        sequence_cols_train = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])
        sequence_cols_test = test_FD.columns.difference(['unit_nr', 'os_1', 'os_2', 'cycles'])
        # print(train_FD.columns.tolist(),test_FD.columns.tolist(),sequence_cols_train,sequence_cols_test)

        ## generator for the sequences
        # transform each id of the train dataset in a sequence
        seq_gen = (list(gen_sequence(train_FD[train_FD['unit_nr'] == id], self.sequence_length, sequence_cols_train))
                   for id in train_FD['unit_nr'].unique())

        # generate sequences and convert to numpy array in training set
        seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)
        self.seq_array_train = seq_array_train.transpose(0, 2, 1)  # shape = (samples, sensors, sequences)
        # print("seq_array_train.shape", self.seq_array_train.shape)

        # generate label of training samples
        label_gen = [gen_labels(train_FD[train_FD['unit_nr'] == id], self.sequence_length, ['RUL'])
                     for id in train_FD['unit_nr'].unique()]
        self.label_array_train = np.concatenate(label_gen).astype(np.float32)
        #print("label_array_train.shape", self.label_array_train.shape)
        #print(self.label_array_train)
        # generate sequences and convert to numpy array in test set (only the last sequence for each engine in test set)
        seq_array_test_last = [test_FD[test_FD['unit_nr'] == id][sequence_cols_test].values[-self.sequence_length:]
                               for id in test_FD['unit_nr'].unique() if
                               len(test_FD[test_FD['unit_nr'] == id]) >= self.sequence_length]

        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
        self.seq_array_test_last = seq_array_test_last.transpose(0, 2, 1)  # shape = (samples, sensors, sequences)
        # print("seq_array_test_last.shape", self.seq_array_test_last.shape)
        # generate label of test samples
        #修改在这
        y_mask = [len(test_FD[test_FD['unit_nr'] == id]) >= sequence_length for id in test_FD['unit_nr'].unique()]

        mapper1 = {}
        for unit_nr in test_FD['unit_nr'].unique():
            mapper1[unit_nr] = test_FD['cycles'].loc[test_FD['unit_nr'] == unit_nr].max()
        mapper1_df = pd.DataFrame(list(mapper1.items()))
        # testmax=test_FD['unit_nr'].apply(lambda nr: mapper1[nr])
        # testmax=testmax.loc[testmax.shift(1)!=testmax]
        # #print(testmax[:50])
        testmax = test_FD['unit_nr'].drop_duplicates(keep='first')
        testmax = testmax.reset_index()
        testmax.loc[:, 'maxlen'] = 0
        testmax['maxlen'] = mapper1_df[1]
        
        label_array_test_last = RUL_FD['RUL_truth'][y_mask].values/(RUL_FD['RUL_truth'][y_mask].values+testmax['maxlen'][y_mask].values)
        self.label_array_test = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
        #print("label_array_test.shape", self.label_array_test.shape)
        #print(self.label_array_test.shape)
        #print(self.label_array_test)

        # print('\nseq_array_train.shape:{} \n''label_array_train.shape:{} \n''seq_array_train.shape:{} \n''label_array_test.shape:{}\n'.
        #       format(self.seq_array_train.shape, self.label_array_train.shape, self.seq_array_train.shape, self.label_array_test.shape))

    def input_seq(self): # batchsize * sensors * sequence
        train_samples = self.seq_array_train
        train_label = self.label_array_train
        test_samples = self.seq_array_test_last
        test_label = self.label_array_test

        return train_samples, train_label, test_samples, test_label



def load_seq(data_path_list, sequence_length, sensor_drop):
    sample_class = input_trans(data_path_list=data_path_list, sequence_length=sequence_length, sensor_drop=sensor_drop)
    #直接seq输入 exactor

    train_samples, train_label, test_samples, test_label = sample_class.input_seq()

    # print('\ntrain_samples.shape:{} \n''train_label.shape:{} \n''test_samples.shape:{} \n''test_label.shape:{}\n'.
    #     format(train_samples.shape, train_label.shape, test_samples.shape, test_label.shape))

    return train_samples, train_label, test_samples, test_label

class seq_Dataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, phase, data_path_list, sequence_length, sensor_drop):
        super(seq_Dataset, self).__init__()

        if phase == "train" :
            train_samples, train_label, test_samples, test_label = load_seq(data_path_list, sequence_length, sensor_drop)
            self.seqs, self.labels =  train_samples, train_label
        else:
            train_samples, train_label, test_samples, test_label = load_seq(data_path_list,sequence_length,sensor_drop)
            self.seqs, self.labels = test_samples, test_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq = seq.astype(np.float32)
        label = self.labels[index]
        label = float(label)
        return seq, label

def load_training(data_path_list, sequence_length, sensor_drop, batch_size,suffle = True):
    train_dataset = seq_Dataset("train", data_path_list, sequence_length, sensor_drop)
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=suffle )
    return loader  # torch.Size([700, 14, 50])


def load_testing(data_path_list, sequence_length, sensor_drop, batch_size,suffle = False):
    test_dataset = seq_Dataset("test", data_path_list, sequence_length, sensor_drop)
    loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0, shuffle=suffle)
    return loader  # torch.Size([700, 14, 50])


# import os
# if __name__ == "__main__":
#     batch_size = 700
#     phase = 'train'
#     sequence_length=50
#     sensor_drop=[]
#     visualize=0
#     flatten= False
#     thres_type = None
#     thres_percentage = 50
#     # Path
#     dataset_dir = os.getcwd()
#
#     ## Dataset path
#     train_FD001_path = dataset_dir + '/cmapss/train_FD001.csv'
#     test_FD001_path = dataset_dir + '/cmapss/test_FD001.csv'
#     RUL_FD001_path = dataset_dir + '/cmapss/RUL_FD001.txt'
#     FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]
#
#     train_FD002_path = dataset_dir + '/cmapss/train_FD002.csv'
#     test_FD002_path = dataset_dir + '/cmapss/test_FD002.csv'
#     RUL_FD002_path = dataset_dir + '/cmapss/RUL_FD002.txt'
#     FD002_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]
#
#     data1 = seq_Dataload(phase, FD001_path, sequence_length, sensor_drop, batch_size, thres_type, thres_percentage, flatten)
#     data2 = seq_Dataload(phase, FD002_path, sequence_length, sensor_drop, batch_size, thres_type, thres_percentage,
#                          flatten)
#     # for i in data:
#     #     # print(i)
#     #     print(i[0].shape)  # torch.Size([700, 14, 50])
#     #     break
#     print("data_loader1:", len(data1))
#     print("data_loader2:", len(data2))
#     test_data = (zip(data1, data2))
#     test_data = list(test_data)
#     n_test = len(test_data)
#     print("data_:", n_test)
#     # print("data[0]:", data[0])
#
