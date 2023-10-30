import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler


Dataset_Root = "data_div"

def get_dataset_path(dataset_name: str):
    path = os.path.join(Dataset_Root, dataset_name)    
    return path

def sample_dataframe(df_all, number):
    number = min(number, df_all.shape[0]) # 不够number了就全取出来
    df_all_sample = df_all.sample(number)
    df_all_remain = df_all.drop(df_all_sample.index, axis=0)
    return df_all_sample, df_all_remain


def sample_10_train_test_split(X_all_01, y):
    XY_all_01 = pd.concat([X_all_01, y], axis=1)

    XY_all_01.reset_index()
    train_content = pd.DataFrame()
    for index in range(5):
        index_content = XY_all_01.loc[XY_all_01['Activity'] == index]
        index_content_sample = index_content.sample(10)
        train_content = train_content.append(index_content_sample)
    test_content = XY_all_01.drop(train_content.index, axis=0)

    y_train2, y_test2 = train_content.pop('Activity'), test_content.pop('Activity')
    X_train2, X_test2 = train_content, test_content

    return X_train2, X_test2, y_train2, y_test2
    

class MyDataSet(Dataset):
    def __init__(self, X, Y):
        self.sample_list_x = torch.tensor(np.array(X), dtype=torch.float32)
        self.sample_list_y = torch.tensor(np.array(Y), dtype=torch.int64)
        self.x_size = self.sample_list_x[0].size()

    def __getitem__(self, index):
        x = torch.reshape(self.sample_list_x[index], (1, self.x_size[0]))
        y = self.sample_list_y[index]
        return x, y

    def __len__(self):
        return len(self.sample_list_y)


def apt_MinMaxScaler(data, pop_lable=None): 
    # pop_lable设为列名时，先去掉这列，归一化后再加上
    if(pop_lable):
        y = data.pop(pop_lable)
    
    scaler = MinMaxScaler()
    X_all_01 = scaler.fit_transform(data)
    X_all_01 = pd.DataFrame(X_all_01, columns=data.columns, index=data.index)
    
    if(pop_lable):
        return pd.concat([X_all_01, y], axis=1)
    else:
        return X_all_01


def label_astype(data, original_column, new_column):
    value = data[original_column].astype('category')
    data[new_column] = value.cat.codes
    data_drop = data.drop(labels=[original_column], axis=1)
    return data_drop


def load_DAPT_2020(client_id: int, clients_4_training_num: int, dataset_root: str):

    if client_id < clients_4_training_num:
        csv_path = os.path.join(dataset_root, "train", str(client_id) + ".csv")
        data = pd.read_csv(csv_path, index_col=0)
        data = label_astype(data, 'Activity', 'Label')
        Label = data.pop('Label')
    else:
        csv_path = os.path.join(dataset_root, "test", str(client_id) + ".csv")
        data = pd.read_csv(csv_path, index_col=0)
        Label = data.pop('Stage')
    
    data_01 = apt_MinMaxScaler(data)
    X_train, X_test, y_train, y_test = train_test_split(data_01, Label, train_size=0.67, stratify=Label,)  # random_state = 0) #shuffle=False

    if y_test.size > 25: # 改数据集 后检查这里
        from imblearn.over_sampling import SMOTE
        smote=SMOTE(sampling_strategy={4:350})
        X_train, y_train = smote.fit_resample(X_train, y_train)   # 插值平衡第5类

    return X_train, X_test, y_train, y_test


def load_CICIDS2017(client_id: int, clients_4_training_num: int, args_dataset_root: str):
    if client_id < clients_4_training_num:
        csv_path = os.path.join(args_dataset_root, "train", str(client_id) + ".csv")
        data = pd.read_csv(csv_path, index_col=0) 
        data = label_astype(data, 'Activity', 'Label')
        Label = data.pop('Label')
        data_01 = apt_MinMaxScaler(data)    
        X_train2, X_test2, y_train2, y_test2 = train_test_split(data_01, Label, train_size = 0.67, stratify=Label,) #random_state = 0) #shuffle=False
    
    else:
        csv_path = os.path.join(args_dataset_root, "test", str(client_id) + ".csv")
        data = pd.read_csv(csv_path, index_col=0)
        data = label_astype(data, 'Activity', 'Label')
        data_01 = apt_MinMaxScaler(data, pop_lable='Label')

        data_01.reset_index()
        train_content = pd.DataFrame()
        for index in range(5):
            index_content = data_01.loc[data_01['Label'] == index]
            index_content_sample = index_content.sample(10)
            train_content = train_content.append(index_content_sample)
        test_content = data_01.drop(train_content.index, axis=0)

        y_train2, y_test2 = train_content.pop('Label'), test_content.pop('Label')
        X_train2, X_test2 = train_content, test_content

    return X_train2, X_test2, y_train2, y_test2


# def get_dataloader(dataset: str, client_id: int, batch_size=20, valset_ratio=0.1):
def get_dataloader(client_id: int, clients_4_training_num: int, dataset_root: str):

    if(dataset_root.find('dapt2020') != -1):
        X_train, X_test, y_train, y_test = load_DAPT_2020(client_id, clients_4_training_num, dataset_root)
    elif(dataset_root.find('CICIDS2017') != -1):
        X_train, X_test, y_train, y_test = load_CICIDS2017(client_id, clients_4_training_num, dataset_root)
    else:
        print("No dataset name found in dataset_root (dapt2020/CICIDS2017)")

    dataset_train = MyDataSet(X_train, y_train)
    dataset_val = MyDataSet(X_test, y_test)
    
    data_loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=y_test.size, shuffle=False) # 保证一次取完，eval函数的f1分数代码里没写多轮的计算

    return data_loader_train, data_loader_val
