import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from joblib import Parallel, delayed
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from einops import rearrange
import torch
import joblib


# 文件抬头
calIndex = 500

model_name = 'randomforest'

def set_seed(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

deviceString = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceString)


# Function to create train and test sets for features and labels from given indices
def create_train_test_sets(data, train_indices, test_indices, feature_cols, label_col):
    X_train = data.iloc[train_indices][feature_cols]
    y_train = data.iloc[train_indices][label_col]
    X_test = data.iloc[test_indices][feature_cols]
    y_test = data.iloc[test_indices][label_col]
    return X_train, y_train, X_test, y_test

# 重新定义自定义分割策略，以获得整体的训练集和测试集
def custom__kfold(data, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    grouped_data = data.groupby('Reference ID')

    # 初始化每个fold的训练集和测试集索引列表
    fold_indices = [[] for _ in range(n_splits)]

    # 对每个组分别应用KFold
    for _, group in grouped_data:
        if len(group) < n_splits:
            # 如果组的样本数小于n_splits，将所有样本放入每个fold的训练集
            for fold_idx in range(n_splits):
                train_indices = group.index
                fold_indices[fold_idx].append((train_indices, np.array([])))
        else:
            # 如果组的样本数足够，正常应用KFold
            for fold_idx, (train_index, test_index) in enumerate(kf.split(group)):
                original_train_index = group.iloc[train_index].index
                original_test_index = group.iloc[test_index].index
                fold_indices[fold_idx].append((original_train_index, original_test_index))

    # 合并每个fold的索引
    final_fold_indices = []
    for fold in fold_indices:
        train_indices = np.concatenate([train_idx for train_idx, _ in fold])
        test_indices = np.concatenate([test_idx for _, test_idx in fold if len(test_idx) > 0])
        final_fold_indices.append((train_indices, test_indices))

    return final_fold_indices

# Load the dataset
file_path = '../data/train_set_8.csv'
data = pd.read_csv(file_path)

# Define the KFold cross-validator
folds = custom__kfold(data)


# 随机森林模型的类
class RandomForestModule(nn.Module):
    def __init__(self, n_estimators=100, random_state=42,batch_size = 512):
        super(RandomForestModule, self).__init__()

        self.batch_size = batch_size
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, src, real):
            # 将 src 转换为二维数组，形状为 [batch_size, num_features]
            src_reshaped = src.view(src.size(1), -1).numpy()  # [512, 6]
            real = real.numpy()  # 转换为 numpy 数组
            self.model.fit(src_reshaped, real)

    def forward(self, src):
        # 将 src 转换为二维数组，形状为 [batch_size, num_features]
        src_reshaped = src.view(src.size(1), -1).numpy()  # [512, 6]
        # 使用 sklearn 的随机森林模型进行预测

        prediction = self.model.predict(src_reshaped)
        return torch.tensor(prediction, dtype=torch.float)
 

def train_and_evaluate(model, train_loader, val_loader, criterion, epochs, device, callIndex, model_name):
    best_val_loss = float('inf')
    model.to(device)
    patience = epochs
    patience_counter = 0  # 初始化早停计数器
    train_losses = []
    val_losses = []
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=150,verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.96)
    epochs = 1
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            # for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.unsqueeze(0)  # 适合任务的序列长度
            model.fit(batch_x,batch_y)

        for batch_x, batch_y in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{epochs}", leave=False):
            # for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.unsqueeze(0)  # 适合任务的序列长度
            model.fit(batch_x,batch_y)

        joblib.dump(model, f'../model/{model_name}_best_model_{callIndex}.pkl')
    return '', '', '', model


def doTrain(feature_columns = ['Tube Diameter' ,'Heated Length', 'Pressure' ,'Mass Flux', 'Inlet Subcooling'] , model_name = 'randomforest',callIndex='500') :
    
    print(f'开始训练 callIndex : {callIndex} feature_columns: {feature_columns}')
    # Load the dataset
    file_path = '../data/train_set_8.csv'

    # Model hyperparameters setup
    # 超参数的搜索范围
    batch_size = 512
    output_size = 1
    n_estimators = 100
    random_state = 42

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(file_path)

    # Define the KFold cross-validator
    folds = custom__kfold(data)
    label_column = 'CHF'
    input_size = len(feature_columns)
    dropout = 0.01


    X_train, y_train, X_val, y_val = create_train_test_sets(data, folds[1][0], folds[1][1], feature_columns,label_column)

    # Convert DataFrame/Series to numpy array before converting to Tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) if isinstance(X_train,pd.DataFrame) else torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) if isinstance(y_train,pd.Series) else torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32) if isinstance(X_val, pd.DataFrame) else torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)

    # Define data loaders for PyTorch
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.SmoothL1Loss()
    model = RandomForestModule(n_estimators=n_estimators, random_state=random_state, batch_size = batch_size)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)

    # Train and evaluate the model
    train_losses, val_losses, best_val_loss, model = train_and_evaluate(model, train_loader, val_loader, criterion, epochs=epochsCount, device= device, callIndex=callIndex,model_name = model_name)

    return train_losses, val_losses, best_val_loss, model


if __name__ == "__main__":
    print("测试 rwky")

    epochsCount = 10

    # 300 6-1
    # Tube Diameter, Heated Length, Enthalpy value of water, Enthalpy value of air , Mass Flux, Inlet Subcooling
    doTrain(feature_columns = ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], callIndex = '500')



