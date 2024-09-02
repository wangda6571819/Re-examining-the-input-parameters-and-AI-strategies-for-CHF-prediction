import pandas as pd
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
import os
import torch
from jamba.model import Jamba,JambaBlock
from zeta.nn import OutputHead

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 文件抬头
calIndex = 400
# epochsCount 学习次数
epochsCount = 500



data_file_path = '../data/slice_real_value (copy).xlsx'
input_path = '../input/'
public_sheet_name = 'chf_public_LUT'

def set_seed(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def readDataIfExist(name) :
    pkl_name = input_path + name + '.pkl'

    exist = os.path.exists(pkl_name)
    if exist :
        data  = pd.read_pickle(pkl_name)
        return data

    # 读取xlsx 获取数据
    data = pd.read_excel(data_file_path, sheet_name=name, engine='openpyxl')
    data.to_pickle(pkl_name) #格式另存
    return data

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
file_path = '../data/train_set_6.csv'
data = pd.read_csv(file_path)


# Define the KFold cross-validator
folds = custom__kfold(data)

# Define the feature columns and the label column
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']
# feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
# 管径、加热长度、压力、质量流量、出口质量
feature_columns = ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling']
label_column = 'CHF'

input_size = len(feature_columns)
output_size = 1


# 定义Mamba模型
class JambaModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(JambaModel, self).__init__()

        self.linear_in = nn.Linear(input_features, d_model)

        # dim=512,
        # depth=6 -> num_encoder_layers 2,
        # num_tokens=100,
        # d_state=256 ->64,
        # d_conv=128,
        # heads=8,
        # num_experts=8,
        # num_experts_per_token=2

        self.jamba_layer = nn.ModuleList(
            [
                JambaBlock(
                    d_model,
                    64,
                    128,
                    nhead,
                    8,
                    2,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.linear_f2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 减小到 256 -> 1
        self.linear_mid2 = nn.Linear(d_model, d_model)
        self.linear_mid3 = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, output_size)

    def forward(self, src):
        # print(src.size())
        src = self.linear_in(src)
        # Apply the layers
        for layer in self.jamba_layer:
            src = layer(src)
        output = src
        # 通过额外的线性层和激活函数
        # output = F.gelu(self.linear_f1(output))
        output = F.softsign(self.linear_f2(output))
        output = self.dropout(output)
        output = F.softsign(self.linear_mid2(output))
        output = F.softsign(self.linear_mid3(output))
        output = self.linear_out(output)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear_in = nn.Linear(input_size, 16)

        self.lstm = nn.LSTM(16, hidden_size, num_layers, batch_first=True)

        self.linear_f2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # 减小到 256 -> 1
        self.linear_mid2 = nn.Linear(d_model, d_model)
        self.linear_mid3 = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, output_size)

    def forward(self, src):

        src = self.linear_in(src)

        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, src.size(0), self.hidden_size)
        
        output, _ = self.lstm(src, (h0, c0))

        output = F.softsign(self.linear_f2(output))
        output = self.dropout(output)
        output = F.softsign(self.linear_mid2(output))
        output = F.softsign(self.linear_mid3(output))
        output = self.linear_out(output)
        return output



def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    model.to(device)
    patience = 1000
    patience_counter = 0  # 初始化早停计数器
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=150,verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.96)

    for epoch in range(epochs):

        start_time = time.time()  # 记录程序开始时间
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            # for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x = batch_x.unsqueeze(0)  # 适合任务的序列长度

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            train_loss += loss.item() * batch_x.size(0)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                batch_x = batch_x.unsqueeze(0)  # 适合任务的序列长度

                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)


        total_loss = train_loss + val_loss
        # 更新学习率
        scheduler.step(total_loss)
        # Early Stopping check
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patience_counter = 0  # 重置早停计数器
            # Optional: Save best model
            torch.save(model.state_dict(), f'../model/Mamba_best_model_{calIndex}.pth')
            print(f'Best Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}')
        else:
            patience_counter += 1  # 增加早停计数器
            if patience_counter >= patience:
                print("Early stopping triggered")
                break  # 达到容忍周期，停止训练

        end_time = time.time()  # 记录程序结束时间
        # 计算程序执行的总时间
        elapsed_time = end_time - start_time  
        # 打印损失
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}  time:{elapsed_time} s')

    return train_losses, val_losses, best_val_loss



set_seed(42)
# Create train and test sets

# X_train_fold_1, y_train_fold_1, X_val_fold_1, y_val_fold_1 = create_train_test_sets(data, folds[0][0], folds[0][1], feature_columns, label_column)
# X_train_fold_2, y_train_fold_2, X_val_fold_2, y_val_fold_2 = create_train_test_sets(data, folds[1][0], folds[1][1], feature_columns, label_column)
# X_train_fold_3, y_train_fold_3, X_val_fold_3, y_val_fold_3 = create_train_test_sets(data, folds[2][0], folds[2][1], feature_columns, label_column)
# X_train_fold_4, y_train_fold_4, X_val_fold_4, y_val_fold_4 = create_train_test_sets(data, folds[3][0], folds[3][1], feature_columns, label_column)
# X_train_fold_5, y_train_fold_5, X_val_fold_5, y_val_fold_5 = create_train_test_sets(data, folds[4][0], folds[4][1], feature_columns, label_column)

X_train, y_train, X_val, y_val = create_train_test_sets(data, folds[1][0], folds[1][1], feature_columns,label_column)

# Convert DataFrame/Series to numpy array before converting to Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) if isinstance(X_train,pd.DataFrame) else torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) if isinstance(y_train,pd.Series) else torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32) if isinstance(X_val, pd.DataFrame) else torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)

# Define data loaders for PyTorch
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

# Model hyperparameters setup
# 超参数的搜索范围
d_model = 64
nhead = 8
num_encoder_layers = 6
dim_feedforward = 4096
dropout = 0.01
last_batch_size = batch_size # 最后一个批次大小
current_batch_size = batch_size
different_batch_size = False
# torch.nn.SmoothL1Loss()平滑L1损失
criterion = torch.nn.SmoothL1Loss()

# model = TransformerModel(input_features=input_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
# model = JambaModel(input_features=input_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
model = LSTM(input_size = input_size, hidden_size = d_model, num_layers = num_encoder_layers, output_size = 1)



optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)

# Train and evaluate the model
train_losses, val_losses, best_val_loss = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=epochsCount)

print(best_val_loss)

#
# 保存训练和验证损失
with open(f'../model/lstm_train_losses_{calIndex}.txt', 'w') as f:
    for loss in train_losses:
        f.write(f"{loss}\n")

with open(f'../model/lstm_val_losses_{calIndex}.txt', 'w') as f:
    for loss in val_losses:
        f.write(f"{loss}\n")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Losses Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('../model/lstm_loss_plot_{calIndex}.png')
plt.show()



