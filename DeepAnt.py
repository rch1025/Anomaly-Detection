## 경로에 맞는 데이터를 불러오고 결측치 대체와 같은 1차 전처리 수행
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os



def read_modulate_data(data_file):
    """
        Data ingestion : Function to read and formulate the data
    """
    data = pd.read_csv(data_file)
    data.fillna(data.mean(), inplace=True)
    df = data.copy()
    data.set_index("LOCAL_DATE", inplace=True)
    data.index = pd.to_datetime(data.index)
    return data, df # df와 data의 차이는 index가 LOCAL_DATE라는 것 말고는 없다.

def data_pre_processing(df):
    """
        Data pre-processing : Function to create data for Model
    """
    try:
        scaled_data = MinMaxScaler(feature_range = (0, 1))
        data_scaled_ = scaled_data.fit_transform(df)
        df.loc[:,:] = data_scaled_
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,LOOKBACK_SIZE,df.shape[1]))
        Y = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,df.shape[1]))
        timesteps = []
        ## LOOKBACK_SIZE가 10이기에 0~9를 X로 두고 10번 째 값(_data_[9+1])이 첫 번째 Y값이 된다.
        for i in range(LOOKBACK_SIZE-1, df.shape[0]-1):
            timesteps.append(df.index[i])
            # print(f'############## {i}')
            # print('y-step :', i-LOOKBACK_SIZE+1)
            # print(_data_[i+1])
            # print()
            Y[i-LOOKBACK_SIZE+1] = _data_[i+1] # 첫 번째 i=9

            ## LOOKBACK_SIZE만큼 x값을 채운다
            for j in range(i-LOOKBACK_SIZE+1, i+1):
                # print(f'##### {j}')
                # print('y-step :', i-LOOKBACK_SIZE+1)
                # print('x-step :', LOOKBACK_SIZE-1-i+j)
                # print(_data_[j])
                # 첫 배치부터 window size만큼의 값을 채우기 시작한다.
                X[i-LOOKBACK_SIZE+1][LOOKBACK_SIZE-1-i+j] = _data_[j] # X[0][0] = 26으로 변수의 개수
        return X,Y,timesteps
    except Exception as e:
        print("Error while performing data pre-processing : {0}".format(e))
        return None, None, None


"""DeepAnt"""
## 1D-CNN에서 channel은 feature를 의미
class DeepAnT(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """
    def __init__(self, LOOKBACK_SIZE, DIMENSION):
        super(DeepAnT, self).__init__()
        # Conv1d layer
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=DIMENSION, out_channels=32, padding =1, kernel_size=3)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        # Conv1d layer
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=32, out_channels=64, padding =1, kernel_size=3)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        # Linear first
        self.dense_1_layer = torch.nn.Linear(128, 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        # Linear second
        self.dense_2_layer = torch.nn.Linear(40, DIMENSION)
        
    def forward(self, x):
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.dense_2_layer(x)

"""훈련 함수"""
def make_train_step(model, loss_fn, optimizer):
    """
        Computation : Function to make batch size data iterator
    """
    def train_step(x, y):
        model.train()
        yhat = model(x) # 모델 예측값
        loss = loss_fn(y, yhat) # 손실함수 적용
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step ## 함수를 반환

"""loss 계산 및 최적화"""
def compute(X,Y):
    """
        Computation : Find Anomaly using model based computation 
    """
    if str(MODEL_SELECTED) == "lstmae":
        ## LSTMAE 부분은 생략
        # model = LSTMAE(10,26)
        # criterion = torch.nn.MSELoss(reduction='mean')
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(X.astype(np.float32)))
        # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        # train_step = make_train_step(model, criterion, optimizer)
        # for epoch in range(30):
        #     loss_sum = 0.0
        #     ctr = 0
        #     for x_batch, y_batch in train_loader:
        #         loss_train = train_step(x_batch, y_batch)
        #         loss_sum += loss_train
        #         ctr += 1
        #     print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
        # hypothesis = model(torch.tensor(X.astype(np.float32))).detach().numpy()
        # loss = np.linalg.norm(hypothesis - X, axis=(1,2))
        # return loss.reshape(len(loss),1)
        return None
    elif str(MODEL_SELECTED) == "deepant":
        model = DeepAnT(LOOKBACK_SIZE = 10,DIMENSION = 26)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
        # transpose로 차원 바꾸기
            # [batch_size, window, feature] -> [batch_size, feature, window]로 바꾸기
        train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)).transpose(1,2), torch.tensor(Y.astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        train_step = make_train_step(model, criterion, optimizer) # train_step은 함수를 의미한다.
        for epoch in range(30):
            loss_sum = 0.0
            ctr = 0
            ## batch별 최적화 적용
            for x_batch, y_batch in train_loader:
                loss_train = train_step(x_batch, y_batch) # loss 값 반환
                loss_sum += loss_train # 총 loss 값
                ctr += 1
            print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
        ## 가설: [예측값-실제값]에 대해서 맨허튼 거리를 계산한다.
        hypothesis = model(torch.tensor(X.astype(np.float32)).transpose(1,2)).detach().numpy()
        loss = np.linalg.norm(hypothesis - Y, axis=1) # norm 구하기 (L1 norm : 각 성분의 절대값 더히기)
        return loss.reshape(len(loss),1)
    else:
        print("Selection of Model is not in the set")
        return None