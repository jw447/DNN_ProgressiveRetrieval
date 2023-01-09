# -*- coding: utf-8 -*-
"""DNN_MGARD
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import *

import copy
from sklearn import preprocessing

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2

import seaborn as sns
sns.set_style("whitegrid")

def preprocess(dname, testing_ratio, level_idx, shuffle):
    """
    testing ratio = 0.5, shuffle=False - for splitting the training and testing set by half;
    testing ratio = 0.01, shuffle=True - to shuffle the training dataset;
    testing_ratio = 0.99, shuffle=False - for inference;
    target: bp_0, bp_1, bp_2, bp_3, bp_4, all
    """
    df_features = pd.read_csv(dname)
    num_record = len(df_features)

    # split training and testing set
    if level_idx == 0:
        df_data_feature = df_features[["minvalue", "maxvalue", "avgvalue", "maxerror"]].copy()
        df_data_target = df_features[["bp_0"]].copy()
    elif level_idx == 1:
        df_data_feature = df_features[["minvalue", "maxvalue", "avgvalue", "maxerror", "bp_0"]].copy()
        df_data_target = df_features[["bp_1"]].copy()
    elif level_idx == 2:
        df_data_feature = df_features[["minvalue", "maxvalue", "avgvalue", "maxerror", "bp_0", "bp_1"]].copy()
        df_data_target = df_features[["bp_2"]].copy()
    elif level_idx == 3:
        df_data_feature = df_features[["minvalue", "maxvalue", "avgvalue", "maxerror", "bp_0", "bp_1", "bp_2"]].copy()
        df_data_target = df_features[["bp_3"]].copy()
    elif level_idx == 4:
        df_data_feature = df_features[["minvalue", "maxvalue", "avgvalue", "maxerror", "bp_0", "bp_1", "bp_2", "bp_3"]].copy()
        df_data_target = df_features[["bp_4"]].copy()
    else:
        print("error config in level_idx. Please check again.")
        return -1
    
    df_data_feature  = np.array(df_data_feature.astype("float32"))
    df_data_target   = np.array(df_data_target.astype("float32"))

    df_training_feature = df_data_feature
    df_training_target = df_data_target

    X_train, X_test, Y_train, Y_test = train_test_split(df_training_feature, df_training_target, test_size=testing_ratio, 
                                                        shuffle=shuffle, random_state=level_idx+233)
    
    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(X_train)
    normalized_test_X  = normalizer.transform(X_test)

    return normalized_train_X, normalized_test_X, Y_train, Y_test

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.deep0 = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

        self.deep1 = torch.nn.Sequential(
            torch.nn.Linear(5, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

        self.deep2 = torch.nn.Sequential(
            torch.nn.Linear(6, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

        self.deep3 = torch.nn.Sequential(
            torch.nn.Linear(7, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

        self.deep4 = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

        self.wide0 = torch.nn.Sequential(
            torch.nn.Linear(4, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1),
        )

        self.wide1 = torch.nn.Sequential(
            torch.nn.Linear(5, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1),
        )

        self.wide2 = torch.nn.Sequential(
            torch.nn.Linear(6, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1),
        )

        self.wide3 = torch.nn.Sequential(
            torch.nn.Linear(7, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1),
        )

        self.wide4 = torch.nn.Sequential(
            torch.nn.Linear(8, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1),
        )
        
    def forward(self, x):
        output = self.main(x)
        output = torch.sigmoid(output)*32
        return output

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    
    def forward(self, prediction, target):
        loss1 = torch.nn.MSELoss()(prediction, target)
        # loss2 = torch.max(torch.abs(prediction - target))
        huber_loss = torch.nn.functional.huber_loss(prediction, target, reduction='mean', delta=1)
        return huber_loss

def train(X_train, Y_train, level_idx, num_epoch, learning_rate):

    x_train = Variable(torch.from_numpy(X_train))
    y_train = Variable(torch.from_numpy(Y_train))

    if level_idx == 0:
        net = MLP().deep0
    elif level_idx == 1:
        net = MLP().deep1
    elif level_idx == 2:
        net = MLP().deep2
    elif level_idx == 3:
        net = MLP().deep3
    elif level_idx == 4:
        net = MLP().deep4
        
    print(net)
    #net = net.to("cuda")

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = MyLoss()

    BATCH_SIZE = 256
    EPOCH = num_epoch

    torch_dataset = Data.TensorDataset(x_train, y_train)

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=1,)
    
    # start training
    print("Training model for level:{}".format(level_idx))
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            #b_x = b_x.to("cuda")
            #b_y = b_y.to("cuda")
            prediction = net(b_x)                 # input x and predict based on x
            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()                 # clear gradients for next train
            loss.backward()                       # backpropagation, compute gradients
            optimizer.step()                      # apply gradients
            
        if epoch % 100 == 0:
            print(loss.to("cpu").detach().numpy())
    return net

def inference(feature, net, target):
    feature = Variable(torch.from_numpy(feature))
    target = Variable(torch.from_numpy(target))

    #feature = feature.to("cuda")
    #target = target.to("cuda")
    prediction = net(feature)
    return target.to("cpu").detach().numpy(), prediction.to("cpu").detach().numpy()

def validate_var(dname, net0, net1, net2, net3, net4):
        X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.9, level_idx=0, shuffle=False)
        target0, prediction0 = inference(X_test, net0, Y_test)

        X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.9, level_idx=1, shuffle=False)
        target1, prediction1 = inference(X_test, net1, Y_test)

        X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.9, level_idx=2, shuffle=False)
        target2, prediction2 = inference(X_test, net2, Y_test)

        X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.9, level_idx=3, shuffle=False)
        target3, prediction3 = inference(X_test, net3, Y_test)

        X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.9, level_idx=4, shuffle=False)
        target4, prediction4 = inference(X_test, net4, Y_test)

if __name__ == "__main__":
    dname = "expr_data/mgard_WarpX_laser64_Jx_0_512.csv"
    
    X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.5, level_idx=0, shuffle=False)
    net0 = train(X_train, Y_train, level_idx=0, num_epoch=500, learning_rate=0.00005)

    X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.5, level_idx=1, shuffle=False)
    net1 = train(X_train, Y_train, level_idx=1, num_epoch=500, learning_rate=0.00005)
    
    X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.5, level_idx=2, shuffle=False)
    net2 = train(X_train, Y_train, level_idx=2, num_epoch=500, learning_rate=0.00005)
    
    X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.5, level_idx=3, shuffle=False)
    net3 = train(X_train, Y_train, level_idx=3, num_epoch=500, learning_rate=0.00005)
    
    X_train, X_test, Y_train, Y_test = preprocess(dname=dname, testing_ratio=0.5, level_idx=4, shuffle=False)
    net4 = train(X_train, Y_train, level_idx=4, num_epoch=500, learning_rate=0.00005)

    #validate_var("expr_data/mgard_WarpX_laser64_Bx_0_512.csv", net0, net1, net2, net3, net4)
    #validate_var("expr_data/mgard_WarpX_laser64_Ex_0_512.csv", net0, net1, net2, net3, net4)

    #validate_var("expr_data/mgard_WarpX_laser64_Bx_0_512.csv", net0.to("cuda"), net1.to("cuda"), net2.to("cuda"), net3.to("cuda"), net4.to("cuda"))
    #validate_var("expr_data/mgard_WarpX_laser64_Ex_0_512.csv", net0.to("cuda"), net1.to("cuda"), net2.to("cuda"), net3.to("cuda"), net4.to("cuda"))
    
