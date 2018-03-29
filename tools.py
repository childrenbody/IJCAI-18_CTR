#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:55:20 2018

@author: childrenbody
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import copy, time

class XGBModel:
    def __init__(self):
        self.param = {'objective': 'binary:logistic',
                      'eval_metric': 'logloss'
                      }
        
    def make_train(self, train, label):
        self.xgbtrain = xgb.DMatrix(train, label)
        
    def make_watch_list(self, x_train, y_train, x_test, y_test):
        watch_train = xgb.DMatrix(x_train, y_train)
        watch_test = xgb.DMatrix(x_test, y_test)
        self.watch_list = [(watch_train, 'train'), (watch_test, 'val')]
        
    def fit(self, iters):
        self.model = xgb.train(self.param, self.xgbtrain, iters, self.watch_list)
        
    def predict(self, test):
        xgbtest = xgb.DMatrix(test)
        return self.model.predict(xgbtest)
    
class Models:
    def __init__(self):
        self.models = []
        
    def get_model(self, model):
        self.models.append(model)
        
    def predict(self, data):
        res = pd.DataFrame()
        for i in range(len(self.models)):
            res[i] = self.models[i].predict(data)
        return res
        
class DataClass:
    def __init__(self, file_path):
        self.data_split(file_path)
        
    def data_split(self, file_path):
        data = pd.read_csv(file_path)
        self.train = data[data.is_trade.notnull()]
        self.test = data[data.is_trade.isnull()]
        
    def feature_label(self, label):
        self.label = ['is_trade']
        self.feature = [_ for _ in self.train.columns if _ not in label]
        
    def positive_negative(self):
        self.positive = self.train[self.train.is_trade == 1].instance_id
        self.negative = self.train[self.train.is_trade == 0].instance_id.tolist()
        
    def random_group(self, quantity):
        negative = copy.copy(self.negative)
        np.random.shuffle(negative)
        self.negative_list = [0] * quantity
        end = len(self.negative)
        for start in range(quantity):
            temp = list(range(start, end, quantity))
            self.negative_list[start] = [negative[_] for _ in temp]
        
    def get_valid_data(self, test_size):
        x_train, x_test, y_train, y_test = train_test_split(self.train[self.feature], self.train[self.label], test_size=0.3)
        return x_test, y_test
        
    def logloss(self, y_true, y_pred):
        return np.sum(log_loss(y_true, y_pred)) / len(y_true)
    
    def make_subset(self, index):
        return pd.concat([self.train.loc[self.positive, :], self.train.loc[self.negative_list[index], :]], axis=0)
        
    @staticmethod
    def convert_timestamp(x): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))
        
    def user_query_day(self):
        query_day = self.train.groupby(['user_id', 'day']).size().reset_index().rename(columns={0:'user_query_day'})
        return query_day
        
        