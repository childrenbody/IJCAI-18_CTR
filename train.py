#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:39:36 2018

@author: childrenbody
"""
from tools import XGBModel, DataClass
import pandas as pd

data = DataClass('data/idea12.csv')
data.feature_label(['is_trade'])


train = data.train[(data.train.day < 24)&(data.train.day >= 22)]
test = data.train[data.train.day == 24]
submission = data.make_submission()
result = data.test[['instance_id']]

model = XGBModel()
model.make_train(data.train[data.feature], data.train[data.label])
model.make_watch_list(train[data.feature], train[data.label], test[data.feature], test[data.label])
model.fit(10)
result['predicted_score'] = model.predict(data.test[data.feature])

x_test , y_test = data.get_valid_data(24)
print('logloss: {}'.format(data.logloss(y_test, model.predict(x_test))))

submission = pd.merge(submission, result, how='left', on='instance_id')
submission.to_csv('result12.csv', index=False, sep=' ')

# =============================================================================
# # idea 1.1
# data = DataClass('data/idea11.csv')
# data.feature_label(['is_trade', 'user_id', 'instance_id'])
# submission = pd.read_csv('input/round1_ijcai_18_test_a_20180301.txt', sep=' ', usecols=['instance_id'])
# 
# train = data.train[(data.train.day < 24)&(data.train.day >= 22)]
# test = data.train[data.train.day == 24]
# result = data.test[['instance_id']]
# 
# model = XGBModel()
# model.make_train(data.train[data.feature], data.train[data.label])
# model.make_watch_list(train[data.feature], train[data.label], test[data.feature], test[data.label])
# model.fit(50)
# result['predicted_score'] = model.predict(data.test[data.feature])
# 
# submission = pd.merge(submission, result, how='left', on='instance_id')
# submission.to_csv('result11.csv', index=False, sep=' ')
# =============================================================================

# =============================================================================
# import lightgbm as lgb
# 
# data = DataClass('data/idea20.csv')
# 
# feature = [_ for _ in data.train.columns if _ not in ['user_id', 'is_trade', 'datetime', 'instance_id']]
# 
# # num_leaves=63, max_depth=7, n_estimators=80
# model = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80)
# model.fit(data.train[feature], data.train['is_trade'], categorical_feature=['user_gender_id', ])
# data.test['predicted_score'] = model.predict_proba(data.test[feature])[:, 1]
# 
# submission = pd.read_csv('input/round1_ijcai_18_test_a_20180301.txt', sep=' ', usecols=['instance_id'])
# submission = pd.merge(submission, data.test[['instance_id', 'predicted_score']], how='left', on='instance_id')
# submission.to_csv('result20.csv', index=False, sep=' ')
# =============================================================================

# =============================================================================
# # idea 1.0
# quantity = 50
# 
# data = DataClass('data/idea10.csv')
# data.feature_label(['is_trade'])
# data.positive_negative()
# data.random_group(quantity)
# 
# models = Models()
# for i in range(quantity):
#     subset = data.make_subset(i)
#     model = XGBModel()
#     model.make_train(subset[data.feature], subset[data.label])
#     model.make_watch_list(subset[data.feature], subset[data.label], 0.3)
#     model.train_model(10)
#     models.get_model(model)
#     print('model {} completed'.format(i))
# 
# x_test, y_test = data.get_valid_data(0.0385)
# val = models.predict(x_test)
# val.score = val.mean(axis=1)
# 
# print('logloss: {}'.format(data.logloss(y_test, val.score)))
# 
# result = models.predict(data.test[data.feature])
# result['score'] = result.mean(axis=1)
# 
# submission = pd.DataFrame({'instance_id': data.test.index, 'predicted_score': result.score})
# test = pd.read_csv('input/round1_ijcai_18_test_a_20180301.txt', sep=' ', usecols=['instance_id'])
# result = pd.merge(test, submission, on='instance_id')
# =============================================================================
