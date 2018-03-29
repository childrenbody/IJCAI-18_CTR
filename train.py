#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:39:36 2018

@author: childrenbody
"""
from tools import DataClass
import lightgbm as lgb
import pandas as pd

data = DataClass('data/idea20.csv')

feature = [_ for _ in data.train.columns if _ not in ['user_id', 'is_trade', 'datetime', 'instance_id']]

# num_leaves=63, max_depth=7, n_estimators=80
model = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80)
model.fit(data.train[feature], data.train['is_trade'], categorical_feature=['user_gender_id', ])
data.test['predicted_score'] = model.predict_proba(data.test[feature])[:, 1]

submission = pd.read_csv('input/round1_ijcai_18_test_a_20180301.txt', sep=' ', usecols=['instance_id'])
submission = pd.merge(submission, data.test[['instance_id', 'predicted_score']], how='left', on='instance_id')
submission.to_csv('result20.csv', index=False, sep=' ')



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
#     model.train_model(100)
#     models.get_model(model)
#     print('model {} completed'.format(i))
# 
# result = models.predict(data.test)
# result['score'] = result.mean(axis=1)
# 
# submission = pd.DataFrame({'instance_id': data.test.index, 'predicted_score': result.score})
# =============================================================================
