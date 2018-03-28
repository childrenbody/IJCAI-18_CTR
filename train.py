#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:39:36 2018

@author: childrenbody
"""
from tools import XGBModel, Models, DataClass
import pandas as pd       

quantity = 50

data = DataClass('data/idea10.csv')
data.feature_label(['is_trade'])
data.positive_negative()
data.random_group(quantity)

models = Models()
for i in range(quantity):
    subset = data.make_subset(i)
    model = XGBModel()
    model.make_train(subset[data.feature], subset[data.label])
    model.make_watch_list(subset[data.feature], subset[data.label], 0.3)
    model.train_model(100)
    models.get_model(model)
    print('model {} completed'.format(i))

result = models.predict(data.test)
result['score'] = result.mean(axis=1)

submission = pd.DataFrame({'instance_id': data.test.index, 'predicted_score': result.score})