#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:56:17 2018

@author: childrenbody
"""
import pandas as pd
import gc, time

def convert_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

ad_feature = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
user_feature = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'user_id']
context_feature = ['context_page_id', 'predict_category_property', 'context_timestamp']
shop_feature = ['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
                'shop_score_service', 'shop_score_delivery', 'shop_score_description']

columns = ['instance_id']
columns = columns + ad_feature + user_feature + context_feature + shop_feature

train = pd.read_csv('input/round1_ijcai_18_train_20180301.txt', sep=' ', usecols=columns + ['is_trade'])
test = pd.read_csv('input/round1_ijcai_18_test_a_20180301.txt', sep=' ', usecols=columns)
data = pd.concat([train, test], axis=0)

del train, test
gc.collect()

data.instance_id = data.instance_id.astype(str)

# handing context feature
data['datetime'] = data.context_timestamp.apply(convert_datetime)
data['datetime'] = pd.to_datetime(data.datetime)
data['day'] = data.datetime.dt.day
data['hour'] = data.datetime.dt.hour
data.drop(['datetime', 'context_timestamp'], axis=1, inplace=True)

# handing ad feature
item_category = data.item_category_list.str.get_dummies(sep=';')
item_category.columns = ['item_category_list_' + _ for _ in item_category.columns]
data.drop(['item_category_list'], axis=1, inplace=True)
data = pd.concat([data, item_category], axis=1)

# handing user feature
user = data[user_feature]
user.user_gender_id = user.user_gender_id.astype(str)
user.user_occupation_id = user.user_occupation_id.astype(str)
user = pd.get_dummies(user)
data.drop(user_feature, axis=1, inplace=True)
data = pd.concat([data, user], axis=1)
user = data[data.is_trade.notnull()].groupby(['user_id', 'hour']).size().reset_index().rename(columns={0:'user_query_hour'})
data = pd.merge(data, user, how='left', on=['user_id', 'hour'])

# handing shop feature
data['count_category'] = data.predict_category_property.apply(lambda x: len(x.split(';')))
data.drop(['predict_category_property'], axis=1, inplace=True)

# other feature
# convert rate of hour
convert_rate = data[data.is_trade.notnull()].groupby(['hour'])['is_trade'].mean().reset_index().rename(columns={'is_trade':'convert_rate'})
data = pd.merge(data, convert_rate, how='left', on='hour')

del item_category, user
gc.collect()

data.to_csv('data/idea11.csv', index=False)

# =============================================================================
# # idea 1.0
# ad_feature = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
# user_feature = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
# context_feature = ['context_page_id', 'predict_category_property']
# shop_feature = ['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
#                 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
# 
# columns = ['instance_id']
# columns = columns + ad_feature + user_feature + context_feature + shop_feature
# 
# train = pd.read_csv('input/round1_ijcai_18_train_20180301.txt', sep=' ', usecols=columns + ['is_trade'])
# test = pd.read_csv('input/round1_ijcai_18_test_a_20180301.txt', sep=' ', usecols=columns)
# data = pd.concat([train, test], axis=0)
# 
# del train, test
# gc.collect()
# 
# data.instance_id = data.instance_id.astype(str)
# 
# # handing ad feature
# item_category = data.item_category_list.str.get_dummies(sep=';')
# item_category.columns = ['item_category_list_' + _ for _ in item_category.columns]
# data.drop(['item_category_list'], axis=1, inplace=True)
# data = pd.concat([data, item_category], axis=1)
# 
# # handing user feature
# user = data[user_feature]
# user.user_gender_id = user.user_gender_id.astype(str)
# user.user_occupation_id = user.user_occupation_id.astype(str)
# user = pd.get_dummies(user)
# data.drop(user_feature, axis=1, inplace=True)
# data = pd.concat([data, user], axis=1)
# 
# # handing shop feature
# data['count_category'] = data.predict_category_property.apply(lambda x: len(x.split(';')))
# data.drop(['predict_category_property'], axis=1, inplace=True)
# 
# del item_category, user
# gc.collect()
# 
# data.to_csv('data/idea10.csv', index=False)
# 
# =============================================================================
