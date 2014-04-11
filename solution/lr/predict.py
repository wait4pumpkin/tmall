#!/usr/bin/env python
# -*- coding: utf-8 -*-  
import csv
import random
import glob
import os
import sys
import time
import math
import Queue

from collections import Counter

import numpy
import pylab as pl
import matplotlib.font_manager
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import lil_matrix

from sklearn import svm
from sklearn import linear_model, cross_validation, datasets, metrics
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import convolve
from sklearn.cross_validation import train_test_split


N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4
CLICK_TAG = 0
BUY_TAG = 1
FAVOR_TAG = 2
CART_TAG = 3

class User(object):
    def __init__(self, id, info):
        self.id = id;
        self.info = info;
        self.brands = info.keys()
        self.data = dict()
        for brand_id in self.brands:
            brand = info[brand_id]

            for month, day, action in brand:
                if month not in self.data:
                    self.data[month] = dict()

                if brand_id not in self.data[month]:
                    self.data[month][brand_id] = [0, 0, 0, 0]

                self.data[month][brand_id][action] += 1

    def data_per_day(self, brand_id, n_day=1, n_month=3, base_month=BASE_MONTH):
        bin_per_month = int(math.ceil(N_DAY_PER_MONTH / float(n_day)))
        data_raw = [0 for num in xrange(n_month * bin_per_month * TYPE_LENGTH)]
        data_sum = [0 for num in xrange(TYPE_LENGTH)]
        data_click = [0 for num in xrange(n_month * bin_per_month)]
        data_buy = [0 for num in xrange(n_month * bin_per_month)]
        data_favor = [0 for num in xrange(n_month * bin_per_month)]
        data_cart = [0 for num in xrange(n_month * bin_per_month)]
        for month, day, action in self.info[brand_id]:
            if month not in range(base_month, base_month + n_month):
                continue
            idx = ((month - base_month) * bin_per_month \
                + int(math.ceil(float(day) / n_day)) - 1)
            data_raw[idx * TYPE_LENGTH + action] += 1
            data_sum[action] += 1
            if action == CLICK_TAG:
                data_click[idx] += 1
            elif action == BUY_TAG:
                data_buy[idx] += 1
            elif action == FAVOR_TAG:
                data_favor[idx] += 1
            elif action == CART_TAG:
                data_cart[idx] += 1

        data_click_buy = [1 if a > 0 and b > 0 else 0 for a, b in zip(data_click, data_buy)]
        data_click_favor = [1 if a > 0 and b > 0 else 0 for a, b in zip(data_click, data_favor)]
        data_click_cart = [1 if a > 0 and b > 0 else 0 for a, b in zip(data_click, data_cart)]
        data_buy_favor = [1 if a > 0 and b > 0 else 0 for a, b in zip(data_buy, data_favor)]
        data_buy_cart = [1 if a > 0 and b > 0 else 0 for a, b in zip(data_buy, data_cart)]
        data_favor_cart = [1 if a > 0 and b > 0 else 0 for a, b in zip(data_favor, data_cart)]
        data_click_buy_favor = [1 if a > 0 and b > 0 and c > 0 else 0 for a, b, c in zip(data_click, data_buy, data_favor)]
        data_click_buy_cart = [1 if a > 0 and b > 0 and c > 0 else 0 for a, b, c in zip(data_click, data_buy, data_cart)]
        data_click_favor_cart = [1 if a > 0 and b > 0 and c > 0 else 0 for a, b, c in zip(data_click, data_favor, data_cart)]
        data_buy_favor_cart = [1 if a > 0 and b > 0 and c > 0 else 0 for a, b, c in zip(data_buy, data_favor, data_cart)]
        data_click_buy_favor_cart = [1 if a > 0 and b > 0 and c > 0 and d > 0 else 0 for a, b, c, d in zip(data_click, data_buy, data_favor, data_cart)]

        return data_raw + data_sum + data_click_buy + data_click_favor + data_click_cart + \
               data_buy_favor + data_buy_cart + data_favor_cart + \
               data_click_buy_favor + data_click_buy_cart + data_click_favor_cart + data_buy_favor_cart + \
               data_click_buy_favor_cart

        # return data_sum

    def data_per_week(self, brand_id, n_month=3, base_month=BASE_MONTH):
        return self.data_per_day(brand_id, n_day=7, n_month=n_month, base_month=base_month)

    def data_per_month(self, brand_id, n_month=3, base_month=BASE_MONTH):
        return self.data_per_day(brand_id, n_day=N_DAY_PER_MONTH, n_month=n_month, base_month=base_month)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))


if __name__ == '__main__':
    userInfo = dict()
    with open('../../dataset/t_alibaba_data.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            userID, brandID, actionType, month, day = [int(field) for field in row]

            if userID not in userInfo:
                userInfo[userID] = dict()

            user = userInfo[userID]
            if brandID not in user:
                user[brandID] = []

            if month in (4, 5, 6):
                day = day - 14
            else:
                day = day - 15
            if day <= 0:
                month -= 1
                day += 31

            band = user[brandID]
            band.append((month, day, actionType))

    users = dict()
    for (userID, info) in userInfo.iteritems():
        users[userID] = User(userID, info)

    history = dict()
    for user_id, user in users.items():
        if user_id not in history:
            history[user_id] = dict()

        for brand_id in user.brands:
            if brand_id not in history[user_id]:
                history[user.id][brand_id] = \
                    { 'view': 0, 'click': 0, 'cart': 0, 'favor': 0, 'buy': 0, 'label': 0 }

            history[user_id][brand_id]['view'] = 1 if \
                sum([sum(user.data[month][brand_id]) - user.data[month][brand_id][BUY_TAG] \
                    for month in \
                    xrange(BASE_MONTH, BASE_MONTH + N_MONTH - 1)
                    if month in user.data and \
                        brand_id in user.data[month]]) > 0 \
                    else 0

            history[user_id][brand_id]['click'] = \
                sum([user.data[month][brand_id][CLICK_TAG] \
                    for month in \
                    xrange(BASE_MONTH + N_MONTH - 2, BASE_MONTH + N_MONTH - 1)
                    if month in user.data and \
                        brand_id in user.data[month]])

            history[user_id][brand_id]['cart'] = 1 if \
                sum([user.data[month][brand_id][CART_TAG] \
                    for month in \
                    xrange(BASE_MONTH, BASE_MONTH + N_MONTH - 1)
                    if month in user.data and \
                        brand_id in user.data[month]]) > 0 \
                    else 0

            history[user_id][brand_id]['favor'] = 1 if \
                sum([user.data[month][brand_id][FAVOR_TAG] \
                    for month in \
                    xrange(BASE_MONTH, BASE_MONTH + N_MONTH - 1)
                    if month in user.data and \
                        brand_id in user.data[month]]) > 0 \
                    else 0

            history[user.id][brand_id]['buy'] = 1 if \
                sum([user.data[month][brand_id][1] for month in \
                    xrange(BASE_MONTH, BASE_MONTH + N_MONTH - 1)
                    if month in user.data and \
                        brand_id in user.data[month]]) > 0 \
                    else 0

            history[user.id][brand_id]['label'] = \
                user.data[BASE_MONTH + N_MONTH - 1][brand_id][BUY_TAG] \
                if BASE_MONTH + N_MONTH - 1 in user.data and \
                    brand_id in user.data[BASE_MONTH + N_MONTH - 1] \
                else 0

    view_before_pos = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['view'] > 0]
    view_before_neg = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['view'] > 0]
    click_before_pos = [(user_id, brand_id, counter['click']) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['click'] > 0]
    click_before_neg = [(user_id, brand_id, counter['click']) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['click'] > 0]
    cart_before_pos = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['cart'] > 0]
    cart_before_neg = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['cart'] > 0]
    favor_before_pos = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['favor'] > 0]
    favor_before_neg = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['favor'] > 0]
    buy_before = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['buy'] > 0]
    buy_total = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0]
    
    # print 'View Not Buy before: ', len(view_before_pos), len(view_before_neg), len(buy_total), \
    #     '{:.2f}%'.format(float(len(view_before_pos)) / len(buy_total) * 100)
    # print 'Click Not Buy before: ', len(click_before_pos), len(click_before_neg), len(buy_total), \
    #     '{:.2f}%'.format(float(len(click_before_pos)) / len(buy_total) * 100)
    # print 'Cart Not Buy before: ', len(cart_before_pos), len(cart_before_neg), len(buy_total), \
    #     '{:.2f}%'.format(float(len(cart_before_pos)) / len(buy_total) * 100)
    # print 'Favor Not Buy before: ', len(favor_before_pos), len(favor_before_neg), len(buy_total), \
    #     '{:.2f}%'.format(float(len(favor_before_pos)) / len(buy_total) * 100)
    # print 'Buy before: ', len(buy_before), len(buy_total), \
    #     '{:.2f}%'.format(float(len(buy_before)) / len(buy_total) * 100)
    # print ''
    
    data = []
    label = []
    view_before = view_before_pos + view_before_neg
    for user_id, brand_id in view_before:
        user = users[user_id]
        data.append(user.data_per_day(brand_id) + 
            user.data_per_week(brand_id) + 
            user.data_per_month(brand_id) + 
            user.data_per_day(brand_id, n_day=10) + 
            user.data_per_day(brand_id, n_day=15))
        label.append(1 if (user_id, brand_id) in view_before_pos else 0)
    train_data = numpy.asarray(data)
    train_label = numpy.asarray(label)

    train_data_pos = [x for x, y in zip(train_data, train_label) if y > 0]
    train_data_neg = [x for x, y in zip(train_data, train_label) if y < 1]
    n_pos_ori = len(train_data_pos)
    n_pos_gen = 2 * len(train_data_neg) - n_pos_ori

    K_NEIGHTBOR = 5
    neigh = NearestNeighbors(K_NEIGHTBOR)
    neigh.fit(train_data)

    n_gen = []
    for n in xrange(n_pos_ori):
        _, idx = neigh.kneighbors(train_data_pos[n])
        n_gen.append(1 - numpy.sum(train_label[idx]) / float(K_NEIGHTBOR))
    
    probs_sum = numpy.sum(numpy.asarray(n_gen))
    n_gen = [int(prob / probs_sum * n_pos_gen) for prob in n_gen]
        
    neigh.fit(train_data_pos)
    for n in xrange(n_pos_ori):
        x = train_data_pos[n]
        _, idx = neigh.kneighbors(x)
        idx = idx[0]

        for cnt in xrange(n_gen[n]):
            neighbor = train_data_pos[idx[random.randint(0, K_NEIGHTBOR - 1)]]
            x_new = x + (neighbor - x) * random.random()
            train_data_pos.append(x_new)

    train_data = train_data_pos + train_data_neg
    train_label = [1 for n in xrange(len(train_data_pos))] \
                + [0 for n in xrange(len(train_data_neg))]
    logistic = linear_model.LogisticRegression(class_weight=None)
    logistic.fit(train_data, train_label)
    predict = logistic.predict(train_data)

    print("Training Report:\n%s\n" % (
        metrics.classification_report(
            train_label, 
            predict)))

    test_data = []
    mark = []
    for user_id, user in users.items():
        for brand_id in user.brands:
            test_data.append(user.data_per_day(brand_id, base_month=BASE_MONTH + 1) + 
                user.data_per_week(brand_id, base_month=BASE_MONTH + 1) + 
                user.data_per_month(brand_id, base_month=BASE_MONTH + 1) + 
                user.data_per_day(brand_id, n_day=10, base_month=BASE_MONTH + 1) + 
                user.data_per_day(brand_id, n_day=15, base_month=BASE_MONTH + 1))
            mark.append((user_id, brand_id))
    test_data = numpy.asarray(test_data)
    predict = logistic.predict(test_data)
    
    result = dict()
    for label, tag in zip(predict, mark):
        if label < 1: continue
        if tag[0] not in result:
            result[tag[0]] = []
        result[tag[0]].append(tag[1])

    f = open('result.txt', 'w')
    for user_id, brands in result.items():
        print >> f, str(user_id) + '\t' + ','.join([str(brand) for brand in brands])


