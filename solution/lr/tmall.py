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

    def data_per_day(self, brand_id, n_day=1, n_month=3):
        bin_per_month = int(math.ceil(N_DAY_PER_MONTH / float(n_day)))
        data_raw = [0 for num in xrange(n_month * bin_per_month * TYPE_LENGTH)]
        data_sum = [0 for num in xrange(TYPE_LENGTH)]
        data_click = [0 for num in xrange(n_month * bin_per_month)]
        data_buy = [0 for num in xrange(n_month * bin_per_month)]
        data_favor = [0 for num in xrange(n_month * bin_per_month)]
        data_cart = [0 for num in xrange(n_month * bin_per_month)]
        for month, day, action in self.info[brand_id]:
            if month not in range(BASE_MONTH, BASE_MONTH + n_month):
                continue
            idx = ((month - BASE_MONTH) * bin_per_month \
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

        return data_raw + data_sum + data_click_buy + data_click_favor + data_click_cart + \
               data_buy_favor + data_buy_cart + data_favor_cart
        # return data_sum

    def data_per_week(self, brand_id, n_month=3):
        return self.data_per_day(brand_id, n_day=7, n_month=n_month)

    def data_per_month(self, brand_id, n_month=3):
        return self.data_per_day(brand_id, n_day=N_DAY_PER_MONTH, n_month=n_month)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))


if __name__ == '__main__':
    user_info = dict()
    with open('../../dataset/t_alibaba_data.csv', 'rb') as csvfile:
        user_table = dict()
        brand_table = dict()
        user_counter = 0
        brand_counter = 0
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            user_id, brand_id, action_type, month, day = [int(field) for field in row]

            if user_id not in user_table:
                user_table[user_id] = user_counter
                user_counter += 1
            if brand_id not in brand_table:
                brand_table[brand_id] = brand_counter
                brand_counter += 1
            user_id = user_table[user_id]
            brand_id = brand_table[brand_id]

            if user_id not in user_info:
                user_info[user_id] = dict()

            user = user_info[user_id]
            if brand_id not in user:
                user[brand_id] = []

            if month in (4, 5, 6):
                day = day - 14
            else:
                day = day - 15
            if day <= 0:
                month -= 1
                day += 31

            brand = user[brand_id]
            brand.append((month, day, action_type))

    users = dict()
    for user_id, info in user_info.items():
        users[user_id] = User(user_id, info)

    history = dict()
    for user_id, user in users.items():
        if user_id not in history:
            history[user_id] = dict()

        for brand_id in user.brands:
            if brand_id not in history[user_id]:
                history[user.id][brand_id] = \
                    { 'view': 0, 'click': 0, 'cart': 0, 'favor': 0, 'buy': 0, 'label': 0 }

            history[user_id][brand_id]['view'] = 1 if \
                sum([sum(user.data[month][brand_id]) \
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
                sum([user.data[month][brand_id][BUY_TAG] for month in \
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
    
    print 'View Not Buy before: ', len(view_before_pos), len(view_before_neg), len(buy_total), \
        '{:.2f}%'.format(float(len(view_before_pos)) / len(buy_total) * 100)
    print 'Click Not Buy before: ', len(click_before_pos), len(click_before_neg), len(buy_total), \
        '{:.2f}%'.format(float(len(click_before_pos)) / len(buy_total) * 100)
    print 'Cart Not Buy before: ', len(cart_before_pos), len(cart_before_neg), len(buy_total), \
        '{:.2f}%'.format(float(len(cart_before_pos)) / len(buy_total) * 100)
    print 'Favor Not Buy before: ', len(favor_before_pos), len(favor_before_neg), len(buy_total), \
        '{:.2f}%'.format(float(len(favor_before_pos)) / len(buy_total) * 100)
    print 'Buy before: ', len(buy_before), len(buy_total), \
        '{:.2f}%'.format(float(len(buy_before)) / len(buy_total) * 100)
    print ''
    
    view_before = view_before_pos + view_before_neg
    k_fold = cross_validation.KFold(len(view_before), n_folds=5, shuffle=True)
    data = []
    label = []
    for user_id, brand_id in view_before:
        user = users[user_id]
        data.append(user.data_per_day(brand_id) + 
            user.data_per_week(brand_id) + 
            user.data_per_month(brand_id) + 
            user.data_per_day(brand_id, n_day=10) + 
            user.data_per_day(brand_id, n_day=15))
        label.append(1 if (user_id, brand_id) in view_before_pos else 0)
    data = numpy.asarray(data)
    label = numpy.asarray(label)

    """
    for train_index, test_index in k_fold:
        train_data = data[train_index]
        train_label = label[train_index]

        train_data_pos = [x for x, y in zip(train_data, train_label) if y > 0]
        train_data_neg = [x for x, y in zip(train_data, train_label) if y < 1]

        classifiers = []
        train_data_boost = []
        train_label_boost = []
        for n in xrange(100):
            logistic = linear_model.LogisticRegression(class_weight='auto')

            train_data_part = train_data_pos + random.sample(train_data_neg, len(train_data_pos))
            train_label_part = [1 for n in xrange(len(train_data_pos))] \
                             + [0 for n in xrange(len(train_data_pos))]

            logistic.fit(train_data_part, train_label_part)
            predict = logistic.predict(train_data_part)

            # print("Training Report:\n%s\n" % (
            #     metrics.classification_report(
            #         train_label_part, 
            #         predict)))

            classifiers.append(logistic)
            train_data_boost.append(predict)
            train_label_boost += train_label_part

        predict = numpy.zeros(len(train_label))
        for classifier in classifiers:
            predict += numpy.asarray(classifier.predict(train_data))
        predict[predict > 5] = 1
        predict[predict <= 5] = 0

        print len(train_label)
        print predict.shape
        print len(predict.tolist())
        print("Training Report:\n%s\n" % (
                metrics.classification_report(
                    train_label, 
                    predict.tolist())))

        pos_idx = [idx for idx, tag in enumerate(label[train_index]) if tag > 0]
        neg_idx = [idx for idx, tag in enumerate(label[train_index]) if tag < 1]
        
        pos2neg = [(a, b) for a, b in zip(label[train_index][pos_idx], predict[pos_idx]) if a != b]
        neg2pos = [(a, b) for a, b in zip(label[train_index][neg_idx], predict[neg_idx]) if a != b]

        print 'Pos2neg: ', len(pos2neg), ' ', 'Neg2pos: ', len(neg2pos)
        print ''

        predict = logistic.predict(data[test_index])

        print("Validation Report:\n%s\n" % (
            metrics.classification_report(
                label[test_index], 
                predict)))

        pos_idx = [idx for idx, tag in enumerate(label[test_index]) if tag > 0]
        neg_idx = [idx for idx, tag in enumerate(label[test_index]) if tag < 1]
        
        pos2neg = [(a, b) for a, b in zip(label[test_index][pos_idx], predict[pos_idx]) if a != b]
        neg2pos = [(a, b) for a, b in zip(label[test_index][neg_idx], predict[neg_idx]) if a != b]

        print 'Pos2neg: ', len(pos2neg), ' ', 'Neg2pos: ', len(neg2pos)

        print '-------------------------------------------------------------'
        print ''

    """
    for train_index, test_index in k_fold:
        logistic = linear_model.LogisticRegression(class_weight=None)

        train_data = data[train_index]
        train_label = label[train_index]

        train_data_pos = [x for x, y in zip(train_data, train_label) if y > 0]
        train_data_neg = [x for x, y in zip(train_data, train_label) if y < 1]

        logistic.fit(train_data, train_label)
        predict = logistic.predict(train_data)

        print("Training Report:\n%s\n" % (
            metrics.classification_report(
                train_label, 
                predict)))

        pos_idx = [idx for idx, tag in enumerate(label[train_index]) if tag > 0]
        neg_idx = [idx for idx, tag in enumerate(label[train_index]) if tag < 1]
        
        pos2neg = [(a, b) for a, b in zip(train_label[pos_idx], predict[pos_idx]) if a != b]
        neg2pos = [(a, b) for a, b in zip(train_label[neg_idx], predict[neg_idx]) if a != b]

        print 'Pos2neg: ', len(pos2neg), ' ', 'Neg2pos: ', len(neg2pos)
        print ''



        predict = logistic.predict(data[test_index])

        print("Validation Report:\n%s\n" % (
            metrics.classification_report(
                label[test_index], 
                predict)))

        pos_idx = [idx for idx, tag in enumerate(label[test_index]) if tag > 0]
        neg_idx = [idx for idx, tag in enumerate(label[test_index]) if tag < 1]
        
        pos2neg = [(a, b) for a, b in zip(label[test_index][pos_idx], predict[pos_idx]) if a != b]
        neg2pos = [(a, b) for a, b in zip(label[test_index][neg_idx], predict[neg_idx]) if a != b]

        print 'Pos2neg: ', len(pos2neg), ' ', 'Neg2pos: ', len(neg2pos)

        print '-------------------------------------------------------------'
        print ''
    
    """
    # Load Data
    
    # Models we will use
    logistic = linear_model.LogisticRegression(class_weight='auto')
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    # logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(data, label)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(class_weight='auto')
    logistic_classifier.fit(data, label)

    ###############################################################################
    # Evaluation

    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            label,
            classifier.predict(data))))

    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            label,
            logistic_classifier.predict(data))))

    ###############################################################################
    # Plotting

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()
    """
