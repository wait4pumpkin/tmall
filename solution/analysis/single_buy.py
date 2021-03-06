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
from scipy.ndimage import convolve
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


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
        # data = [0 for num in xrange(n_month * bin_per_month * TYPE_LENGTH)]
        data = [0 for num in xrange(n_month * bin_per_month)]
        if brand_id not in self.info.keys(): return data
        for month, day, action in self.info[brand_id]:
            if month not in range(BASE_MONTH, BASE_MONTH + n_month):
                continue
            # data[((month - BASE_MONTH) * bin_per_month + int(math.ceil(float(day) / n_day)) - 1) * TYPE_LENGTH + action] += 1
            data[(month - BASE_MONTH) * bin_per_month + int(math.ceil(float(day) / n_day)) - 1] = 1
        return data

    def data_per_week(self, brand_id, n_month=3):
        return self.data_per_day(brand_id, n_day=7, n_month=n_month)

    def data_per_month(self, brand_id, n_month=3):
        return self.data_per_day(brand_id, n_day=N_DAY_PER_MONTH, n_month=n_month)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))


if __name__ == '__main__':
    userInfo = dict()
    with open('../../dataset/t_alibaba_data.csv', 'rb') as csvfile:
    # with open('../../dataset/dataset/demo.csv', 'rb') as csvfile:
        user_table = dict()
        brand_table = dict()
        user_counter = 0
        brand_counter = 0
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            userID, brandID, actionType, month, day = [int(field) for field in row]

            if userID not in user_table:
                user_table[userID] = user_counter
                user_counter += 1
            if brandID not in brand_table:
                brand_table[brandID] = brand_counter
                brand_counter += 1
            userID = user_table[userID]
            brandID = brand_table[brandID]

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
                sum([sum(user.data[month][brand_id]) - user.data[month][brand_id][1] \
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
                user.data[BASE_MONTH + N_MONTH - 1][brand_id][1] \
                if BASE_MONTH + N_MONTH - 1 in user.data and \
                    brand_id in user.data[BASE_MONTH + N_MONTH - 1] \
                else 0

    view_before_pos = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['view'] > 0 and counter['buy'] < 1]
    view_before_neg = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['view'] > 0 and counter['buy'] < 1]
    click_before_pos = [(user_id, brand_id, counter['click']) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['click'] > 0 and counter['buy'] < 1]
    click_before_neg = [(user_id, brand_id, counter['click']) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['click'] > 0 and counter['buy'] < 1]
    cart_before_pos = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['cart'] > 0 and counter['buy'] < 1]
    cart_before_neg = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['cart'] > 0 and counter['buy'] < 1]
    favor_before_pos = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['favor'] > 0 and counter['buy'] < 1]
    favor_before_neg = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] < 1 and counter['favor'] > 0 and counter['buy'] < 1]
    buy_before = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['buy'] > 0]
    buy_total = [(user_id, brand_id) for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0]
    print Counter([times for _, _, times in click_before_pos])
    print Counter([times for _, _, times in click_before_neg])
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
        logistic = linear_model.LogisticRegression(class_weight='auto')
        logistic.fit(data[train_index], label[train_index])

        print 'Training: ', sum(label[train_index]), '/', len(label)
        print 'Validation: ', sum(label[test_index]), '/', len(label)

        print '-------------------------------------------------------------'

        pos_idx = [idx for idx, tag in enumerate(label[train_index]) if tag > 0]
        neg_idx = [idx for idx, tag in enumerate(label[train_index]) if tag < 1]
        predict = logistic.predict(data[train_index])
        pos2neg = [(a, b) for a, b in zip(label[train_index][pos_idx], predict[pos_idx]) if a != b]
        neg2pos = [(a, b) for a, b in zip(label[train_index][neg_idx], predict[neg_idx]) if a != b]

        error = numpy.sum(numpy.absolute(predict - label[train_index]))
        print 'Training:', error, len(train_index), \
            '{:.2f}%'.format(float(error) / len(train_index) * 100)
        print 'Pos2neg: ', len(pos2neg), ' ', 'Neg2pos: ', len(neg2pos)

        print '-------------------------------------------------------------'

        pos_idx = [idx for idx, tag in enumerate(label[test_index]) if tag > 0]
        neg_idx = [idx for idx, tag in enumerate(label[test_index]) if tag < 1]
        predict = logistic.predict(data[test_index])
        pos2neg = [(a, b) for a, b in zip(label[test_index][pos_idx], predict[pos_idx]) if a != b]
        neg2pos = [(a, b) for a, b in zip(label[test_index][neg_idx], predict[neg_idx]) if a != b]

        error = numpy.sum(numpy.absolute(predict - label[test_index]))
        print 'Validation:', error, len(test_index), \
            '{:.2f}%'.format(float(error) / len(test_index) * 100)
        print 'Pos2neg: ', len(pos2neg), ' ', 'Neg2pos: ', len(neg2pos)

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
    