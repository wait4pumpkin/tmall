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

from scipy import sparse
from scipy.sparse import lil_matrix

from sklearn import svm
from sklearn import linear_model, cross_validation, datasets


N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

class User(object):
    def __init__(self, id, info):
        self.id = id;
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

    users = []
    for (userID, info) in userInfo.iteritems():
        users.append(User(userID, info))

    history = dict()
    for user in users:
        if user.id not in history:
            history[user.id] = dict()

        for brand_id in user.brands:
            if brand_id not in history[user.id]:
                history[user.id][brand_id] = \
                    { 'view': 0, 'buy': 0, 'label': 0 }

            history[user.id][brand_id]['view'] = 1 if \
                sum([sum(user.data[month][brand_id]) - user.data[month][brand_id][1] \
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

    view_before = sum([1 for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['view'] > 0 and counter['buy'] < 1])
    buy_before = sum([1 for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0 and counter['buy'] > 0])
    buy_total = sum([1 for user_id, brands in history.items() \
        for brand_id, counter in brands.items() \
        if counter['label'] > 0])
    print 'View Not Buy before: ', view_before, buy_total, \
        '{:.2f}%'.format(float(view_before) / buy_total * 100)
    print 'Buy before: ', buy_before, buy_total, \
        '{:.2f}%'.format(float(buy_before) / buy_total * 100)
    

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    print X.shape
    print Y.shape

    h = .02  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
