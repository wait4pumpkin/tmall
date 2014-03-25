#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys
import time
import math

import numpy
import pylab

from sklearn import linear_model, cross_validation

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
    with open('/home/pumpkin/Documents/project/tmall/dataset/clean.csv', 'rb') as csvfile:
    # with open('/home/pumpkin/Documents/project/tmall/dataset/demo.csv', 'rb') as csvfile:
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

    # count = 0
    # for user in users:
    #     if len(user.test_label):
    #         count += 1

    # print count, len(users), '{:.2f}%'.format(float(count) / len(users) * 100)

    data = []
    label = []
    for user in users:
        sample = []
        for month in (BASE_MONTH, BASE_MONTH + 1):
            total = [0, 0, 0, 0]
            if month in user.data:
                for brand, actions in user.data[month].items():
                    total = [a + b for a, b in zip(total, actions)]
                total += [len(user.data[month])]
            else:
                total += [0]
            sample += total
        data.append(sample)
        print sample

        flag = 0
        if BASE_MONTH + 2 in user.data:
            for brand, actions in user.data[BASE_MONTH + 2].items():
                if actions[1] > 0:
                    flag = 1
        label.append(flag)

    data = numpy.asarray(data)
    label = numpy.asarray(label)

    # Training
    logistic = None
    k_fold = cross_validation.KFold(len(users), n_folds=5, shuffle=True)
    for train_index, test_index in k_fold:
        logistic = linear_model.LogisticRegression()
        logistic.fit(data[train_index], label[train_index])

        error = numpy.sum(numpy.absolute(
            logistic.predict(data[train_index]) - label[train_index]))
        print 'Training:', error, len(train_index), \
            '{:.2f}%'.format(float(error) / len(train_index) * 100)

        error = numpy.sum(numpy.absolute(
            logistic.predict(data[test_index]) - label[test_index]))
        print 'Validation:', error, len(test_index), \
            '{:.2f}%'.format(float(error) / len(test_index) * 100)

        print ''

    # Test
    data = []
    label = []
    for user in users:
        sample = []
        for month in (BASE_MONTH + 1, BASE_MONTH + 2):
            if month in user.data:
                sample.append(len(user.data[month]))
            else:
                sample.append(0)
        data.append(sample)

        flag = 0
        if BASE_MONTH + 3 in user.data:
            for brand, actions in user.data[BASE_MONTH + 3].items():
                if actions[1] > 0:
                    flag = 1
        label.append(flag)

    data = numpy.asarray(data)
    label = numpy.asarray(label)

    logistic.fit(data[train_index], label[train_index])

    error = numpy.sum(numpy.absolute(
        logistic.predict(data) - label))
    print error, len(users), '{:.2f}%'.format(float(error) / len(users) * 100)