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

from collections import Counter

from svmutil import *

N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

class User(object):
    def __init__(self, id, info):
        self.id = id;
        self.brands = info.keys()
        self.data = dict()
        self.test_label = set()
        self.train_label = set()
        self.weight = [0, 0, 0, 0]
        for brandID in self.brands:
            brand = info[brandID]

            for month, day, action in brand:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8

                if month == BASE_MONTH + N_MONTH - 1:
                    if action == 1:
                        self.test_label.add(brandID)
                else:
                    if not brandID in self.data:
                        self.data[brandID] = [0, 0, 0, 0]
                    self.data[brandID][action] += 1
                    self.weight[action] += 1
                    self.train_label.add(brandID)
        
        for brand in self.data.keys():
            self.data[brand] = [num * weight for num, weight in zip(self.data[brand], self.weight)]

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))

if __name__ == '__main__':
    userInfo = dict()
    with open('/home/pumpkin/Documents/project/tmall/dataset/t_alibaba_data.csv', 'rb') as csvfile:
    # with open('/home/pumpkin/Documents/project/tmall/dataset/demo.csv', 'rb') as csvfile:
        user_table = dict()
        brand_table = dict()
        user_counter = 0
        brand_counter = 0
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            userID, brandID, actionType, month, day = [int(field) for field in row]

            if not userID in user_table:
                user_table[userID] = user_counter
                user_counter += 1
            if not brandID in brand_table:
                brand_table[brandID] = brand_counter
                brand_counter += 1
            userID = user_table[userID]
            brandID = brand_table[brandID]

            if not userID in userInfo:
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

    num_folds = 5
    fold_size = len(users) / num_folds
    print 'Num for test: ', fold_size
    for n in range(num_folds):
        random.shuffle(users)
        user_test = users[n * fold_size: (n + 1) * fold_size]
        user_train = users[:n * fold_size] + users[(n + 1) * fold_size:]
        
        train_data = [[len(user.train_label)] + user.weight for user in user_train]
        train_label = [len(user.test_label) > 0 for user in user_train]
        svm = svm_train(train_label, train_data, '-t 0 -q')

        test_data = [[len(user.train_label)] + user.weight for user in user_test]
        test_label = [len(user.test_label) > 0 for user in user_test]
        predict, acc, prob = svm_predict(test_label, test_data, svm)
