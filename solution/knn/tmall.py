#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys

import numpy
import pylab

import sklearn.neighbors

N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

class User(object):
    def __init__(self, id, info):
        self.id = id;
        self.bands = info.keys()
        self.data = numpy.zeros((len(info), 4), dtype=int)
        self.train_label = []
        self.test_label = []
        for idx, brandID in enumerate(self.bands):
            band = info[brandID]
            train_label = 0
            test_label = 0
            for month, day, action in band:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8

                if month is BASE_MONTH + N_MONTH - 1:
                    if action is 1:
                        test_label = 1
                else:
                    if action is 0:
                        self.data[idx, action] += 1
                    elif action is 1:
                        self.data[idx, action] += 5
                    elif action is 2:
                        self.data[idx, action] += 2
                    else:
                        self.data[idx, action] += 3
                    if action is 1:
                        train_label = 1

            self.train_label.append(train_label)
            self.test_label.append(test_label)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))

def eval(user_train, user_test, k=1):
    data_train = numpy.vstack([user.data for user in user_train])
    label_train = []
    for user in user_train:
        label_train.extend(user.train_label)

    clf = sklearn.neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(data_train, label_train)

    pBands = []
    bBands = []
    hitBands = []
    for user in user_test:
        bBands.append(sum(user.test_label))

        hit = 0
        total = 0
        for idx, label in enumerate(user.test_label):
            predict = clf.predict(user.data[idx, :])[0]
            if predict == 1:
                total += 1
                if label == 1:
                    hit += 1

        hitBands.append(hit)
        pBands.append(total)

    print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)

    precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
    recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

    return precision, recall, f1


if __name__ == '__main__':
    userInfo = dict()
    with open('/home/pumpkin/Documents/project/tmall/dataset/t_alibaba_data.csv', 'rb') as csvfile:
    # with open('/home/pumpkin/Documents/project/tmall/dataset/demo.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            userID, brandID, actionType, month, day = [int(field) for field in row]
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

    random.shuffle(users)
    num_folds = 5
    fold_size = len(users) / num_folds
    print 'Num for test: ', fold_size

    for n in range(num_folds):
        user_test = users[n * fold_size: (n + 1) * fold_size]
        user_train = users[:n * fold_size] + users[(n + 1) * fold_size:]
        
        for k in (1, 3, 5, 7, 9, 11, 15):
            precision, recall, f1 = eval(user_train, user_test, k)

            print 'Test(k = %d): %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (k, precision * 100, recall * 100, f1 * 100)

        print ''
        sys.stdout.flush()
