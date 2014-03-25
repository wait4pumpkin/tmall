#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys

import numpy
import scipy.io
import pylab

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
        self.times = []
        for idx, brandID in enumerate(self.bands):
            band = info[brandID]
            train_label = 0
            test_label = 0
            times_buy = 0
            for month, day, action in band:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8

                self.data[idx, action] += 1
                if action is 1:
                    times_buy += 1

                if action is 1:
                    test_label = 1

                # if month is BASE_MONTH + N_MONTH - 1:
                #     if action is 1:
                #         test_label = 1
                # else:
                #     self.data[idx, action] += 1
                #     if action is 1:
                #         train_label = 1

            self.train_label.append(train_label)
            self.test_label.append(test_label)
            self.times.append(times_buy)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))

def eval(user_test, svm):
    pBands = []
    bBands = []
    hitBands = []
    for user in user_test:
        bBands.append(sum(user.test_label))

        hit = 0
        total = 0
        for idx, label in enumerate(user.test_label):
            predict, acc, prob = svm_predict([0], [user.data[idx, :].flatten().tolist()], svm, '-q')
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

    times = []
    label = []
    for user in users:
        times.extend(user.times)
        label.extend(user.test_label)

    random.shuffle(users)
    data_train = numpy.vstack([user.data for user in users])
    
    print 'Num users: ', len(users)
    print 'Num bands: ', data_train.shape[0]
    print numpy.sum(data_train, axis=0)
    print numpy.sum(times)
    
    idx = 0
    print numpy.count_nonzero(times)
    for num in numpy.bincount(times):
        if not idx is 0:
            print idx, num, '{:0.2f}%'.format(float(num) / numpy.count_nonzero(times) * 100)
        idx += 1
