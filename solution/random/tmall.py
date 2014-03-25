#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys

import numpy
import pylab

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from svmutil import *

N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

class User(object):
    def __init__(self, id, info):
        self.id = id;
        self.bands = info.keys()
        self.data = numpy.zeros((len(info), N_MONTH * 3 * TYPE_LENGTH * 8), dtype=int)
        self.label = []
        for idx, brandID in enumerate(self.bands):
            band = info[brandID]
            row = [0 for n in range(48)]
            label = 0
            for month, day, action in band:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8
                row[p + action] = min(255, row[p + action] + 1)

                if month == BASE_MONTH + N_MONTH - 1 and action == 1:
                    label = 1

            self.label.append(label)
            self.data[idx, :] = numpy.mat([list(format(num, '08b')) for num in row]).flatten()
        self.data = self.data.astype(numpy.float32)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))

def eval(users):
    pBands = []
    bBands = []
    hitBands = []
    for user in users:
        bBands.append(sum(user.label))
        ratio = sum(user.label) / float(len(user.label))

        hit = 0
        total = 0
        for idx, label in enumerate(user.label):
            predict = 1 if random.random() < ratio * 2 else 0
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

    print 'Num of users: ', len(users)
    
    precision, recall, f1 = eval(users)
    print 'All:  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (precision * 100, recall * 100, f1 * 100)
    print ''
    sys.stdout.flush()

