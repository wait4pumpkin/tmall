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
        self.brands = info.keys()
        self.train_label = []
        self.test_label = []
        self.flag = []
        self.brands_train = []
        self.brands_test = []
        for idx, brandID in enumerate(self.brands):
            band = info[brandID]
            train_label = 0
            test_label = 0
            begin = None
            end = None
            for month, day, action in band:
                if action is 1:
                    counter = day + (month - BASE_MONTH) * N_DAY_PER_MONTH
                    if begin is None or counter < begin:
                        begin = counter
                    if end is None or counter > end:
                        end = counter

                    if month is BASE_MONTH + N_MONTH - 1:
                        test_label = 1
                    else:
                        train_label = 1

            self.train_label.append(train_label)
            self.test_label.append(test_label)
            if test_label is 1 and begin > (N_MONTH - 1) * N_DAY_PER_MONTH:
                self.flag.append(1)
            else:
                self.flag.append(0)

            if begin < (N_MONTH - 1) * N_DAY_PER_MONTH:
                self.brands_train.append(brandID)
            if end > (N_MONTH - 1) * N_DAY_PER_MONTH:
                self.brands_test.append(brandID)


    def __str__(self):
        return str(self.id) + ' ' + str(len(self.brands))

def eval(user_test, svm):
    pbrands = []
    bbrands = []
    hitbrands = []
    for user in user_test:
        bbrands.append(sum(user.test_label))

        hit = 0
        total = 0
        for idx, label in enumerate(user.test_label):
            predict, acc, prob = svm_predict([0], [user.data[idx, :].flatten().tolist()], svm, '-q')
            if predict == 1:
                total += 1
                if label == 1:
                    hit += 1

        hitbrands.append(hit)
        pbrands.append(total)

    print sum(hitbrands), ' ', sum(pbrands), ' ', sum(bbrands)

    precision = float(sum(hitbrands)) / sum(pbrands) if not sum(pbrands) == 0 else 0
    recall = float(sum(hitbrands)) / sum(bbrands) if not sum(bbrands) == 0 else 0
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

    train_label = []
    test_label = []
    for user in users:
        train_label.extend(user.train_label)
        test_label.extend(user.test_label)
    
    print 'Total:', len(train_label)
    counter = 0
    for train, test in zip(train_label, test_label):
        if train is 1 and test is 1:
            counter += 1
    print 'Twice:', counter, '{:.2f}%'.format(counter / float(len(train_label)) * 100)

    flag = []
    for user in users:
        flag.extend(user.flag)
    print 'Total:', sum(test_label)
    print 'New:', sum(flag), '{:.2f}%'.format(sum(flag) / float(sum(test_label)) * 100)

    brands_train = []
    brands_test = []
    for user in users:
        brands_train.extend(user.brands_train)
        brands_test.extend(user.brands_test)
    brands_train = set(brands_train)
    brands_test = set(brands_test)
    diff = brands_test.difference(brands_train)
    print 'Train brands:', len(brands_train)
    print 'Test brands:', len(brands_test)
    print 'New:', len(diff), '{:.2f}%'.format(float(len(diff)) / len(brands_test) * 100)

    

    # Test if brand brought appear before for specific user, not all
    total = 0
    counter = 0
    for user in users:
        for brand in user.brands_test:
            total += 1
            if brand in user.brands_train:
                counter += 1
    print total, counter, '{:.2f}%'.format(float(counter) / total * 100)

    # Test pairwise brand overlap
    overlap = numpy.zeros((len(users), len(users)), dtype=numpy.float32)
    for idxA in range(len(users)):
        for idxB in range(len(users)):
            if idxA is idxB:
                continue
            brandA = set(users[idxA].brands_train + users[idxA].brands_test)
            brandB = set(users[idxB].brands_train + users[idxB].brands_test)
            overlap[idxA, idxB] = len(brandA.intersection(brandB)) / float(len(brandA))
    print sorted(numpy.amax(overlap, axis=1).tolist())
    print sum(numpy.amax(overlap, axis=1).tolist()) / float(len(users))
