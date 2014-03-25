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
        self.label = []
        self.time_last = []
        self.time_last_buy = []
        self.time_last_non = []
        for idx, brandID in enumerate(self.brands):
            band = info[brandID]
            label = 0
            begin = None
            end = None
            for month, day, action in band:
                counter = day + (month - BASE_MONTH) * N_DAY_PER_MONTH
                if begin is None or counter < begin:
                    begin = counter
                if end is None or counter > end:
                    end = counter
                
                if action is 1:
                    label = 1

            self.label.append(label)
            self.time_last.append(end - begin)
            if label is 1:
                self.time_last_buy.append(end - begin)
            else:
                self.time_last_non.append(end - begin)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.brands))

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

    print 'Total'
    times = []
    for user in users:
        times.extend(user.time_last)
    idx = 0
    for num in numpy.bincount(times):
        print idx, num, '{:0.2f}%'.format(float(num) / len(times) * 100)
        idx += 1
    print ''

    print 'None'
    times = []
    for user in users:
        times.extend(user.time_last_non)
    idx = 0
    for num in numpy.bincount(times):
        print idx, num, '{:0.2f}%'.format(float(num) / len(times) * 100)
        idx += 1

    print 'Buy'
    times = []
    for user in users:
        times.extend(user.time_last_buy)
    idx = 0
    for num in numpy.bincount(times):
        print idx, num, '{:0.2f}%'.format(float(num) / len(times) * 100)
        idx += 1