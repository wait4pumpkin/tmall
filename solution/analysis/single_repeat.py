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
        self.day = dict()
        self.label = set()
        for brandID in self.brands:
            brand = info[brandID]

            for month, day, action in brand:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8

                if action == 1:
                    if month >= BASE_MONTH + N_MONTH - 1:
                        self.label.add(brandID)
                    else:
                        if brandID not in self.data:
                            self.data[brandID] = 0
                            self.day[brandID] = []
                        self.data[brandID] += 1
                        
                        self.day[brandID].append(day + (month - BASE_MONTH) * N_DAY_PER_MONTH)

        self.data = sorted(self.data.items(), key=lambda e: e[1], reverse=True)

        self.period_brand = set()
        for brand, days in self.day.items():
            days.sort()

            wait = [days[idx+1] - days[idx] for idx in range(len(days)-1)]
            repeat = [num for num in wait if num > 0]
            if len(repeat) > 0:
                if days[-1] < (N_MONTH - 2) * N_DAY_PER_MONTH:
                    if len(repeat) > 2 or sum(repeat) > 10:
                        self.period_brand.add(brand)
                        print repeat
                else:
                    self.period_brand.add(brand)
                    print '!', repeat
        
    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))

if __name__ == '__main__':
    userInfo = dict()
    with open('/home/pumpkin/Documents/project/tmall/dataset/t_alibaba_data.csv', 'rb') as csvfile:
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

    counter = 0
    for user in users:
        if len(user.data) <= 0:
            continue

        if user.data[0][1] > 1:
            counter += 1
    print counter, '{:.2f}%'.format(float(counter) / len(users) * 100)


    # counter = 0
    # for user in users:
    #     if len(user.data) <= 0 or user.data[0][1] < 2:
    #         continue

    #     flag = False
    #     for brand, time in user.data:
    #         if time < 2:
    #             break
    #         day = sorted(user.day[brand])
    #         wait = [day[idx+1] - day[idx] for idx in range(len(day)-1)]
    #         if len([num for num in wait if num > 0]) > 0:
    #             flag = True
    #             repeat = [num for num in wait if num > 0]
    #             if day[-1] < (N_MONTH - 1) * N_DAY_PER_MONTH:
    #                 if len(repeat) < 3 and sum(repeat) < 10:
    #                     flag = False
    #                 else:
    #                     print repeat

    #     if flag:
    #         counter += 1
    #         print '================================================================'
            
    # print counter, '{:.2f}%'.format(float(counter) / len(users) * 100)
    

    pBands = []
    bBands = []
    hitBands = []
    for user in users:
        bBands.append(len(user.label))

        hit = 0
        total = len(user.period_brand)
        for predict in user.period_brand:
            if predict in user.label:
                hit += 1

        hitBands.append(hit)
        pBands.append(total)

    print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)

    precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
    recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

    print 'All:  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (precision * 100, recall * 100, f1 * 100)