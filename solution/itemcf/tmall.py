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
        self.weight = [1, 1, 1, 1]
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

        total_buy_action = float(self.weight[1])
        self.weight = [1 / (self.weight[idx] / total_buy_action) for idx, num in enumerate(self.weight)]
        
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

    users_train = []
    for (userID, info) in userInfo.iteritems():
        users_train.append(User(userID, info))

    overlap = dict()
    n_users = dict()
    for user in users_train:
        brands = user.train_label
        for i in brands:
            if i not in n_users:
                n_users[i] = 0
            if i not in overlap:
                overlap[i] = dict()

            n_users[i] += 1
            for j in brands:
                if i is j:
                    continue
                if j not in overlap[i]:
                    overlap[i][j] = 0
                overlap[i][j] += 1

    w = dict()
    for i, related_brands in overlap.items():
        if i not in w:
            w[i] = dict()

        for j, val in related_brands.items():
            w[i][j] = float(val) / math.sqrt(n_users[i] * n_users[j])


    for top_n in (1, 2, 3, 4, 5, 10, 20, 40, 80, 160):
        pBands = []
        bBands = []
        hitBands = []
        for user in users_train:
            bBands.append(len(user.test_label))

            rank = dict()
            for i in user.train_label:
                weight = sum(user.data[i])
                for j, w_ij in sorted(w[i].items(), key=lambda e: e[1], reverse=True)[0:top_n]:
                    if j in user.train_label:
                        continue
                    if j not in rank:
                        rank[j] = 0
                    rank[j] += w_ij * weight

            hit = 0
            total = top_n
            for brand, prob in rank.items():
                if brand in user.test_label:
                    hit += 1
            
            hitBands.append(hit)
            pBands.append(total)

        print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)

        precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
        recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

        print 'All(%d):  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (top_n, precision * 100, recall * 100, f1 * 100)
