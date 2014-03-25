#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys

import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

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
                    self.train_label.add(brandID)

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

    data = Data()
    for user in users:
        print user.id
        for brand in user.train_label:
            if user.data[brand][1] > 0:
                data.add_tuple((10, brand, user.id))
            else:
                data.add_tuple((1, brand, user.id))

    train, test = data.split_train_test(percent=0.8)

    svd = SVD()
    svd.set_data(data)
    svd.compute(k=100, min_values=1)

    # rmse = RMSE()
    # mae = MAE()
    # for rating, item_id, user_id in test.get():
    #     try:
    #         pred_rating = svd.predict(item_id, user_id)
    #         rmse.add(rating, pred_rating)
    #         mae.add(rating, pred_rating)
    #     except KeyError:
    #         continue

    # print 'RMSE=%s' % rmse.compute()
    # print 'MAE=%s' % mae.compute()

    top_n = 10
    pBands = []
    bBands = []
    hitBands = []
    for user in users:
        # predicts = [svd.predict(brandID, user.id) for brandID in range(brand_counter)]
        predicts = svd.recommend(user.id, n=top_n, only_unknowns=True, is_row=False)
        print predicts
        # result = sorted(zip(predicts, [num for num in range(brand_counter)]), reverse=True)[0:top_n]
        bBands.append(len(user.test_label))

        hit = 0
        total = top_n
        for brand_id, prob in predicts:
            if brand_id in user.test_label:
                hit += 1

        hitBands.append(hit)
        pBands.append(total)

    print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)
    precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
    recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0
    print 'All:  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (precision * 100, recall * 100, f1 * 100)
