#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random

import numpy

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
        self.train_label_pro = set()
        self.train_label_neg = set()
        self.action_weight = [1, 1, 1, 1]
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
                    self.action_weight[action] += 1
                    if action is 1:
                        self.train_label_pro.add(brandID)
                    else:
                        self.train_label_neg.add(brandID)

        total_buy_action = float(self.action_weight[1])
        self.action_weight = [1 / max(self.action_weight[idx] / total_buy_action, 5) for idx, num in enumerate(self.action_weight)]
        self.action_weight[1] = 1

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

    brands = set([num for num in range(brand_counter)])
    writer = open('data.train', 'w')
    for user in users:
        info = []
        for brand in user.train_label_pro.union(user.train_label_neg):
            for action, time in enumerate(user.data[brand]):
                if time <= 0: continue
                info.append(str(user_counter + brand_counter + action * brand_counter + brand) + ':' + str(time))
        info = ' '.join(info)

        brands_pro = user.train_label_pro
        brands_neg = user.train_label_neg

        if len(brands_neg) < len(brands_pro):
            brands_neg.union(random.sample(
                    brands.difference(brands_pro).difference(brands_neg), 
                len(brands_pro) - len(brands_neg)))
        else:
            brands_neg = random.sample(brands_neg, len(brands_pro))

        for pro, neg in zip(random.sample(brands_pro, len(brands_pro)), 
                random.sample(brands_neg, len(brands_neg))):
            print >> writer, '1', 
            print >> writer, str(user.id) + ':1', 
            print >> writer, str(pro + user_counter) + ':1', 
            print >> writer, info

            print >> writer, '-1', 
            print >> writer, str(user.id) + ':1', 
            print >> writer, str(neg + user_counter) + ':1', 
            print >> writer, info
    writer.close()

    writer = open('data.test', 'w')
    for user in users:
        info = []
        for brand in user.train_label_pro.union(user.train_label_neg):
            for action, time in enumerate(user.data[brand]):
                if time <= 0: continue
                info.append(str(user_counter + brand_counter + action * brand_counter + brand) + ':' + str(time))
        info = ' '.join(info)

        for brand in xrange(brand_counter):
            if brand in user.test_label:
                print >> writer, '1', 
            else:
                print >> writer, '-1', 
            print >> writer, str(user.id) + ':1', 
            print >> writer, str(pro + user_counter) + ':1', 
            print >> writer, info
    writer.close()

    writer = open('meta', 'w')
    for num in xrange(user_counter):
        print >> writer, 0
    for idx in range(5):
        for num in xrange(brand_counter):
            print >> writer, idx + 1
    writer.close()
