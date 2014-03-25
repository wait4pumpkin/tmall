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


def PersonalRank(G, alpha, root, max_step):
    rank = dict()
    rank = {x:0.0 for x in G.keys()}
    rank[root] = 1.0
    for k in range(max_step):
        tmp = {x:0.0 for x in G.keys()}
        for i,ri in G.items():
            for j,wij in ri.items():
                if j not in tmp: tmp[j] = 0.0
                tmp[j] += alpha * rank[i] / (len(ri)*1.0)
                if j == root: tmp[j] += 1.0 - alpha
        rank = tmp
        # print(k, rank)
    return rank


class Graph:
    def __init__(self):
        self.G = dict()
    
    def addEdge(self, p, q):
        if p not in self.G: self.G[p] = dict()
        if q not in self.G: self.G[q] = dict()
        self.G[p][q] = 1
        self.G[q][p] = 1

    def getGraphMatrix(self):
        return self.G

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

    graph = Graph()
    for user in users:
        for brand in user.train_label:
            if user.data[brand][1] > 0:
                graph.addEdge(user.id, brand + 1000)
    
    G = graph.getGraphMatrix()

    top_n = 2000
    pBands = []
    bBands = []
    hitBands = []
    for user in users:
        bBands.append(len(user.test_label))

        rank = dict()
        result = sorted(PersonalRank(G, 0.9, user.id, 20).items(), key=lambda e: e[1], reverse=True)
        hit = 0
        total = 0
        for brand, prob in result:
            if total >= top_n: break

            if brand < 1000: continue
            if brand in user.train_label: continue

            total += 1
            if brand in user.test_label: 
                hit += 1
        
        hitBands.append(hit)
        pBands.append(total)

    print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)

    precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
    recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

    print 'All(%d):  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (top_n, precision * 100, recall * 100, f1 * 100)
