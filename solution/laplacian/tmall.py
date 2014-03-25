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
import scipy.linalg
import scipy.sparse.linalg
import scipy.io

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csgraph

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
                    self.train_label.add(brandID)

        total_buy_action = float(self.action_weight[1])
        self.action_weight = [1 / max(self.action_weight[idx] / total_buy_action, 5) for idx, num in enumerate(self.action_weight)]
        self.action_weight[1] = 1
        
        for brand in self.data.keys():
            self.data[brand] = [num * weight for num, weight in zip(self.data[brand], self.action_weight)]

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

    graph = lil_matrix((user_counter + brand_counter, user_counter + brand_counter))
    for user in users:
        for brand in user.train_label:
            graph[brand, user.id + brand_counter] = sum(user.data[brand])
            graph[user.id + brand_counter, brand] = sum(user.data[brand])
    
    l = lil_matrix(csgraph.laplacian(graph, normed=True))
    # l_pinv_base = scipy.linalg.pinv(l.todense())

    # scipy.io.savemat('data.mat', dict(l=l))
    # exit()
    
    # m = brand_counter
    # k = user_counter
    # subgraph = -l[0:m, m:]

    # u_sub, s_sub, v_sub = scipy.linalg.svd(subgraph.todense())
    # print u_sub
    # print s_sub
    # print v_sub.T

    # u_sub, s_sub, vt_sub = scipy.sparse.linalg.svds(subgraph, k=k-1)
    # m1 = 0
    # for num in s_sub:
    #     if not numpy.allclose(num, 1):
    #         break
    #     else:
    #         m1 += 1
    # m2 = k - m1
    # m3 = m - m1 - m2
    # sigma = numpy.diagflat(s_sub[m1:])

    # u = lil_matrix((m + k, m + k))
    # print u_sub
    # print s_sub
    # print vt_sub
    # print m, k
    
    # u[0:m, 0:m] = u_sub
    # u[m:, m:] = v_sub

    # sigma1 = numpy.linalg.inv(numpy.eye(m2) - sigma.dot(sigma))
    # sigma2 = sigma.dot(sigma1)

    # s = lil_matrix((m + k, m + k))
    # if m1 > 0:
    #     s[0:m1, 0:m1] = 0.25 * numpy.eye(m1)
    #     s[0:m1, m:m+m1] = -0.25 * numpy.eye(m1)
    #     s[m:m+m1, 0:m1] = -0.25 * numpy.eye(m1)
    #     s[m:m+m1, m:m+m1] = 0.25 * numpy.eye(m1)
    # s[m1:k, m1:k] = sigma1
    # s[m1:k, m+m1:] = sigma2
    # s[m+m1:, m1:k] = sigma2
    # s[m+m1:, m+m1:] = sigma1
    # s[k:m, k:m] = numpy.eye(m3)

    # s = lil_matrix((m + k, m + k))
    # if m1 > 0:
    #     s[0:m1, 0:m1] = numpy.eye(m1)
    #     s[0:m1, m:m+m1] = -numpy.eye(m1)
    #     s[m:m+m1, 0:m1] = -numpy.eye(m1)
    #     s[m:m+m1, m:m+m1] = numpy.eye(m1)
    # s[:m, :m] = numpy.eye(m)
    # s[m1:k, m+m1:] = -sigma
    # s[m+m1:, m1:k] = -sigma
    # s[m+m1:, m+m1:] = numpy.eye(m2)

    # l_pinv = u.dot(s).dot(u.T)
    l_pinv = scipy.io.loadmat('pinv.mat')['l_pinv']
    print type(l_pinv)
    print l_pinv.shape

    top_n = 1000
    pBands = []
    bBands = []
    hitBands = []
    for user in users:
        bBands.append(len(user.test_label))

        total = 0
        hit = 0
        predicts = [(abs(l_pinv[user.id + brand_counter][brand]) / \
            math.sqrt(abs(l_pinv[user.id + brand_counter][user.id + brand_counter] * \
                l_pinv[brand][brand])), brand) for brand in range(brand_counter)]
        predicts.sort()
        for idx in xrange(min(top_n, len(predicts))):
            if predicts[idx][1] in user.train_label: continue
            if predicts[idx][1] in user.test_label:
                hit += 1
            total += 1
        
        hitBands.append(hit)
        pBands.append(total)

    print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)

    precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
    recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

    print 'All(%d):  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (top_n, precision * 100, recall * 100, f1 * 100)