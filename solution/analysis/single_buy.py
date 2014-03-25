#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys
import time
import math
import Queue

from collections import Counter

import numpy
import pylab as pl
import matplotlib.font_manager

from scipy import sparse
from scipy.sparse import lil_matrix

from sklearn import svm
from sklearn import linear_model, cross_validation

N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

class User(object):
    def __init__(self, id, info):
        self.id = id;
        self.brands = info.keys()
        self.data = dict()
        for brand_id in self.brands:
            brand = info[brand_id]

            for month, day, action in brand:
                if month not in self.data:
                    self.data[month] = dict()

                if brand_id not in self.data[month]:
                    self.data[month][brand_id] = [0, 0, 0, 0]

                self.data[month][brand_id][action] += 1

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))


class Graph(object):
    def __init__(self):
        self.adjList = dict()

    def add_edge(self, u, v):
        if u not in self.adjList:
            self.adjList[u] = set()
        if v not in self.adjList:
            self.adjList[v] = set()
        self.adjList[u].add(v)
        self.adjList[v].add(u)

    def adj_matrix(self):
        result = lil_matrix((len(self.adjList), len(self.adjList)))
        for u in self.adjList:
            for v in self.adjList[u]:
                result[u][v] = 1


class BFS(object):
    def __init__(self, graph):
        self.graph = graph
        self.components = []
        isVisited = set()
        for u in graph.adjList.keys():
            if u in isVisited: continue

            comp = set()
            comp.add(u)
            isVisited.add(u)
            queue = [u]
            
            while len(queue) > 0:
                u = queue.pop(0)
                for v in graph.adjList[u]:
                    if v not in isVisited:
                        queue.append(v)
                        isVisited.add(v)
                        comp.add(v)
            self.components.append(comp)

    def distance(self, start, end):
        if start not in self.graph.adjList or \
            end not in self.graph.adjList: return -1

        self.distances = dict()
        queue = [start]
        self.distances[start] = 0
        while len(queue) > 0:
            u = queue.pop(0)
            print 'u', u, self.distances[u]
            for v in self.graph.adjList[u]:
                print 'v', v
                if v not in self.distances:
                    queue.append(v)
                    self.distances[v] = self.distances[u] + 1
                    if v is end: break
            if end in self.distances: break
            print '==========================='
        return self.distances[end] if end in self.distances else -1

# class Dijkstra(object):
#     def __init(self, graph, start):
#         self.distances = [sys.maxint for num in range(len(graph.adjList))]
#         pq = Queue.PriorityQueue()
#         self.distances[start] = 0
#         pq.put(start, 0)
#         while not pq.empty():
#             u, dist = pq.get()
#             if dist > self.distances[u]: continue
#             for v in graph.adjList[u]:
#                 if self.distances[v] > self.distances[u] + 1:
#                     self.distances


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

    print 'Start: %s' % time.clock()

    g = Graph()
    data_train = []
    for user in users:
        for month in range(BASE_MONTH, BASE_MONTH + N_MONTH - 1):
            if month not in user.data: continue

            for brand, actions in user.data[month].items():
                if actions[1] < 1: continue
                g.add_edge(user.id, brand + user_counter)
                data_train.append([user.id, brand + user_counter])

    data_train = numpy.asarray(data_train)
    print data_train.shape, user_counter, brand_counter

    prob = 0.9
    M = lil_matrix((user_counter + brand_counter, user_counter + brand_counter))
    for i in xrange(user_counter + brand_counter):
        for j in xrange(user_counter + brand_counter):
            if i in g.adjList: M[i, j] = prob / len(g.adjList[i])

    Q = 1 - prob

    n_step = 10
    k_half = M.dot(Q).dot(M.T * Q)
    for n in xrange(1, n_step):
        print n
        k_half += M.dot(k[n-1]).dot(M.T)

    # print k_half.todense()
    print k_half.shape

    print 'Graph: %s' % time.clock()

    def graph_kernel(x, y):
        gram = numpy.empty((x.shape[0], y.shape[0]))
        for i, u in enumerate(x):
            for j, v in enumerate(y):
                gram[i, j] = k_half[u[0], v[0]] * k_half[u[1], v[1]]
            # for j in xrange(len(data_train)):
            #     u = data_train[i]
            #     v = data_train[j]
            #     gram[i, j] = k_half[u[0], v[0]] * k_half[u[1], v[1]]
        # return np.dot(x, y.T)
        # print gram
        return gram

    clf = svm.OneClassSVM(kernel=graph_kernel)
    clf.fit(data_train)
    print Counter(clf.predict(data_train))

    print 'SVM: %s' % time.clock()







    # gram = numpy.empty((len(data_train), len(data_train)))
    # for i in xrange(len(data_train)):
    #     for j in xrange(len(data_train)):
    #         u = data_train[i]
    #         v = data_train[j]
    #         gram[i, j] = k_half[u[0], v[0]] * k_half[u[1], v[1]]

    # clf = svm.OneClassSVM(kernel='precomputed')
    # clf.fit(gram)
    # print clf

    # # user = users[0]
    # # data_test = numpy.asarray([[0, 1]])
    # print clf.predict(gram)
    # # svm.libsvm.predict_proba
    
    # X = numpy.array([[0, 0], [1, 1]])
    # y = [0, 1]
    # clf = svm.SVC(kernel='precomputed')

    # gram = numpy.dot(X, X.T)
    # print clf.fit(gram, y) 

    # print clf.predict(gram)
