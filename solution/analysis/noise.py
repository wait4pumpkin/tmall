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
import pylab

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

    def addEdge(self, u, v):
        if u not in self.adjList:
            self.adjList[u] = set()
        if v not in self.adjList:
            self.adjList[v] = set()
        self.adjList[u].add(v)
        self.adjList[v].add(u)



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
            # print 'u', u, self.distances[u]
            for v in self.graph.adjList[u]:
                # print 'v', v
                if v not in self.distances:
                    queue.append(v)
                    self.distances[v] = self.distances[u] + 1
                    if v is end: break
            if end in self.distances: break
            # print '==========================='
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

    # Test if buy before for one user
    # total = 0
    # hit = 0
    # for user in users:
    #     if BASE_MONTH + N_MONTH - 1 not in user.data: continue

    #     for brand, actions in user.data[BASE_MONTH + N_MONTH - 1].items():
    #         if actions[1] < 1: continue
    #         total += 1

    #         for month in range(BASE_MONTH, BASE_MONTH + N_MONTH - 1):
    #             if month not in user.data: continue
    #             if brand not in user.data[month]: continue
    #             if user.data[month][brand][1] > 0: 
    #                 hit += 1
    #                 break
    # print hit, total, '{:.2f}%'.format(float(hit) / total * 100)

    # Test if visit before for one user
    # total = 0
    # hit = 0
    # for user in users:
    #     if BASE_MONTH + N_MONTH - 1 not in user.data: continue

    #     for brand, actions in user.data[BASE_MONTH + N_MONTH - 1].items():
    #         if actions[1] < 1: continue
    #         total += 1

    #         for month in range(BASE_MONTH, BASE_MONTH + N_MONTH - 1):
    #             if month not in user.data: continue
    #             if brand not in user.data[month]: continue
    #             if sum(user.data[month][brand]) > 0: 
    #                 hit += 1
    #                 break
    # print hit, total, '{:.2f}%'.format(float(hit) / total * 100)

    # Test if other user had brought before
    before = set()
    for user in users:
        for month in range(BASE_MONTH, BASE_MONTH + N_MONTH - 1):
            if month not in user.data: continue

            for brand, actions in user.data[month].items():
                if actions[1] < 1: continue
                before.add(brand)
    print len(before)
    total = 0
    hit = 0
    for user in users:
        if BASE_MONTH + N_MONTH - 1 not in user.data: continue

        for brand, actions in user.data[BASE_MONTH + N_MONTH - 1].items():
            if actions[1] < 1: continue
            total += 1
            if brand in before: hit += 1
    print hit, total, '{:.2f}%'.format(float(hit) / total * 100)

    g = Graph()
    for user in users:
        for month in range(BASE_MONTH, BASE_MONTH + N_MONTH - 1):
            if month not in user.data: continue

            for brand, actions in user.data[month].items():
                if actions[1] < 1: continue
                g.addEdge(user.id, brand + user_counter)

    bfs = BFS(g)
    bfs.components.sort(key=lambda e: len(e), reverse=True)
    print [len(comp) for comp in bfs.components]

    print len(users), len(g.adjList), len(bfs.components)

    distances = []
    for user in users:
        if BASE_MONTH + N_MONTH - 1 not in user.data: continue

        for brand, actions in user.data[BASE_MONTH + N_MONTH - 1].items():
            if actions[1] < 1: continue
            distances.append(bfs.distance(user.id, brand + user_counter))

    print Counter(distances)
    print len(distances)


    pBands = 0
    bBands = 0
    hitBands = 0
    for user in users:
        if BASE_MONTH + N_MONTH - 1 not in user.data: continue
        label = set()
        for brand, actions in user.data[BASE_MONTH + N_MONTH - 1].items():
            if actions[1] < 1: continue
            label.add(brand + user_counter)

        hit = 0
        predicts = set()
        if user.id in g.adjList:
            brands = g.adjList[user.id]
            for brand in brands:
                for neighbour in g.adjList[brand]:
                    if neighbour is user.id: continue
                    predicts.update(g.adjList[neighbour])

                    for abc in g.adjList[neighbour]:
                        if bfs.distance(user.id, abc) is 5:
                            print user.id, abc

        bBands += len(label)
        pBands += len(predicts)
        hitBands += len(label.intersection(predicts))

    print hitBands, ' ', pBands, ' ', bBands

    precision = float(hitBands) / pBands if not pBands is 0 else 0
    recall = float(hitBands) / bBands if not bBands is 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

    print 'All:  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (precision * 100, recall * 100, f1 * 100)
