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
            queue = []
            queue.append(u)
            
            while len(queue) > 0:
                u = queue.pop(0)
                for v in graph.adjList[u]:
                    if v not in isVisited:
                        queue.append(v)
                        isVisited.add(v)
                        comp.add(v)
            self.components.append(comp)


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

    g = Graph()
    for user in users:
        for brand in user.train_label:
            g.addEdge(user.id, brand + user_counter)

    bfs = BFS(g)
    sorted(bfs.components, key=lambda e: len(e), reverse=True)
    print len(bfs.components)
    print bfs.components[1]
    # users = [user for user in users if user.id is in bfs.components[0]]

    # clean = bfs.components[1].union(bfs.components[2])
    clean = bfs.components[1]
    user_table_inv = { v: k for k, v in user_table.items() }
    brand_table_inv = { v: k for k, v in brand_table.items() }
    for user in clean:
        if user < user_counter: 
            print user_table_inv[user]
        else:
            print brand_table_inv[user - user_counter]
