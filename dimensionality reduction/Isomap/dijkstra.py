#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:01:46 2017

@author: plz
"""

import numpy as np

inf=999

class Dijkstra(object):

    def __init__(self, connect_array, start_point):
        self.e = connect_array
        self.num = connect_array.shape[0]
        self.dis = np.zeros((self.num))
        for j in range(self.num):
            inf = 999
            self.dis[j] = self.e[start_point][j] if self.e[
                start_point][j] != inf else inf
        self.book = np.zeros((self.num))
        self.book[start_point] = 1

    def find_min_dis_id(self):
        min_dis = inf
        min_dis_id = 0
        for i in range(self.num):
            if self.book[i] == 0:
                if self.dis[i] < min_dis:
                    min_dis = self.dis[i]
                    min_dis_id = i
        self.book[min_dis_id] = 1
        return min_dis_id

    def renew_connect_array(self, min_dis_id):
        for i in range(self.num):
            if self.e[min_dis_id][i] < inf:
                if self.dis[i] > self.dis[min_dis_id] + self.e[min_dis_id][i]:
                    self.dis[i] = self.dis[min_dis_id] + self.e[min_dis_id][i]

    def find_shortest_path(self):
        for i in range(self.num - 1):
            min_dis_id = self.find_min_dis_id()
            self.renew_connect_array(min_dis_id)
            if np.sum(self.book) == self.num:
                break
        print("final dis is:{}".format(self.dis))
        return self.dis


def build_connect_array():

    points_num = input("how many points totally:")
    edges_num = input("how many edges totally:")
    connect_array = np.zeros((points_num, points_num))
    for i in range(edges_num):
        start_id = input("start_id:")
        end_id = input("end_id:")
        weight = input("weight:")
        connect_array[start_id][end_id] = weight

    for i in range(points_num):
        for j in range(points_num):
            if connect_array[i][j] == 0:
                if i != j:
                    connect_array[i][j] = inf
    return connect_array


if __name__ == '__main__':
    inf = 999
    connect_array = build_connect_array()
    instance = Dijkstra(connect_array, start_point=0)
    instance.find_shortest_path()
