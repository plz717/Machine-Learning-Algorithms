#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:01:46 2017

@author: plz
"""


import numpy as np
from dijkstra import Dijkstra
from mds import Mds
from pca import load_data


inf = 999
int=long

class Isomap(object):
    """Input:
        Data: array of samples with shape (num_samples,sample_dimension).
        k: how many neareset neighbors to found in the k-nearest algorithm.
        d_: the low dimension expected.
    """

    def __init__(self, Data, k, d_):
        self.Data = Data
        self.sample_num = Data.shape[0]
        print("self.sample_num is:{}".format(self.sample_num))
        self.k = k
        self.d_ = d_
        self.eucli_dis_array = np.zeros((self.sample_num, self.sample_num))
        print("eucli_dis_array shape is:{}".format(self.eucli_dis_array.shape))
        self.dijkstra_dis_array = np.array((self.sample_num, self.sample_num))
        
    def search_k_mins_of_a_point(self,point,i):
        """search the k-nearest neighbors of a point.
        Input:
        i:index of the input point
        point:the coordinate of the point.
        Return:
        index_sorted:the indexes of the k-nearest neighbors of the input point
        """
        dis_list = []
        for j in range(self.sample_num):
            if i != j:
                dis_ij = np.sum((self.Data[i] - self.Data[j])**2)
                dis_ij = np.sqrt(dis_ij)
                dis_list.append(dis_ij)
        # return an array of index of items from smallest to largest
        index_sorted = np.argsort(np.array(dis_list))
        return index_sorted, dis_list

    def search_k_mins(self):
        """search the k-nearest neighbors of every sample,  calculate the
         Euclide distances between them and set the other distances to inf.
         Return of this function is to be sent to the Dijkstra algorithm.
        """
        for i in range(self.sample_num):
            # return an array of index of items from smallest to largest
            index_sorted,dis_list = self.search_k_mins_of_a_point(self.Data[i],i)
            for j in range(self.sample_num):
                if j in index_sorted[-self.d_:]:
                    self.eucli_dis_array[i][j] = dis_list[j]
                elif i == j:
                    self.eucli_dis_array[i][j] = 0
                else:
                    self.eucli_dis_array[i][j] = inf
        return index_sorted, self.eucli_dis_array

    def cal_dijkstra_dis_array(self):
        dijkstra_dis_list = []
        for i in range(self.sample_num):
            instance = Dijkstra(self.eucli_dis_array,i)
            dijkstra_dis_array = instance.find_shortest_path()
            dijkstra_dis_list.append(dijkstra_dis_array)
        self.dijkstra_dis_array = np.array(dijkstra_dis_list).reshape(
            self.sample_num, self.sample_num)
        return self.dijkstra_dis_array

    def obtain_mds_projection(self):
        instance = Mds(self.dijkstra_dis_array, self.d_)
        instance.calcu_B()
        low_dimension_projection = instance.eigen_decomp(self.d_)
        return low_dimension_projection


if __name__ == '__main__':
    train_faces = load_data('./dataset', 1, 10)
    d_ = 100
    instance = Isomap(train_faces, 3, d_)
    instance.search_k_mins()
    instance.cal_dijkstra_dis_array()
    instance.obtain_mds_projection()
