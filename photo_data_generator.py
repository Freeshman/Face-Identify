#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:43:43 2018

@author: hu-tom
"""

# -*-coding:utf-8-*-
import numpy
#import theano
from PIL import Image
from pylab import hstack
import os
#import theano.tensor as T
import random

def dataresize(path):
    # test path
    path_t ="./te_data"
    datas = []
    train_x= []
    train_y= []
    valid_x= []
    valid_y= []
    test_x= []
    test_y= []
    for dirs in os.listdir(path):
        # print dirs
        for filename in os.listdir(os.path.join(path,dirs)):
            imgpath =os.path.join(os.path.join(path,dirs),filename)
            img = Image.open(imgpath)
            img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = numpy.asarray(img,dtype='float64')/256.
    
            tmp = img.reshape(1, hight*width)[0]
            if dirs=='0':
                #print("dirs=0")
                dirs2bin=[1,0]
            else:
                #print("dirs=1")
                dirs2bin=[0,1]
            tmp =hstack((dirs2bin,tmp))  # 在此将标签加在数据的前面。
    
            datas.append(tmp)
       # datas.append(img.reshape(1, hight*width)[0])
        #在此处取出第一行的数据否则在后面的转换的过程中会出现叠加的情况，在成在转换成矩阵时宝类型转换的错误
    #将数据打乱顺序
    random.shuffle(datas)
    random.shuffle(datas)
    random.shuffle(datas)
    # 将数据和标签进行分离
    label=[]
    for num in range(len(datas)):
        label.append((datas[num])[:2])
        datas[num] =(datas[num])[2:]
        #将数据的标签项去掉
    tests = []
        # #读取测试集
    for dirs in os.listdir(path_t):
        for filename in os.listdir(os.path.join(path_t,dirs)):
            imgpath =os.path.join(os.path.join(path_t,dirs),filename)
            img = Image.open(imgpath)
            img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = numpy.asarray(img,dtype='float64')/256.
            tmp = img.reshape(1, hight*width)[0]
            # 在此如果不是取出[0]的话在后面会发现其实其是一个多维的数据的叠加，
            # 在后面使用theano中的cnn在调用时会出现数据的异常（转换的异常），
            # 在此是跟原始的mnist的数据集的形式做了比较修改才发现的。。。
            if dirs=='0':
                #print("dirs=0")
                dirs2bin=[1,0]
            else:
                #print("dirs=1")
                dirs2bin=[0,1]
            tmp =hstack((dirs2bin,tmp))
            tests.append(tmp)
    #将数据打乱顺序
    random.shuffle(tests)
    random.shuffle(tests)
    random.shuffle(tests)
    #  将数据和标签进行分离
    label_t=[]
    for num in range(len(tests)):
        label_t.append((tests[num])[:2])
        tests[num] =(tests[num])[2:]
        #将数据的标签项去掉
        '''    将数据进行打乱，拆分成train test valid    '''
    for num in range(len(label)):
        train_x.append(datas[num])
        train_y.append(label[num])
    
    for num in range(len(tests)):
        if num%2==0:
            valid_x.append(tests[num])
            valid_y.append(label_t[num])
        if num%2==1:
            test_x.append(tests[num])
            test_y.append(label_t[num])
    train_x=numpy.asarray(train_x,dtype='float64')
    train_y=numpy.asarray(train_y,dtype='int64')
    valid_x=numpy.asarray(valid_x,dtype='float64')
    valid_y=numpy.asarray(valid_y,dtype='int64')
    test_x=numpy.asarray(test_x,dtype='float64')
    test_y=numpy.asarray(test_y,dtype='int64')
    
    rval = [(train_x, train_y), (valid_x, valid_y),(test_x, test_y)]
    return rval
rva=dataresize("./tr_data")