#! /usr/bin/env python
# coding=utf-8

"""
模型训练方法
使用命令行命令：
python test_lqa.py
返回：
     train_losses - 训练集上每个epoch的loss
     train_accs - 测试集上每个epoch的ACC
     test_losses - 测试集上每个epoch的loss
     test_accs - 测试集上每个epoch的ACC
     run_times - 每个epoch训练时间
并输出记录为json文件 res_resnet18-cifar10-LQA
"""
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


import keras

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import json
import time
import copy
import random
import sys


# 定义的resnet模型
from net_models import *
# 获取数据
from net_dataset import *
# 引入lqa
import lqa


def log_refresh(filename):
    # 更新日志
    with open(filename, 'w') as f:
        f.write('Start \n')

def log(filename, info):
    # 记录信息
    print(info)
    with open(filename, 'a') as f:
        f.write(info+'\n')
        

def test(_model, batch_size, X0_test, Y0_test):
    # 测试在训练集上的loss、ACC
    global test_loss
    global test_accuracy
    
    # reset
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    # 先测试训练集上
    K = int(len(X0_test) / batch_size)    
    
    for k in range(K):
        _X,_Y = X0_test[k*batch_size:(k+1)*batch_size,],Y0_test[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        # 在batch上计算loss、acc
        test_step(_model, test_loss, test_accuracy, _X, _Y)
   
    # 最终的loss和acc
    return test_loss.result().numpy(), test_accuracy.result().numpy()

def record(model, train_data, test_data, train_loss, train_acc, test_loss, test_acc):
    # 记录当前模型在测试集训练集上的loss与acc
    _loss, _acc = test(model, 2000, train_data[0], train_data[1])
    _loss_test, _acc_test = test(model, 2000, test_data[0], test_data[1])
    train_loss.append(_loss)
    train_acc.append(_acc)
    test_loss.append(_loss_test)
    test_acc.append(_acc_test)


# 给定一个batch的数据，利用batch数据单次迭代训练方法
@tf.function
def train_step_grads(model, _X, _Y):
    with tf.GradientTape() as tape:
        Y_pred = model(_X, training=True)                                                         
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=_Y, y_pred=Y_pred)    
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads
    
# 给定一个batch的数据，测试loss、acc的方法   
@tf.function
def test_step(model, test_loss, test_accuracy, _X, _Y):
    Y_pred = model(_X, training=False)
    t_loss = tf.keras.losses.sparse_categorical_crossentropy(_Y, Y_pred)
    test_loss(t_loss)
    test_accuracy(_Y, Y_pred)
    
def main(flows, X0, X1, Y0, Y1, log_file):
    # 模型训练主方法

    global test_loss
    global test_accuracy
    test_loss.reset_states()
    test_accuracy.reset_states()


    # 定义模型
    model = ResNet18()
    
    # 声明LQA对象实例
    _lqa = lqa.lqa()
    
    # 记录初始的loss、acc
    losses = []
    accs = []
    losses_test = []
    accs_test = []
    run_times = []
    # 记录初始loss、acc
    t0 = time.time()
    record(model, [X0, Y0], [X1, Y1], losses, accs, losses_test, accs_test)
    t1 = time.time()
    log(log_file, '* Initial. Loss: %.7f. ACC: %.3f. Loss_test: %.7f. ACC_test: %.3f. Time cost: %.2fs' % 
          (losses[-1], accs[-1], losses_test[-1], accs_test[-1], t1-t0))
    
    # 定义优化器
    optimizer = tf.keras.optimizers.SGD(lr=0.1)

    # batch大小
    batch_size = 64
    # batch个数数
    K = int(len(X0)/batch_size)

    # 迭代训练
    for epoch in range(200):

        t0 = time.time()
   
        tc0 = time.process_time()
        for k in range(K):
            # 获取单个batch的数据
            for flow in flows:
                _X, _Y = flow
                break
            _X, _Y = flow
            
            # 获取当前梯度
            loss, grads = train_step_grads(model, _X, _Y)

            # 计算基于LQA的学习率
            delta_lqa = _lqa.delta(loss, model, _X, _Y, grads, epoch)
            
            # 将LQA学习率输入给优化器
            optimizer.lr = delta_lqa
            
            # 调用优化器更新模型参数估计
            optimizer.apply_gradients(zip(grads, model.trainable_variables))   
    

        tc1 = time.process_time()
        t1 = time.time()

        # 记录loss、acc
        record(model, [X0, Y0], [X1, Y1], losses, accs, losses_test, accs_test)
        run_times.append(tc1-tc0)
        log(log_file, '* [LQA]Epoch %d. Loss: %.7f. ACC: %.3f. Loss_test: %.7f. ACC_test: %.3f. Time cost: %.2fs (%.2fs)' % 
              (epoch, losses[-1], accs[-1], losses_test[-1], accs_test[-1], t1-t0, tc1-tc0))
        
    return losses, accs, losses_test, accs_test, run_times


def change_float(x):
    new_x = []
    for i in range(len(x)):
        new_x.append(float(x[i]))
    return new_x

if __name__ == '__main__': 
    # 使用的GPU序号
    gpu_id = "1"
    # 指定GPU进行计算
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # 使用的GPU序号
    print("Num GPUs Availiable: ",len(tf.config.experimental.list_physical_devices('GPU')))
    
    # 日志文件
    pid = os.getpid()
    log_file = 'log_resnet18-cifar10-LQA-%s' % (gpu_id)
    res_file = 'res_resnet18-cifar10-LQA-%s' % (gpu_id)
    log_refresh(log_file)
    log(log_file, '[LQA] - gpu_id: %s | pid: %d' % (gpu_id, pid))
    
    # 获取cifar10数据
    _t0 = time.time()
    batch_size = 64
    X0, X1, Y0, Y1, flows = get_dataflow(batch_size)
    
    # loss和acc计算
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 开始模型训练
    # 返回依次为： 
    # train_losses - 训练集上每个epoch的loss
    # train_accs - 测试集上每个epoch的ACC
    # test_losses - 测试集上每个epoch的loss
    # test_accs - 测试集上每个epoch的ACC
    # run_times - 每个epoch训练时间
    train_losses, train_accs, test_losses, test_accs, run_times = main(flows, X0, X1, Y0, Y1, log_file)
    
    
    # 写为json文件保存loss和ACC记录
    with open(res_file, 'w') as f:
        f.write(json.dumps(
            {
             'train_loss': change_float(train_losses),
             'train_acc': change_float(train_accs),
             'test_loss': change_float(test_losses),
             'test_acc': change_float(test_accs),
             'run_time': change_float(run_times)
            }
        
        ))
    _t1 = time.time()

    log(log_file, 'OK. %.2fh' % ((_t1-_t0) / 3600.0))
