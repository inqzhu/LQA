#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import keras
import copy
import random

class lqa(object):
    
    # 内置优化器，用于在当前参数估计附近作小范围变动，进而计算loss
    _optimizer = tf.keras.optimizers.SGD(learning_rate=0.1) 

    def get_loss(self, _model, _X, _Y):
        # 计算模型给定参数估计下的loss
        Y_pred = _model(_X, training=False)   
        t_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=_Y, y_pred=Y_pred)
        loss = tf.reduce_mean(t_loss)      
        return loss

    def grad_add(self, grads1, grads2):
        # 合并两个grad（两个梯度相加）
        for i in range(len(grads1)):
            grads1[i] = grads1[i] + grads2[i]
        return grads1

    def grad_scale(self, grads, scale):
        # 对梯度每层乘以一个数值 scale
        for i in range(len(grads)):
            grads[i] = grads[i] * scale
        return grads


    def get_ab(self, loss0, model, x, y, grads, delta_try):
        # 计算lqa、BB中共同部分的项
        
        # 计算loss_p（loss(θ+δ_0)）
        self._optimizer.lr = delta_try
        self._optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables)) 
        loss_p = self.get_loss(model, x, y)

        # 计算loss_n（loss(θ-δ_0)）
        self._optimizer.lr = -2*delta_try
        self._optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables)) 
        loss_n = self.get_loss(model, x, y)

        # * 复原模型参数估计
        self._optimizer.lr = delta_try
        self._optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables)) 

        _e = 1e-12
        _term = (loss_p - loss_n) / ( 2.0 * (loss_p + loss_n - 2*loss0) + _e )

        return _term.numpy()

    # LQA方法
    # 尝试的学习率 δ_0
    #@tf.function
    def delta(self, loss0, model, x, y, grads, epoch):
        # 基于当前loss值，计算lqa学习率

        # 设置尝试性学习率，用于产生当前参数估计附近的loss值，进而计算LQA学习率
        delta_try = max(0.05 * (1-epoch*0.01), 0.01)
        # 计算LQA学习率
        delta_lqa = delta_try * self.get_ab(loss0, model, x, y, grads, delta_try)

        # 若计算学习率为负，可能为局部（或近似的局部）非凸，替换为默认学习率
        if delta_lqa < 0:
            delta_lqa = delta_try * 1.05
        
        # 若计算学习率过大，替换为默认学习率
        if delta_lqa > 1:
            delta_lqa = 0.05
            if epoch >= 50:
                delta_lqa = 0.01

        return abs(delta_lqa)