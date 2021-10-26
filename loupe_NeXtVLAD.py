# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Learnable mOdUle for Pooling fEatures (LOUPE)
Contains a collection of models (NetVLAD, NetRVLAD, NetFV and Soft-DBoW)
which enables pooling of a list of features into a single compact 
representation.

Reference:

Learnable pooling method with Context Gating for video classification
Antoine Miech, Ivan Laptev, Josef Sivic

"""


import math
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import utils.tf_util as tf_util
from utils.transform_nets import input_transform_net, feature_transform_net, neural_feature_net
from utils.pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module



class PoolingBaseModel(object):
    """Inherit from this class when implementing new models."""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True, expansion=2, groups=8):
        """Initialize a NetVLAD block.

        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """

        self.feature_size = feature_size            # 特征维度 1024
        self.max_samples = max_samples              # 点数 4096
        self.output_dim = output_dim                # 输出维度 256
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size            # 第二阶段邻域聚合聚类点数 64
        self.expansion = expansion
        self.groups = groups


    def forward(self, reshaped_input):
        raise NotImplementedError("Models should implement the forward pass.")

    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """

        """ 第6步：gating_weights_1，注意维度变化 """
        input_dim = input_layer.get_shape().as_list()[1]        # 这里的维度与NeXtVLAD不一定一致         (18, K*(2D)//G)  (18, K*2*1024//8)

        gating_weights = tf.get_variable("gating_weights",
                                         [input_dim, input_dim],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(input_dim)))

        gates = tf.matmul(input_layer, gating_weights)      # (18, 256) * (256, 256) = (18, 256)
        if self.add_batch_norm:
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="gating_bn")
        else:
            gating_biases = tf.get_variable("gating_biases",
                                            [input_dim],
                                            initializer=tf.random_normal(stddev=1 / math.sqrt(input_dim)))
            gates += gating_biases


        """ 第7步：gating_weights_2，注意维度变化 """


        print("!!!!!!!!!!!!!!!!gates before: !!!!!!!!!!!!!!!!!", gates)
        """ 第8步：gates 与 gating_weights_2相乘得到最终的gates，注意维度变化 """
        ## 少了一步相乘，这一步把维度乘回去了， (18, 256) * (128, 1024) = (18, 1024)
        gates = tf.sigmoid(gates)
        print("!!!!!!!!!!!!!!!!gates after: !!!!!!!!!!!!!!!!!", gates)


        """ 第9步：gates 与 输入activation相乘得到最终的activate，注意维度变化 """
        activation = tf.multiply(input_layer, gates)        # (18, 256) * (18*256) = (18, 256)
        print("!!!!!!!!!!!!!!!!activation after: !!!!!!!!!!!!!!!!!", activation)

        return activation


# Edited based on the original version   继承自上面的PoolingBaseModel
class NetVLAD(PoolingBaseModel):

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True, expansion=2, groups=None):
        super(self.__class__, self).__init__(
            feature_size=feature_size,      # 1024
            max_samples=max_samples,        # 4096
            cluster_size=cluster_size,      # 第二阶段邻域聚合聚类点数 64
            output_dim=output_dim,          # 256
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
            expansion=expansion,
            groups=groups)


    # def forward(self, xyz, reshaped_input, bn_decay):
    def forward(self, reshaped_input):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """

        """
        input = tf.reshape(reshaped_input, [-1,
                                            self.max_samples, self.feature_size])

        # msg grouping
        l1_xyz, l1_points = pointnet_sa_module_msg(xyz, input, 256, [0.1, 0.2, 0.4], [16, 32, 64],
                                                   [[16, 16, 32], [32, 32, 64], [32, 64, 64]], self.is_training,
                                                   bn_decay,
                                                   scope='layer3', use_nchw=True)

        l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None,
                                                  mlp=[256, 512], mlp2=None, group_all=True, is_training=self.is_training,
                                                  bn_decay=bn_decay, scope='layer4')

        print('l2_points:', l2_points)

        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], self.is_training, bn_decay,
                                       scope='fa_layer2')
        l0_points = pointnet_fp_module(xyz, l1_xyz, tf.concat([xyz, input], axis=-1), l1_points, [128, 128],
                                       self.is_training, bn_decay, scope='fa_layer3')

        print('l0_points shape', l0_points)

        net = tf_util.conv1d(l0_points, 1, 1, padding='VALID', bn=True, is_training=self.is_training, scope='fc3',
                             bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dp2')
        net = tf_util.conv1d(net, 1, 1, padding='VALID', activation_fn=None, scope='fc4')

        m = tf.reshape(net, [-1, 1])
        print('m:', m)

        # constrain weights to [0, 1]
        m = tf.nn.sigmoid(m)
        weights = m
        m = tf.tile(m, [1, self.cluster_size])

        print('m:', m)
        """

        """ 第1步：得到reshaped_input_myself 和 cluster_weights, 二者相乘得到 activation """

        reshaped_input = slim.fully_connected(reshaped_input, self.expansion * self.feature_size, activation_fn=None,
                                              weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(reshaped_input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())

        attention = tf.reshape(attention, [-1, self.max_samples * self.groups, 1])      # (18*N, 8)  ==>  (18, N*8, 1)
        feature_size = self.expansion * self.feature_size // self.groups  # 2D//G  (论文中的λN/G)
        print("!!!!!!!!!feature_size: !!!!!!!!!!!!", feature_size)


        #cluster_weights = tf.get_variable("cluster_weights",
        #                                  [self.expansion*self.feature_size, self.groups*self.cluster_size],
        #                                  initializer=tf.random_normal_initializer(
        #                                      stddev=1 / math.sqrt(self.feature_size)))     # (2D, 8*K) = (2D, G*K)
        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion * self.feature_size, self.groups * self.cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )  # (2D, 8*K) = (2D, G*K)

        reshaped_input_1 = tf.reshape(reshaped_input, [-1, self.expansion * self.feature_size])       # (18*4096, 2D) = (18*N, 2D)
        activation = tf.matmul(reshaped_input_1, cluster_weights)     # (73728, 1024)   (1024, 64)   ==>   (73728, 64)  (18*N, G*K)

        # activation = tf.contrib.layers.batch_norm(activation, 
        #         center=True, scale=True, 
        #         is_training=self.is_training,
        #         scope='cluster_bn')

        # activation = slim.batch_norm(
        #       activation,
        #       center=True,
        #       scale=True,
        #       is_training=self.is_training,
        #       scope="cluster_bn")


        """ 第2步：通过activation得到a_sum并且与cluster_weights2相乘得到a """
        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn",
                fused=False)
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            activation += cluster_biases

        activation = tf.reshape(activation, [-1, self.max_samples * self.groups,
                                             self.cluster_size])  # (18*N, 8*K)  ==> (18, N*8, K)
        # activation = tf.nn.softmax(activation)                  # 黄色部分，实现α拟合        (73728, 64)     (18*N, G*K)
        activation = tf.nn.softmax(activation, axis=-1)

        ### activation_crn = tf.multiply(activation, m)         # 新添加的模块

        ###activation = tf.reshape(activation,
        ###                        [-1, self.max_samples, self.cluster_size])  # 这里进行了修改 (73728, 64) ==> (18, 4096, 64)  (N*K)   原有的

        ## ...........  这里少了一步

        activation = tf.multiply(activation, attention)  # △△△ 多了这一步 ☆☆☆     (18, N*8, K) * (18, N*8, 1)  = (18, N*8, K)

        ###################################################### 开始求c ##################################################
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)       # (18, 4096, 64)  ==> (18, 1, 64)       这里就是论文中的ck
        #cluster_weights2 = tf.get_variable("cluster_weights2",
        #                                   [1, feature_size, self.cluster_size],
        #                                   initializer=tf.random_normal_initializer(
        #                                       stddev=1 / math.sqrt(self.feature_size)))    # (1, 1024, 64)     这里的self.feature_size 更改为 feature_size   # (1, 2D//G, K)
        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )  # (1, 2D//G, K)
        a = tf.multiply(a_sum, cluster_weights2)                    # (18, 1024, 64)        # (18, 1, K) * (1, 2D//G, K) = (18, 2D//G, K)
        ################################################################################################################


        """ 第3步：得到最终的vlad """
        activation = tf.transpose(activation, perm=[0, 2, 1])       # (18, 64, 4096)    ==> N*K   (18, K, N)

        reshaped_input_2 = tf.reshape(reshaped_input, [-1, self.max_samples * self.groups, feature_size])  # (73728, 1024)  ==> (18, 4096, 1024)
        vlad = tf.matmul(activation, reshaped_input_2)                # (18, 64, 1024)     (18, K, N) * (18, N, D) = (18, K, D)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])                   # (18, 1024, 64)    (18, D, K)
        vlad = tf.subtract(vlad, a)                                 # 做减法，(18, 1024, 64)     (18, D, K)


        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])    # (18, 1024, 64) = (18, 65536) = (18, D*K)   # (18, K*(2D)//G)   (18, 16384)
        #vlad = tf.nn.l2_normalize(vlad, 1)                          # 这里不一样
        vlad = slim.batch_norm(vlad,
                               center=True,
                               scale=True,
                               is_training=self.is_training,
                               scope="vlad_bn",
                               fused=False)  # △△△ 这里的正则化不同 ☆☆☆


        """ 第4步：dropout层 （后面的注释好像尝试过并且去掉了）"""



        """ 第5步：vlad与hidden1_weights相乘再次得到activation """
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [self.cluster_size * feature_size, self.output_dim],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.cluster_size)))     # (64*1024, 256) = (65536, 256) =(K*D, H)    (K*(2D)//G, 256)  (64*2048//8, 256) = (16384, 256)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hidden1_weights: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!", hidden1_weights)
        ## Tried using dropout
        # vlad=tf.layers.dropout(vlad,rate=0.5,training=self.is_training)
        vlad = tf.matmul(vlad, hidden1_weights)                     # (18, 65536) * (65536, 256) = (18, 256)       (18, K*(2D)//G) * (K*(2D)//G, 256) = (18, 256)
        ## Added a batch norm
        """vlad = tf.contrib.layers.batch_norm(vlad, 
                                          center=True, scale=True, 
                                          is_training=self.is_training,
                                          scope='bn')
        """
        vlad = tf.compat.v1.layers.batch_normalization(vlad,
                                                       center=True, scale=True,
                                                       training=self.is_training,
                                                       name='bn')


        if self.gating:
            vlad = super(self.__class__, self).context_gating(vlad)     # (18, 256)

        return vlad


