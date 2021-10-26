"""
    LPD-Net Model: FN-SF-VLAD
    Feature Network + FN-Parallel structure (P) + Series-FC structure (SF)
    # Thanks to Mikaela Angelina Uy, modified from PointNetVLAD
    # Reference: LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis, ICCV 2019
    author: suo_ivy
    created: 10/26/18
"""
import os
import sys
import time
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# sys.path.append(os.path.join(MODELS_DIR, '../utils'))
import utils.tf_util as tf_util
from utils.transform_nets import input_transform_net, feature_transform_net, neural_feature_net
from utils.pointnet_util import pointnet_sa_module, pointnet_sa_module_msg


# Taken from Charles Qi's pointnet code
MODELS_DIR = os.path.dirname(__file__)
sys.path.append(MODELS_DIR)

# Adopted from Antoine Meich  采纳自安托万·梅奇
import loupe as lp

dimension = 13

# △△△ 换维度 △△△
def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, dimension))
    return pointclouds_pl

# 将max_pooling 池化改为 attentive pooling
# @staticmethod
def att_pooling(feature_set, d_out, name, is_training):
    batch_size = tf.shape(feature_set)[0]
    num_points = tf.shape(feature_set)[1]
    num_neigh = tf.shape(feature_set)[2]
    d = feature_set.get_shape()[3].value
    f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
    att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
    att_scores = tf.nn.softmax(att_activation, axis=1)
    f_agg = f_reshaped * att_scores
    f_agg = tf.reduce_sum(f_agg, axis=1)
    f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
    f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
    return f_agg



# Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None):
    """LPD-Net: FNSF,    INPUT is batch_num_queries X num_pointclouds_per_query X num_points_per_pointcloud X 13,
                         OUTPUT batch_num_queries X num_pointclouds_per_query X output_dim """
    print("very begin point_cloud: ", point_cloud)                      # (1, 18, 4096, 13)
    batch_num_queries = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_points = point_cloud.get_shape()[2].value

    CLUSTER_SIZE = 64
    OUTPUT_DIM = 256
    k = 25
    point_cloud = tf.reshape(point_cloud, [batch_num_queries * num_pointclouds_per_query, num_points, 13])


    point_cloud, feature_cloud = tf.split(point_cloud, [3, 10], 2)

    # 
    ###with tf.variable_scope('transform_net1') as sc:
    ###     input_transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)  # (18, 3, 3)


    ###point_cloud_transformed = tf.matmul(point_cloud, input_transform)           # (18, 4096, 3)
    point_cloud_transformed = point_cloud

    # Neural Network to learn neighborhood features
    # feature_cloud = neural_feature_net(point_cloud, is_training, bn_decay, knn_k=20, F=10)


    point_cloud_input = tf.concat([point_cloud_transformed, feature_cloud], 2)  # (18, 4096, 13)

    point_cloud_input = tf.expand_dims(point_cloud_input, -1)               # (18, 4096, 13, 1)

    net = tf_util.conv2d(point_cloud_input, 64, [1, 13],
                         padding='VALID', stride=[1, 1],
                         is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)                  # (18, 4096, 1, 64)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    print('~~~second net~~~: ', net.shape)                                  # (18, 4096, 1, 64)


    ### with tf.variable_scope('transform_net2') as sc:
    ###    feature_transform = feature_transform_net(net, is_training, bn_decay, K=64) # (18, 64, 64)
    ### feature_transform = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)     # (18, 4096, 64)
    feature_transform = tf.squeeze(net, axis=[2])                           # (18, 4096, 64)

    feature_transform = tf.expand_dims(feature_transform, axis=-2)                  ###  (18, 4096, 1, 64)
    point_cloud = tf.expand_dims(point_cloud, axis=-2)                              ###  (18, 4096, 1, 3)


    # Serial structure                Series-FC structure (SF)
    #                                                       FC: Feature space / Cartesian space
    # Dynamic Graph cnn for feature space
    with tf.variable_scope('DGfeature') as sc:
        adj_matrix = tf_util.pairwise_distance(feature_transform)               # (18, 4096, 4096)
        nn_idx = tf_util.knn(adj_matrix, k=k)                                   # (18, 4096, 20)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)        # (18, 4096, 20, 64+64)
        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='dgmlp1', bn_decay=bn_decay)                 # (18, 4096, 20, 64)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)                     # (18, 4096, 1, 64)
        # net = att_pooling(net, 64, 'dgmlp1'+'att_pooling_2', is_training)       # (18, 4096, 1, 64)
        print("!!!!!!!!!!!!!!After net att_pooling: ", net)
        net1 = net

        adj_matrix = tf_util.pairwise_distance(net)                             # (18, 4096, 4096)
        nn_idx = tf_util.knn(adj_matrix, k=k)                                   # (18, 4096, 20)
        net = tf.concat([feature_transform, net1], axis=-1)                     # (18, 4096, 1, 128)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)        # (18, 4096, 1, 128+128)
        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='dgmlp2', bn_decay=bn_decay)                 # (18, 4096, 20, 64)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)                       # (18, 4096, 1, 64)
        # net = att_pooling(net, 64, 'dgmlp2' + 'att_pooling_2', is_training)
        net2 = net


    # Spatial Neighborhood fusion for cartesian space
    with tf.variable_scope('SNfeature') as sc:
        adj_matrix = tf_util.pairwise_distance(point_cloud)                 # (18, 4096, 4096)
        nn_idx = tf_util.knn(adj_matrix, k=k)                               # (18, 4096, 20)
        ###idx_ = tf.range(batch_num_queries * num_pointclouds_per_query) * num_points     # (1*18)*4096
        ###idx_ = tf.reshape(idx_, [batch_num_queries * num_pointclouds_per_query, 1, 1])  # (18*4096, 1, 1)

        ###feature_cloud = tf.reshape(net, [-1, 64])                           # (73728, 64)
        ###edge_feature  = tf.gather(feature_cloud, nn_idx + idx_)             # (18, 4096, 20, 64)
        net = tf.concat([point_cloud, net1, net2], axis=-1)                 # (18, 4096, 1, 131)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)    # (18, 4096, 1, 131+131)
        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='snmlp1', bn_decay=bn_decay)             # (18, 4096, 20, 64)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)                   # (18, 4096,  1, 64)
        # net = att_pooling(net, 64, 'snmlp1' + 'att_pooling_2', is_training)
        net3 = net

        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        net = tf.concat([point_cloud, net1, net2, net3], axis=-1)           # (18, 4096, 20, 195)  这里可不可以只有net3，与上面的net2对应
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)    # (18, 4096, 20, 390)
        print("edge_feature: ", edge_feature)
        net = tf_util.conv2d(edge_feature, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='snmlp2', bn_decay=bn_decay)             # (18, 4096, 20, 128)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)                   # (18, 4096,  1, 128)
        # net = att_pooling(net, 128, 'snmlp2' + 'att_pooling_2', is_training)
        net4 = net


    # input: B*N*1*323
    # net:   B*N*1*1024
    net = tf_util.conv2d(tf.concat([point_cloud, net1, net2, net3,
                                    net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)                  # (18, 4096,  1, 1024)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)                  # (18, 4096,  1, 128)
    # MLP for fusion
    ###net = tf_util.conv2d(net, 64, [1, 1],
    ###                     padding='VALID', stride=[1, 1],
    ###                     is_training=is_training,
    ###                    scope='conv3', bn_decay=bn_decay)
    ###net = tf_util.conv2d(net, 128, [1, 1],
    ###                     padding='VALID', stride=[1, 1],
    ###                     is_training=is_training,
    ###                     scope='conv4', bn_decay=bn_decay)
    ###net = tf_util.conv2d(net, 1024, [1, 1],
    ###                     padding='VALID', stride=[1, 1],
    ###                     is_training=is_training,
    ###                     scope='conv5', bn_decay=bn_decay)
    ###point_wise_feature = net                # (18, 4096, 1, 1024)


    ###NetVLAD = lp.NetVLAD(feature_size=1024, max_samples=num_points, cluster_size=CLUSTER_SIZE,
    ###                     output_dim=OUTPUT_DIM, gating=True, add_batch_norm=True,
    ###                     is_training=is_training, expansion=2, groups=4)
    NetVLAD = lp.NetVLAD(feature_size=1024, max_samples=num_points, cluster_size=CLUSTER_SIZE,
                         output_dim=OUTPUT_DIM, gating=True, add_batch_norm=True,
                         is_training=is_training)

    net = tf.reshape(net, [-1, 1024])       #    (18, 4096, 1, 1024) ==> (73728, 1024)
    net = tf.nn.l2_normalize(net, 1)
    output = NetVLAD.forward(net)
    print(output)                           # shape=(18, 256)


    # normalize to have norm 1
    output = tf.nn.l2_normalize(output, 1)
    output = tf.reshape(output, [batch_num_queries, num_pointclouds_per_query, OUTPUT_DIM])


    return output



def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        # batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1, int(num_pos), 1])  # shape num_pos x output_dim
        best_pos = tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies), 2), 1)
        # best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos


# Losses for PointNetVLAD ###########
# Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
# △△△
def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    # ''', end_points, reg_weight=0.001):
    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_sum(
        tf.maximum(tf.add(m, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))),
                   tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss

# △△△
def softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    ones = tf.fill([int(batch), int(num_neg)], 1.0)
    soft_loss = tf.reduce_mean(tf.reduce_sum(
        tf.log(tf.exp(tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))) + 1.0),
        1))
    return soft_loss

# △△△
def lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    ones = tf.fill([int(batch), int(num_neg)], 1.0)
    soft_loss = tf.reduce_mean(tf.reduce_max(
        tf.log(tf.exp(tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))) + 1.0),
        1))
    return soft_loss

# △△△
def quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss = softmargin_loss(q_vec, pos_vecs, neg_vecs)

    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = soft_loss + second_loss

    return total_loss

# △△△
def lazy_quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss = lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs)

    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = soft_loss + second_loss

    return total_loss

# △△△
def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss = triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss

    return total_loss


# Lazy variant  △△△
def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_max(
        tf.maximum(tf.add(m, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))),
                   tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss


def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss

    return total_loss
