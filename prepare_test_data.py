'''
	Pre-processing: prepare_data in LPD-Net
	generate KNN neighborhoods and calculate feature as the feature matrix of point
	Reference: LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis, ICCV 2019

	author: Chuanzhe Suo(suo_ivy@foxmail.com)
	created: 10/26/18
'''

# !usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import multiprocessing as multiproc
from copy import deepcopy
import glog as logger
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
import math
import errno


def calculate_features_old(pointcloud, nbrs_index, eigens):
    ### calculate handcraft feature with eigens and statistics data

    # features using eigens
    eig3d = eigens['eigens'][:3]
    eig2d = eigens['eigens'][3:5]
    vetors = eigens['vectors']
    # 3d
    C_ = eig3d[2] / (eig3d.sum())
    O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)), 1.0 / 3)
    L_ = (eig3d[0] - eig3d[1]) / eig3d[0]
    E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum()
    D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())
    # 2d
    S_2 = eig2d.sum()
    L_2 = eig2d[1] / eig2d[0]
    # features using statistics data
    neighborhood = pointcloud[nbrs_index]
    nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
    dZ_ = nbr_dz.max()
    vZ_ = np.var(nbr_dz)
    V_ = vetors[2][2]

    features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_])
    return features


def calculate_features_(pointcloud, nbrs_index, eigens_, vectors_):
    ### calculate handcraft feature with eigens and statistics data

    # features using eigens
    eig3d = eigens_[:3]
    eig2d = eigens_[3:5]

    # 3d
    C_ = eig3d[2] / (eig3d.sum())
    O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)), 1.0 / 3)
    L_ = (eig3d[0] - eig3d[1]) / eig3d[0]
    E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum()
	# P_ = (eig3d[1] - eig3d[2]) / eig3d[0]
	# S_ = eig3d[2] / eig3d[0]
	# A_ = (eig3d[0] - eig3d[2]) / eig3d[0]
	# X_ = eig3d.sum()
    D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())
    # 2d
    S_2 = eig2d.sum()
    L_2 = eig2d[1] / eig2d[0]
    # features using statistics data
    neighborhood = pointcloud[nbrs_index]
    nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
    dZ_ = nbr_dz.max()
    vZ_ = np.var(nbr_dz)
    V_ = vectors_[2][2]

    features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_])  # ([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    return features


def calculate_features(pointcloud, nbrs_index, eigens_, vectors_, distance_, raster_size):
    # calculate handcraft feature with eigens and statistics data

    point_ID_max = pointcloud.shape[0]  # 4096原始点云数量
    # 取出原始点云的xyz各个分量
    X_vals = np.array([pointcloud[:, 0]]).T
    ## print('X_vals: ', X_vals.shape)
    # X_vals11 = np.array([pointcloud[:, 0]])
    # print('X_vals11: ', X_vals11.shape)
    # Y_vals = pointcloud[:, 1]
    # Z_vals = pointcloud[:, 2]
    Y_vals = np.array([pointcloud[:, 1]]).T
    Z_vals = np.array([pointcloud[:, 2]]).T
    X = X_vals - np.min(X_vals) * np.ones([point_ID_max, 1])
    Y = Y_vals - np.min(Y_vals) * np.ones([point_ID_max, 1])
    X_new = np.floor(X / raster_size) + 1
    Y_new = np.floor(Y / raster_size) + 1
    # print('X.shape: ', X.shape)
    # print('X_new: ', X_new)

    radius_kNN = distance_[len(nbrs_index) - 1] + 1e-6  # 3D            18

    # get size of observed area
    min_X = np.min(X_new)
    max_X = np.max(X_new)
    min_Y = np.min(Y_new)
    max_Y = np.max(Y_new)
    r_acc = max_Y - min_Y + 2  # ????
    c_acc = max_X - min_X + 2  # ????

    # accumulate
    # Acc = np.zeros([int(r_acc), int(c_acc)])
    # for i in range(0, point_ID_max):
    #     Acc[int(Y_new[i, 0]), int(X_new[i, 0])] = Acc[int(Y_new[i]), int(X_new[i])] + 1

    # return a suitable vector representation
    # frequency_acc_map = np.zeros([point_ID_max, 1])
    h_max = np.zeros([point_ID_max, 1])
    h_min = np.zeros([point_ID_max, 1])
    std_z = np.zeros([point_ID_max, 1])

    # use another loop  :-(  for getting accumulation map based 2D features
    # for i in range(0, point_ID_max):
    bound = np.logical_and(X_new == X_new[0], Y_new == Y_new[0])
    r = np.array(np.where(bound == True))
    r = r[0, :]
    h_max = np.max(Z_vals[r])
    h_min = np.min(Z_vals[r])
    # frequency_acc_map[i, 0] = Acc[int(Y_new[i, 0]), int(X_new[i, 0])]
    std_z = np.std(Z_vals[r])  # 19

    # height difference in the respective 2D bins (compare to cylindrical 3D neighborhood -> e.g. [Mallet et al., 2011])
    delta_z = h_max - h_min  # 20

    """ 描述的都是以当前3D点维中心 的 局部点云特征 """
    # features using eigens
    eig3d = eigens_[:3]  # 补充10维特征中的 前三维   3D特征
    eig2d = eigens_[3:5]  # 补充10维特征中的 中间两维 2D特征
    # print('eig3d before', eig3d)
    # print('len(eig3d)', len(eig3d))
    sum_EVs = np.sum(eig3d)
    sum_EVs_2D = np.sum(eig2d)
    eig3d = 1.0 * eig3d / sum_EVs
    eig2d = 1.0 * eig2d / sum_EVs_2D
    # print('eigen3d after: ', eig3d)

    # 3d 三维特征
    C_ = eig3d[2] / (eig3d.sum())  # 1 change of curvature    1
    O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)),
                  1.0 / 3)  # 2 全方差  这个全方差与原参考论文不同  lambda4 = pow(evalue1 * evalue2 * evalue3, 1 / 3.0)
    L_ = (eig3d[0] - eig3d[1]) / eig3d[0]  # 3                       3
    E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum()  # 4
    P_ = (eig3d[1] - eig3d[2]) / eig3d[0]
    S_ = eig3d[2] / eig3d[0]
    A_ = (eig3d[0] - eig3d[2]) / eig3d[0]
    X_ = eig3d.sum()  # 三维特征和                         8

    D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())  # 5                    9

    # 2d  二维特征
    S_2 = eig2d.sum()  # 6 2D散射性                           10
    L_2 = eig2d[1] / eig2d[0]  # 7                         11


    # features using statistics data
    neighborhood = pointcloud[nbrs_index]
    nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
    dZ_ = nbr_dz.max()  # 9 △Zi_max                  12
    vZ_ = np.var(nbr_dz)  # △Zi_max                    13

    dist_x = neighborhood[:, 0] - np.kron(np.ones([int(nbrs_index.shape[0]), 1]), X_vals[nbrs_index[0]])[:, 0]
    # dist_X = X_vals[nbrs_index, 0] - np.kron(np.ones([int(nbrs_index.shape[0]), 1]), X_vals[nbrs_index[0]])[:, 0]
    ## print('dist_x: ', dist_x)
    ## print('len(dist_x): ', len(dist_x))
    # print('dist_X: ', dist_X)
    # print('len(dist_X): ', len(dist_X))
    ## print('neighborhood[:, 0]: ', neighborhood[:, 0])
    # print('X_vals[nbrs_index, 0]: ', X_vals[nbrs_index, 0])
    ## print('len(neighborhood[:, 0]): ', len(neighborhood[:, 0]))
    ## print('neighborhood[:, 0].shape: ', neighborhood[:, 0].shape)
    # test = np.kron(np.ones([int(nbrs_index.shape[0])+1, 1]), X_vals[nbrs_index[0]])[:, 0]
    # print('test: ', test)
    # print('len(test): ', len(test))
    dist_y = neighborhood[:, 1] - np.kron(np.ones([int(nbrs_index.shape[0]), 1]), Y_vals[nbrs_index[0]])[:, 0]
    dist_2D = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
    radius_kNN_2D = np.max(dist_2D) + 1e-6  # 15 2D半径           14
    density_2D = 1.0 * (nbrs_index.shape[0] + 1) / (np.pi * np.power(radius_kNN_2D, 2))  # 16        15
    # print('dist_2D: ', dist_2D)
    # print('radius_kNN_2D: ', radius_kNN_2D)
    # print('density_2D: ', density_2D)

    # Z1_  = pointcloud[nbrs_index][:, 2]
    ## Z_  = neighborhood[:, 2]
    Z_ = Z_vals[nbrs_index[0], 0]  # 16
    # print('Z_: ', Z_)
    # print('len(Z_): ', len(Z_))
    # print('Z1_: ', Z1_)
    # print('len(Z1_): ', len(Z1_))

    V_ = vectors_[2][2]  #   17

    features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_])  # ([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    # features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_, Z_, radius_kNN])  # ([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    # features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_, Z_])

    # features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, S_, A_, Z_, dZ_, vZ_, V_, density_2D])
    features_ = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, S_, A_, Z_, dZ_, vZ_, V_, density_2D])  #
    ### features = np.asarray([C_, O_, L_, E_, D_, density_2D, L_2, dZ_, vZ_, V_])
    features_15 = np.asarray(
        [L_, A_, E_, C_, O_, D_, vZ_, dZ_, L_2, V_, radius_kNN_2D, eig3d[0], eig3d[2], eig2d[0], eig2d[1]])  # 15维

    # print('features.shape: ', features.shape)

    return features


def calculate_entropy(eigen):
    L_ = (eigen[0] - eigen[1]) / eigen[0]
    P_ = (eigen[1] - eigen[2]) / eigen[0]
    S_ = eigen[2] / eigen[0]
    Entropy = -L_ * np.log(L_) - P_ * np.log(P_) - S_ * np.log(S_)
    return Entropy


def calculate_entropy_array(eigen):
    L_ = (eigen[:, 0] - eigen[:, 1]) / eigen[:, 0]
    P_ = (eigen[:, 1] - eigen[:, 2]) / eigen[:, 0]
    S_ = eigen[:, 2] / eigen[:, 0]
    Entropy = -L_ * np.log(L_) - P_ * np.log(P_) - S_ * np.log(S_)
    return Entropy


def covariation_eigenvalue(neighborhood_index, pointcloud):
    ### calculate covariation and eigenvalue of 3D and 2D
    # prepare neighborhood
    neighborhoods = pointcloud[neighborhood_index]

    # 3D cov and eigen by matrix
    Ex = np.average(neighborhoods, axis=1)
    Ex = np.reshape(np.tile(Ex, [neighborhoods.shape[1]]), neighborhoods.shape)
    P = neighborhoods - Ex
    cov_ = np.matmul(P.transpose((0, 2, 1)), P) / (neighborhoods.shape[1] - 1)
    eigen_, vec_ = np.linalg.eig(cov_)
    indices = np.argsort(eigen_)
    indices = indices[:, ::-1]
    pcs_num_ = eigen_.shape[0]
    indx = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind = indices + indx * 3
    vec_ind = np.reshape(eig_ind * 3, [-1, 1]) + np.full((pcs_num_ * 3, 3), [0, 1, 2])
    vec_ind = np.reshape(vec_ind, [-1, 3, 3])
    eigen3d_ = np.take(eigen_, eig_ind)
    vectors_ = np.take(vec_, vec_ind)
    entropy_ = calculate_entropy_array(eigen3d_)

    # 2D cov and eigen
    cov2d_ = cov_[:, :2, :2]
    eigen2d, vec_2d = np.linalg.eig(cov2d_)
    indices = np.argsort(eigen2d)
    indices = indices[:, ::-1]

    pcs_num_ = eigen2d.shape[0]
    indx = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind = indices + indx * 2
    eigen2d_ = np.take(eigen2d, eig_ind)

    eigens_ = np.append(eigen3d_, eigen2d_, axis=1)

    return cov_, entropy_, eigens_, vectors_


def build_neighbors_NN(k, pointcloud):
    ### using KNN NearestNeighbors cluster according k
    nbrs = NearestNeighbors(n_neighbors=k).fit(pointcloud)
    distances, indices = nbrs.kneighbors(pointcloud)
    covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, pointcloud)
    neighbors = {}
    neighbors['k'] = k
    neighbors['indices'] = indices
    neighbors['covs'] = covs
    neighbors['entropy'] = entropy
    neighbors['eigens_'] = eigens_
    neighbors['vectors_'] = vectors_
    neighbors['distances'] = distances
    # logger.info("KNN:{}".format(k))
    return neighbors


def build_neighbors_KDT(k, args):
    ### using KNN KDTree cluster according k
    nbrs = KDTree(args.pointcloud)
    distances, indices = nbrs.query(args.pointcloud, k=k)
    covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, args)
    neighbors = {}
    neighbors['k'] = k
    neighbors['indices'] = indices
    neighbors['covs'] = covs
    neighbors['entropy'] = entropy
    neighbors['eigens_'] = eigens_
    neighbors['vectors_'] = vectors_
    # logger.info("KNN:{}".format(k))
    return neighbors


def prepare_file(pointcloud_file, args):
    # logger.info("Processing pointcloud file:{}".format(pointcloud_file))
    # load pointcloud file
    pointcloud = np.fromfile(pointcloud_file, dtype=np.float64)
    pointcloud = np.reshape(pointcloud, (pointcloud.shape[0] // 3, 3))

    #
    k_nbrs = []
    for k in args.cluster_number:
        k_nbr = build_neighbors_NN(k, pointcloud)
        k_nbrs.append(k_nbr)

    # get argmin k according E, different points may have different k
    k_entropys = []
    for k_nbr in k_nbrs:
        k_entropys.append(k_nbr['entropy'])
    argmink_ind = np.argmin(np.asarray(k_entropys), axis=0)


    raster_size = 0.5
    points_feature = []
    for index in range(pointcloud.shape[0]):
        ### per point
        neighborhood = k_nbrs[argmink_ind[index]]['indices'][index]
        eigens_ = k_nbrs[argmink_ind[index]]['eigens_'][index]
        vectors_ = k_nbrs[argmink_ind[index]]['vectors_'][index]
        distance_ = k_nbrs[argmink_ind[index]]['distances'][index]


        # calculate point feature
        # feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_)
        feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_, distance_, raster_size)
        points_feature.append(feature)
    points_feature = np.asarray(points_feature)

    # save to point feature folders and bin files
    feature_cloud = np.append(pointcloud, points_feature, axis=1)
    pointfile_path, pointfile_name = os.path.split(pointcloud_file)
    filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
    feature_cloud.tofile(filepath)


# build KDTree and store for the knn query
# kdt = KDTree(pointcloud, leaf_size=50)
# treepath = os.path.splitext(filepath)[0] + '.pickle'
# with open(treepath, 'wb') as handel:
# 	pickle.dump(kdt, handel)

# logger.info("Feature cloud file saved:{}".format(filepath))


def prepare_dataset(args):
    # load folder csv file
    df_locations = pd.read_csv(os.path.join(args.dataset_path, args.runs_folder, args.pointcloud_folder, args.filename),
                               sep=',')
    df_locations[
        'timestamp'] = args.base_path + args.runs_folder + args.pointcloud_folder + '/' + args.pointcloud_fols + \
                       df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    # creat feature_cloud folder
    pointcloud_files = df_locations['file'].tolist()
    featurecloud_path = os.path.join(args.dataset_path, args.runs_folder, args.pointcloud_folder,
                                     args.featurecloud_fols)
    if not os.path.exists(featurecloud_path):
        try:
            os.makedirs(featurecloud_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    # multiprocessing pool to parallel process pointcloud_files
    pool = multiproc.Pool(args.bin_core_num)
    for file in pointcloud_files:
        file = os.path.join(args.BASE_DIR, file)
        pointfile_path, pointfile_name = os.path.split(file)
        filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
        if not os.path.exists(filepath):
            pool.apply_async(prepare_file, (file, deepcopy(args)))
        else:
            a = 1
        # logger.info("{} exists, skipped".format(file))
    pool.close()
    pool.join()
    logger.info("finish {} processing".format(args.pointcloud_folder))


def run_all_processes(all_p):
    try:
        for p in all_p:
            p.start()
        for p in all_p:
            p.join()
    except KeyboardInterrupt:
        for p in all_p:
            if p.is_alive():
                p.terminate()
            p.join()
        exit(-1)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "./benchmark_datasets/"
## runs_folder = "oxford/"
runs_folder = "inhouse_datasets/"
## filename = "pointcloud_locations_20m.csv"
filename = "pointcloud_centroids_25.csv"
# 点云
## pointcloud_fols = "pointcloud_20m/"
pointcloud_fols = "pointcloud_25m_25/"
# 人工特征
featurecloud_fols = "featurecloud_20m/"


def main(args):
    # prepare dataset folders
    args.BASE_DIR = BASE_DIR
    args.base_path = base_path
    args.dataset_path = os.path.join(BASE_DIR, base_path)
    args.runs_folder = runs_folder
    args.pointcloud_fols = pointcloud_fols
    args.featurecloud_fols = featurecloud_fols
    args.filename = filename

    # prepare KNN cluster number k
    args.cluster_number = []
    for ind in range(((args.k_end - args.k_start) // args.k_step) + 1):
        args.cluster_number.append(args.k_start + ind * args.k_step)

    # All runs are used for training (both full and partial)
    ### oxford
    ###all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
    ###index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    ###folders = []
    ###for index in index_list:
    ###    folders.append(all_folders[index])

    ### U.S.
    ###all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
    ###uni_index = range(10, 15)
    ###folders = []
    ###for index in uni_index:
    ###    folders.append(all_folders[index])

    ### R.A.
    ###all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
    ###res_index = range(5, 10)
    ###folders = []
    ###for index in res_index:
    ###    folders.append(all_folders[index])

    ### B.D.
    all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
    bus_index = range(5)
    folders = []
    for index in bus_index:
        folders.append(all_folders[index])

    # multiprocessing dataset folder
    all_p = []
    for folder in folders:
        args.pointcloud_folder = folder
        print(args.pointcloud_folder + " start")
        prepare_dataset(args)
    # all_p.append(multiproc.Process(target=prepare_dataset, args=(deepcopy(args),)))
    # run_all_processes(all_p)

    logger.info("Dataset preparation Finised")


if __name__ == '__main__':
    parse = argparse.ArgumentParser(sys.argv[0])

    parse.add_argument('--k_start', type=int, default=20,
                       help="KNN cluster k range start point")
    parse.add_argument('--k_end', type=int, default=100,
                       help="KNN cluster k range end point")
    parse.add_argument('--k_step', type=int, default=10,
                       help="KNN cluster k range step")

    parse.add_argument('--bin_core_num', type=int, default=4, help="Parallel process file Pool core num")

    args = parse.parse_args(sys.argv[1:])
    main(args)
