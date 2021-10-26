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
import csv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
base_path = "./benchmark_datasets/"
runs_folder = "oxford/"
filename = "pointcloud_locations_20m_10overlap.csv"

pointcloud_fols = "pointcloud_20m_10overlap/"
featurecloud_fols = "featurecloud_20m_10overlap/"


def calculate_features_old(pointcloud, nbrs_index, eigens):
    # calculate handcraft feature with eigens and statistics data

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


def calculate_features(pointcloud, nbrs_index, eigens_, vectors_, distance_, raster_size):
    # calculate handcraft feature with eigens and statistics data

    point_ID_max = pointcloud.shape[0]
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

    radius_kNN = distance_[len(nbrs_index) - 1] + 1e-6                  # 3D            18


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
    std_z = np.std(Z_vals[r])                                                           # 19

    # height difference in the respective 2D bins (compare to cylindrical 3D neighborhood -> e.g. [Mallet et al., 2011])
    delta_z = h_max - h_min                                                             # 20


    # features using eigens
    eig3d = eigens_[:3]
    eig2d = eigens_[3:5]
    # print('eig3d before', eig3d)
    # print('len(eig3d)', len(eig3d))
    sum_EVs = np.sum(eig3d)
    sum_EVs_2D = np.sum(eig2d)
    eig3d = 1.0 * eig3d / sum_EVs
    eig2d = 1.0 * eig2d / sum_EVs_2D
    #print('eigen3d after: ', eig3d)


    C_ = eig3d[2] / (eig3d.sum())
    O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)), 1.0 / 3)
    L_ = (eig3d[0] - eig3d[1]) / eig3d[0]
    E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum()
    P_ = (eig3d[1] - eig3d[2]) / eig3d[0]
    S_ = eig3d[2] / eig3d[0]
    A_ = (eig3d[0] - eig3d[2]) / eig3d[0]
    X_ = eig3d.sum()

    D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())

    # 2d
    S_2 = eig2d.sum()
    L_2 = eig2d[1] / eig2d[0]


    # features using statistics data
    neighborhood = pointcloud[nbrs_index]
    nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
    dZ_ = nbr_dz.max()
    vZ_ = np.var(nbr_dz)

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
    radius_kNN_2D = np.max(dist_2D) + 1e-6
    density_2D = 1.0 * (nbrs_index.shape[0] + 1) / (np.pi * np.power(radius_kNN_2D, 2))
    #print('dist_2D: ', dist_2D)
    #print('radius_kNN_2D: ', radius_kNN_2D)
    #print('density_2D: ', density_2D)


    #Z1_  = pointcloud[nbrs_index][:, 2]
    ## Z_  = neighborhood[:, 2]
    Z_ = Z_vals[nbrs_index[0], 0]                                                       # 16
    # print('Z_: ', Z_)
    # print('len(Z_): ', len(Z_))
    # print('Z1_: ', Z1_)
    # print('len(Z1_): ', len(Z1_))


    V_ = vectors_[2][2]

    features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_])  # ([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    # features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_, Z_, radius_kNN])  # ([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    # features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, dZ_, vZ_, V_, Z_])

    # features = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, S_, A_, Z_, dZ_, vZ_, V_, density_2D])
    features_ = np.asarray([C_, O_, L_, E_, D_, S_2, L_2, S_, A_, Z_, dZ_, vZ_, V_, density_2D])     #
    ### features = np.asarray([C_, O_, L_, E_, D_, density_2D, L_2, dZ_, vZ_, V_])
    features_15 = np.asarray([L_, A_, E_, C_, O_, D_, vZ_, dZ_, L_2, V_, radius_kNN_2D, eig3d[0], eig3d[2], eig2d[0], eig2d[1]])

    # print('features.shape: ', features.shape)

    return features


def calculate_entropy(eigen):
    L_ = (eigen[0] - eigen[1]) / eigen[0]
    P_ = (eigen[1] - eigen[2]) / eigen[0]
    S_ = eigen[2] / eigen[0]
    Entropy = -L_ * np.log(L_) - P_ * np.log(P_) - S_ * np.log(S_)
    return Entropy


def calculate_entropy_array(eigen):
    # EVs = 1.0 * eigen / sum(eigen)
    # L =
    # eigen = 1.0 * eigen / sum(eigen)
    L_ = (eigen[:, 0] - eigen[:, 1]) / eigen[:, 0]
    P_ = (eigen[:, 1] - eigen[:, 2]) / eigen[:, 0]
    S_ =  eigen[:, 2] / eigen[:, 0]
    Entropy = -L_ * np.log(L_) - P_ * np.log(P_) - S_ * np.log(S_)
    return Entropy


def covariation_eigenvalue(neighborhood_index, args):
    # calculate covariation and eigenvalue of 3D and 2D
    # prepare neighborhood
    neighborhoods = args.pointcloud[neighborhood_index]
    # print('neighborhoods.shape: ', neighborhoods.shape)
    # print('neighborhoods.shape[1]: ', neighborhoods.shape[1])


    # 3D cov and eigen by matrix
    Ex = np.average(neighborhoods, axis=1)
    Ex = np.reshape(np.tile(Ex, [neighborhoods.shape[1]]), neighborhoods.shape)
    P  = neighborhoods - Ex
    cov_ = np.matmul(P.transpose((0, 2, 1)), P) / (neighborhoods.shape[1] - 1)
    # print('cov_: ', cov_)
    # print('len(cov_): ', len(cov_))


    eigen_, vec_ = np.linalg.eig(cov_)
    indices  = np.argsort(eigen_)
    indices  = indices[:, ::-1]

    # print('indices: ', indices)
    # print('len(indices): ', len(indices))
    """
    indices:  [[1 2 0]
               [1 2 0]
               [1 2 0]
               ...
               [1 2 0]
               [1 0 2]
               [0 2 1]]
    len(indices):  4096
    """

    pcs_num_ = eigen_.shape[0]
    indx     = np.reshape(np.arange(pcs_num_), [-1, 1])
    # print('indx: ', indx)
    # print('len(indx): ', len(indx))
    """ 
    indx:  [[   0]
            [   1]
            [   2]
             ...
            [4093]
            [4094]
            [4095]]
            len(indx):  4096       * 1
    """

    eig_ind  = indices + indx * 3
    # print('eig_ind: ', eig_ind)
    # print('len(eig_ind): ', len(eig_ind))
    """  
    eig_ind:  [[    0     1     2]        =  [  0   1   2  ]  +  [ 0 * 3 ]
               [    3     4     5]        =  [  0   1   2  ]  +  [ 1 * 3 ]
               [    6     7     8]        =  
               ...
               [12281 12280 12279]
               [12282 12284 12283]
               [12285 12287 12286]]
    len(eig_ind):  4096         * 3
    """
    vec_ind  = np.reshape(eig_ind * 3, [-1, 1]) + np.full((pcs_num_ * 3, 3), [0, 1, 2])
    # print('vec_ind.origin: ', vec_ind.shape)
    # print('len(vec_ind.origin): ', len(vec_ind))
    # print('np.full((pcs_num_ * 3, 3), [0, 1, 2]): ', np.full((pcs_num_ * 3, 3), [0, 1, 2]))
    # print('len(np.full((pcs_num_ * 3, 3), [0, 1, 2])): ', len(np.full((pcs_num_ * 3, 3), [0, 1, 2])))
    """
    vec_ind.origin: [[    0     1     2]          =    [0]  +   [   0   1   2  ]
                     [    6     7     8]          =    [6]  +   [   0   1   2  ]
                     [    3     4     5]          =    [3]  +   [   0   1   2  ]
                     ...
                     [36855 36856 36857]
                     [36861 36862 36863]
                     [36858 36859 36860]]
    len(vec_ind.origin):  12288
    np.full((pcs_num_ * 3, 3), [0, 1, 2]):  
                    [[0 1 2]
                     [0 1 2]
                     [0 1 2]
                     ...
                     [0 1 2]
                     [0 1 2]
                     [0 1 2]]
    len(np.full((pcs_num_ * 3, 3), [0, 1, 2])):  12288
    """
    vec_ind  = np.reshape(vec_ind, [-1, 3, 3])
    # print('vec_ind: ', vec_ind)
    # print('len(vec_ind): ', len(vec_ind))             # 回退到4096*3*3
    """
    vec_ind:  [[[    0     1     2] 
                [    3     4     5]
                [    6     7     8]]
        
               [[    9    10    11]
                [   12    13    14]
                [   15    16    17]]
                ...
               [[36855 36856 36857]   
                [36861 36862 36863]
                [36858 36859 36860]]]
    len(vec_ind):  4096         * 3 * 3
    """
    eigen3d_ = np.take(eigen_, eig_ind)
    # print('eigen3d_: ', eigen3d_)
    # print('len(eigen3d_): ', len(eigen3d_))
    """
    eigen3d_:  [[2.93607743e-02 1.52603867e-02 3.15769222e-03]
                [1.83774088e-03 1.18676149e-03 5.63060862e-04]
                [2.93347026e-02 1.49005268e-02 2.88457068e-03]
                ...
                [8.47681560e-04 7.86129274e-04 1.33812231e-05]
                [1.11460833e-03 4.37772255e-04 1.78745881e-04]
                [8.56013225e-04 5.02086801e-04 1.37112385e-04]]
    len(eigen3d_):  4096        * 3
    """

    vectors_ = np.take(vec_,   vec_ind)
    # print('vec_: ', vec_)
    # print('len(vec_): ', len(vec_))
    # print('vectors_: ', vectors_)
    # print('len(vectors_): ', len(vectors_))
    """  
    vec_:  [[[-0.61139214  0.64329536 -0.46083699]         [ 0
             [ 0.65322209  0.73898327  0.16493827]           1
             [-0.44665485  0.20018694  0.8720233 ]]          2 ]
            
            [[ 0.24818246 -0.89545047  0.3695591 ]
             [-0.81082733 -0.40076824 -0.42654878]
             [-0.53006086  0.19378669  0.82551936]]
             ...
            [[ 0.98236395 -0.14724553 -0.11523809]
             [-0.1346498  -0.98472497  0.11039098]
             [ 0.1297324   0.09292733  0.98718489]]]
    len(vec_):  4096
    vectors_:  [[[-0.61139214  0.64329536 -0.46083699]     [ 0
                 [-0.44665485  0.20018694  0.8720233 ]       2
                 [ 0.65322209  0.73898327  0.16493827]]      1 ]
    
                [[-0.81082733 -0.40076824 -0.42654878]
                 [-0.53006086  0.19378669  0.82551936]
                 [ 0.24818246 -0.89545047  0.3695591 ]]
                 ...
                [[ 0.98236395 -0.14724553 -0.11523809]
                 [ 0.1297324   0.09292733  0.98718489]
                 [-0.1346498  -0.98472497  0.11039098]]]
    len(vectors_):  4096
    """
    entropy_ = calculate_entropy_array(eigen3d_)


    # 2D cov and eigen
    cov2d_   = cov_[:, :2, :2]
    eigen2d, vec_2d = np.linalg.eig(cov2d_)
    indices  = np.argsort(eigen2d)
    indices  = indices[:, ::-1]

    pcs_num_ = eigen2d.shape[0]
    indx     = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind  = indices + indx * 2
    eigen2d_ = np.take(eigen2d, eig_ind)

    eigens_  = np.append(eigen3d_, eigen2d_, axis=1)
    # print('eigens_: ', eigens_)
    # print('len(eigens_): ', len(eigens_))
    """
    eigens_:  [[0.00124353 0.00100129 0.00021696 0.00100788 0.00021757]
               [0.00123442 0.00100745 0.00022247 0.00101793 0.00022445]
               [0.00123442 0.00100745 0.00022247 0.00101793 0.00022445]
               ...
               [0.01003382 0.00298235 0.00050072 0.00711221 0.00109644]
               [0.00096965 0.00050044 0.00016672 0.00078933 0.00023554]
               [0.00072316 0.0004405  0.00026194 0.00063725 0.00030785]]
    len(eigens_):  4096
    """

    return cov_, entropy_, eigens_, vectors_


def build_neighbors_NN(k, args):
    # using KNN NearestNeighbors cluster according k
    nbrs = NearestNeighbors(n_neighbors=k).fit(args.pointcloud)
    distances, indices = nbrs.kneighbors(args.pointcloud)
    # print('distances:', distances)
    # print('len(distances):', len(distances))
    """
        [[0.         0.00377083 0.02116227 ... 0.13322983 0.13449705 0.1385918 ]
         [0.         0.00853505 0.01428977 ... 0.04876285 0.04877476 0.05002452]
         ...
         [0.         0.0039868  0.01440182 ... 0.04310825 0.04381387 0.04808963]
         [0.         0.00973217 0.01115481 ... 0.03023292 0.03167693 0.03503766]]
        len(distances): 4096
    """
    covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, args)
    neighbors = {}
    neighbors['k']         = k
    neighbors['indices']   = indices
    neighbors['covs']      = covs
    neighbors['entropy']   = entropy
    neighbors['eigens_']   = eigens_
    neighbors['vectors_']  = vectors_
    neighbors['distances'] = distances
    logger.info("KNN:{}".format(k))
    return neighbors


def build_neighbors_KDT(k, args):
    # using KNN KDTree cluster according k
    nbrs = KDTree(args.pointcloud)
    distances, indices = nbrs.query(args.pointcloud, k=k)
    covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, args)
    neighbors = {}
    neighbors['k']        = k
    neighbors['indices']  = indices
    neighbors['covs']     = covs
    neighbors['entropy']  = entropy
    neighbors['eigens_']  = eigens_
    neighbors['vectors_'] = vectors_
    logger.info("KNN:{}".format(k))
    return neighbors


def prepare_file(pointcloud_file, args):
    # Parallel process pointcloud files
    # load pointcloud file      点云加载，变为N×3行列格式
    pointcloud = np.fromfile(pointcloud_file, dtype=np.float64)
    pointcloud = np.reshape(pointcloud, (pointcloud.shape[0] // 3, 3))
    args.pointcloud = pointcloud

    # prepare KNN cluster number k
    cluster_number = []
    for ind in range(((args.k_end - args.k_start) // args.k_step) + 1):
        cluster_number.append(args.k_start + ind * args.k_step)
    # print("cluster_number:", cluster_number)        # cluster_number: [20, 30, 40, 50, 60, 70, 80, 90, 100]


    k_nbrs = []
    for k in cluster_number:
        k_nbr = build_neighbors_NN(k, args)
        k_nbrs.append(k_nbr)

    # print('len(k_nbr): ', len(k_nbr))
    # print('k_nbr: ', k_nbr)
    # return;

    logger.info("Processing pointcloud file:{}".format(pointcloud_file))
    # multiprocessing pool to parallel cluster pointcloud
    # pool = multiproc.Pool(len(cluster_number))
    # build_neighbors_func = partial(build_neighbors, args=deepcopy(args))
    # k_nbrs = pool.map(build_neighbors_func, cluster_number)
    # pool.close()
    # pool.join()

# ================================================================================

    # get argmin k according E, different points may have different k
    k_entropys = []
    for k_nbr in k_nbrs:
        k_entropys.append(k_nbr['entropy'])
    # print('k_entropys: ', k_entropys)
    # print('len(k_entropys)', len(k_entropys))
    """
    k_entropys:  [array([0.60572515, 0.75214906, 0.53681936, ..., 0.6933713 , 0.40027387,0.30938291]), 
                  array([0.66228989, 0.88394138, 0.81495463, ..., 0.60256185, 0.34173076,0.41542797]), 
                  array([0.7901194 , 0.90208596, 0.91585912, ..., 0.59108632, 0.39905651,0.69711053]), 
                  array([0.78070991, 0.94590139, 1.01010286, ..., 0.55083555, 0.35251558,0.56071454]), 
                  array([0.8121596 , 0.97297046, 1.04518914, ..., 0.61314235, 0.29813672,0.70194436]), 
                  array([0.87037548, 0.99693005, 0.98659956, ..., 0.60277552, 0.18200434,0.56466811]), 
                  array([0.8358722 , 0.96834927, 1.0028928 , ..., 0.70582815, 0.56534536,0.59555046]), 
                  array([0.79953146, 0.9100608 , 1.05661841, ..., 0.8383317 , 0.54457751,0.64295721]), 
                  array([0.82496195, 0.93082751, 1.04186762, ..., 0.8627731 , 0.63966064,0.82549187])]
    len(k_entropys.shape[1])： 9
    """
    argmink_ind = np.argmin(np.asarray(k_entropys), axis=0)
    # print('argmink_ind: ', argmink_ind)
    # print('len(argmink_ind): ', len(argmink_ind))
    """
    argmink_ind:  [2 2 0 ... 5 2 6]
    len(argmink_ind):  4096
    """

    raster_size = 0.5
    points_feature = []
    for index in range(pointcloud.shape[0]):
        # per point
        neighborhood = k_nbrs[argmink_ind[index]]['indices'][index]
        eigens_      = k_nbrs[argmink_ind[index]]['eigens_'][index]
        vectors_     = k_nbrs[argmink_ind[index]]['vectors_'][index]
        distance_    = k_nbrs[argmink_ind[index]]['distances'][index]

        # calculate point feature
        # feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_)
        feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_, distance_, raster_size)
        points_feature.append(feature)
    points_feature = np.asarray(points_feature)


    # save to point feature folders and bin files
    feature_cloud = np.append(pointcloud, points_feature, axis=1)
    # print('feature_cloud: ', feature_cloud.shape)
    pointfile_path, pointfile_name = os.path.split(pointcloud_file)
    filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
    feature_cloud.tofile(filepath)


    # build KDTree and store fot the knn query
    # kdt = KDTree(pointcloud, leaf_size=50)
    # treepath = os.path.splitext(filepath)[0] + '.pickle'
    # with open(treepath, 'wb') as handle:
    #     pickle.dump(kdt, handle)

    logger.info("Feature cloud file saved:{}".format(filepath))



def prepare_dataset(args):
    # Parallel process dataset folders
    # Initialize pandas DataFrame
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_locations = pd.read_csv(os.path.join(args.dataset_path, args.runs_folder, args.pointcloud_folder, args.filename),
                               sep=',')

    # args.runs_folder = "oxford/"
    # args.pointcloud_folder = "2015-04-17-09-06-25"
    # args.pointcloud_fols = "pointcloud_20m_10overlap/"
    df_locations[
        'timestamp'] = args.base_path + args.runs_folder + args.pointcloud_folder + '/' + args.pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    # creat feature_cloud folder
    featurecloud_path = os.path.join(args.dataset_path, args.runs_folder, args.pointcloud_folder,
                                     args.featurecloud_fols)
    if not os.path.exists(featurecloud_path):
        try:
            os.makedirs(featurecloud_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    pointcloud_files = df_locations['file'].tolist()
    # print(len(pointcloud_files))


    # multiprocessing pool to parallel process pointcloud_files
    pool = multiproc.Pool(args.bin_core_num)
    for file in pointcloud_files:
        file = os.path.join(args.BASE_DIR, file)
        pointfile_path, pointfile_name = os.path.split(file)
        filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
        # print('filepath:', filepath)            # /home/ubuntu/PointCloud/LPD-Net/./benchmark_datasets/oxford/2015-11-10-14-15-57/featurecloud_20m_10overlap/1447165160762755.bin

        if not os.path.exists(filepath):
            pool.apply_async(prepare_file, (file, deepcopy(args)))
        else:
            logger.info("{} exists, skipped".format(file))


    pool.close()
    logger.info("Cloud folder processing:{}".format(args.pointcloud_folder))
    pool.join()
    logger.info("end folder processing")



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



def main(args):
    # prepare dataset folders
    args.BASE_DIR = BASE_DIR
    args.base_path = base_path
    args.dataset_path = os.path.join(BASE_DIR, base_path)
    args.runs_folder = runs_folder
    args.pointcloud_fols = pointcloud_fols
    args.featurecloud_fols = featurecloud_fols
    args.filename = filename


    # All runs are used for training (both full and partial)
    all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
    index_list = range(len(all_folders))        #  - 1
    folders = []
    print("Number of runs: " + str(len(index_list)))
    print(all_folders)
    for index in index_list:
        folders.append(all_folders[index])



    # multiprocessing dataset folder
    all_p = []
    for folder in folders:
        args.pointcloud_folder = folder
        all_p.append(multiproc.Process(target=prepare_dataset, args=(deepcopy(args),)))

    run_all_processes(all_p)

    logger.info("Dataset preparation Finished")


if __name__ == '__main__':
    parse = argparse.ArgumentParser(sys.argv[0])

    parse.add_argument('--k_start', type=int, default=20,
                       help="KNN cluster k range start point")
    parse.add_argument('--k_end', type=int, default=100,
                       help="KNN cluster k range end point")
    parse.add_argument('--k_step', type=int, default=10,
                       help="KNN cluster k range step")

    parse.add_argument('--bin_core_num', type=int, default=10, help="Parallel process file Pool core num")

    args = parse.parse_args(sys.argv[1:])

    main(args)
