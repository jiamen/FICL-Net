import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from tensorflow.python import pywrap_tensorflow

import time

print("plt.get_backend():", plt.get_backend())

# plt.switch_backend('agg')
# plt.switch_backend('agg')
# print("plt.get_backend():", plt.get_backend())

file_length = 0

class PLOT:
    def __init__(self, dirIdx=0, pclIdx=0):
        # 路径设置
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.base_path = "./benchmark_datasets/"
        self.runs_folder = "oxford/"
        self.filename = "pointcloud_locations_20m_10overlap.csv"
        self.pointcloud_fols = "/pointcloud_20m_10overlap/"
        self.featurecloud_fols = "/featurecloud_20m_10overlap/"
        all_folders = sorted(os.listdir(os.path.join(self.BASE_DIR, self.base_path, self.runs_folder)))

        # print(len(all_folders))
        self.folder = all_folders[dirIdx]


        self.df_locations = pd.read_csv(os.path.join(self.base_path, self.runs_folder, self.folder, self.filename), sep=',')
        self.df_locations['timestamp'] = self.runs_folder + self.folder + self.featurecloud_fols + self.df_locations[
            'timestamp'].astype(str) + '.bin'
        self.df_locations = self.df_locations.rename(columns={'timestamp': 'file'})


        self.file_location = os.path.join(self.base_path, self.runs_folder, self.folder, 'pointcloud_20m_10overlap/')
        print('file_location: ', self.file_location)
        self.filelist = os.listdir(self.file_location)
        self.filelist.sort(key=lambda x: int(x[:-4]))

        file_length = len(self.filelist)
        print('file_length:', file_length)


        self.dp_locations = pd.read_csv(os.path.join(self.base_path, self.runs_folder, self.folder, self.filename), sep=',')

        self.dp_locations['timestamp'] = self.runs_folder + self.folder + self.pointcloud_fols + self.dp_locations[
            'timestamp'].astype(str) + '.bin'
        self.dp_locations = self.dp_locations.rename(columns={'timestamp': 'file'})


        self.pointcloud_files = self.dp_locations['file'].tolist()
        self.northings = np.asarray(self.dp_locations['northing'].tolist())
        self.eastings = np.asarray(self.dp_locations['easting'].tolist())
        self.dirIdx = dirIdx
        self.pclIdx = pclIdx

        self.p1 = [5735712.768124, 620084.402381]
        self.p2 = [5735611.299219, 620540.270327]
        self.p3 = [5735237.358209, 620543.094379]
        self.p4 = [5734749.303802, 619932.693364]
        self.width = 150


        # for index, row in self.df_locations.iterrows():
        #     print(index)
        #     print(row.index)

    def plotsquare(self, p):
        plt.scatter(p[0], p[1], s=10, c='red')
        # plt.scatter(p[0]-self.width, p[1]-self.width, s=10,c='red')
        # plt.scatter(p[0]+self.width, p[1]-self.width, s=10,c='red')
        # plt.scatter(p[0]-self.width, p[1]+self.width, s=10,c='red')
        # plt.scatter(p[0]+self.width, p[1]+self.width, s=10,c='red')

    def plotcsv(self):
        for i in range(10):
            a = np.asarray(self.northings[i], self.eastings[i])
            b = np.asarray(self.northings[i + 1], self.eastings[i + 1])
            dist = np.sqrt(np.sum(np.square(a - b)))
            print(dist)
        plt.scatter(self.northings, self.eastings, s=2)
        self.plotsquare(self.p1)
        self.plotsquare(self.p2)
        self.plotsquare(self.p3)
        self.plotsquare(self.p4)
        plt.show()

    def plot_place(self, pclIdx=0):
        for i in range(10):
            a = np.asarray(self.northings[i], self.eastings[i])
            b = np.asarray(self.northings[i + 1], self.eastings[i + 1])
            dist = np.sqrt(np.sum(np.square(a - b)))
            print(dist)

        plt.scatter(self.northings, self.eastings, s=2)
        ax = plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        place = [self.northings[pclIdx], self.eastings[pclIdx]]
        print("file:", self.pointcloud_files[pclIdx])
        self.plotsquare(place)
        plt.show()

    def plotpcl(self):
        pointcloud_file = os.path.join(self.BASE_DIR, self.base_path, self.pointcloud_files[self.pclIdx])
        pointcloud = np.fromfile(pointcloud_file, dtype=np.float64)
        pointcloud = np.reshape(pointcloud, (pointcloud.shape[0] // 3, 3)).T
        print(np.max(pointcloud))
        """
            0.975330493219281
            0.8831555599640089
            0.9533693392123237
            0.8977215943788069
            0.9996848189469281
            0.9997043461276511
            0.875185700283391
            0.9525375801834162
            0.8468831686908642
            0.988124380533747
            0.9994679345591356
            0.917918340100378
            0.8182202276787661
            0.9562811124031871
            0.9697224681707275
            0.9978868011200089
            0.7378168156040453
            0.9943420475028548
            0.9991047925175376
            0.9998103885705251
            0.7892840407507425
            0.842944113402326
            0.9994865422663635
        """
        fig = plt.figure(figsize=(10, 30))
        ax = fig.add_subplot(111, projection='3d')
        colors = np.random.rand(10)
        ax.scatter(pointcloud[0], pointcloud[1], pointcloud[2], color='blue', s=1)      # 显示点云3D图
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def returnTrajectory(self):
        return [self.northings, self.eastings]


if __name__ == '__main__':
    index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    import random

    random.shuffle(index_list)
    print(index_list)
    print(random.shuffle(index_list))



    for i in index_list:
        # for j in len(index_list[i]):
        script = PLOT(dirIdx=i, pclIdx=0)
        script.plotpcl()
        for j in range(file_length):
            # plt.scatter(script.returnTrajectory()[0], script.returnTrajectory()[1], s=2)
            script = PLOT(dirIdx=i, pclIdx=j)
            script.plotpcl()
            time.sleep(1)
            plt.show()
            break


    for j in range(file_length):
        # plt.scatter(script.returnTrajectory()[0], script.returnTrajectory()[1], s=2)
        script = PLOT(dirIdx=0, pclIdx=j)
        time.sleep(100)
        script.plotpcl()
        # plt.show()

    ###pcl_Idx = 265
    ###script = PLOT(dirIdx=15, pclIdx=pcl_Idx)
    # plt.scatter(script.returnTrajectory()[0], script.returnTrajectory()[1], s=2)    # 显示
    ###script.plotpcl()
    # script.plotcsv()
    ###script.plot_place(pclIdx=pcl_Idx)

    ###plt.show()

    #pointcloud_file = '/home/zlc/ROSFiles/sample/ldmrs/1418381798113072.bin'
    #pointcloud = np.fromfile(pointcloud_file, dtype=np.float64)
    #pointcloud = np.reshape(pointcloud, (pointcloud.shape[0] // 3, 3)).T
    #print(np.max(pointcloud))

    #fig = plt.figure(figsize=(10, 30))
    #ax = fig.add_subplot(111, projection='3d')
    #colors = np.random.rand(10)
    #ax.scatter(pointcloud[0], pointcloud[1], pointcloud[2], color='blue', s=1)      # 显示点云3D图
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    #ax.set_zlabel("z")
    #plt.show()
