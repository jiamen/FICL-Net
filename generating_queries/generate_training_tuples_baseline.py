import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = ".././benchmark_datasets/"

runs_folder = "oxford/"
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/pointcloud_20m_10overlap/"
featurecloud_fols = "/featurecloud_20m_10overlap/"

all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))

folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders))
print("Number of runs: " + str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)
"""
['2014-05-19-13-20-57', '2014-06-26-09-31-18', '2014-06-26-09-53-12', '2014-07-14-14-49-50', '2014-07-14-15-16-36',
 '2014-11-14-16-34-33', '2014-11-18-13-20-12', '2014-12-02-15-30-08', '2014-12-05-15-42-07', '2014-12-09-13-21-02',
 '2014-12-10-18-10-50', '2014-12-12-10-45-15', '2014-12-16-09-14-09', '2014-12-16-18-44-24', '2015-02-03-08-45-10',
 '2015-02-10-11-58-05', '2015-02-13-09-16-26', '2015-02-17-14-42-12', '2015-03-10-14-18-10', '2015-03-17-11-08-44',
 '2015-04-17-09-06-25', '2015-04-24-08-15-07', '2015-05-19-14-06-38', '2015-05-22-11-14-30', '2015-06-09-15-06-29',
 '2015-06-12-08-52-55', '2015-06-26-08-09-43', '2015-07-03-15-23-28', '2015-07-08-13-37-17', '2015-07-10-10-01-59',
 '2015-07-14-16-17-39', '2015-08-12-15-04-18', '2015-08-13-16-02-58', '2015-08-14-14-54-57', '2015-08-17-10-26-38',
 '2015-08-17-10-42-18', '2015-08-17-13-30-19', '2015-08-20-12-00-47', '2015-08-28-09-50-22', '2015-10-30-13-52-14',
 '2015-11-10-10-32-52', '2015-11-10-11-55-47', '2015-11-10-14-15-57', '2015-11-12-11-22-05', '2015-11-13-10-28-08']"""



# For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124, 620084.402381]
p2 = [5735611.299219, 620540.270327]
p3 = [5735237.358209, 620543.094379]
p4 = [5734749.303802, 619932.693364]
p = [p1, p2, p3, p4]


""" 调用第①个函数 """
def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if point[0] - x_width < northing and northing < point[0] + x_width and point[1] - y_width < easting and easting < point[1] + y_width:
            in_test_set = True
            break
    return in_test_set


##########################################
""" 调用第②个函数 """
def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=10)
    ind_r  = tree.query_radius(df_centroids[['northing', 'easting']], r=50)
    print("len(ind_nn):", len(ind_nn))
    print("len(ind_r):",  len(ind_r))
    """
    len(ind_nn): 21711
    len(ind_r): 21711
    """
    print("len(ind_nn):", ind_nn[0])
    """
    len(ind_nn): [18663  2587  1448 10730 20269     1 14442 13389  8974 21187  5005  3279
                   588 11862 19181  3842 15488  6138  4428 18664  2019 16057 12326   117
                 13923 17150 17230  9548  7276 20610 19750   346  2588 16620  3108   996
                 11301 20268     0 12867  8382 10120 13388  7833  5571  6708  1447 10729
                 14441 21186  8973 17149  2018 17229  3841  7275 20609   587 11861 12325
                  6137  3278   116 13922 16056 19180 19749  9547   345 15487  4427   995
                  5004 16619  7832  8381  3107 10119 11300  6707]
    """
    print("len(ind_r):",  ind_r[0])

    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()

        random.shuffle(negatives)
        queries[i] = {"query": query, "positives": positives, "negatives": negatives}

    with open(filename, 'wb') as handle:        # filename: "training_queries_baseline.pickle"
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)



# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
df_test  = pd.DataFrame(columns=['file', 'northing', 'easting'])


for folder in folders:
    df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
    df_locations['timestamp'] = runs_folder + folder + featurecloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        if check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)


print("Number of training submaps: " + str(len(df_train['file'])))
print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
"""
Number of training submaps: 22278
Number of non-disjoint test submaps: 9989
"""
construct_query_dict(df_train, "training_queries_baseline.pickle")
construct_query_dict(df_test,  "test_queries_baseline.pickle")
