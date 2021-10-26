# FICL-Net

## Benchmark Datasets
The benchmark datasets introdruced in this work can be downloaded [here](https://drive.google.com/open?id=1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D).
* All submaps are in binary file format
* Ground truth GPS coordinate of the submaps are found in the corresponding csv files for each run
* Filename of the submaps are their timestamps which is consistent with the timestamps in the csv files
* Use CSV files to define positive and negative point clouds
* All submaps are preprocessed with the road removed and downsampled to 4096 points

### Oxford Dataset
* 45 sets in total of full and partial runs
* Used both full and partial runs for training but only used full runs for testing/inference
* Training submaps are found in the folder "pointcloud_20m_10overlap/" and its corresponding csv file is "pointcloud_locations_20m_10overlap.csv"
* Training submaps are not mutually disjoint per run
* Each training submap ~20m of car trajectory and subsequent submaps are ~10m apart
* Test/Inference submaps found in the folder "pointcloud_20m/" and its corresponding csv file is "pointcloud_locations_20m.csv"
* Test/Inference submaps are mutually disjoint

### Evaluate

https://drive.google.com/file/d/18_bgT2snyrEpR-Gwc3UIYQYMksaPFZe2/view?usp=sharing

python evaluate.py --log_dir log/lpd_FNSF-21-04-24-16-05
