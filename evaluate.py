# Thanks to Mikaela Angelina Uy, modified from PointNetVLAD and LPD-Net

import argparse
import os
import sys
import importlib
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils.loading_pointclouds import *
from sklearn.neighbors import KDTree

dimension = 13

# params
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 1]')
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--positives_per_query', type=int, default=4,
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=10,
                    help='Number of definite negatives in each training tuple [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--dimension', type=int, default=256)
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

# BATCH_SIZE = FLAGS.batch_size
BATCH_NUM_QUERIES = FLAGS.batch_num_queries         # 3
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
POSITIVES_PER_QUERY = FLAGS.positives_per_query     # 4
NEGATIVES_PER_QUERY = FLAGS.negatives_per_query     # 10
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# DATABASE_FILE = 'generating_queries/oxford_evaluation_database.pickle'
# QUERY_FILE = 'generating_queries/oxford_evaluation_query.pickle'
# DATABASE_FILE = 'generating_queries/university_evaluation_database.pickle'
# QUERY_FILE = 'generating_queries/university_evaluation_query.pickle'
# DATABASE_FILE = 'generating_queries/residential_evaluation_database.pickle'
# QUERY_FILE = 'generating_queries/residential_evaluation_query.pickle'
DATABASE_FILE = 'generating_queries/business_evaluation_database.pickle'
QUERY_FILE = 'generating_queries/business_evaluation_query.pickle'


LOG_DIR = FLAGS.log_dir
model = LOG_DIR.split('/')[1]
RESULTS_FOLDER = os.path.join("results/", model)
model = model.split('-')[0]
print(LOG_DIR)
MODEL = importlib.import_module(model)
print('Model: ', MODEL)
if not os.path.exists(RESULTS_FOLDER): os.makedirs(RESULTS_FOLDER)
output_file = RESULTS_FOLDER + '/results.txt'
model_file = "model.ckpt"

DATABASE_SETS = get_sets_dict(DATABASE_FILE)
QUERY_SETS = get_sets_dict(QUERY_FILE)

global DATABASE_VECTORS
DATABASE_VECTORS = []

global QUERY_VECTORS
QUERY_VECTORS = []

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


result_DIR = "./my_result"
if not os.path.exists(result_DIR): os.makedirs(result_DIR)
LOG_FOUT = open(os.path.join(result_DIR, 'my_result.txt'), 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)




def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_NUM_QUERIES,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def evaluate():
    global DATABASE_VECTORS
    global QUERY_VECTORS

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            print("In Graph")
            query = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
            positives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
            negatives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
            eval_queries = MODEL.placeholder_inputs(EVAL_BATCH_SIZE, 1, NUM_POINTS)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            with tf.variable_scope("query_triplets") as scope:
                vecs = tf.concat([query, positives, negatives], 1)
                print('vecs: ', vecs)
                out_vecs = MODEL.forward(vecs, is_training_pl, bn_decay=bn_decay)                   # △△△△
                q_vec, pos_vecs, neg_vecs = tf.split(out_vecs, [1, POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY], 1)

                print('q_vec: ', q_vec)
                print('pos_vecs: ', pos_vecs)
                print('neg_vecs: ', neg_vecs)

            saver = tf.train.Saver()


        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)


        saver.restore(sess, os.path.join(LOG_DIR, model_file))
        print("Model restored:{}".format(os.path.join(LOG_DIR, model_file)))

        # flops = tf.profiler.profile(tf.Graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
        # params = tf.profiler.profile(tf.Graph(),
        #                              options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        # print('original flops before: ', flops.total_float_ops)
        # print('original params before: ', params.total_parameters)

        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'is_training_pl': is_training_pl,
               'eval_queries': eval_queries,
               'q_vec': q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs}
        recall = np.zeros(25)
        count = 0
        similarity = []
        one_percent_recall = []     #
        ### print('len(DATABASE_SETS): ', len(DATABASE_SETS))
        for i in range(len(DATABASE_SETS)):
            ### print('i: ', i)
            DATABASE_VECTORS.append(get_latent_vectors(sess, ops, DATABASE_SETS[i]))

        ### print('len(QUERY_SETS): ', len(QUERY_SETS))
        for j in range(len(QUERY_SETS)):
            ### print('j: ', j)
            QUERY_VECTORS.append(get_latent_vectors(sess, ops, QUERY_SETS[j]))


        for m in range(len(QUERY_SETS)):
            for n in range(len(QUERY_SETS)):
                if m == n:
                    continue
                pair_recall, pair_similarity, pair_opr = get_recall(sess, ops, m, n)
                recall += np.array(pair_recall)
                count  += 1
                one_percent_recall.append(pair_opr)
                for x in pair_similarity:
                    similarity.append(x)


        ave_recall = recall / count
        print('ave_recallrecall')
        print(ave_recall)

        # print('similarity:')
        # print(similarity)

        average_similarity = np.mean(similarity)
        print('average_similarity: ', average_similarity)



        ave_one_percent_recall = np.mean(one_percent_recall)
        print('ave_one_percent_recall: ', ave_one_percent_recall)

        # flops = tf.profiler.profile(tf.Graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
        # params = tf.profiler.profile(tf.Graph(),
        #                              options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        # print('original flops after: ', flops.total_float_ops)
        # print('original params after: ', params.total_parameters)


        # filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
        with open(output_file, "a") as output:
            output.write(model)
            output.write("\n\n")
            output.write("Average Recall @N:\n")
            output.write(str(ave_recall))
            output.write("\n\n")
            output.write("Average Similarity:\n")
            output.write(str(average_similarity))
            output.write("\n\n")
            output.write("Average Top 1% Recall:\n")
            output.write(str(ave_one_percent_recall))
            output.write("\n\n")


def get_latent_vectors(sess, ops, dict_to_process):
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    print('dict_to_process: ', dict_to_process)
    print('len(dict_to_process): ', len(dict_to_process))

    # print(len(train_file_idxs))
    batch_num = BATCH_NUM_QUERIES * (1 + POSITIVES_PER_QUERY + NEGATIVES_PER_QUERY)
    q_output = []
    count = 0
    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index * batch_num : (q_index + 1) * batch_num]
        file_names = []
        count = count + 1
        ### print('count: ', count)
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])

        ### print('file_names[0]:', file_names[0])
        ### print('file_names:', file_names)
        ### print('len(file_names):', len(file_names))
        queries = load_pc_files(file_names)

        # queries= np.expand_dims(queries, axis=1)
        q1 = queries[0 : BATCH_NUM_QUERIES]             # before q1.shape:  (3, 4096, 18)
        q1 = np.expand_dims(q1, axis=1)                 # after  q1.shape:  (3, 1, 4096, 18)

        q2 = queries[BATCH_NUM_QUERIES : BATCH_NUM_QUERIES * (POSITIVES_PER_QUERY + 1)]         # before q2.shape:  (12, 4096, 18)
        q2 = np.reshape(q2, (BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, dimension))    # after  q2.shape:  (3, 4, 4096, 18)

        q3 = queries[BATCH_NUM_QUERIES * (POSITIVES_PER_QUERY + 1) : BATCH_NUM_QUERIES * (
                    NEGATIVES_PER_QUERY + POSITIVES_PER_QUERY + 1)]                             # before q3.shape:  (30, 4096, 18)
        q3 = np.reshape(q3, (BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, dimension))    # after  q3.shape:  (3, 10, 4096, 18)

        feed_dict  = {ops['query']: q1, ops['positives']: q2, ops['negatives']: q3, ops['is_training_pl']: is_training}

        o1, o2, o3 = sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict)

        o1 = np.reshape(o1, (-1, o1.shape[-1]))
        o2 = np.reshape(o2, (-1, o2.shape[-1]))
        # before o3.shape:  (3, 10, 256)
        o3 = np.reshape(o3, (-1, o3.shape[-1]))
        # after  o3.shape:  (30, 256)

        out = np.vstack((o1, o2, o3))
        # print('out.shape: ', out.shape)         # out.shape:  (45, 256)
        q_output.append(out)

    q_output = np.array(q_output)
    # print('before q_output.shape: ', q_output.shape)
    if len(q_output) != 0:
        q_output = q_output.reshape(-1, q_output.shape[-1])
    # print('after  q_output.shape: ', q_output.shape)
    # after  q_output.shape:  (405, 256)

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)
        # print(query.shape)
        # exit()
        fake_queries = np.zeros((BATCH_NUM_QUERIES - 1, 1, NUM_POINTS, dimension))
        fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, dimension))
        fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, dimension))
        q = np.vstack((queries, fake_queries))
        # print(q.shape)
        feed_dict = {ops['query']: q, ops['positives']: fake_pos, ops['negatives']: fake_neg,
                     ops['is_training_pl']: is_training}
        output = sess.run(ops['q_vec'], feed_dict=feed_dict)

        # print(output.shape)
        output = output[0]
        output = np.squeeze(output)
        if q_output.shape[0] != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    # q_output=np.array(q_output)
    # q_output=q_output.reshape(-1,q_output.shape[-1])
    ### print('q_output.shape: ', q_output.shape)
    return q_output


def get_recall(sess, ops, m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output = DATABASE_VECTORS[m]
    ### print('database_output: ', database_output)
    queries_output  = QUERY_VECTORS[n]
    ### print('queries_output: ', queries_output)


    ### print('len(queries_output): ', len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    # print('recall: ', recall)     recall:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0

    print("m, n = ", m, n)  # m, n =  0 1
    ### print('Query sets:', QUERY_SETS)
    log_string('m = %d' % m)
    log_string('n = %d' % n)

    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        # print('len(indices): ', len(indices))                 len(indices): 1
        # print('len(indices[0]): ', len(indices[0]))           len(indices[0]): 25
        # print('indices: ', indices)
        # indices:  [[368 367 124  21 366 252  20  73  30 365  34 187   0 357 107 106  97  22
        #   375  99   5  33 389 374 268]]

        print('QUERY_SETS[n][i]', QUERY_SETS[n][i])
        log_string('QUERY_SETS[n][i]: %s' % QUERY_SETS[n][i])
        ### QUERY_SETS[n][i] {'query': 'oxford/2014-11-18-13-20-12/pointcloud_20m/1416317029804631.bin', 'northing': 5735752.225992, 'easting': 619994.940317,
        #   0: [0], 2: [3, 2], 3: [], 4: [6, 5], 5: [6, 7], 6: [], 7: [6, 5], 8: [], 9: [8, 7, 6], 10: [], 11: [], 12: [],
        #   13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: []}
        print('indices[0]', indices[0])
        log_string('indices[0]: %s' % indices[0])
        # indices[0] [  0 335 226 271 234 336 274  89  90 140 310 307 244 308 311 205 227 225 142 147 309 312  77  86 334]
        print('true_neighbors', true_neighbors)
        log_string('true_neighbors: %s' % true_neighbors)


        for j in range(len(indices[0])):
            if (j < 3):
                print('DATABASE_SETS[m][indices[0][j]]', DATABASE_SETS[m][indices[0][j]])
                log_string('DATABASE_SETS[m][indices[0][j]]: %s' % DATABASE_SETS[m][indices[0][j]])
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len( list( set(indices[0][0:threshold]).intersection(set(true_neighbors)) ) ) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    print('recall: ', recall)
    print('top1_simlar_score: ')
    print(np.mean(top1_similarity_score))
    print('one_percent_recall: ', one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


"""  """
def get_similarity(sess, ops, m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    threshold = len(queries_output)
    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    similarity = []
    for i in range(len(queries_output)):
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=1)
        for j in range(len(indices[0])):
            q_sim = np.dot(q_output[i], database_output[indices[0][j]])
            similarity.append(q_sim)
    average_similarity = np.mean(similarity)
    print(average_similarity)
    return average_similarity


if __name__ == "__main__":
    evaluate()
    #flops = tf.profiler.profile(tf.Graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
    #params = tf.profiler.profile(tf.Graph(),
    #                             options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    #print('original flops before: ', flops.total_float_ops)
    #print('original params before: ', params.total_parameters)
