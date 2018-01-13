from __future__ import print_function

import tensorflow as tf
from base import Model
from util import get_data
from sparse_util import *
import math
import time
from tqdm import tqdm
from collections import OrderedDict

def sample_submatrix(mask_,#mask, used for getting concentrations
                     maxN, maxM,
                     sample_uniform=False):
    '''
    sampling mini-batches. Currently it is sampling rows and columns based on the number of non-zeros.
    In the sparse implementation, we could sample non-zero entries and directly.
    '''
    pN, pM = mask_.sum(axis=1)[:,0], mask_.sum(axis=0)[:,0]
    pN /= pN.sum()#non-zero dist per row
    pM /= pM.sum()#non-zero dist per column
    
    N, M, _ = mask_.shape
    for n in range(N // maxN):
        for m in range(M // maxM):
            if N == maxN:
                ind_n = np.arange(N)
            else:
                ind_n = np.random.choice(N, size=maxN, replace=False, p = pN)#select a row based on density of nonzeros
            if M == maxM:
                ind_m = np.arange(M)
            else:
                ind_m = np.random.choice(M, size=maxM, replace=False, p = pM)
            yield ind_n, ind_m 


def sample_dense_values_uniform(mask_indices, minibatch_size, iters_per_epoch):
    num_vals = mask_indices.shape[0]
    minibatch_size = np.minimum(minibatch_size, num_vals)
    for n in range(iters_per_epoch):
        sample = np.random.choice(num_vals, size=minibatch_size, replace=False)
        yield sample

def sample_dense_values_uniform_val(mask_indices, mask_tr_val_split, minibatch_size, iters_per_epoch):
    num_vals_tr = mask_indices[mask_tr_val_split == 0].shape[0]
    num_vals_val = mask_indices[mask_tr_val_split == 1].shape[0]
    minibatch_size_tr = np.minimum(int(.9 * minibatch_size), num_vals_tr)
    minibatch_size_val = np.minimum(int(.1 * minibatch_size), num_vals_val)
    for n in range(iters_per_epoch):
        sample_tr = np.random.choice(num_vals_tr, size=minibatch_size_tr, replace=False)
        sample_val = np.random.choice(num_vals_val, size=minibatch_size_val, replace=False)
        yield sample_tr, sample_val


def ce_loss(log_prob, counts):
    return - tf.reduce_mean(counts * log_prob)

def masked_inner_product(nvec, mvec, mask, log_inp=False):
    ng = tf.gather(nvec, mask[:,0], axis=0)
    mg = tf.gather(mvec, mask[:,1], axis=1)
    if log_inp:
        # assumes that ng and mg are in log domain so we add and take exp instead of multiply to get inner product
        return tf.reduce_sum(tf.exp(ng + tf.transpose(mg, (1, 0, 2))), axis=2, keep_dims=False)
    else:
        return tf.reduce_sum(ng * tf.transpose(mg, (1, 0, 2)), axis=2, keep_dims=False)

def main(opts, data=None):        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)    
    if data is None:
        data = read_data()
    
    N, M, num_features = data['mat_shape']

    with tf.Graph().as_default():
            mat_values_tr = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr')
            mask_indices_tr = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_tr')

            with tf.variable_scope("model"):
                tr_dict = {'input':mat_values_tr,
                           'mask_indices':mask_indices_tr,
                           'units':1,
                           'shape':[N,M]}


                model = Model(layers=opts['architecture'], layer_defaults=opts['defaults'], verbose=2) #define the model
                model_output = model.get_output(tr_dict) #build the model
                words = tf.nn.log_softmax(model_output['nvec'])
                docs = tf.nn.log_softmax(model_output['mvec'], dim=0)

            # log_prob_topics = words + docs # gather
            # take sum
            total_prob = tf.clip_by_value(masked_inner_product(words, docs, mask_indices_tr, log_inp=True), 0., 1.)
            topic_loss = ce_loss(total_prob, mat_values_tr)
            #loss and training
            reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
            total_loss = topic_loss + reg_loss

            # train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
            train_step = tf.train.RMSPropOptimizer(opts['lr']).minimize(total_loss)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.global_variables_initializer())

            if 'by_row_column_density' in opts['sample_mode']:
                iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
            elif 'uniform_over_dense_values' in opts['sample_mode']:
                minibatch_size = np.minimum(opts['minibatch_size'], data['mask_indices_tr'].shape[0])
                iters_per_epoch = data['mask_indices_tr'].shape[0] // minibatch_size
            
            
            min_loss = 5
            min_loss_epoch = 0
            losses = OrderedDict()
            losses["train"] = []
            losses["valid"] = []
            
            for ep in range(opts['epochs']):
                begin = time.time()
                loss_tr_, topic_loss_tr_, loss_val_, loss_ts_ = 0,0,0,0

                for sample_ in tqdm(sample_dense_values_uniform(data['mask_indices_tr'], minibatch_size, iters_per_epoch), total=iters_per_epoch):

                    mat_values = data['mat_values_tr'][sample_]
                    mask_indices = data['mask_indices_tr'][sample_]

                    tr_dict = {mat_values_tr:mat_values,
                                mask_indices_tr:mask_indices}
                    
                    _, bloss_, btopic_loss_ = sess.run([train_step, total_loss, topic_loss], feed_dict=tr_dict)

                    loss_tr_ += bloss_
                    topic_loss_tr_ += btopic_loss_

                loss_tr_ /= iters_per_epoch
                topic_loss_tr_ /= iters_per_epoch

                losses['train'].append(loss_tr_)
                losses['valid'].append(loss_val_)

                print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f})".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_))            
    return losses, {"sess":sess, "total_loss": total_loss, "rec_loss": rec_loss, "rec_loss_val":rec_loss_val, 
                    "mat_values_tr": mat_values_tr, "mask_indices_tr": mask_indices_tr,
                    "mat_values_val":mat_values_val, "mask_indices_val":mask_indices_val,
                    "mask_indices_tr_val":mask_indices_tr_val}

def read_data():
    '''read data'''
    data = {}
    # contains index of words appearing in that document and the number of times they appear
    with open('./data/nyt/nyt_data.txt') as f:
        documents = f.readlines()
    documents = [x.strip().strip('\n').strip("'") for x in documents] 

    # contains vocabs with rows as index
    with open('./data/nyt/nyt_vocab.dat') as f:
        vocabs = f.readlines()
    vocabs = [x.strip().strip('\n').strip("'") for x in vocabs]
    data['mat_shape'] = (len(vocabs), len(documents), 1)

    mask = []
    values = []

    for col, doc in enumerate(documents):
        for item in doc.split(","):
            word, freq = [int(i) for i in item.split(":")]
            word -= 1
            mask.append([word, col])
            values.append(freq)
    
    data['mask_indices_tr'] = np.array(mask)
    data['mat_values_tr'] = np.array(values)
    return data

if __name__ == "__main__":
    minibatch_size = 10000
    skip_connections = True
    units = 64
    latent_features = 5
    learning_rate = 0.005

    opts ={'epochs': 5000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/topic_model',
           'model_name':'test_topic_model',
           'verbose':2,
           'minibatch_size':minibatch_size,
           'visualize':False,
           'save':False,
           'output_file':'output',
           'architecture':[
               {'type':'matrix_sparse', 'units':units, 'activation':tf.nn.relu},
               #{'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units, 'activation':tf.nn.relu},
               #{'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':latent_features, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool_sparse'},
               ],
            'defaults':{#default values for each layer type (see layer.py)
                'matrix_sparse':{
                    # 'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    'activation':tf.nn.relu,
                    # 'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'max',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                    'skip_connections':skip_connections,
                },
                'dense':{#not used
                    'activation':tf.nn.elu,
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'matrix_pool_sparse':{
                    'pool_mode':'max',
                },
                'matrix_dropout_sparse':{
                    'rate':.3,
                }
            },
           'lr':learning_rate,
           'sample_mode':'uniform_over_dense_values' # by_row_column_density, uniform_over_dense_values
           
    }
    
    main(opts)


