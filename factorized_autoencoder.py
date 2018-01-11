from __future__ import print_function

import tensorflow as tf
import numpy as np
from base import Model
from util import get_data, to_indicator, to_number
from sparse_util import sparse_tensordot_sparse, get_mask_indices
import math
import time
from tqdm import tqdm
from copy import deepcopy

def masked_crossentropy(mat, mask, rec):
    '''
    Average crossentropy over non-zero entries
    '''
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=mat, logits=rec) * tf.squeeze(mask) ) / tf.reduce_sum(mask)

def normalize(data):
    data_mean = data.mean()
    data_std = data.std()
    standardize = lambda x: (x - data_mean)/data_std
    inverse_trans = lambda x: (x * data_std) + data_mean
    return standardize, inverse_trans

def get_loss_function(loss="mse"):
    if loss == "ce":
        return masked_crossentropy
    elif loss == "mse":
        return rec_loss_fn
    else:
        raise ValueError("Unrecognized loss: %s" % loss)

def expected_value(probs):
    print(probs.sum(axis=2).min(), probs.sum(axis=2).max(), probs.max(), probs.min())
    return np.dot(probs, np.arange(1,6).reshape((5,1)))

def sample_submatrix(mask_,#mask, used for getting concentrations
                     maxN, maxM):
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



def sample_masked_entries(user_inds, movie_inds, maxN, maxM):
    N = user_inds.shape[0]
    M = movie_inds.shape[0]
    
    if N < maxN: maxN = N
    if M < maxM: maxM = M
    
    for n in range(N // maxN):
        for m in range(M // maxM):
            ind_n = np.random.choice(user_inds, size=maxN, replace=False)
            ind_m = np.random.choice(movie_inds, size=maxM, replace=False)
            yield ind_n, ind_m



def rec_loss_fn(mat, mask, rec):
    return (tf.reduce_sum(((mat - rec)**2)*mask)) / tf.reduce_sum(mask)#average l2-error over non-zero entries


def main(opts):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    path = opts['data_path']
    data = get_data(path, train=.8, valid=.2, test=.0)
    
    standardize = inverse_trans = lambda x: x # defaults
    if opts.get("loss", "mse") == "mse":
        input_data = data['mat']
        raw_input_data = data['mat'].copy()
        if opts.get('normalize', False):
            print("Normalizing data")
            standardize, inverse_trans = normalize(input_data)
    else:
        raw_input_data = data['mat'].copy()
        input_data = to_indicator(data['mat'])

    loss_fn = get_loss_function(opts.get("loss", "mse"))
    #build encoder and decoder and use VAE loss
    N, M, num_features = input_data.shape
    opts['decoder'][-1]['units'] = num_features
    maxN, maxM = opts['maxN'], opts['maxM']

    if N < maxN: maxN = N
    if M < maxM: maxM = M

    N_users = np.sum(data['users_split'])
    M_movies = np.sum(data['movies_split'])

    if N_users < maxN: maxN = N_users
    if M_movies < maxM: maxM = M_movies

    if opts['verbose'] > 0:
        print('\nRun Settings:')
        print('dataset: ', path)
        print('drop mask: ', opts['defaults']['matrix_dense']['drop_mask'])
        print('Exchangable layer pool mode: ', opts['defaults']['matrix_dense']['pool_mode'])
        print('Pooling layer pool mode: ', opts['defaults']['matrix_pool']['pool_mode'])
        print('learning rate: ', opts['lr'])
        print('activation: ', opts['defaults']['matrix_dense']['activation'])
        print('maxN: ', opts['maxN'])
        print('maxM: ', opts['maxM'])
        print('')
        

    with tf.Graph().as_default():
        mat_raw = tf.placeholder(tf.float32, shape=(None, None, 1), name='mat_raw')#data matrix for training
        mat_raw_valid = tf.placeholder(tf.float32, shape=(None, None, 1), name='mat_raw_valid')#data matrix for training

        mat = tf.placeholder(tf.float32, shape=(None, None, num_features), name='mat')#data matrix for training
        mask_tr = tf.placeholder(tf.float32, shape=(None, None, 1), name='mask_tr')
        # For validation, since we need less memory (forward pass only), 
        # we are feeding the whole matrix. This is only feasible for this smaller dataset. 
        # In the long term we could perform validation on CPU to avoid memory problems
        mat_val = tf.placeholder(tf.float32, shape=(None, None, num_features), name='mat')##data matrix for validation: 
        mask_val = tf.placeholder(tf.float32, shape=(None, None, 1), name='mask_val')#the entries not present during training
        mask_tr_val = tf.placeholder(tf.float32, shape=(None, None, 1), name='mask_tr_val')#both training and validation entries

        indn = tf.placeholder(tf.int32, shape=(None), name='indn')
        indm = tf.placeholder(tf.int32, shape=(None), name='indm')
        
        with tf.variable_scope("encoder"):
            tr_dict = {'input':mat,
                       'mask':mask_tr,
                       'total_shape':[N,M],
                       'indn':indn,
                       'indm':indm}
            val_dict = {'input':mat_val,
                        'mask':mask_tr_val,
                        'total_shape':[N,M],
                        'indn':indn,
                        'indm':indm}

            encoder = Model(layers=opts['encoder'], layer_defaults=opts['defaults'], verbose=2) #define the encoder

            out_enc_tr = encoder.get_output(tr_dict) #build the encoder
            out_enc_val = encoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#get encoder output, reusing the neural net
            

        with tf.variable_scope("decoder"):
            tr_dict = {'nvec':out_enc_tr['nvec'],
                       'mvec':out_enc_tr['mvec'],
                       'mask':out_enc_tr['mask'],
                       'total_shape':[N,M],
                       'indn':indn,
                       'indm':indm}
            val_dict = {'nvec':out_enc_val['nvec'],
                        'mvec':out_enc_val['mvec'],
                        'mask':out_enc_val['mask'],                        
                        'total_shape':[N,M],
                        'indn':indn,
                        'indm':indm}
            # tr_dict = {'input':mat,
            #            'mask':mask_tr,
            #            'total_shape':[N,M],
            #            'indn':indn,
            #            'indm':indm}
            # val_dict = {'input':mat_val,
            #             'mask':mask_tr_val,
            #             'total_shape':[N,M],
            #             'indn':indn,
            #             'indm':indm}


            decoder = Model(layers=opts['decoder'], layer_defaults=opts['defaults'], verbose=2)#define the decoder

            out_tr = decoder.get_output(tr_dict)['input']#build it
            out_val = decoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)['input']#reuse it for validation

        #loss and training
        rec_loss = loss_fn(inverse_trans(mat), mask_tr, inverse_trans(out_tr))# reconstruction loss
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
        rec_loss_val = loss_fn(inverse_trans(mat_val), mask_val, inverse_trans(out_val))
        total_loss = rec_loss + reg_loss

        rng = tf.range(1,6,1, dtype=tf.float32)
        idx = tf.convert_to_tensor([[2],[0]], dtype=np.int32)
        mse_loss_train = rec_loss_fn(mat_raw, mask_tr, tf.reshape(tf.tensordot(tf.nn.softmax(out_tr), rng, idx), (maxN,maxM,1)))
        mse_loss_valid = rec_loss_fn(mat_raw_valid, mask_val, tf.reshape(tf.tensordot(tf.nn.softmax(out_val), rng, idx), (N,M,1)))

        train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
        merged = tf.summary.merge_all()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        # iters_per_epoch = (N//maxN) * (M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
        iters_per_epoch = (N_users//maxN) * (M_movies//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
        
        min_loss_tr = 5
        min_loss_epoch_tr = 0
        loss_threshold = 1.13 # Short circuit the training loop if validation loss gets low enough  

        start_time = time.time()
        for ep in range(opts['epochs']):
            begin = time.time()
            loss_tr_, rec_loss_tr_, loss_val_, mse_tr = 0,0,0,0

            for indn_, indm_ in tqdm(sample_masked_entries(data['user_inds_tr'], data['movie_inds_tr'], maxN, maxM), total=iters_per_epoch):#go over mini-batches
                inds_ = np.ix_(indn_,indm_,range(num_features))
                inds_mask = np.ix_(indn_,indm_,[0])                

                tr_dict = {mat:standardize(input_data[inds_]),
                           mask_tr:data['mask_ratings'][inds_mask],
                           mat_raw:raw_input_data[inds_mask],
                           indn:indn_,
                           indm:indm_}

                if opts.get("loss", "mse") == "mse":
                    _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)
                    loss_tr_ += np.sqrt(bloss_)
                    rec_loss_tr_ += np.sqrt(brec_loss_)
                elif opts.get("loss", "mse") == "ce":
                    _, bloss_, brec_loss_, mse = sess.run([train_step, total_loss, rec_loss, mse_loss_train],feed_dict=tr_dict)
                    loss_tr_ += np.sqrt(mse)
                    rec_loss_tr_ += brec_loss_

            loss_tr_ /= iters_per_epoch
            rec_loss_tr_ /= iters_per_epoch

            if loss_tr_ < min_loss_tr: # keep track of the best validation loss 
                min_loss_tr = loss_tr_
                min_loss_epoch_tr = ep

            print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t minimum training loss {:.3f} at epoch {:d}".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_, min_loss_tr, min_loss_epoch_tr))

            if loss_tr_ < loss_threshold: # When trainnig loss is sufficiently low, check validation 
                break

        print("training for {:d} epochs took {:.1f} final training loss {:.3f} \t best training loss {:.3f} at epoch {:d}".format(ep, time.time() - start_time, loss_tr_, min_loss_tr, min_loss_epoch_tr))    

        users_inds_val = np.random.permutation(data['user_inds_val'])
        movies_inds_val = np.random.permutation(data['movie_inds_val'])

        iters = 10 # How many 'mixed' submatricies to try validation on 

        NN = np.sum(1 - data['users_split']) # Number of users not used in training
        MM = np.sum(1 - data['movies_split']) # Number of movies not used in training

        users_cutoffs = np.ndarray.astype(np.ceil(np.linspace(0, NN, iters)), np.int32)
        movies_cutoffs = np.ndarray.astype(np.ceil(np.linspace(0, MM, iters)), np.int32)

        mask_tr_ = data['mask_ratings'] * np.matmul(data['users_split'][:,None], data['movies_split'][None,:])[:,:,None]
        mask_val_ = data['mask_ratings'] * (1-data['users_split'][:,None,None]) * (1-data['movies_split'][None,:,None])
        

        for i in reversed(range(iters)):
            loss_val_ = 0

            users_cutoff = users_cutoffs[i]
            movies_cutoff = movies_cutoffs[i]

            users_overlap = users_inds_val[0:users_cutoff]
            users_val_ = deepcopy(data['users_split'][:,None])            
            users_val_[users_overlap,:] = 1            

            movies_overlap = movies_inds_val[0:movies_cutoff]
            movies_val_ = deepcopy(data['movies_split'][None,:])
            movies_val_[:,movies_overlap] = 1

            mask_predict = np.matmul(users_val_, movies_val_)[:,:,None]
            mask_tr_val_ = data['mask_ratings'] * mask_predict * (1 - mask_val_)

            val_dict = {mat_val:standardize(input_data),
                        mask_val:mask_val_,
                        mask_tr_val:mask_tr_val_,
                        mat_raw_valid:raw_input_data,
                        indn:np.arange(N),
                        indm:np.arange(M)}

            if opts.get("loss", "mse") == "mse":
                bloss_, = sess.run([rec_loss_val], feed_dict=val_dict)
            else:
                bloss_true, bloss_ = sess.run([rec_loss_val, mse_loss_valid], feed_dict=val_dict)
            loss_val_ = np.sqrt(bloss_)
            print("validation loss using {:.1f}% user and {:.1f}% movie overlap: {:.8f}".format(users_cutoff / NN * 100, movies_cutoff / MM * 100, loss_val_))


if __name__ == "__main__":

    # path = 'movielens-TEST'
    path = 'movielens-100k'
    # path = 'movielens-1M'


    ## TEST configurations 
    if 'movielens-TEST' in path:
        # np.set_printoptions(threshold=np.nan)
        maxN = 100
        maxM = 100
        skip_connections = True
        units = 32
        latent_features = 5
        learning_rate = 0.01

    ## 100k Configs
    if 'movielens-100k' in path:
        maxN = 943
        maxM = 1682
        # maxN = 100
        # maxM = 100
        skip_connections = True
        units = 32
        latent_features = 20
        learning_rate = 0.01

    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 320
        maxM = 220
        skip_connections = True
        units = 32
        latent_features = 10
        learning_rate = 0.001

    
    opts ={'epochs': 30000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/factorized_ae',
           'model_name':'test_fac_ae',
           'verbose':2,
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
           'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':maxM,#num movies per submatrix
           'visualize':False,
           'save':False,
           'loss':'mse',
           'data_path':path,
           'encoder':[
               {'type':'matrix_dense', 'units':units, "theta_0": True, "theta_3": True, 'overparam':False},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':False, 'overparam':False},
               # {'type':'matrix_dropout'},
               # {'type':'matrix_dense', 'units':units, 'skip_connections':False, 'overparam':False},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':latent_features, 'activation':None, "theta_0": True, "theta_3": True, 'overparam':False},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool'},
               ],
            'decoder':[
               {'type':'matrix_dense', 'units':units},
               {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':False, 'overparam':False},
               {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':False, 'overparam':False},
               {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':False, 'overparam':False},
               {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':False, 'overparam':False},
               {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':1, 'activation':None, 'theta_4': True, 'theta_5': True, "bilinear": False, 'overparam':False},
               # {'type':'predict_mean'}
            ],
            'defaults':{#default values for each layer type (see layer.py)
                'matrix_dense':{
                    #'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    'activation':tf.nn.relu,
                    'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'max',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),# tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                    'skip_connections':False,
                    'overparam':False, # whether to use the different parameters for each movie/user 
                },
                'dense':{#not used
                    'activation':tf.nn.elu, 
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'matrix_pool':{
                    'pool_mode':'mean',
                },
                'matrix_dropout':{
                    'rate':.5, #.6
                },
                'predict_mean':{},                
            },
           'lr':learning_rate,
    }
    
    main(opts)


