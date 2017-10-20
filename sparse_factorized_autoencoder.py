from __future__ import print_function

import tensorflow as tf
from base import Model
from util import get_data
from sparse_util import *
import math
import time
from tqdm import tqdm

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


def sample_submatrix_sp(mask_indices, N, M, maxN, maxM):
    max_val = mask_indices.shape[0]
    num_samples = np.minimum(max_val, maxN*maxM)    
    for n in range(N // maxN):
        for m in range(M // maxM):
            sample = np.random.choice(max_val, size=num_samples, replace=False)
            sample = np.sort(sample)
            yield sample


def rec_loss_fn_sp(mat_values, mask_indices, mask_split, rec_values):
    mask_indices = tf.cast(mask_indices, tf.float32) # Casting to floats so calculation can be run on GPU
    mask_split = tf.cast(mask_split, tf.float32)
    diffs = (mat_values - rec_values)**2
    return tf.squeeze(tf.matmul(tf.reshape(diffs, [1,-1]), tf.reshape(mask_split, [-1,1]))) / tf.cast(tf.shape(mask_indices)[0], tf.float32)

# def vae_loss_fn_sp(mat_values, mask_indices, mask_split, rec_values):
#     return 0


def main(opts):        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)    
    path = opts['data_path']
    data = get_data(path, train=.8, valid=.2, test=.001)
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat_tr_val'].shape
    maxN, maxM = opts['maxN'], opts['maxM']

    if N < maxN: maxN = N
    if M < maxM: maxM = M

    if opts['verbose'] > 0:
        print('\nFactorized Autoencoder run settings:')
        print('dataset: ', path)
        print('drop mask: ', opts['defaults']['matrix_sparse']['drop_mask'])
        print('Exchangable layer pool mode: ', opts['defaults']['matrix_sparse']['pool_mode'])
        print('Pooling layer pool mode: ', opts['defaults']['matrix_pool_sparse']['pool_mode'])
        print('learning rate: ', opts['lr'])
        print('activation: ', opts['defaults']['matrix_sparse']['activation'])
        #print('use skip connections is middle layers: ', skip_connections)
        #print('number of features: ', units)
        print('number of latent features: ', opts['encoder'][-2]['units'])
        print('maxN: ', opts['maxN'])
        print('maxM: ', opts['maxM'])
        print('')



    output_filename = opts['output_file'] + str(time.time())
    f = open(output_filename, 'w')
    f.write('\nFactorized Autoencoder run settings:')
    f.write('dataset: ' + path + '\n')
    f.write('drop mask: ' + str(opts['defaults']['matrix_sparse']['drop_mask']) + '\n')
    f.write('Exchangable layer pool mode: ' + opts['defaults']['matrix_sparse']['pool_mode'] + '\n')
    f.write('Pooling layer pool mode: ' + opts['defaults']['matrix_pool_sparse']['pool_mode'] + '\n')
    f.write('learning rate: ' + str(opts['lr']) + '\n')
    f.write('activation: ' + str(opts['defaults']['matrix_sparse']['activation']) + '\n')
    #f.write('use skip connections is middle layers: ' + str(skip_connections) + '\n')
    #f.write('number of features: ' + str(units) + '\n')
    #f.write('number of latent features: ' + str(latent_features) + '\n')
    f.write('maxN: ' + str(opts['maxN']) + '\n')
    f.write('maxM: ' + str(opts['maxM']) + '\n')
    f.write('\n')

    with tf.Graph().as_default():
        # with tf.device('/gpu:0'):

        mat_values_tr = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr')
        mask_indices_tr = tf.placeholder(tf.int64, shape=[None, 2], name='mask_indices_tr')
        # mat_values_val = tf.placeholder(tf.float32, shape=[None], name='mat_values_val')
        mask_indices_val = tf.placeholder(tf.int64, shape=[None, 2], name='mask_indices_val')
        mat_values_tr_val = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr_val')
        mask_indices_tr_val = tf.placeholder(tf.int64, shape=[None, 2], name='mask_indices_tr_val')
        mask_tr_val_split = tf.placeholder(tf.int64, shape=[None], name='mask_tr_val_split')

        mat_values_ts = tf.placeholder(tf.float32, shape=[None], name='mat_values_ts')
        mask_indices_ts = tf.placeholder(tf.int64, shape=[None, 2], name='mask_indices_ts')

        ##.....
        mat_tr = tf.SparseTensorValue(expand_tensor_indices(mask_indices_tr, 1), mat_values_tr, [maxN,maxM,num_features])
        mat_val = tf.SparseTensorValue(expand_tensor_indices(mask_indices_val, 1), mat_values_tr_val, [N,M,num_features])
        # mat_ts = tf.SparseTensorValue(expand_tensor_indices(mask_indices_ts, 1), mat_values_ts, [N,M,num_features])

        with tf.variable_scope("encoder"):
            tr_dict = {'input':mat_tr,
                       'mask_indices':mask_indices_tr,
                       'shape':[maxN,maxM,num_features]} ## Passing in shape to be used in sparse functions
            val_dict = {'input':mat_val,
                        'mask_indices':mask_indices_tr_val,
                        'shape':[N,M,num_features]}

            # ts_dict = {'input':mat_ts,
            #             'mask_indices':mask_indices_ts,
            #             'shape':[N,M,num_features]}

            encoder = Model(layers=opts['encoder'], layer_defaults=opts['defaults'], verbose=2) #define the encoder
            out_enc_tr = encoder.get_output(tr_dict) #build the encoder
            out_enc_val = encoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#get encoder output, reusing the neural net

            # out_enc_ts = encoder.get_output(ts_dict, reuse=True, verbose=0, is_training=False)
            
        with tf.variable_scope("decoder"):
            tr_dict = {'nvec':out_enc_tr['nvec'],
                       'mvec':out_enc_tr['mvec'],
                       'mask_indices':out_enc_tr['mask_indices'],
                       'shape':[maxN,maxM,out_enc_tr['shape'][2]]} ## Passing in shape to be used in sparse functions 
            val_dict = {'nvec':out_enc_val['nvec'],
                        'mvec':out_enc_val['mvec'],
                        'mask_indices':out_enc_val['mask_indices'],
                        'shape':[N,M,out_enc_val['shape'][2]]}

            # ts_dict = {'nvec':out_enc_ts['nvec'],
            #             'mvec':out_enc_ts['mvec'],
            #             'mask_indices':out_enc_ts['mask_indices'],
            #             'shape':[N,M,out_enc_ts['shape'][2]]}

            decoder = Model(layers=opts['decoder'], layer_defaults=opts['defaults'], verbose=2)#define the decoder
            out_dec_tr = decoder.get_output(tr_dict)#build it
            out_tr = out_dec_tr['input']

            out_dec_val = decoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#reuse it for validation
            out_val = out_dec_val['input']

            # out_dec_ts = decoder.get_output(ts_dict, reuse=True, verbose=0, is_training=False)
            # out_ts = out_dec_ts['input']

        #loss and training
        rec_loss = rec_loss_fn_sp(mat_values_tr, mask_indices_tr, tf.ones_like(mat_values_tr, dtype=tf.float32), out_tr.values)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
        rec_loss_val = rec_loss_fn_sp(mat_values_tr_val, mask_indices_val, mask_tr_val_split, out_val.values)
        total_loss = rec_loss + reg_loss

        # rec_loss_ts = rec_loss_fn_sp(mat_values_ts, mask_indices_ts, tf.ones_like(mat_values_ts, dtype=tf.float32), out_ts.values)

        train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
        # train_step = tf.train.AdamOptimizer(opts['lr'], opts['adam_b1'], opts['adam_b2']).minimize(total_loss)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch

        min_loss = 5
        min_loss_epoch = 0
        
        for ep in range(opts['epochs']):
            begin = time.time()
            loss_tr_, rec_loss_tr_, loss_val_, loss_ts_ = 0,0,0,0
            # for sample_ in tqdm(sample_submatrix_sp(data['mask_indices_tr'], N, M, maxN, maxM), total=iters_per_epoch):# Sampling dense indices directly doesnt work right now.

            #     tr_dict = {mat_values_tr:data['mat_values_tr'][sample_],
            #                 mask_indices_tr:data['mask_indices_tr'][sample_]}

            for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM), total=iters_per_epoch):#go over mini-batches
                inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies

                mat_sp = data['mat_tr_val'][inds_] * data['mask_tr'][inds_]
                mat_sp = dense_array_to_sparse(mat_sp)
                mask_tr_sp = dense_array_to_sparse(data['mask_tr'][inds_])
                
                tr_dict = {mat_values_tr:mat_sp['values'],
                            mask_indices_tr:mask_tr_sp['indices'][:,0:2]}



                _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)


                loss_tr_ += np.sqrt(bloss_)
                rec_loss_tr_ += np.sqrt(brec_loss_)

            loss_tr_ /= iters_per_epoch
            rec_loss_tr_ /= iters_per_epoch

            ## Validation Loss
            val_dict = {mat_values_tr_val:data['mat_values_tr_val'],
                        mask_indices_val:data['mask_indices_val'],
                        mask_indices_tr_val:data['mask_indices_tr_val'],
                        mask_tr_val_split:data['mask_tr_val_split']}

            bloss_, = sess.run([rec_loss_val], feed_dict=val_dict)
            loss_val_ += np.sqrt(bloss_)
            if loss_val_ < min_loss: # keep track of the best validation loss 
                min_loss = loss_val_
                min_loss_epoch = ep

            # ## Test Loss
            # ts_dict = {mat_values_ts:data['mat_values_ts'],
            #             mask_indices_ts:data['mask_indices_ts']}

            # bloss_, = sess.run([rec_loss_ts], feed_dict=ts_dict)
            # loss_ts_ += np.sqrt(bloss_)                        

            print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t validation: {:.3f} \t minimum validation loss: {:.3f} at epoch: {:d} \t test loss: {:.3f}".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_,  loss_val_, min_loss, min_loss_epoch, loss_ts_))            
            f.write("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t validation: {:.3f} \t minimum validation loss: {:.3f} at epoch: {:d} \t test loss: {:.3f}\n".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_,  loss_val_, min_loss, min_loss_epoch, loss_ts_))
        f.close()



if __name__ == "__main__":

    # path = 'movielens-TEST'
    path = 'movielens-100k'
    # path = 'movielens-1M'
    # path = 'netflix-full'

    ## Val loss: .891 after 270 epochs
    # if 'movielens-100k' in path:
    #     maxN = 100    
    #     maxM = 100
    #     skip_connections = True
    #     units = 32
    #     latent_features = 5
    #     learning_rate = 0.001

    ## 100k Configs
    if 'movielens-100k' in path:
        maxN = 100    
        maxM = 100
        skip_connections = True
        units = 32
        latent_features = 5
        learning_rate = 0.001

    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 640
        maxM = 220
        skip_connections = True
        units = 54
        latent_features = 10
        learning_rate = 0.001


    if 'netflix-full' in path:
        maxN = 100    
        maxM = 100
        skip_connections = True
        units = 32
        latent_features = 5
        learning_rate = 0.001

    opts ={'epochs': 5000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/factorized_ae',
           'model_name':'test_fac_ae',
           'verbose':2,
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
           'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':maxM,#num movies per submatrix
           'visualize':False,
           'save':False,
           'data_path':path,
           'output_file':'output',
           'encoder':[
               {'type':'matrix_sparse', 'units':units},
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               # {'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':latent_features, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool_sparse'},
               ],
            'decoder':[
               {'type':'matrix_sparse', 'units':units},
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               # {'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':1, 'activation':None},
            ],
            'defaults':{#default values for each layer type (see layer.py)                
                'matrix_sparse':{
                    # 'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    'activation':tf.nn.relu,
                    'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'mean',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                    'skip_connections':False,
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
                    'rate':.5,
                }
            },
           'lr':learning_rate,
           # 'adam_b1':.2,
           # 'adam_b2':.7,
           # 'adam_ep':1,
    }
    
    main(opts)


