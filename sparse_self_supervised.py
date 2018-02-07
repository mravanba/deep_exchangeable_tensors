from __future__ import print_function

import tensorflow as tf
from base import Model
from util import get_data
from sparse_util import *
import math
import time
from tqdm import tqdm
from collections import OrderedDict
import sys
from sparse_factorized_autoencoder import sample_dense_values_uniform, sample_k_neighbours
from sparse_factorized_autoencoder import one_hot, expected_value

from scipy.sparse import csr_matrix

def sample_dense_values_uniform_val(mask_indices, mask_tr_val_split, minibatch_size, iters_per_epoch):
    num_vals_tr = mask_indices.shape[0]
    for n in range(iters_per_epoch):
        sample_tr = np.random.choice(num_vals_tr, size=minibatch_size, replace=False)
        yield sample_tr, mask_tr_val_split[sample_tr]

def conditional_sample_sparse(mask_indices, tr_val_split, shape, maxN, maxM, valid=False): # AKA Kevin sampling 
    N,M,_ = shape
    num_vals = mask_indices.shape[0]

    for n in range(N // maxN):
        for m in range(M // maxM):

            pN = np.bincount(mask_indices[:,0], minlength=N).astype(np.float32)
            pN /= pN.sum()
            ind_n = np.arange(N)[pN!=0] # If there are 0s in p and replace is False, we cant select N=maxN unique values. Filter out 0s.
            pN = pN[pN!=0]
            maxN = min(maxN, ind_n.shape[0])
            ind_n = np.random.choice(ind_n, size=maxN, replace=False, p=pN)

            select_row = np.in1d(mask_indices[:,0], ind_n)
            rows = mask_indices[select_row==True]

            pM = np.bincount(rows[:,1], minlength=M).astype(np.float32)
            pM /= pM.sum()
            ind_m = np.arange(M)[pM!=0] # If there are 0s in p and replace is False, we cant select M=maxM unique values. Filter out 0s.
            pM = pM[pM!=0] 
            maxM = min(maxM, ind_m.shape[0])
            ind_m = np.random.choice(ind_m, size=maxM, replace=False, p=pM)

            select_col = np.in1d(mask_indices[:,1], ind_m)
            select_row_col = np.logical_and(select_row, select_col)

            inds_all = np.arange(num_vals)[select_row_col==True]
            
            split = tr_val_split[inds_all]
            
            inds_tr = inds_all[split==0]
            inds_val = inds_all[split==1]
            inds_tr_val = inds_all[split<=1]
            inds_ts = inds_all[split==2]
            if valid:
                yield inds_all, split
            else:
                yield inds_all

def conditional_validation(tf_dict, mat_values_both, mask_indices_both_, mask_tr_val_split_, split_id, draw_sample, iters_per_epoch):
    entries_val = np.zeros(mask_indices_both_.shape[0])
    entries_val_count = np.zeros(mask_indices_both_.shape[0])
    entries_tr_val_count = np.zeros(mask_indices_both_.shape[0])
    num_entries_tr_val = mask_indices_both_.shape[0]

    while np.sum(entries_tr_val_count) < .95 * num_entries_tr_val:
        for sample_, split  in tqdm(draw_sample(mask_indices_both_, mask_tr_val_split_), 
                                                    total=iters_per_epoch):
            mat_values = one_hot(mat_values_both[sample_]).reshape((-1, 5))
            mask_indices = mask_indices_both_[sample_]

            noise_mask = (mask_tr_val_split_[sample_] == split_id) * 1.
            no_noise_mask = np.ones_like(noise_mask) - noise_mask
            mat_values_noisy = (mat_values * no_noise_mask[:, None]).flatten()

            val_dict = {tf_dict["mat_values_val"]:mat_values.flatten(),
                        tf_dict["mat_values_val_noisy"]:mat_values_noisy,
                        tf_dict["mask_indices_val"]:mask_indices,
                        tf_dict["noise_mask_val"]:noise_mask
                        }

            beout_val, = tf_dict["sess"].run([tf_dict["ev"]], feed_dict=val_dict)
            sample_val_ = sample_[split == split_id]
            losses_val = (mat_values_both[sample_val_] - beout_val[noise_mask == 1.])**2
            entries_val[sample_val_] = losses_val
            entries_val_count[sample_val_] = 1
            entries_tr_val_count[sample_] = 1

    return np.sqrt(np.sum(entries_val) / np.sum(entries_val_count))

def ce_loss(mat_values, rec_values):
    out = tf.reshape(rec_values, shape=[-1,5])
    mat_values = tf.reshape(mat_values, shape=[-1,5])
    return -tf.reduce_sum(mat_values * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)), axis=1)

def dae_loss_fn_sp(mat_values, rec_values, noise_mask, alpha, mask_split, valid=False):
    noise_mask = tf.cast(noise_mask, tf.float32)
    no_noise_mask = tf.ones_like(noise_mask) - noise_mask
    if valid:
        ev = expected_value(tf.nn.softmax(tf.reshape(rec_values, shape=[-1,5])))
        av = expected_value(tf.reshape(rec_values, shape=[-1,5]))
        diff = (av - ev)**2
    else:
        diff = ce_loss(mat_values, rec_values)
    diff_c = diff * noise_mask * mask_split # mask split = 1 iff sample part of seed set in neighbourhood sampling
    diff_u = diff * no_noise_mask * mask_split
    loss_c = tf.reduce_sum(diff_c)
    loss_u = tf.reduce_sum(diff_u)
    eps = 1e-10
    return alpha * loss_c / (tf.reduce_sum(noise_mask) + eps) + (1-alpha) * loss_u / (tf.reduce_sum(no_noise_mask) + eps)

def ordinal_hinge_loss_fn_sp(mat_values, rec_values, noise_mask, alpha, num_values):
    # num_values = noise_mask.shape[0]
    noise_mask = tf.cast(noise_mask, tf.float32)
    no_noise_mask = tf.ones_like(noise_mask) - noise_mask
    categories = tf.cast(np.reshape(np.tile(range(1,6), reps=num_values), [-1,5]), tf.float32)
    mat_values = tf.transpose(tf.reshape(tf.tile(mat_values, [5]), [-1,num_values]))
    greater = tf.cast(tf.greater_equal(categories, mat_values), tf.float32)
    less = tf.cast(tf.less_equal(categories, mat_values), tf.float32)
    not_equal = tf.cast(tf.not_equal(categories, mat_values), tf.float32)
    rec_values = tf.cast(tf.transpose(tf.reshape(tf.tile(rec_values, [5]), [-1,num_values])), tf.float32)
    rec_values = rec_values * (greater - less)
    rec_values = (rec_values + 1) * not_equal
    out = categories * (less - greater) + rec_values
    out = tf.maximum(out, tf.zeros_like(out))
    out = tf.reduce_sum(out, axis=1)
    out_c = out * noise_mask
    out_u = out * no_noise_mask
    return alpha * tf.reduce_sum(out_c) + (1-alpha) * tf.reduce_sum(out_u) 

def main(opts, logfile=None, restore_point=None):
    if logfile is not None:
        LOG = open(logfile, "w", 0)
    else:
        LOG = sys.stdout
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) 
    path = opts['data_path']
    if 'movielens-100k' in path:
        data = get_data(path, train=.85, valid=.05, test=.1, mode='sparse', fold=1) # ml-100k uses official test set so only the valid paramter matters
    else: 
        data = get_data(path, train=.85, valid=.05, test=.1, mode='sparse', fold=1)
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat_shape']
    maxN, maxM = opts['maxN'], opts['maxM']


    if N < maxN: maxN = N
    if M < maxM: maxM = M

    if opts['verbose'] > 0:
        print('\nSelf supervised run settings:')
        print('dataset: ', path)
        print('Exchangable layer pool mode: ', opts['defaults']['matrix_sparse']['pool_mode'])
        print('learning rate: ', opts['lr'])
        print('activation: ', opts['defaults']['matrix_sparse']['activation'])
        print('dae_noise_rate: ', opts['dae_noise_rate'])
        print('dae_loss_alpha: ', opts['dae_loss_alpha'])
        print('l2_regularization: ', opts['l2_regularization'])
        print('')

    with tf.Graph().as_default():
        # with tf.device('/gpu:0'):
            mat_values_tr = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr')
            mask_split = tf.placeholder(tf.float32, shape=[None], name='mask_split')
            mat_values_tr_noisy = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr_noisy')
            mask_indices_tr = tf.placeholder(tf.int64, shape=[None, 2], name='mask_indices_tr')
            mat_shape_tr = tf.placeholder(tf.int32, shape=[3], name='mat_shape_tr')
            noise_mask_tr = tf.placeholder(tf.int64, shape=(None), name='noise_mask_tr')

            mat_values_val = tf.placeholder(tf.float32, shape=[None], name='mat_values_val')
            mat_values_val_noisy = tf.placeholder(tf.float32, shape=[None], name='mat_values_val_noisy')
            mask_indices_val = tf.placeholder(tf.int64, shape=[None, 2], name='mask_indices_val')
            mat_shape_val = tf.placeholder(tf.int32, shape=[3], name='mat_shape_val')
            noise_mask_val = tf.placeholder(tf.int64, shape=(None), name='noise_mask_val')
            
            
            with tf.variable_scope("network"):
                tr_dict = {'input':mat_values_tr_noisy,
                           'mask_indices':mask_indices_tr,
                           'units':5,
                           'shape':[N,M]}

                val_dict = {'input':mat_values_val_noisy,
                            'mask_indices':mask_indices_val,
                            'units':5,
                            'shape':[N,M]}

                network = Model(layers=opts['network'], layer_defaults=opts['defaults'], verbose=2) #define the network
                out_tr = network.get_output(tr_dict)['input'] #build the network
                
                out_val = network.get_output(val_dict, reuse=True, verbose=0, is_training=False)['input']#get network output, reusing the neural net
            

            iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM)

            #loss and training
            rec_loss = dae_loss_fn_sp(mat_values_tr, out_tr, noise_mask_tr, opts['dae_loss_alpha'], mask_split)
            #rec_loss = ordinal_hinge_loss_fn_sp(mat_values_tr, out_tr, noise_mask_tr, opts['dae_loss_alpha'], minibatch_size)
            reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization            
            total_loss = rec_loss + reg_loss

            ev = expected_value(tf.nn.softmax(tf.reshape(out_val, shape=[-1,5])))
            av = expected_value(tf.reshape(mat_values_val, shape=[-1,5]))
            nm = tf.cast(noise_mask_val, tf.float32)
            rec_loss_val = tf.reduce_sum((av - ev)**2 * nm) / tf.reduce_sum(nm)
            # rec_loss_val = dae_loss_fn_sp(mat_values_val, out_val, noise_mask_val, 1, valid=True)

            train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
            sess.run(tf.global_variables_initializer())
            
            min_loss = np.inf
            min_train = np.inf
            min_loss_epoch = 0
            losses = OrderedDict()
            losses["train"] = []
            losses["valid"] = []
            losses["test"] = []
            min_ts_loss = np.inf
            min_val_ts = np.inf

            noise_rate = opts['dae_noise_rate']
            sample_mode = opts.get('sample_mode', 'conditional_sample_sparse')
            if 'conditional_sample_sparse' in sample_mode:
                iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
            elif 'uniform_over_dense_values' in sample_mode:
                minibatch_size = np.minimum(opts['minibatch_size'], data['mask_indices_tr'].shape[0])
                iters_per_epoch = data['mask_indices_tr'].shape[0] // minibatch_size
            elif 'neighbourhood' in sample_mode:
                minibatch_size = np.minimum(opts['minibatch_size'], data['mask_indices_tr'].shape[0])
                hops = opts.get("n_hops", 3)
                n_samp = opts.get("n_neighbours", None) # None for "get all neighbours"
                print("Using neighbourhood sampling with %d hops and %s samples" % (hops, n_samp))
                sp_mat = csr_matrix((data['mat_values_all'], 
                    (data['mask_indices_all'][:,0], 
                    data['mask_indices_all'][:, 1])),
                    data["mat_shape"][0:2])

            saver = tf.train.Saver()
            if restore_point is not None:
                saver.restore(sess, restore_point)

            best_log = "logs/best_" + opts.get("model_name", "TEST") + ".log"
            print("epoch,train,valid,test\n", file=open(best_log, "a"))
            restore_point_epoch = opts.get('restore_point_epoch', 0)
            
            for ep in range(restore_point_epoch, opts['epochs'] + restore_point_epoch):
                begin = time.time()
                loss_tr_, rec_loss_tr_, loss_val_, loss_ts_ = 0.,0.,0.,0.
                if 'conditional_sample_sparse' in sample_mode or 'uniform_over_dense_values' in sample_mode:
                    if 'conditional_sample_sparse' in sample_mode:
                        # set up helper for drawing sample with common interface so we can reuse code between 
                        # 'conditional_sample_sparse' and 'uniform_over_dense_values'
                        draw_sample = lambda mask, split: conditional_sample_sparse(mask, split, [N,M,1], maxN, maxM)
                        draw_sample_val = lambda mask, split: conditional_sample_sparse(mask, split, [N,M,1], maxN, maxM, valid=True)
                    else:
                        draw_sample = lambda mask, split: sample_dense_values_uniform(mask, minibatch_size, iters_per_epoch)
                        draw_sample_val = lambda mask, split: sample_dense_values_uniform_val(mask, split, minibatch_size, iters_per_epoch)
                    
                    for sample_ in tqdm(draw_sample(data['mask_indices_tr'], data['mask_tr_val_split']), total=iters_per_epoch):
                        mat_values = one_hot(data['mat_values_tr'][sample_])
                        mask_indices = data['mask_indices_tr'][sample_]

                        # which entries to 'corrupt' by dropping out 
                        noise_mask = np.random.choice([0,1], size=mask_indices.shape[0], p=[1-noise_rate, noise_rate])
                        no_noise_mask = np.ones_like(noise_mask) - noise_mask
                        mat_values_noisy = (mat_values.reshape((-1, 5)) * no_noise_mask[:, None]).flatten()
                        
                        tr_dict = {mat_values_tr:mat_values,
                                    mat_values_tr_noisy:mat_values_noisy,
                                    mask_indices_tr:mask_indices,
                                    noise_mask_tr:noise_mask,
                                    mask_split:np.ones_like(noise_mask)
                                    }

                        _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)

                        loss_tr_ += np.sqrt(bloss_)
                        rec_loss_tr_ += np.sqrt(brec_loss_)
                
                elif 'neighbourhood' in sample_mode:
                    iters_per_epoch = max(1,data['mask_indices_tr'].shape[0] /  minibatch_size)
    
                    for seed_set_idx in tqdm(sample_dense_values_uniform(data['mask_indices_tr'], minibatch_size, iters_per_epoch),
                                        total=iters_per_epoch):
                        seed_set = data['mask_indices_tr'][seed_set_idx]
                        neighbours = sample_k_neighbours(seed_set, data['mask_indices_tr'], hops, n_samp)
                        mask_indices_ = np.concatenate([seed_set, neighbours], axis=0)
                        mask_split_ = np.concatenate([np.ones(seed_set.shape[0]), np.zeros(neighbours.shape[0])]) # only evaluate the seed set
                        mat_values_ = np.array(sp_mat[mask_indices_[:,0], mask_indices_[:,1]]).flatten()
                        mat_values_ = one_hot(mat_values_)

                        # which entries to 'corrupt' by dropping out 
                        noise_mask = mask_split_
                        no_noise_mask = np.ones_like(noise_mask) - noise_mask
                        mat_values_noisy = (mat_values_.reshape((-1, 5)) * no_noise_mask[:, None]).flatten()

                        tr_dict = {mat_values_tr:mat_values_,
                                   mat_values_tr_noisy:mat_values_noisy,
                                   noise_mask_tr:noise_mask,
                                   mask_indices_tr:mask_indices_,
                                   mask_split:mask_split_
                                  }
                        
                        _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)

                        loss_tr_ += bloss_
                        rec_loss_tr_ += np.sqrt(brec_loss_)
                else:
                    raise KeyError("Unrecognized sample mode: %s" % sample_mode)

                loss_tr_ /= iters_per_epoch
                rec_loss_tr_ /= iters_per_epoch
                losses['train'].append(loss_tr_)

                print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f})".format(ep+1, time.time() - begin, loss_tr_, rec_loss_tr_))            

                if (ep+1) % opts.get("checkpoint_interval", 10000000) == 0:
                    save_path = saver.save(sess, opts['ckpt_folder'] + "/%s_checkpt_ep_%05d.ckpt" % (opts.get('model_name', "test"), ep + 1))
                    print("Model saved in file: %s" % save_path, file=LOG)      
                    
                if (ep+1) % opts['validate_interval'] == 0:
                    tf_dict = {"sess": sess, "mat_values_val":mat_values_val, "mat_values_val_noisy":mat_values_val_noisy,
                               "mask_indices_val":mask_indices_val, "noise_mask_val":noise_mask_val, "ev":ev}
                    
                    if 'conditional_sample_sparse' in sample_mode:
                        loss_val_ = conditional_validation(tf_dict, data['mat_values_tr_val'], data['mask_indices_tr_val'], 
                                                        data['mask_tr_val_split'], split_id=1, draw_sample=draw_sample_val, 
                                                        iters_per_epoch=iters_per_epoch)
                        loss_ts_ = conditional_validation(tf_dict, data['mat_values_all'], data['mask_indices_all'], 
                                                        data['mask_tr_val_split'], split_id=2, draw_sample=draw_sample_val, 
                                                        iters_per_epoch=iters_per_epoch)
                    elif 'uniform_over_dense_values' in sample_mode:
                        loss_val_ = conditional_validation(tf_dict, data['mat_values_tr_val'], data['mask_indices_tr_val'], 
                                                        data['mask_tr_val_split'], split_id=1, draw_sample=draw_sample_val, 
                                                        iters_per_epoch=iters_per_epoch)
                        loss_ts_ = conditional_validation(tf_dict, data['mat_values_all'], data['mask_indices_all'], 
                                                        data['mask_tr_val_split'], split_id=2, draw_sample=draw_sample_val, 
                                                        iters_per_epoch=iters_per_epoch)
                    elif 'neighbourhood' in sample_mode:

                        mask_idx_val = data['mask_indices_all'][data['mask_tr_val_split']==1]
                        mask_idx_train = data['mask_indices_tr']
                        mat_values_val_ = data['mat_values_all'][data['mask_tr_val_split']==1]
                        idxes = np.random.permutation(np.arange(mat_values_val_.shape[0]))
                        iters_per_val = max(1, mat_values_val_.shape[0] / minibatch_size)
                        for seed_set_id in tqdm(np.array_split(idxes, iters_per_val), total=iters_per_val):
                            seed_set = mask_idx_val[seed_set_id]
                            neighbours = sample_k_neighbours(seed_set, mask_idx_train, hops, n_samp)
                            mask_indices_ = np.concatenate([seed_set, neighbours], axis=0)
                            mat_values_ = np.array(sp_mat[mask_indices_[:,0], mask_indices_[:,1]]).flatten()
                            mat_values_ = one_hot(mat_values_)

                            noise_mask = np.concatenate([np.ones(seed_set.shape[0]), np.zeros(neighbours.shape[0])]) # only evaluate the seed set
                            no_noise_mask = np.ones_like(noise_mask) - noise_mask
                            mat_values_noisy = (mat_values_.reshape((-1,5)) * no_noise_mask[:, None]).flatten()

                            test_dict = {mat_values_val:mat_values_.flatten(),
                                        mat_values_val_noisy:mat_values_noisy,
                                        mask_indices_val:mask_indices_,
                                        noise_mask_val:noise_mask
                                        }
                            bloss_val_, = sess.run([rec_loss_val], feed_dict=test_dict)
                            loss_val_ = np.sqrt(bloss_val_)

                    losses['valid'].append(loss_val_)
                    losses['test'].append(loss_ts_)

                    if loss_val_ < min_loss: # keep track of the best validation loss 
                        min_loss = loss_val_
                        min_loss_epoch = ep+1
                        min_train = rec_loss_tr_
                        min_test = loss_ts_
                        print("{:d},{:4},{:4},{:4}\n".format(ep, loss_tr_, loss_val_, loss_ts_), file=open(best_log, "a"))
                        if opts.get("save_best", False): 
                            save_path = saver.save(sess, opts['ckpt_folder'] + "/%s_best.ckpt" % opts.get('model_name', "test"))
                            print("Model saved in file: %s" % save_path, file=LOG)
                
                    if loss_ts_ < min_ts_loss: # keep track of the best test loss 
                        min_ts_loss = loss_ts_
                        min_val_ts = loss_val_     
                
                    print("Validation: epoch {:d} took {:.1f} train loss {:.3f} (rec:{:.3f}); valid: {:.3f}; min valid loss: {:.3f} (train: {:.3}, test: {:.3}) at epoch: {:d}; test loss: {:.3f} (best test: {:.3f} with val {:.3f})"
                                    .format(ep+1, 
                                            time.time() - begin, 
                                            loss_tr_, 
                                            rec_loss_tr_, 
                                            loss_val_, 
                                            min_loss, 
                                            min_train, 
                                            min_test, 
                                            min_loss_epoch, 
                                            loss_ts_,
                                            min_ts_loss, 
                                            min_val_ts), 
                                    file=LOG)
    return losses    

if __name__ == "__main__":

    auto_restore = False

    # path = 'movielens-TEST'
    # path = 'movielens-100k'
    path = 'movielens-1M'
    # path = 'netflix/6m'

    ## 100k Configs
    if 'movielens-100k' in path:
        maxN = 300
        maxM = 300
        minibatch_size = 2000000
        skip_connections = False
        units = 12
        learning_rate = 0.005
        dae_noise_rate = .1 # drop out this proportion of input values 
        dae_loss_alpha = 1.  # proportion of loss assigned to predicting droped out values 
        l2_regularization = .00001
        validate_interval = 5
        checkpoint_interval = 5


    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 800
        maxM = 800
        minibatch_size = 10000
        skip_connections = True
        units = 256
        learning_rate = 0.0005
        dae_noise_rate = .1 # drop out this proportion of input values 
        dae_loss_alpha = 1.  # proportion of loss assigned to predicting droped out values 
        l2_regularization = .00001
        validate_interval = 1
        checkpoint_interval = 20

    if 'netflix/6m' in path:
        maxN = 300
        maxM = 300
        minibatch_size = 500000
        skip_connections = True
        units = 128
        learning_rate = 0.0005
        dae_noise_rate = .1 # drop out this proportion of input values 
        dae_loss_alpha = .7  # proportion of loss assigned to predicting droped out values 
        l2_regularization = .00001
        validate_interval = 5
        checkpoint_interval = 1

    if 'netflix/full' in path:
        maxN = 300
        maxM = 300
        minibatch_size = 2000000
        skip_connections = True
        units = 32
        learning_rate = 0.001
        dae_noise_rate = .1 # drop out this proportion of input values 
        dae_loss_alpha = .7  # proportion of loss assigned to predicting droped out values 
        l2_regularization = .00001
        validate_interval = 5
        checkpoint_interval = 1


    opts ={'epochs': 50000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/self_supervised',
           'model_name':'ss_ae',
           'verbose':2,
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
           'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':maxM,#num movies per submatrix
           'minibatch_size':minibatch_size,
           'visualize':False,
           'save':False,
           'data_path':path,
           'output_file':'output',
           'validate_interval':validate_interval,
           'checkpoint_interval':checkpoint_interval,
           'save_best':True,
           'network':[
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               {'type':'channel_dropout_sparse'},               
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               {'type':'channel_dropout_sparse'},                              
               #{'type':'channel_dropout_sparse'},
               #{'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               #{'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units, 'skip_connections':skip_connections},
               {'type':'matrix_sparse', 'units':5, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               ],
            'defaults':{#default values for each layer type (see layer.py)
                'matrix_sparse':{
                    # 'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    'activation':tf.nn.relu,
                    # 'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'mean',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(l2_regularization),
                    'skip_connections':False,
                },
                'dense':{#not used
                    'activation':tf.nn.elu,
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },                
                'matrix_dropout_sparse':{
                    'rate':.2,
                },
                'channel_dropout_sparse':{
                    'rate':.5,
                },
                'matrix_pool_sparse':{
                    'pool_mode':'max',
                },
                
            },
           'lr':learning_rate,
           'sample_mode': 'neighbourhood', # conditional_sample_sparse, by_row_column_density, uniform_over_dense_values
           'n_hops':6,
           'n_neighbours':10000,
           'dae_noise_rate':dae_noise_rate,
           'dae_loss_alpha':dae_loss_alpha,
           'l2_regularization':l2_regularization,
    }
    if auto_restore:        

        restore_point_epoch = sorted(glob.glob(opts['ckpt_folder'] + "/%s_checkpt_ep_*.ckpt*" % (opts.get('model_name', "test"))))[-1].split(".")[0].split("_")[-1]
        restore_point = opts['ckpt_folder'] + "/%s_checkpt_ep_" % (opts.get('model_name', "test")) + restore_point_epoch + ".ckpt"
        print("Restoring from %s" % restore_point)

        opts["restore_point_epoch"] = int(restore_point_epoch) # Pass num_epochs so far to start counting from there. In case of another crash 


    else:
        restore_point = None
    main(opts, restore_point=restore_point)



