import tensorflow as tf
from base import Model
from util import *

def sample_submatrix(mask_,#mask, used for getting concentrations
                     maxN, maxM):
    pN, pM = mask_.sum(axis=1)[:,0], mask_.sum(axis=0)[:,0]
    pN /= pN.sum()
    pM /= pM.sum()
    N, M, _ = mask_.shape
    for n in range(N // maxN):
        for m in range(M // maxM):
            if N == maxN:
                ind_n = np.arange(N)
            else:
                #ind_n = np.arange((n*maxN),(n+1)*maxN)#
                ind_n = np.random.choice(N, size=maxN, replace=False, p = pN)
            if M == maxM:
                ind_m = np.arange(M)
            else:
                #ind_m = np.arange((m*maxM),(m+1)*maxM)
                ind_m = np.random.choice(M, size=maxM, replace=False, p = pM)
            yield ind_n, ind_m #inmat_[ind_n,...][:, ind_m, :], mask_[ind_n,...][:, ind_m, :] # see whether ix_ is faster


def kld_loss_fn(mean_vec, std_vec):
    return -0.5  * tf.reduce_mean((1 + tf.log(std_vec) - (mean_vec)**2 - std_vec))
    
def rec_loss_fn(mat, mask, rec):
    return ((tf.reduce_sum(((mat - rec)**2)*mask))/tf.reduce_sum(mask))

def rec_loss_val_fn(mat, mask, rec):
    return ((tf.reduce_sum(((mat - rec)**2)*mask))/tf.reduce_sum(mask))

def main(opts):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    data = get_data('movielens-100k', train=.8, valid=.2, test=.001)
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat'].shape
    maxN, maxM = opts['maxN'], opts['maxM']
    with tf.Graph().as_default():
        mat = tf.placeholder(tf.float32, shape=(maxN, maxM, num_features), name='mat')
        mask_tr = tf.placeholder(tf.float32, shape=(maxN, maxM, 1), name='mask_tr')
        mat_val = tf.placeholder(tf.float32, shape=(N, M, num_features), name='mat')        
        mask_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_val')
        mask_tr_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_tr_val')
        
        with tf.variable_scope("encoder"):
            encoder = Model(layers=opts['encoder'],
                            layer_defaults=opts['defaults'],verbose=2)
            out_enc_tr = encoder.get_output({'input':mat, 'mask':mask_tr})
            out_enc_val = encoder.get_output({'input':mat_val, 'mask':mask_tr_val}, reuse=True, verbose=0, is_training=False)
            num_hid = out_enc_tr['nvec'].get_shape().as_list()[2]//2
            num_hid_val = out_enc_val['nvec'].get_shape().as_list()[2]//2
            mean_n_tr, std_n_tr = out_enc_tr['nvec'][:,:,:num_hid], tf.exp(out_enc_tr['nvec'][:,:,num_hid:])
            mean_m_tr, std_m_tr = out_enc_tr['mvec'][:,:,:num_hid], tf.exp(out_enc_tr['mvec'][:,:,num_hid:])
            mean_n_val = out_enc_val['nvec'][:,:,:num_hid_val]
            mean_m_val = out_enc_val['mvec'][:,:,:num_hid_val]

        rnd_n_tr = mean_n_tr + tf.random_normal([maxN,1,num_hid]) * std_n_tr
        rnd_m_tr = mean_m_tr + tf.random_normal([1,maxM,num_hid]) * std_m_tr
        with tf.variable_scope("decoder"):
            decoder = Model(layers=opts['decoder'],
                            layer_defaults=opts['defaults'],verbose=2)
            out_tr = decoder.get_output({'nvec':rnd_n_tr, 'mvec':rnd_m_tr, 'mask':out_enc_tr['mask']})['input']
            out_val = decoder.get_output({'nvec':mean_n_val, 'mvec':mean_m_val, 'mask':out_enc_val['mask']}, reuse=True, verbose=0, is_training=False)['input']

        #loss and training
        rec_loss = rec_loss_fn(mat, mask_tr, out_tr)# * maxN * maxM
        kld_loss = kld_loss_fn(mean_n_tr, std_n_tr) + kld_loss_fn(mean_m_tr, std_m_tr)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        rec_loss_val = rec_loss_val_fn(mat_val, mask_val, out_val)# * maxN * maxM
        total_loss = rec_loss + reg_loss + kld_loss

        train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        iters_per_epoch = N*M//(maxN * maxM)
        for ep in range(opts['epochs']):
            begin = time.time()
            loss_tr_, rec_loss_tr_, kld_loss_tr_, loss_val_ = 0,0,0,0
            for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM), total=iters_per_epoch):
                inds_ = np.ix_(indn_,indm_,[0])
                _, bloss_, brec_loss_, bkld_loss_ = sess.run([train_step, total_loss, rec_loss, kld_loss], feed_dict={mat: data['mat'][inds_],  mask_tr:data['mask_tr'][inds_]})
                loss_tr_ += bloss_
                rec_loss_tr_ += brec_loss_
                kld_loss_tr_ += bkld_loss_

            loss_tr_ /= iters_per_epoch
            rec_loss_tr_ /= iters_per_epoch
            kld_loss_tr_ /= iters_per_epoch            
        
            #for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], N, maxM), total=iters_per_epoch):
            #    inds_ = np.ix_(indn_,indm_,[0])                
            #bloss_, = sess.run([tf.sqrt(rec_loss_val)], feed_dict={mat:data['mat'][inds_] ,  mask_val:data['mask_val'][inds_], mask_tr:data['mask_tr'][inds_]})
            bloss_, = sess.run([tf.sqrt(rec_loss_val)], feed_dict={mat_val:data['mat'],  mask_val:data['mask_val'], mask_tr_val:data['mask_tr']})
            loss_val_ += bloss_
            #loss_val_ /= iters_per_epoch
            print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f} kld:{:.3f}) \t validation: {:.3f}".format(ep, time.time() - begin,
                                                                                                    loss_tr_, rec_loss_tr_, kld_loss_tr_, loss_val_), flush=True)
            
            
if __name__ == "__main__":
    
    opts ={'batch_size': 16,
           'epochs': 5000,
           'ckpt_folder':'checkpoints/factorized_ae',
           'model_name':'test_fac_ae',
           'verbose':2,
           'maxN':50,
           'maxM':500,
           'visualize':False,
           'save':False,
           'encoder':[
               {'type':'matrix_dense', 'units':128},
               {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':128},
               {'type':'matrix_dropout'},               
               {'type':'matrix_dense', 'units':20, 'activation':None},
               {'type':'matrix_pool'},
               ],
            'decoder':[
               {'type':'matrix_dense', 'units':128},
               {'type':'matrix_dropout'},               
               {'type':'matrix_dense', 'units':128},
               {'type':'matrix_dropout'},               
               {'type':'matrix_dense', 'units':1, 'activation':None},
            ],
            'defaults':{
                'matrix_dense':{
                    'activation':tf.nn.tanh,
                    'drop_mask':True,
                    'mode':'mean',
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'dense':{
                    'activation':tf.nn.elu,                    
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'matrix_pool':{
                    'mode':'mean',
                },
                'matrix_dropout':{
                    'rate':.5,
                }
            },
           'lr':.0003,
    }
    
    main(opts)

