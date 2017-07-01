import tensorflow as tf
from base import Model
from util import *

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


def rec_loss_fn(mat, mask, rec):
    return ((tf.reduce_sum(((mat - rec)**2)*mask))/tf.reduce_sum(mask))#average l2-error over non-zero entries


def main(opts):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # data = get_data('movielens-100k', train=.8, valid=.2, test=.001)
    data = get_data('movielens-1M', train=.8, valid=.2, test=.001)
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat'].shape
    maxN, maxM = opts['maxN'], opts['maxM']
    with tf.Graph().as_default():
        mat = tf.placeholder(tf.float32, shape=(maxN, maxM, num_features), name='mat')#data matrix for training
        mask_tr = tf.placeholder(tf.float32, shape=(maxN, maxM, 1), name='mask_tr')
        #for validation, since we need less memory (forward pass only), we are feeding the whole matrix. This is only feasible for this smaller dataset. In the long term we could perform validation on CPU to avoid memory problems
        mat_val = tf.placeholder(tf.float32, shape=(N, M, num_features), name='mat')##data matrix for validation: 
        mask_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_val')#the entries not present during training
        mask_tr_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_tr_val')#both training and validation entries
        
        with tf.variable_scope("encoder"):
            encoder = Model(layers=opts['encoder'],
                            layer_defaults=opts['defaults'],verbose=2) #define the encoder
            out_enc_tr = encoder.get_output({'input':mat, 'mask':mask_tr}) #build the encoder
            out_enc_val = encoder.get_output({'input':mat_val, 'mask':mask_tr_val}, reuse=True, verbose=0, is_training=False)#get encoder output, reusing the neural net
            

        with tf.variable_scope("decoder"):
            decoder = Model(layers=opts['decoder'],
                            layer_defaults=opts['defaults'],verbose=2)#define the decoder
            out_tr = decoder.get_output({'nvec':out_enc_tr['nvec'], 'mvec':out_enc_tr['mvec'], 'mask':out_enc_tr['mask']})['input']#build it
            out_val = decoder.get_output({'nvec':out_enc_val['nvec'], 'mvec':out_enc_val['mvec'], 'mask':out_enc_val['mask']}, reuse=True, verbose=0, is_training=False)['input']#reuse it for validation

        #loss and training
        rec_loss = rec_loss_fn(mat, mask_tr, out_tr)# reconstruction loss
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
        rec_loss_val = rec_loss_fn(mat_val, mask_val, out_val)
        total_loss = rec_loss + reg_loss 

        train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        iters_per_epoch = N//maxN * M//maxM # a bad heuristic: the whole matrix is in expectation covered in each epoch
        
        for ep in range(opts['epochs']):
            begin = time.time()
            loss_tr_, rec_loss_tr_, loss_val_ = 0,0,0
            for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM), total=iters_per_epoch):#go over mini-batches
                inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies
                _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict={mat: data['mat'][inds_],  mask_tr:data['mask_tr'][inds_]})
                loss_tr_ += np.sqrt(bloss_)
                rec_loss_tr_ += np.sqrt(brec_loss_)

            loss_tr_ /= iters_per_epoch
            rec_loss_tr_ /= iters_per_epoch
        
            bloss_, = sess.run([rec_loss_val], feed_dict={mat_val:data['mat'],  mask_val:data['mask_val'], mask_tr_val:data['mask_tr']})
            loss_val_ += np.sqrt(bloss_)
            print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t validation: {:.3f}".format(ep, time.time() - begin,
                                                                                                    loss_tr_, rec_loss_tr_,  loss_val_), flush=True)
            
            
if __name__ == "__main__":
    
    opts ={'epochs': 5000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/factorized_ae',
           'model_name':'test_fac_ae',
           'verbose':2,
           'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':1682,#num movies per submatrix
           'visualize':False,
           'save':False,
           'encoder':[
               {'type':'matrix_dense', 'units':32},
               {'type':'matrix_dropout'},
               #{'type':'matrix_dense', 'units':32},
               #{'type':'matrix_dropout'},               
               {'type':'matrix_dense', 'units':32},
               {'type':'matrix_dropout'},               
               {'type':'matrix_dense', 'units':5, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool'},
               ],
            'decoder':[
               {'type':'matrix_dense', 'units':32},
               {'type':'matrix_dropout'},               
               #{'type':'matrix_dense', 'units':32},
               #{'type':'matrix_dropout'},               
               {'type':'matrix_dense', 'units':32},
               {'type':'matrix_dropout'},               
                {'type':'matrix_dense', 'units':1, 'activation':None},
            ],
            'defaults':{#default values for each layer type (see layer.py)
                'matrix_dense':{
                    'activation':tf.nn.tanh,
                    'drop_mask':True,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'mode':'mean',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'dense':{#not used
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
           'lr':.001,
    }
    
    main(opts)

