import tensorflow as tf
from base import Model
from util import *
from sparse_util import *

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


def sample_submatrix_sp(mask_sp, maxN, maxM):
    '''
    returns sparse indices to be sampled 
    '''
    N,M,_ = mask_sp['dense_shape']
    max_val = mask_sp['values'].shape[0]
    num_samples = np.minimum(max_val, maxN*maxM)
    # num_samples = 10000
    for n in range(N // maxN):
        for m in range(M // maxM):
            sample = np.random.choice(max_val, size=num_samples, replace=False)
            sample = np.sort(sample)
            yield sample


def rec_loss_fn(mat, mask, rec):
    return ((tf.reduce_sum(((mat - rec)**2)*mask))/tf.reduce_sum(mask))#average l2-error over non-zero entries


def rec_loss_fn_sp(mat_sp, mask_sp, rec_sp, sparse_indices=None, shape=None):
    if shape is None:
        shape = mat_sp.dense_shape
    rec_sp = tf.SparseTensorValue(rec_sp.indices, tf.negative(rec_sp.values), rec_sp.dense_shape)
    sq_diffs = tf.square(tf.sparse_add(mat_sp, rec_sp))
    masked_diffs = sparse_tensor_mask_to_sparse(sq_diffs, mask_sp, sparse_indices=sparse_indices, shape=shape)
    return tf.sparse_reduce_sum(masked_diffs) / tf.sparse_reduce_sum(mask_sp)


def sparse_placeholders(num_features=3): ##name=....
    inds = tf.placeholder(tf.int64, shape=[None, num_features], name='inds')
    vals = tf.placeholder(tf.float32, shape=[None], name='vals')
    # shape = tf.placeholder(tf.int64, shape=[num_features], name='shape')
    # return inds, vals, shape
    return inds, vals


def main(opts):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # path = 'movielens-TEST'
    path = 'movielens-100k'
    # path = 'movielens-1M'
    data = get_data(path, mode=opts['mode'], train=.8, valid=.2, test=.001)
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat'].shape
    maxN, maxM = opts['maxN'], opts['maxM']

    mode = opts['mode']

    # if N < maxN: maxN = N
    # if M < maxM: maxM = M

    if opts['verbose'] > 0:
        print('\nRun Settings:')
        print('mode: ', mode)
        print('dataset: ', path)
        print('drop mask: ', opts['defaults']['matrix_dense']['drop_mask'])
        print('Exchangable layer pool mode: ', opts['defaults']['matrix_dense']['pool_mode'])
        print('Pooling layer pool mode: ', opts['defaults']['matrix_pool']['pool_mode'])
        print('dense channels: ', units)
        print('latent features: ', latent_features)
        print('number of layers: ', len(opts['decoder'])-1)
        print('learning rate: ', opts['lr'])
        print('activation: ', opts['defaults']['matrix_dense']['activation'])
        print('maxN: ', opts['maxN'])
        print('maxM: ', opts['maxM'])
        print('')
        

    with tf.Graph().as_default():

        if 'dense' in mode:
            mat = tf.placeholder(tf.float32, shape=(maxN, maxM, num_features), name='mat')#data matrix for training
            mask_tr = tf.placeholder(tf.float32, shape=(maxN, maxM, 1), name='mask_tr')
            #for validation, since we need less memory (forward pass only), we are feeding the whole matrix. This is only feasible for this smaller dataset. In the long term we could perform validation on CPU to avoid memory problems
            mat_val = tf.placeholder(tf.float32, shape=(N, M, num_features), name='mat')##data matrix for validation: 
            mask_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_val')#the entries not present during training
            mask_tr_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_tr_val')#both training and validation entries
            
            with tf.variable_scope("encoder"):
                tr_dict = {'input':mat,
                           'mask':mask_tr}
                val_dict = {'input':mat_val,
                            'mask':mask_tr_val}

                encoder = Model(layers=opts['encoder'], layer_defaults=opts['defaults'], mode=mode, verbose=2) #define the encoder

                out_enc_tr = encoder.get_output(tr_dict) #build the encoder
                out_enc_val = encoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#get encoder output, reusing the neural net
                

            with tf.variable_scope("decoder"):
                tr_dict = {'nvec':out_enc_tr['nvec'],
                           'mvec':out_enc_tr['mvec'],
                           'mask':out_enc_tr['mask']}
                val_dict = {'nvec':out_enc_val['nvec'],
                            'mvec':out_enc_val['mvec'],
                            'mask':out_enc_val['mask']}

                decoder = Model(layers=opts['decoder'], layer_defaults=opts['defaults'], mode=mode, verbose=2)#define the decoder

                out_tr = decoder.get_output(tr_dict)['input']#build it
                out_val = decoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)['input']#reuse it for validation

            #loss and training
            rec_loss = rec_loss_fn(mat, mask_tr, out_tr)# reconstruction loss
            reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
            rec_loss_val = rec_loss_fn(mat_val, mask_val, out_val)
            total_loss = rec_loss + reg_loss 

            train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.global_variables_initializer())

            iters_per_epoch = N//maxN * M//maxM # a bad heuristic: the whole matrix is in expectation covered in each epoch
            
            min_loss = 5
            min_loss_epoch = 0

            for ep in range(opts['epochs']):
                begin = time.time()
                loss_tr_, rec_loss_tr_, loss_val_ = 0,0,0
                for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM), total=iters_per_epoch):#go over mini-batches
                    inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies

                    tr_dict = {mat: data['mat'][inds_],
                               mask_tr:data['mask_tr'][inds_]}

                    _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)

                    loss_tr_ += np.sqrt(bloss_)
                    rec_loss_tr_ += np.sqrt(brec_loss_)

                loss_tr_ /= iters_per_epoch
                rec_loss_tr_ /= iters_per_epoch

                val_dict = {mat_val:data['mat'],
                            mask_val:data['mask_val'],
                            # mask_tr_val:data['mask_tr']}
                            mask_tr_val:data['mask_tr_val']}
            
                bloss_, = sess.run([rec_loss_val], feed_dict=val_dict)
                loss_val_ += np.sqrt(bloss_)
                if loss_val_ < min_loss: # keep track of the best validation loss 
                    min_loss = loss_val_
                    min_loss_epoch = ep
                print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t validation: {:.3f} \t minimum validation loss: {:.3f} at epoch: {:d}".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_,  loss_val_, min_loss, min_loss_epoch), flush=True)



        elif 'sparse' in mode:    
            ## Sparse placeholders 
            mat_inds, mat_vals = sparse_placeholders()
            mat = tf.SparseTensorValue(mat_inds, mat_vals, [maxN,maxM,num_features])

            mask_tr_inds, mask_tr_vals = sparse_placeholders()
            mask_tr = tf.SparseTensorValue(mask_tr_inds, mask_tr_vals, [maxN,maxM,1])

            sparse_indices_tr = tf.placeholder(tf.int64, shape=[None, 2], name='sparse_indices_tr')

            mat_val_inds, mat_val_vals = sparse_placeholders()
            mat_val = tf.SparseTensorValue(mat_val_inds, mat_val_vals, [N,M,num_features])

            mask_val_inds, mask_val_vals = sparse_placeholders()
            mask_val = tf.SparseTensorValue(mask_val_inds, mask_val_vals, [N,M,1])

            sparse_indices_val = tf.placeholder(tf.int64, shape=[None, 2], name='sparse_indices_val')

            mask_tr_val_inds, mask_tr_val_vals = sparse_placeholders()
            mask_tr_val = tf.SparseTensorValue(mask_tr_val_inds, mask_tr_val_vals, [N,M,1])

            sparse_indices_tr_val = tf.placeholder(tf.int64, shape=[None, 2], name='sparse_indices_tr_val')

            with tf.variable_scope("encoder"):
                tr_dict = {'input':mat,
                           'mask':mask_tr,
                           'sparse_indices':sparse_indices_tr,
                           'shape':[maxN,maxM,num_features]} ## Passing in shape to be used in sparse functions
                val_dict = {'input':mat_val,
                            'mask':mask_tr_val,
                            'sparse_indices':sparse_indices_tr_val,
                            'shape':[N,M,num_features]}

                encoder = Model(layers=opts['encoder'], layer_defaults=opts['defaults'], mode=mode, verbose=2) #define the encoder
                out_enc_tr = encoder.get_output(tr_dict) #build the encoder
                out_enc_val = encoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#get encoder output, reusing the neural net
                
            with tf.variable_scope("decoder"):
                tr_dict = {'nvec':out_enc_tr['nvec'],
                           'mvec':out_enc_tr['mvec'],
                           'mask':out_enc_tr['mask'],
                           'sparse_indices':out_enc_tr['sparse_indices'],
                           'shape':[maxN,maxM,out_enc_tr['shape'][2]]} ## Passing in shape to be used in sparse functions 
                val_dict = {'nvec':out_enc_val['nvec'],
                            'mvec':out_enc_val['mvec'],
                            'mask':out_enc_val['mask'],
                            'sparse_indices':out_enc_val['sparse_indices'],
                            'shape':[N,M,out_enc_val['shape'][2]]}

                decoder = Model(layers=opts['decoder'], layer_defaults=opts['defaults'], mode=mode, verbose=2)#define the decoder
                out_dec_tr = decoder.get_output(tr_dict)#build it   
                out_tr = out_dec_tr['input']
                sparse_indices_out_tr = out_dec_tr['sparse_indices']

                out_dec_val = decoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#reuse it for validation
                out_val = out_dec_val['input']
                sparse_indices_out_val = out_dec_val['sparse_indices']

            #loss and training
            # rec_loss = rec_loss_fn_sp(mat, mask_tr, out_tr, sparse_indices=sparse_indices_out_tr, shape=[maxN,maxM,1]) # reconstruction loss
            rec_loss = rec_loss_fn_sp(mat, mask_tr, out_tr, sparse_indices=sparse_indices_tr, shape=[maxN,maxM,1]) # reconstruction loss
            reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
            # rec_loss_val = rec_loss_fn_sp(mat_val, mask_val, out_val, sparse_indices=sparse_indices_out_val, shape=[N,M,1])
            rec_loss_val = rec_loss_fn_sp(mat_val, mask_val, out_val, sparse_indices=sparse_indices_val, shape=[N,M,1])
            total_loss = rec_loss + reg_loss

            train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.global_variables_initializer())

            iters_per_epoch = N//maxN * M//maxM # a bad heuristic: the whole matrix is in expectation covered in each epoch

            min_loss = 5
            min_loss_epoch = 0
            
            for ep in range(opts['epochs']):
                begin = time.time()
                loss_tr_, rec_loss_tr_, loss_val_ = 0,0,0
                # for sample_ in tqdm(sample_submatrix_sp(data['mask_tr_sp'], maxN, maxM), total=iters_per_epoch):#go over mini-batches

                for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM), total=iters_per_epoch):#go over mini-batches
                    inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies


                    # tr_dict = {mat_inds:data['mat_sp']['indices'][sample_],
                    #                 mat_vals:data['mat_sp']['values'][sample_], 
                    #                 mask_tr_inds:data['mask_tr_sp']['indices'][sample_], 
                    #                 mask_tr_vals:data['mask_tr_sp']['values'][sample_]}

                    mat_sp = dense_array_to_sparse(data['mat'][inds_])
                    mask_tr_sp = dense_array_to_sparse(data['mask_tr'][inds_])

                    tr_dict = {mat_inds:mat_sp['indices'],
                                    mat_vals:mat_sp['values'], 
                                    mask_tr_inds:mask_tr_sp['indices'],
                                    mask_tr_vals:mask_tr_sp['values'], 
                                    sparse_indices_tr:mask_tr_sp['indices'][:,0:2]}

                    
                    _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)


                    loss_tr_ += np.sqrt(bloss_)
                    rec_loss_tr_ += np.sqrt(brec_loss_)

                loss_tr_ /= iters_per_epoch
                rec_loss_tr_ /= iters_per_epoch
            

                # val_dict = {mat_val_inds:data['mat_sp']['indices'],
                #                 mat_val_vals:data['mat_sp']['values'],                                
                #                 mask_val_inds:data['mask_val_sp']['indices'],
                #                 mask_val_vals:data['mask_val_sp']['values'],
                #                 mask_tr_val_inds: data['mask_tr_val_sp']['indices'],
                #                 mask_tr_val_vals: data['mask_tr_val_sp']['values']}

                mat_sp = dense_array_to_sparse(data['mat'])
                mask_val_sp = dense_array_to_sparse(data['mask_val'])
                mask_tr_val_sp = dense_array_to_sparse(data['mask_tr_val'])

                val_dict = {mat_val_inds:mat_sp['indices'],
                                mat_val_vals:mat_sp['values'],                                
                                mask_val_inds:mask_val_sp['indices'],
                                mask_val_vals:mask_val_sp['values'],
                                sparse_indices_val:mask_val_sp['indices'][:,0:2],
                                mask_tr_val_inds:mask_tr_val_sp['indices'],
                                mask_tr_val_vals:mask_tr_val_sp['values'],
                                sparse_indices_tr_val:mask_tr_val_sp['indices'][:,0:2]}


                bloss_, = sess.run([rec_loss_val], feed_dict=val_dict)
                loss_val_ += np.sqrt(bloss_)
                if loss_val_ < min_loss: # keep track of the best validation loss 
                    min_loss = loss_val_
                    min_loss_epoch = ep
                print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t validation: {:.3f} \t minimum validation loss: {:.3f} at epoch: {:d}".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_,  loss_val_, min_loss, min_loss_epoch), flush=True)



if __name__ == "__main__":
    
    units = 32
    latent_features = 5

    opts ={'epochs': 1000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/factorized_ae',
           'model_name':'test_fac_ae',
           'verbose':2,
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
            'maxN':100,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':100,#num movies per submatrix
           'visualize':False,
           'save':False,
           'mode':'dense', # use sparse or dense tensor representation
           'encoder':[
               # {'type':'matrix_dense', 'units':units},
               #{'type':'matrix_dropout'},
               # {'type':'matrix_dense', 'units':16},
               # {'type':'matrix_dropout'},
               # {'type':'matrix_dense', 'units':16},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':latent_features, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool'},
               ],
            'decoder':[
               # {'type':'matrix_dense', 'units':units},
               #{'type':'matrix_dropout'},
               # {'type':'matrix_dense', 'units':16},
               # {'type':'matrix_dropout'},
               # {'type':'matrix_dense', 'units':16},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units},
               # {'type':'matrix_dropout'},
                {'type':'matrix_dense', 'units':1, 'activation':None},
            ],
            'defaults':{#default values for each layer type (see layer.py)
                'matrix_dense':{
                    'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    # 'activation':tf.nn.relu,
                    'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'mean',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'dense':{#not used
                    'activation':tf.nn.elu, 
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'matrix_pool':{
                    'pool_mode':'max',
                },
                'matrix_dropout':{
                    'rate':.5,
                }
            },
           'lr':.001,
    }
    
    main(opts)


