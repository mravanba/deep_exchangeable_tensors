from util import *
from sparse_util import *
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope, model_variable


def matrix_dense(
        inputs,
        #units = 100,
        #mode = 'max', #using 'max' or averaging 'avg'
        #activation = None,
        #kernel_initializer = None,
        #regularizer = None,
        reuse = None,
        scope = None,
        verbose = 1,
        #drop_mask = True,#whether or not drop the mask for the next layer
        **kwargs
        ):
    
    ''' 
    mat, # N x M x K input matrix
    mask = None, # N x M x 1 the observed entries
    nvec = None, # N x 1 x K' features for rows
    mvec = None, # 1 x M x K'' features for cols
    '''
    units = kwargs.get('units')
    mode = kwargs.get('mode', 'dense')

    if not scope:
        scope = "matrix_dense"
    with tf.variable_scope(scope, default_name="matrix_dense",
                               initializer=kwargs.get('kernel_initializer', None),
                               regularizer=kwargs.get('regularizer', None),
                               reuse=reuse,
                               ):
        #we should have the input matrix or at least one vector per dimension
        assert(('nvec' in inputs and 'mvec' in inputs) or 'input' in inputs)

        if 'dense' in mode:
            eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
            mat = inputs.get('input', None)#N x M x K        
            mask = inputs.get('mask', None)#N x M
            output =  tf.convert_to_tensor(0, np.float32)

            if mat is not None:#if we have an input matrix. If not, we only have nvec and mvec, i.e., user and movie properties                
                N,M,K = mat.get_shape().as_list()
                norm_N = np.float32(N)
                norm_M = np.float32(M)
                norm_NM = np.float32(N*M)
                if mask is not None:
                    mat = mat * mask
                    norm_N = tf.reduce_sum(mask, axis=0, keep_dims=True) + eps# 1, M, 1
                    norm_M = tf.reduce_sum(mask, axis=1, keep_dims=True) + eps# N, 1, 1
                    norm_NM = tf.reduce_sum(mask, axis=[0,1], keep_dims=True) + eps# 1, 1, 1

                if 'max' in kwargs.get('pool_mode', 'max') and mask is None:
                    mat_marg_0 = tf.reduce_max(mat, axis=0, keep_dims=True)
                    mat_marg_1 = tf.reduce_max(mat, axis=1, keep_dims=True)
                    mat_marg_2 = tf.reduce_max(mat_marg_0, axis=1, keep_dims=True)
                else:
                    mat_marg_0 = tf.reduce_sum(mat, axis=0, keep_dims=True)/norm_N # 1 x M x K
                    mat_marg_1 = tf.reduce_sum(mat, axis=1, keep_dims=True)/norm_M # N x 1 x K
                    mat_marg_2 = tf.reduce_sum(mat_marg_0, axis=1, keep_dims=True)/norm_NM # 1 x 1 x K

                theta_0 = model_variable("theta_0",shape=[K, units],trainable=True)
                theta_1 = model_variable("theta_1",shape=[K, units],trainable=True)
                theta_2 = model_variable("theta_2",shape=[K, units],trainable=True)
                theta_3 = model_variable("theta_3",shape=[K, units],trainable=True)
                
                # output = tf.reshape(tf.matmul(tf.reshape(mat, [N*M, K]), theta_0), [N,M,units]) 
                # output += tf.reshape(tf.matmul(tf.reshape(mat_marg_0, [-1, K]), theta_1), [1,M,units])
                # output += tf.reshape(tf.matmul(tf.reshape(mat_marg_1, [-1, K]), theta_2), [N,1,units])
                # output += tf.reshape(tf.matmul(tf.reshape(mat_marg_2, [-1, K]), theta_3), [1,1,units])

                output = tf.tensordot(mat, theta_0, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # N x M x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!
                output += tf.tensordot(mat_marg_0, theta_1, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # 1 x M x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!            
                output += tf.tensordot(mat_marg_1, theta_2, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # N x 1 x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!            
                output += tf.tensordot(mat_marg_2, theta_3, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # 1 x 1 x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!            

            nvec = inputs.get('nvec', None)
            mvec = inputs.get('mvec', None)
            if nvec is not None:
                N,_,K = nvec.get_shape().as_list()
                theta_4 = model_variable("theta_4",shape=[K, units],trainable=True)
                output_tmp = tf.tensordot(nvec, theta_4, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) + output# N x 1 x units
                output_tmp.set_shape([N,1,units])#because of current tensorflow bug!!
                output = output_tmp + output

            if mvec is not None:
                _,M,K = mvec.get_shape().as_list()
                theta_5 = model_variable("theta_5",shape=[K, units],trainable=True)
                output_tmp = tf.tensordot(mvec, theta_5, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32))# N x 1 x units
                output_tmp.set_shape([1,M,units])#because of current tensorflow bug!!
                output = output_tmp + output
            
            if kwargs.get('activation', None) is not None:
                output = kwargs.get('activation')(output)
            if kwargs.get('drop_mask', True):
                mask = None
            outdic = {'input':output, 'mask':mask}
            return outdic


        elif 'sparse' in mode: 
            eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
            mat = inputs.get('input', None)#N x M x K
            mask = inputs.get('mask', None)#N x M
            output =  tf.convert_to_tensor(0, np.float32)

            N,M,K = inputs['shape'] ## Passing shape as input so that it can be known statically 

            if mat is not None:#if we have an input matrix. If not, we only have nvec and mvec, i.e., user and movie properties
                norm_N = np.float32(N)
                norm_M = np.float32(M)
                norm_NM = np.float32(N*M)
                if mask is not None:
                    mat = sparse_tensor_mask_to_sparse(mat, mask, [N,M,K])
                    norm_N = sparse_reduce(mask, mode='sum', axis=0, shape=[N,M,1]) + eps
                    norm_M = sparse_reduce(mask, mode='sum', axis=1, shape=[N,M,1]) + eps
                    norm_NM = sparse_reduce(mask, mode='sum', axis=None, shape=[N,M,1]) + eps

                if 'max' in kwargs.get('pool_mode', 'max') and mask is None:
                    mat_marg_0 = sparse_reduce(mat, mode='max', axis=0, shape=[N,M,K])
                    mat_marg_1 = sparse_reduce(mat, mode='max', axis=1, shape=[N,M,K])
                    mat_marg_2 = sparse_reduce(mat, mode='max', axis=None, shape=[N,M,K])

                else:
                    mat_marg_0 = sparse_reduce(mat, mode='sum', axis=0, shape=[N,M,K]) / norm_N
                    mat_marg_1 = sparse_reduce(mat, mode='sum', axis=1, shape=[N,M,K]) / norm_M
                    mat_marg_2 = sparse_reduce(mat, mode='sum', axis=None, shape=[N,M,K]) / norm_NM

                theta_0 = model_variable("theta_0",shape=[K, units],trainable=True)
                theta_1 = model_variable("theta_1",shape=[K, units],trainable=True)
                theta_2 = model_variable("theta_2",shape=[K, units],trainable=True)
                theta_3 = model_variable("theta_3",shape=[K, units],trainable=True)
                
                # output = tf.tensordot(mat, theta_0, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # N x M x units
                # output.set_shape([N,M,units])#because of current tensorflow bug!!
                
                output = sparse_tensordot(mat, theta_0, [N*M, K], [N,M,units]) ## mat is sparse but output will be dense
                # output += sparse_tensordot(mat_marg_0, theta_1, [-1,K], [1,M,units])
                
                output += tf.tensordot(mat_marg_0, theta_1, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # 1 x M x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!            
                output += tf.tensordot(mat_marg_1, theta_2, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # N x 1 x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!            
                output += tf.tensordot(mat_marg_2, theta_3, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32)) # 1 x 1 x units
                output.set_shape([N,M,units])#because of current tensorflow bug!!

            nvec = inputs.get('nvec', None)
            mvec = inputs.get('mvec', None)
            if nvec is not None:

                theta_4 = model_variable("theta_4",shape=[K, units],trainable=True)
                output_tmp = sparse_tensordot(nvec, theta_4, [-1,K], [N,1,units])
                output = output_tmp + output

            if mvec is not None:

                theta_5 = model_variable("theta_5",shape=[K, units],trainable=True)
                output_tmp = sparse_tensordot(mvec, theta_5, [-1,K], [1,M,units])
                output = output_tmp + output
            
            if kwargs.get('activation', None) is not None:
                output = kwargs.get('activation')(output)
            if kwargs.get('drop_mask', True):
                mask = None

            ##....
            output = dense_tensor_to_sparse(output, [N,M,units]) ## converting output back to sparse, to be fed into pool or dropout layer. Maybe this doesnt make sense though

            outdic = {'input':output, 'mask':mask, 'shape':[N,M,units]}
            return outdic



def matrix_pool(inputs,#pool the tensor: input: N x M x K along two dimensions
                verbose=1,
                scope=None,
                **kwargs
                ):
    pool_mode = kwargs.get('pool_mode', 'max')#max or average pooling
    mode = kwargs.get('mode', 'dense')

    eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
    with tf.variable_scope(scope, default_name="matrix_dense"):
        
        if 'dense' in mode:
            mask = inputs.get('mask', None)
            inp = inputs['input']
            if 'mean' in pool_mode or mask is not None:
                op = tf.reduce_mean
            else:
                op = tf.reduce_max
            if mask is None:
                nvec = op(inp, axis=1, keep_dims=True)
                mvec = op(inp, axis=0, keep_dims=True)
            else:
                inp = inp * mask
                norm_0 = tf.reduce_sum(mask, axis=0, keep_dims=True) + eps
                norm_1 = tf.reduce_sum(mask, axis=1, keep_dims=True) + eps
                nvec = tf.reduce_sum(inp, axis=1, keep_dims=True)/norm_1
                mvec = tf.reduce_sum(inp, axis=0, keep_dims=True)/norm_0

            outdic = {'nvec':nvec, 'mvec':mvec, 'mask':mask}
            return outdic

        elif 'sparse' in mode:
            mask = inputs.get('mask', None)
            inp = inputs['input']
            N,M,K = inputs['shape']

            if mask is None:
                nvec = sparse_reduce(inp, mode=pool_mode, axis=1) / M
                mvec = sparse_reduce(inp, mode=pool_mode, axis=0) / N
            else:
                inp = sparse_tensor_mask_to_sparse(inp, mask, shape=[N,M,K])
                norm_0 = sparse_reduce(mask, mode='sum', axis=0, shape=[N,M,1]) + eps
                norm_1 = sparse_reduce(mask, mode='sum', axis=1, shape=[N,M,1]) + eps
                nvec = sparse_reduce(inp, mode='sum', axis=1, shape=[N,M,K]) / norm_1
                mvec = sparse_reduce(inp, mode='sum', axis=0, shape=[N,M,K]) / norm_0

            ##....
            nvec = dense_tensor_to_sparse(nvec, [N,1,K]) # Maybe these should just stay dense 
            mvec = dense_tensor_to_sparse(mvec, [1,M,K])

            outdic = {'nvec':nvec, 'mvec':mvec, 'mask':mask, 'shape':[N,M,K]}
            return outdic
                


def matrix_dropout(inputs,#dropout along both axes
                verbose=1,
                scope=None,
                is_training=True,
                **kwargs
                ):
    rate = kwargs.get('rate', .1)
    mode = kwargs.get('mode', 'dense')

    inp = inputs['input']
    mask = inputs.get('mask', None)

    if 'dense' in mode:
        N, M, K = inp.get_shape().as_list()
        out = tf.layers.dropout(inp, rate = rate, noise_shape=[N,1,1], training=is_training)
        out = tf.layers.dropout(out, rate = rate, noise_shape=[1,M,1], training=is_training)

        return {'input':out, 'mask':mask}

    elif 'sparse' in mode:
        
        N,M,K = inputs['shape']

        ##....
        inp = sparse_tensor_to_dense(inp, [N,M,K]) ## Currently converting to dense and back, since sparse_dropout eats too much memory 
        out = tf.layers.dropout(inp, rate = rate, noise_shape=[N,1,1], training=is_training)
        out = tf.layers.dropout(out, rate = rate, noise_shape=[1,M,1], training=is_training)
        ##....
        out = dense_tensor_to_sparse(out, [N,M,K])

        # out = sparse_dropout(inp, rate=rate, training=is_training, shape=[N,M,K]) ## Eating too much memory 


        outdic = {'input':out, 'mask':mask, 'shape':[N,M,K]}

        return outdic


# def dense(
#         inputs,
#         verbose = 1,
#         **kwargs):
    
#     inp = inputs['input']
#     units = kwargs.get('units', 100)
#     outp = tf.layers.dense(inp, units, **kwargs)
#     output = {'input':outp}
#     return output

