from __future__ import print_function
from util import *
from sparse_util import *
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope, model_variable

##### Dense Layers: #####

def matrix_dense(
        inputs,
        layer_params,
        reuse = None,
        scope = None,
        verbose = 1,
        **kwargs
        ):
    
    ''' 
    mat, # N x M x K input matrix
    mask = None, # N x M x 1 the observed entries
    nvec = None, # N x 1 x K' features for rows
    mvec = None, # 1 x M x K'' features for cols
    '''
    units = layer_params.get('units')

    if not scope:
        scope = "matrix_dense"
    with tf.variable_scope(scope, default_name="matrix_dense",
                               initializer=layer_params.get('kernel_initializer', None),
                               regularizer=layer_params.get('regularizer', None),
                               reuse=reuse,
                               ):
        #we should have the input matrix or at least one vector per dimension
        assert(('nvec' in inputs and 'mvec' in inputs) or 'input' in inputs)

        eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
        mat = inputs.get('input', None)#N x M x K        
        mask = inputs.get('mask', None)#N x M
        skip_connections = layer_params.get('skip_connections', False)
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

            if 'max' in layer_params.get('pool_mode', 'max') and mask is None:
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
            output_tmp = tf.tensordot(nvec, theta_4, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32))# N x 1 x units
            output_tmp.set_shape([N,1,units])#because of current tensorflow bug!!
            output = output_tmp + output

        if mvec is not None:
            _,M,K = mvec.get_shape().as_list()
            theta_5 = model_variable("theta_5",shape=[K, units],trainable=True)
            output_tmp = tf.tensordot(mvec, theta_5, axes=tf.convert_to_tensor([[2],[0]], dtype=np.int32))# 1 x M x units
            output_tmp.set_shape([1,M,units])#because of current tensorflow bug!!
            output = output_tmp + output
        
        if layer_params.get('activation', None) is not None:
            output = layer_params.get('activation')(output)
        if layer_params.get('drop_mask', True):
            mask = None
        ##.....
        # else:
        #     output = mask * output
        ##.....

        if skip_connections and mat is not None:
            output = output + mat

        outdic = {'input':output, 'mask':mask}
        return outdic


def matrix_pool(inputs,#pool the tensor: input: N x M x K along two dimensions
                layer_params,
                verbose=1,
                scope=None,
                **kwargs
                ):
    pool_mode = layer_params.get('pool_mode', 'max')#max or average pooling
    mode = layer_params.get('mode', 'dense')

    eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
    with tf.variable_scope(scope, default_name="matrix_dense"):
        
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


def matrix_dropout(inputs,#dropout along both axes
                   layer_params,
                    verbose=1,
                    scope=None,
                    is_training=True,
                    **kwargs
                    ):
    rate = layer_params.get('rate', .1)
    mode = layer_params.get('mode', 'dense')
    inp = inputs['input']
    mask = inputs.get('mask', None)
    N, M, K = inp.get_shape().as_list()
    out = tf.layers.dropout(inp, rate = rate, noise_shape=[N,1,1], training=is_training)
    out = tf.layers.dropout(out, rate = rate, noise_shape=[1,M,1], training=is_training)

    return {'input':out, 'mask':mask}    


# def dense(
#         inputs,
#         verbose = 1,
#         **layer_params):
    
#     inp = inputs['input']
#     units = layer_params.get('units', 100)
#     outp = tf.layers.dense(inp, units, **layer_params)
#     output = {'input':outp}
#     return output

            

##### Sparse Layers: #####

def matrix_sparse(
        inputs,
        layer_params,
        reuse = None,
        scope = None,
        verbose = 1,
        **kwargs
        ):

    units = layer_params.get('units')    

    if not scope:
        scope = "matrix_sparse"
    with tf.variable_scope(scope, default_name="matrix_sparse",
                               initializer=layer_params.get('kernel_initializer', None),
                               regularizer=layer_params.get('regularizer', None),
                               reuse=reuse,
                               ):
        #we should have the input matrix or at least one vector per dimension
        assert(('nvec' in inputs and 'mvec' in inputs) or 'input' in inputs)

        eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
        mat_values = inputs.get('input', None)#N x M x K
        mask_indices = inputs.get('mask_indices', None)
        skip_connections = layer_params.get('skip_connections', False)

        N = inputs['shape'][0]
        M = inputs['shape'][1]
        
        K = inputs['units']        

        output =  tf.convert_to_tensor(0, np.float32)

        if mat_values is not None:#if we have an input matrix. If not, we only have nvec and mvec, i.e., user and movie properties
            norm_N = tf.cast(N, tf.float32)
            norm_M = tf.cast(M, tf.float32)
            norm_NM = norm_N * norm_M

            if mask_indices is not None:
                norm_N = sparse_marginalize_mask(mask_indices, shape=[N,M,K], axis=0, keep_dims=True) + eps
                norm_M = sparse_marginalize_mask(mask_indices, shape=[N,M,K], axis=1, keep_dims=True) + eps
                norm_NM = sparse_marginalize_mask(mask_indices, shape=[N,M,K], axis=None, keep_dims=True) + eps

            if 'max' in layer_params.get('pool_mode', 'max') and mask_indices is None:
                mat_marg_0 = sparse_reduce(mask_indices, mat_values, mode='max', shape=[N,M,K], axis=0, keep_dims=True)
                mat_marg_1 = sparse_reduce(mask_indices, mat_values, mode='max', shape=[N,M,K], axis=1, keep_dims=True)
                mat_marg_2 = sparse_reduce(mask_indices, mat_values, mode='max', shape=[N,M,K], axis=None, keep_dims=True)
            else:
                mat_marg_0 = sparse_reduce(mask_indices, mat_values, mode='sum', shape=[N,M,K], axis=0, keep_dims=True) / norm_N
                mat_marg_1 = sparse_reduce(mask_indices, mat_values, mode='sum', shape=[N,M,K], axis=1, keep_dims=True) / norm_M
                mat_marg_2 = sparse_reduce(mask_indices, mat_values, mode='sum', shape=[N,M,K], axis=None, keep_dims=True) / norm_NM

            theta_0 = model_variable("theta_0",shape=[K, units],trainable=True)
            theta_1 = model_variable("theta_1",shape=[K, units],trainable=True)
            theta_2 = model_variable("theta_2",shape=[K, units],trainable=True)
            theta_3 = model_variable("theta_3",shape=[K, units],trainable=True)
            
            # bias = model_variable("bias",shape=[1],trainable=True)
            
            output = sparse_tensordot_sparse(mat_values, theta_0, K)
            output_0 = tf.tensordot(mat_marg_0, theta_1, axes=[[2],[0]]) # 1 x M x units
            output = sparse_tensor_broadcast_dense_add(output, output_0, mask_indices, units, broadcast_axis=0)
            output_1 = tf.tensordot(mat_marg_1, theta_2, axes=[[2],[0]]) # N x 1 x units
            output = sparse_tensor_broadcast_dense_add(output, output_1, mask_indices, units, broadcast_axis=1)
            output_2 = tf.tensordot(mat_marg_2, theta_3, axes=[[2],[0]]) # 1 x 1 x units
            output = sparse_tensor_broadcast_dense_add(output, output_2, mask_indices, units, broadcast_axis=None)

            # output = tf.SparseTensorValue(output.indices, tf.add(output.values, bias), output.dense_shape)        

        nvec = inputs.get('nvec', None)
        mvec = inputs.get('mvec', None)

        if nvec is not None:
            theta_4 = model_variable("theta_4",shape=[K,units],trainable=True)
            output_tmp = tf.tensordot(nvec, theta_4, axes=[[2],[0]])# N x 1 x units
            # output_tmp.set_shape([N,1,units])#because of current tensorflow bug!!
            if mat_values is not None:
                output = sparse_tensor_broadcast_dense_add(output, output_tmp, mask_indices, broadcast_axis=1, shape=[N,M,units])           
            else:
                output = output_tmp + output

        if mvec is not None:
            theta_5 = model_variable("theta_5",shape=[K, units],trainable=True)
            output_tmp = tf.tensordot(mvec, theta_5, axes=[[2],[0]])# 1 x M x units
            # output_tmp.set_shape([1,M,units])#because of current tensorflow bug!!
            if mat_values is not None:
                output = sparse_tensor_broadcast_dense_add(output, output_tmp, mask_indices, broadcast_axis=0, shape=[N,M,units])
            else:
                output = output_tmp + output
            
        if layer_params.get('activation', None) is not None:
            if mat_values is not None:
                output = layer_params.get('activation')(output)
            else:
                output = layer_params.get('activation')(output)

        if skip_connections and mat_values is not None:
            output = output + mat_values

        if mat_values is None:
            output = dense_tensor_to_sparse_values(output, mask_indices, units)
            
        if layer_params.get('drop_mask', True):
            mask_indices = None

        outdic = {'input':output, 'mask_indices':mask_indices, 'shape':[N,M,units], 'units':units}
        return outdic


def matrix_pool_sparse(inputs,#pool the tensor: input: N x M x K along two dimensions
                        layer_params,
                        verbose=1,
                        scope=None,
                        **kwargs
                        ):
    inp_values = inputs['input']
    N,M,K = inputs['shape']
    units = inputs['units']
    mask_indices = inputs['mask_indices']
    pool_mode = layer_params.get('pool_mode', 'max')#max or average pooling
    mode = layer_params.get('mode', 'dense')

    eps = tf.convert_to_tensor(1e-3, dtype=np.float32)
    with tf.variable_scope(scope, default_name="matrix_sparse"):

        if mask_indices is None:
            nvec = sparse_reduce(mask_indices, inp.values, mode=pool_mode, shape=[N,M,units], axis=1, keep_dims=True) / M
            mvec = sparse_reduce(mask_indices, inp.values, mode=pool_mode, shape=[N,M,units], axis=0, keep_dims=True) / N
        else:
            norm_0 = sparse_marginalize_mask(mask_indices, shape=[N,M,units], axis=0, keep_dims=True) + eps
            norm_1 = sparse_marginalize_mask(mask_indices, shape=[N,M,units], axis=1, keep_dims=True) + eps
            nvec = sparse_reduce(mask_indices, inp_values, mode='sum', shape=[N,M,units], axis=1, keep_dims=True) / norm_1
            mvec = sparse_reduce(mask_indices, inp_values, mode='sum', shape=[N,M,units], axis=0, keep_dims=True) / norm_0

        outdic = {'nvec':nvec, 'mvec':mvec, 'mask_indices':mask_indices, 'shape':[N,M,K], 'units':units}
        return outdic             
        

def matrix_dropout_sparse(inputs,
                          layer_params,
                            verbose=1,
                            scope=None,
                            is_training=True,
                            **kwargs
                            ):
    rate = layer_params.get('rate', .1)
    mode = layer_params.get('mode', 'dense')
    inp_values = inputs['input']
    mask_indices = inputs.get('mask_indices', None)
    N,M,K = inputs['shape']
    units = inputs['units']
   
    out = sparse_dropout(inp_values, units, rate=rate, training=is_training)

    return {'input':out, 'mask_indices':mask_indices, 'shape':[N,M,K], 'units':units}


