import numpy as np
import tensorflow as tf

# Used to get sparse representation of input data
# x should be a np.ndarray-type object
# outputs a dictionary representing a sparse np.ndarray
def dense_array_to_sparse(x, mask_indices=None, expand_dims=False):
    if expand_dims: 
        x = np.expand_dims(x, 2)
    shape = x.shape
    K = shape[2]
    if mask_indices is None:
        vals = x[x.nonzero()]
        inds = np.array(list(zip(*x[:,:,0].nonzero())))
        inds = expand_array_indices(inds, K)
    else:
        inds = expand_array_indices(mask_indices, K)
        vals = x[list(zip(*inds))]
    return {'indices':inds, 'values':vals, 'dense_shape':shape}


# Extract values of x at indices corresponding to mask_indices
def dense_array_to_sparse_values(x, mask_indices):
    shape = x.shape
    K = shape[2]
    inds = expand_array_indices(mask_indices, K)
    return np.reshape(x[list(zip(*inds))], [-1])


# Returns non-zero indices of mask
def get_mask_indices(mask):
    if len(mask.shape) is 3:
        mask = mask[:,:,0]
    return np.array(list(zip(*mask.nonzero())))


# Mostly used for debugging
# x_sp is dictionary of the from:
# {'indices':inds, 'values':vals, 'dense_shape':shape}
# representing a sparse tensor
# output is a np.ndarray
def sparse_array_to_dense(x_sp, shape=None):
    if shape is None:
        shape = x_sp['dense_shape']    
    out = np.zeros(shape)
    inds = list(zip(*x_sp['indices']))
    out[inds] = x_sp['values']
    return out


def dense_tensor_to_sparse(x, mask_indices=None, shape=None):
    if shape is None:
        shape = x.shape
    K = shape[2]        
    if mask_indices is None:
        inds = tf.where(tf.not_equal(x, 0))
    else:
        inds = expand_tensor_indices(mask_indices, K)
    vals = tf.gather_nd(x, inds)
    return tf.SparseTensorValue(inds, vals, shape)

def dense_tensor_to_sparse_values(x, mask_indices, num_features):
    inds = expand_tensor_indices(mask_indices, num_features)
    return tf.gather_nd(x, inds)
    

def dense_vector_to_sparse_values(x, mask_indices, num_features):
    if x.shape[0] == 1:
        unique = tf.unique_with_counts(mask_indices[:,1], out_idx=tf.int32)
        unique_vals = unique[0]
        unique_inds = unique[1]
        inds = tf.cast(tf.gather(unique_vals, unique_inds), tf.int32)
        vals = tf.gather(tf.transpose(x, perm=[1,0,2]), inds)
        return tf.reshape(vals, [-1])
    elif x.shape[1] == 1:
        unique = tf.unique_with_counts(mask_indices[:,0], out_idx=tf.int32)
        unique_vals = unique[0]
        unique_inds = unique[1]
        inds = tf.cast(tf.gather(unique_vals, unique_inds), tf.int32)
        vals = tf.gather(x, inds)
        return tf.reshape(vals, [-1])


def sparse_tensor_to_dense(x_sp, shape=None):
    if shape is None:
        shape = x_sp.dense_shape
    return tf.scatter_nd(x_sp.indices, x_sp.values, shape)


# Given np array mask_indices in [N,M], return equivalent indices in [N,M,num_features]
def expand_array_indices(mask_indices, num_features):    
    num_vals = mask_indices.shape[0]
    inds_exp = np.reshape(np.tile(range(num_features), reps=[num_vals]), newshape=[-1, 1]) # expand dimension of mask indices
    inds = np.tile(mask_indices, reps=[num_features,1]) # duplicate mask_indices num_features times
    inds = np.reshape(inds, newshape=[num_features, num_vals, 2])
    inds = np.reshape(np.transpose(inds, axes=[1,0,2]), newshape=[-1,2])
    inds = np.concatenate((inds, inds_exp), axis=1)
    return inds


# Given tensor mask_indices in [N,M], return equivalent indices in [N,M,num_features]
def expand_tensor_indices(mask_indices, num_features):
    num_vals = tf.shape(mask_indices)[0]
    inds_exp = tf.reshape(tf.tile(tf.range(num_features, dtype=tf.float32), multiples=[num_vals]), shape=[-1, 1]) # expand dimension of mask indices
    mask_indices = tf.cast(mask_indices, dtype=tf.float32) # cast so computation can be done on gpu
    inds = tf.tile(mask_indices, multiples=[num_features,1]) # duplicate mask_indices num_features times
    inds = tf.reshape(inds, shape=[num_features, num_vals, 2])
    inds = tf.reshape(tf.transpose(inds, perm=[1,0,2]), shape=[-1,2])
    inds = tf.concat((inds, inds_exp), axis=1)
    inds = tf.cast(inds, dtype=tf.int32)
    return inds


# Equivalent to tf.reduce_ sum/max/mean, but for sparse tensors.
# mask_indices are [N,M] non-zero indices
# Returns a dense tensor
def sparse_reduce(mask_indices, values, mode, shape, axis=None, keep_dims=False):
    N = shape[0]
    M = shape[1]
    K = shape[2]

    if 'sum' in mode:
        op = tf.unsorted_segment_sum
    elif 'max' in mode:
        op = tf.unsorted_segment_max
    elif 'mean' in  mode:
        if axis is 0:
            d = N
        elif axis is 1:
            d = M
        elif axis is None:
            d = N*M
        else:
            print('\nERROR - unknown <axis> in sparse_reduce()\n')
            return 
        op = lambda *args, **kwargs: tf.unsorted_segment_sum(*args, **kwargs) / d
    else:
        print('\nERROR - unknown <mode> in sparse_reduce()\n')
        return 

    mask_indices = tf.cast(mask_indices, dtype=tf.float32) # cast so computation can be done on gpu
    if axis is 0:
        inds = tf.cast(mask_indices[:,1], dtype=tf.int32)
        vals = tf.reshape(values, shape=[-1,K])
        out = op(vals, inds, num_segments=M)
        if 'max' in mode: ## Hack to avoid tensorflow bug where 0 values are set to large negative number
            out = tf.where(tf.greater(out, -200000000), out, tf.zeros_like(out))
        if keep_dims:
            out = tf.expand_dims(out, axis=0)
        return out
    elif axis is 1:
        inds = tf.cast(mask_indices[:,0], dtype=tf.int32)   
        vals = tf.reshape(values, shape=[-1,K])
        out = op(vals, inds, num_segments=N)
        if 'max' in mode: ## Hack to avoid tensorflow bug where 0 values are set to large negative number
            out = tf.where(tf.greater(out, -200000000), out, tf.zeros_like(out))
        if keep_dims:
            out = tf.expand_dims(out, axis=1)
        return out
    elif axis is None:
        vals = tf.reshape(values, shape=[-1,K])
        if 'sum' in mode:
            out = tf.reduce_sum(vals, axis=0, keep_dims=keep_dims)
        elif 'max' in mode:
            out = tf.reduce_max(vals, axis=0, keep_dims=keep_dims)
        elif 'mean' in mode:
            out = tf.cast(tf.reduce_sum(vals, axis=0, keep_dims=keep_dims), tf.float64) / (N*M)
        else:
            print('\nERROR - unknown <mode> in sparse_reduce()\n')
            return 
        if keep_dims:
            out = tf.expand_dims(out, axis=0)
        return out
    else:
        print('\nERROR - unknown <axis> in sparse_reduce()\n')


# Equivalent to tf.reduce_sum applied to 2d mask
# Returns a dense tensor
def sparse_marginalize_mask(mask_indices, shape, axis=None, keep_dims=True):
    mask_indices = tf.cast(mask_indices, dtype=tf.float32) # cast so computation can be done on gpu
    if axis is 0:
        inds = tf.cast(mask_indices[:,1], dtype=tf.int32)
        marg = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float32), inds, shape[1])
        marg = tf.cast(tf.expand_dims(marg, axis=1), dtype=tf.float32)
        if keep_dims:
            marg = tf.reshape(marg, shape=[1,-1,1])
        return marg
    elif axis is 1:
        inds = tf.cast(mask_indices[:,0], dtype=tf.int32)
        marg = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float32), inds, shape[0])
        marg = tf.cast(tf.expand_dims(marg, axis=1), dtype=tf.float32)
        if keep_dims:
            marg = tf.reshape(marg, shape=[-1,1,1])
        return marg
    elif axis is None:
        return tf.cast(tf.reshape(tf.shape(mask_indices)[0], shape=[1,1,1]), dtype=tf.float32)
    else:
        print('\nERROR - unknown <axis> in sparse_marginalize_mask()\n')



## Like tf.tensordot but with sparse inputs and outputs. Output has non-zero value only where input has non-zero value 
def sparse_tensordot_sparse(tensor_values, param, num_features):    
    vals = tf.matmul(tf.reshape(tensor_values, shape=[-1,num_features]), param)
    vals = tf.reshape(vals, shape=[-1])
    return vals


## x_sp is a sparse tensor of shape [N,M,K], y is dense of shape [1,M,K], [N,1,K], or [1,1,K].
## Broadcast add y onto the sparse coordinates of x_sp.
## Produces a sparse tensor with the same shape as x_sp, and non-zero values corresponding to those of x_sp
def sparse_tensor_broadcast_dense_add(x_values, y, mask_indices, num_features, broadcast_axis=None):
    # if shape is None:
    #     shape = x_sp.dense_shape
    # K = shape[2]
    inds = expand_tensor_indices(mask_indices, num_features)
    num_vals = tf.shape(inds)[0]
    if broadcast_axis is 0:
        temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[num_features,1])
        temp_inds = tf.slice(temp_inds, begin=[0,1], size=[-1,1])
        new_vals = tf.cast(tf.gather_nd(tf.reshape(y, shape=[-1,num_features]), temp_inds), tf.float32)
        vals = tf.reshape(x_values, shape=[-1,num_features]) + new_vals
        vals = tf.reshape(vals, shape=[num_vals])
        # vals = tf.reshape(vals, shape=[-1])
        # return tf.SparseTensorValue(inds, vals, shape)

    elif broadcast_axis is 1:
        temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[num_features,1])
        temp_inds = tf.slice(temp_inds, begin=[0,0], size=[-1,1])
        new_vals = tf.cast(tf.gather_nd(tf.reshape(y, shape=[-1,num_features]), temp_inds), tf.float32)
        vals = tf.reshape(x_values, shape=[-1,num_features]) + new_vals
        vals = tf.reshape(vals, shape=[num_vals])
        # vals = tf.reshape(vals, shape=[-1])
        # return tf.SparseTensorValue(inds, vals, shape)

    else:
        vals = tf.reshape(x_values, shape=[-1,num_features])
        vals = tf.reshape(tf.add(vals, y), shape=[-1])
        # return tf.SparseTensorValue(inds, vals, shape)
    return vals


# Apply dropout to non-zero values
def sparse_dropout(values, num_features, rate=0.0, training=True):
    # rate = 2*rate - rate*rate # match overall dropout rate of dense version
    vals = tf.reshape(values, [-1,num_features])
    num_vals = tf.shape(vals)[0]
    vals = tf.layers.dropout(vals, rate=rate, noise_shape=[num_vals,1], training=training)
    vals = tf.reshape(vals, [-1])
    return vals





