import numpy as np
import tensorflow as tf


def dense_array_to_sparse(x, mask_indices=None, expand_dims=False):
    """Return a sparse representation of an np array."""
    if expand_dims: 
        x = np.expand_dims(x, 2)
    K = x.shape[2]
    if mask_indices is None:
        vals = x[x.nonzero()]
        inds = np.array(list(zip(*x[:,:,0].nonzero())))
        inds = expand_array_indices(inds, K)
    else:
        inds = expand_array_indices(mask_indices, K)
        vals = x[list(zip(*inds))]
    return {'indices':inds, 'values':vals, 'dense_shape':x.shape}


def dense_array_to_sparse_values(x, mask_indices):
    """Return all values of x corresponding to indices in mask_indices."""
    K = x.shape[2]
    inds = expand_array_indices(mask_indices, K)
    return np.reshape(x[list(zip(*inds))], [-1])


def get_mask_indices(mask):
    """Return the non-zero indices of mask."""
    if len(mask.shape) is 3:
        mask = mask[:,:,0]
    return np.array(list(zip(*mask.nonzero())))


# Mostly used for debugging
def sparse_array_to_dense(values, mask_indices, shape):
    """Given sparse representation of an array, return the dense array."""
    out = np.zeros(shape)
    inds = expand_array_indices(mask_indices, shape[2])
    inds = list(zip(*inds))
    out[inds] = values
    return out


def dense_vector_to_sparse_values(x, mask_indices):
    """Collect values from x that correspond to indices of mask_indices."""
    if x.shape[0] == 1:
        vals = tf.gather(tf.transpose(x, perm=[1,0,2]), mask_indices[:,1])
        # vals = tf.gather(x, mask_indices[:,1], axis=1)
    elif x.shape[1] == 1:
        vals = tf.gather(x, mask_indices[:,0])
    return tf.reshape(vals, [-1])    


def dense_tensor_to_sparse(x, mask_indices=None, shape=None):
    """Like dense_array_to_sparse, but for tensorflow tensors."""
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
    """Like dense_array_to_sparse_values, but for tensorflow tensors."""
    inds = expand_tensor_indices(mask_indices, num_features)
    return tf.gather_nd(x, inds)


def sparse_tensor_to_dense(x_sp, shape=None):
    if shape is None:
        shape = x_sp.dense_shape
    return tf.scatter_nd(x_sp.indices, x_sp.values, shape)

 
def expand_array_indices(mask_indices, num_features):    
    """Given np array mask_indices in [N,M], return equivalent indices in [N,M,num_features]."""
    num_vals = mask_indices.shape[0]
    inds_exp = np.reshape(np.tile(range(num_features), reps=[num_vals]), newshape=[-1, 1]) # expand dimension of mask indices
    inds = np.tile(mask_indices, reps=[num_features,1]) # duplicate mask_indices num_features times
    inds = np.reshape(inds, newshape=[num_features, num_vals, 2])
    inds = np.reshape(np.transpose(inds, axes=[1,0,2]), newshape=[-1,2])
    inds = np.concatenate((inds, inds_exp), axis=1)
    return inds


def expand_tensor_indices(mask_indices, num_features):
    """Like expand_array_indices, but for tensorflow tensors."""
    num_vals = tf.shape(mask_indices)[0]
    inds_exp = tf.reshape(tf.tile(tf.range(num_features, dtype=tf.float32), multiples=[num_vals]), shape=[-1, 1]) # expand dimension of mask indices
    mask_indices = tf.cast(mask_indices, dtype=tf.float32) # cast so computation can be done on gpu
    inds = tf.tile(mask_indices, multiples=[num_features,1]) # duplicate mask_indices num_features times
    inds = tf.reshape(inds, shape=[num_features, num_vals, 2])
    inds = tf.reshape(tf.transpose(inds, perm=[1,0,2]), shape=[-1,2])
    inds = tf.concat((inds, inds_exp), axis=1)
    inds = tf.cast(inds, dtype=tf.int32)
    return inds


def sparse_reduce(mask_indices, values, num_features, mode='sum', shape=None, axis=None, keep_dims=False):
    """Equivalent to tf.reduce_sum/max, but for 2D sparse tensors."""
    if 'sum' in mode:
        op = tf.unsorted_segment_sum
    elif 'max' in mode:
        op = tf.unsorted_segment_max
    else:
        print('\nERROR - unknown <mode> in sparse_reduce()\n')
        return 

    mask_indices = tf.cast(mask_indices, dtype=tf.float32) # cast so computation can be done on gpu
    if axis is 0:
        inds = tf.cast(mask_indices[:,1], dtype=tf.int32)
        vals = tf.reshape(values, shape=[-1,num_features])
        if shape is None:
            num_segments = tf.cast(tf.reduce_max(mask_indices[:,1]), tf.int32) + 1
        else:
            num_segments = shape[1]
        out = op(vals, inds, num_segments=num_segments)
        if 'max' in mode: ## Hack to avoid tensorflow bug where 0 values are set to large negative number
            out = tf.where(tf.greater(out, -200000000), out, tf.zeros_like(out))
        if keep_dims:
            out = tf.expand_dims(out, axis=0)
        return out
    elif axis is 1:
        inds = tf.cast(mask_indices[:,0], dtype=tf.int32)
        if shape is None:
            num_segments = tf.cast(tf.reduce_max(mask_indices[:,0]), tf.int32) + 1
        else:
            num_segments = shape[0]
        vals = tf.reshape(values, shape=[-1,num_features])
        out = op(vals, inds, num_segments=num_segments)
        if 'max' in mode: ## Hack to avoid tensorflow bug where 0 values are set to large negative number
            out = tf.where(tf.greater(out, -200000000), out, tf.zeros_like(out))
        if keep_dims:
            out = tf.expand_dims(out, axis=1)
        return out
    elif axis is None:
        vals = tf.reshape(values, shape=[-1,num_features])
        if 'sum' in mode:
            out = tf.reduce_sum(vals, axis=0, keep_dims=keep_dims)
        elif 'max' in mode:
            out = tf.reduce_max(vals, axis=0, keep_dims=keep_dims)
        else:
            print('\nERROR - unknown <mode> in sparse_reduce()\n')
            return 
        if keep_dims:
            out = tf.expand_dims(out, axis=0)
        return out
    else:
        print('\nERROR - unknown <axis> in sparse_reduce()\n')


def sparse_marginalize_mask(mask_indices, shape=None, axis=None, keep_dims=True):
    """Equivalent to tf.reduce_sum applied to 2D mask."""
    mask_indices = tf.cast(mask_indices, dtype=tf.float32) # cast so computation can be done on gpu
    if axis is 0:
        inds = tf.cast(mask_indices[:,1], dtype=tf.int32)
        if shape is None:
            num_segments = tf.cast(tf.reduce_max(mask_indices[:,1]), tf.int32) + 1
        else:
            num_segments = shape[1]
        marg = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float32), inds, num_segments)
        marg = tf.cast(tf.expand_dims(marg, axis=1), dtype=tf.float32)
        if keep_dims:
            marg = tf.reshape(marg, shape=[1,-1,1])
        return marg
    elif axis is 1:
        inds = tf.cast(mask_indices[:,0], dtype=tf.int32)
        if shape is None:
            num_segments = tf.cast(tf.reduce_max(mask_indices[:,0]), tf.int32) + 1
        else:
            num_segments = shape[0]
        marg = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float32), inds, num_segments)
        marg = tf.cast(tf.expand_dims(marg, axis=1), dtype=tf.float32)
        if keep_dims:
            marg = tf.reshape(marg, shape=[-1,1,1])
        return marg
    elif axis is None:
        return tf.cast(tf.reshape(tf.shape(mask_indices)[0], shape=[1,1,1]), dtype=tf.float32)
    else:
        print('\nERROR - unknown <axis> in sparse_marginalize_mask()\n')


def sparse_tensordot_sparse(tensor_values, param, num_features):    
    """Like tf.tensordot but with sparse inputs and outputs. Output has non-zero value only where input has non-zero value."""
    vals = tf.matmul(tf.reshape(tensor_values, shape=[-1,num_features]), param)
    vals = tf.reshape(vals, shape=[-1])
    return vals


def sparse_tensor_broadcast_dense_add(x_values, y, mask_indices, num_features, broadcast_axis=None):
    """Broadcast add y onto the sparse coordinates of x_sp. Produces a sparse tensor with the same shape as x_sp, and non-zero values corresponding to those of x_sp."""
    inds = expand_tensor_indices(mask_indices, num_features)
    num_vals = tf.shape(inds)[0]
    if broadcast_axis is 0:
        temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[num_features,1])
        temp_inds = tf.slice(temp_inds, begin=[0,1], size=[-1,1])
        new_vals = tf.cast(tf.gather_nd(tf.reshape(y, shape=[-1,num_features]), temp_inds), tf.float32)
        vals = tf.reshape(x_values, shape=[-1,num_features]) + new_vals
        vals = tf.reshape(vals, shape=[num_vals])

    elif broadcast_axis is 1:
        temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[num_features,1])
        temp_inds = tf.slice(temp_inds, begin=[0,0], size=[-1,1])
        new_vals = tf.cast(tf.gather_nd(tf.reshape(y, shape=[-1,num_features]), temp_inds), tf.float32)
        vals = tf.reshape(x_values, shape=[-1,num_features]) + new_vals
        vals = tf.reshape(vals, shape=[num_vals])

    else:
        vals = tf.reshape(x_values, shape=[-1,num_features])
        vals = tf.reshape(tf.add(vals, y), shape=[-1])
    return vals


def sparse_dropout(values, num_features, rate=0.0, training=True):
    """Apply dropout to non-zero values of a tensor."""
    # rate = 2*rate - rate*rate # match overall dropout rate of dense version
    vals = tf.reshape(values, [-1,num_features])
    num_vals = tf.shape(vals)[0]
    vals = tf.layers.dropout(vals, rate=rate, noise_shape=[num_vals,1], training=training)
    vals = tf.reshape(vals, [-1])
    return vals


def sparse_dropout_row_col(values, mask_inds, shape, rate=0.0, training=True):
    """Apply dropout to rows and columns independently with probability rate."""
    N,M,K = shape
    vals = tf.reshape(values, [-1,K])
    num_vals = tf.shape(vals)[0]
    
    row_mask = np.random.choice([0,1], size=N, p=(rate, 1-rate)) # Dropout rows where this mask is 0
    row_keep = np.arange(N)[row_mask==1]
    _, row_inds = tf.setdiff1d(mask_inds[:,0], row_keep)

    col_mask = np.random.choice([0,1], size=M, p=(rate, 1-rate)) # Dropout cols where this mask is 0
    col_keep = np.arange(M)[col_mask==1]
    _, col_inds = tf.setdiff1d(mask_inds[:,1], col_keep)

    inds, _ = tf.unique(tf.concat([row_inds, col_inds], axis=0))

    drop_mask = tf.scatter_nd(tf.expand_dims(inds, axis=1), tf.ones_like(inds), shape=[num_vals])
    drop_mask = tf.cast(drop_mask, tf.bool)

    new_vals = tf.where(drop_mask, tf.zeros_like(vals), vals)
    new_vals = tf.reshape(new_vals, [-1]) / (1 - rate * rate)

    return new_vals



