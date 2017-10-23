from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope, model_variable


def print_dims(prefix="", delimiter="\t", **kwargs):
    print(prefix, end=" ")
    for key, t in kwargs.items():
        if t is None:
            print(key,": None", delimiter,end=" ")
        else:
            try:
                shape = [x for x in t.get_shape().as_list()]
                print(key,":",shape, delimiter,end=" ")
            except:
                if hasattr(t, 'shape'):
                    print(key,":",t.get_shape(), delimiter,end=" ")
                else:
                    print(key,":",delimiter,end=" ")
            
    print()
    
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



