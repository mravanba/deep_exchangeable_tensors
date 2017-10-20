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
    



