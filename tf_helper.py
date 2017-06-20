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
                print(key,":",t.get_shape(), delimiter,end=" ")
            
    print(flush=True)
    



