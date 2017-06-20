from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import tf_helper as helper
import tensorflow as tf
import layers as ly
import pdb

class Model:
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, "_"+key, value)

        self.check_vitals()
        self.setup() #1 


    def get_attr(self, attr, default=None):
        if self._verbose > 2:
            print("retrieving %s" % (attr), end=" ")
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            if self._verbose > 2:
                primitive = (int, str, bool, float)
                if type(attr) in primitive:print(" using default value ", default)
                else: print(" using default value")
            return default

        
    def check_vitals(self):
        assert hasattr(self, "_layers")
        assert hasattr(self, "_layer_defaults")     
        assert hasattr(self, "_verbose")

        
    def setup(self):
        for layer in self._layers:
            for key, val in self._layer_defaults[layer['type']].items():
                if key not in layer:
                    layer[key] = val

                    
    def get_output(self, inputs, reuse=None, verbose=None, is_training=True):
        self._inputs = inputs
        products = []
        if verbose is None:
            verbose = self._verbose
        new_product = getattr(self, "_inputs")
        if verbose > 0:
            helper.print_dims(prefix="input: ", **new_product)
        all_layers_returned = self.get_attr("all_layers_returned", False)
        scope = self.get_attr("_scope", "prediction")
        with tf.variable_scope(scope, reuse=reuse):
            for l, layer in enumerate(self._layers):
                if hasattr(ly, layer['type']):
                    with tf.variable_scope(str(l)) as l_scope:
                        layer_params = deepcopy(layer)
                        del layer_params['type']
                        new_product = getattr(ly, layer['type'])(new_product, **layer_params, verbose=self._verbose, scope=l_scope, is_training=is_training)
                        if verbose > 0:
                            helper.print_dims(prefix="layer "+str(l)+" ("+layer['type']+") ", **new_product)
                else:
                    print("layer type %s doesn't exist!" % (layer['type']))
                    raise
                if all_layers_returned:products.append(new_product)
            
        if not all_layers_returned:
                products = new_product
        return products

    # @define_scope
    # def optimization(self):
    #     pass


    # @define_scope
    # def monitor(self):
    #     pass
        
    
