import numpy as np
import json
import os
from collections import OrderedDict
import tensorflow as tf
import sparse_factorized_autoencoder
import argparse

N_jobs = 5
skip_connections = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", default=-1, type=int, help="Number of new jobs to generate")
    parser.add_argument("--id", default=-1, type=int, help="Run job with id")
    return parser.parse_args()

def populate_options(name, maxN=100, maxM=100, units_1=32, units_2=32, lr=0.0001,
                     dropout=0.2, skips=False, latent=8, l2=0.0001, latent_pool="max",
                     regular_pool="mean", use_dropout=False):
    if use_dropout:
        encoder = [
            {'type':'matrix_sparse', 'units':units_1},
            {'type':'matrix_dropout_sparse'},
            {'type':'matrix_sparse', 'units':units_2, 'skip_connections':skips},
            {'type':'matrix_dropout_sparse'},
            {'type':'matrix_sparse', 'units':latent, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
            {'type':'matrix_pool_sparse'},
            ]
        decoder = [
            {'type':'matrix_sparse', 'units':units_2},
            {'type':'matrix_dropout_sparse'},
            {'type':'matrix_sparse', 'units':units_1, 'skip_connections':skips},
            {'type':'matrix_dropout_sparse'},
            {'type':'matrix_sparse', 'units':1, 'activation':None},
        ]
    else:
        encoder = [
            {'type':'matrix_sparse', 'units':units_1},
            {'type':'matrix_sparse', 'units':units_2, 'skip_connections':skips},
            {'type':'matrix_sparse', 'units':latent, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
            {'type':'matrix_pool_sparse'},
            ]
        decoder = [
            {'type':'matrix_sparse', 'units':units_2},
            {'type':'matrix_sparse', 'units':units_1, 'skip_connections':skips},
            {'type':'matrix_sparse', 'units':1, 'activation':None},
        ]
    return {'epochs': 200,#never-mind this. We have to implement look-ahead to report the best result.
        'ckpt_folder':'checkpoints/%s' % name,
        'model_name':'sparse_ae_%s' % name,
        'verbose':2,
        'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
        'maxM':maxM,#num movies per submatrix
        'visualize':False,
        'save':False,
        'data_path': "movielens-1M",
        'output_file':'logs/output_%s' % name,
        'encoder':encoder,
        'decoder':decoder,
        'defaults':{#default values for each layer type (see layer.py)                
            'matrix_sparse':{
                'activation':tf.nn.relu,
                'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                'pool_mode':regular_pool,#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                'kernel_initializer': tf.contrib.layers.xavier_initializer(),
                'regularizer': tf.contrib.keras.regularizers.l2(l2),
                'skip_connections':False,
            },
            'dense':{#not used
                'activation':tf.nn.relu, 
                'kernel_initializer': tf.contrib.layers.xavier_initializer(),
                'regularizer': tf.contrib.keras.regularizers.l2(l2),
            },
            'matrix_pool_sparse':{
                'pool_mode':latent_pool,
            },
            'matrix_dropout_sparse':{
                'rate':dropout,
            }
        },
        'lr':0.001}

def unif(items):
    i = np.random.randint(len(items))
    return items[i]

def make_dirs_if_necessary(path):
    if not os.path.exists(path):
         os.makedirs(path)

def sample_hyperparameters():
    pars = OrderedDict()
    pars['lr'] = unif([0.00001, 0.0001, 0.001, 0.01, 0.1])
    pars['l2'] = unif([0.00001, 0.0001, 0.001, 0.01, 0.1])
    pars['dropout'] = unif([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    pars['maxN'] = unif(range(100, 2000, 200))
    pars['maxN'] = pars['maxN']
    pars['latent_pool'] = unif(["mean", "max"])
    pars['regular_pool'] = unif(["mean", "max"])
    # random number of units with second \leq the first
    unts = [2**i for i in range(4,9)]
    i = np.random.randint(len(unts))
    pars['units_1'] = unts[i]
    pars['skips'] = unif([True, False])
    if pars['skips']:
        pars['units_2'] = pars['units_1']
    else:
        pars['units_2'] = unif(unts[0:i+1])
    pars['use_dropout'] = unif([True, False])
    pars['latent'] = unif([2**i for i in range(2,6)])
    return pars

def generate_jobs(n=1):
    path = "jobs/todo/"
    make_dirs_if_necessary(path)
    for i in range(n):
        job_name = path + "hyperparameters_%d.json" % i
        with open(job_name, "w") as f:
            json.dump(sample_hyperparameters(), f)

def run_job(id=None):
    todo = "jobs/todo/"
    done = "jobs/done/"
    inprogress = "jobs/inprogress/"
    make_dirs_if_necessary(done)
    make_dirs_if_necessary(inprogress)
    if id is not None:
        name = "hyperparameters_%d" % id
        filename = name + ".json"
        opts = populate_options(name, **json.load(open(todo + filename)))
    else:
        name = "random_%d" % np.random.randint(1000, 1000000)
        filename = name + ".json"
        opts = populate_options(name, **sample_hyperparameters())
        json.dump(opts, open(todo + filename,"w"))
    os.rename(todo + filename, inprogress + filename)
    sparse_factorized_autoencoder.main(opts)
    os.rename(inprogress + filename, done + filename)


def main():
    args = parse_args()
    if args.gen > 0:
        generate_jobs(args.gen)
    if args.id >= 0:
        run_job(args.id)

if __name__ == '__main__':
    main()