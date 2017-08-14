from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import itertools
import pdb
import sys
import matplotlib as plt
import scipy.ndimage.filters as filters
from PIL import Image
import pickle as pickle
import os
from fnmatch import fnmatch
from random import shuffle
import itertools
import time
from sys import stderr
import scipy.interpolate as interpolate
import time
import tqdm
from tqdm import trange
from tqdm import tqdm
import glob
from copy import deepcopy
import functools
from sparse_util import *



floatX = "float32"
data_folder = "data"


def get_data(dataset='movielens-small',
                 mode='dense',#returned matrix: dense, sparse, table
                 train=.8,
                 test=.1,
                 valid=.1,
                 ):
    
    if 'movielens-small' in dataset:
        path = os.path.join(data_folder,'ml-latest-small/ratings.csv')
        ratings = pd.read_csv(path, sep=',', usecols=[0,1,2])#drop time-stamp
        if 'table' in mode:
            return ratings
        elif 'dense' in mode:
            n_users = np.max(ratings.userId)
            _, movies = np.unique(ratings.movieId, return_inverse=True)
            n_movies = np.max(movies) + 1
            mat = np.zeros((n_users, n_movies), dtype=np.float32)
            mat[ratings.userId-1, movies] = ratings.rating
            n_ratings = ratings.rating.shape[0]
            n_train = int(np.floor(n_ratings * train))
            n_test = n_train + int(np.floor(n_ratings * test))
            n_valid = n_test + int(np.floor(n_ratings * valid))                        
            rand_perm = np.random.permutation(n_ratings)
            mask_tr = np.zeros((n_users, n_movies), dtype=np.float32)
            mask_ts = np.zeros((n_users, n_movies), dtype=np.float32)
            mask_valid = np.zeros((n_users, n_movies), dtype=np.float32)
            mask_tr[ratings.userId[rand_perm[:n_train]]-1, movies[rand_perm[:n_train]]] = 1
            mask_ts[ratings.userId[rand_perm[n_train:n_test]]-1,movies[rand_perm[n_train:n_test]]] = 1
            mask_valid[ratings.userId[rand_perm[n_test:n_valid]]-1,movies[rand_perm[n_test:n_valid]]] = 1
            data = {'mat':mat[:,:,None], 'mask_tr':mask_tr[:,:,None],
                        'mask_ts':mask_ts[:,:,None], 'mask_val':mask_valid[:,:,None]}
            return data


    elif 'movielens-TEST' in dataset:        
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        path_tr = os.path.join(data_folder,'ml-TEST/u1.base')
        path_valid = os.path.join(data_folder,'ml-TEST/u1.test')
        ratings_tr = pd.read_csv(path_tr, sep='\t', names=r_cols, encoding='latin-1')        
        ratings_valid = pd.read_csv(path_valid, sep='\t', names=r_cols, encoding='latin-1')
        ratings_valid = ratings_valid.sample(frac=1) #random shuffle before test/validation split        
        ratings = pd.concat([ratings_tr, ratings_valid], ignore_index=True)
        n_ratings = ratings.rating.shape[0]
        n_ratings_tr = ratings_tr.rating.shape[0]
        n_ratings_valid = ratings_valid.rating.shape[0]
        n_users = np.max(ratings.user_id)
        _, movies = np.unique(ratings.movie_id, return_inverse=True)
        n_movies = np.max(movies) + 1
        n_train = n_ratings_tr        
        n_test = n_train + int(np.floor(n_ratings * test))
        n_valid = n_test + n_ratings_valid
        mat = np.zeros((n_users, n_movies), dtype=np.float32)
        mat[ratings.user_id-1, movies] = ratings.rating
        

        mask_tr = np.zeros((n_users, n_movies), dtype=np.float32)        
        mask_valid = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_ts = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_tr[ratings.user_id[:n_train]-1, movies[:n_train]] = 1
        mask_ts[ratings.user_id[n_train:n_test]-1, movies[n_train:n_test]] = 1
        mask_valid[ratings.user_id[n_test:n_valid]-1, movies[n_test:n_valid]] = 1

        if 'dense' in mode:
            data = {'mat':mat[:,:,None],
                    'mask_tr':mask_tr[:,:,None],
                    'mask_ts':mask_ts[:,:,None],
                    'mask_val':mask_valid[:,:,None]}
        elif 'sparse' in mode:
            mat_sp = dense_array_to_sparse(mat, expand_dims=True)
            mask_tr_sp = dense_array_to_sparse(mask_tr, expand_dims=True)
            mask_val_sp = dense_array_to_sparse(mask_valid, expand_dims=True)
            mask_ts_sp = dense_array_to_sparse(mask_ts, expand_dims=True)

            data = {'mat':mat[:,:,None],
                    'mask_tr':mask_tr[:,:,None],
                    'mask_val':mask_valid[:,:,None],
                    'mask_ts':mask_ts[:,:,None],
                    'mat_sp':mat_sp,
                    'mask_tr_sp':mask_tr_sp,
                    'mask_val_sp':mask_val_sp}
        # pdb.set_trace()
        return data


    elif 'movielens-100k' in dataset:
        print("\n--> get_data(movielens-100k) : ignoring inputs <train>, <valid> - using u1.base/u1.test training/test split\n")
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        path_tr = os.path.join(data_folder,'ml-100k/u1.base')
        path_valid = os.path.join(data_folder,'ml-100k/u1.test')
        ratings_tr = pd.read_csv(path_tr, sep='\t', names=r_cols, encoding='latin-1')        
        ratings_valid = pd.read_csv(path_valid, sep='\t', names=r_cols, encoding='latin-1')
        ratings_valid = ratings_valid.sample(frac=1) #random shuffle before test/validation split        
        ratings = pd.concat([ratings_tr, ratings_valid], ignore_index=True)
        n_ratings = ratings.rating.shape[0]
        n_ratings_tr = ratings_tr.rating.shape[0]
        n_ratings_valid = ratings_valid.rating.shape[0]
        n_users = np.max(ratings.user_id)
        _, movies = np.unique(ratings.movie_id, return_inverse=True)
        n_movies = np.max(movies) + 1
        n_train = n_ratings_tr        
        n_test = n_train + int(np.floor(n_ratings * test))
        n_valid = n_test + n_ratings_valid
        mat = np.zeros((n_users, n_movies), dtype=np.float32)
        mat[ratings.user_id-1, movies] = ratings.rating
        mask_tr = np.zeros((n_users, n_movies), dtype=np.float32)        
        mask_valid = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_ts = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_tr[ratings.user_id[:n_train]-1, movies[:n_train]] = 1
        mask_ts[ratings.user_id[n_train:n_test]-1, movies[n_train:n_test]] = 1
        mask_valid[ratings.user_id[n_test:n_valid]-1, movies[n_test:n_valid]] = 1
        
        if 'dense' in mode:
            data = {'mat':mat[:,:,None],
                    'mask_tr':mask_tr[:,:,None],
                    'mask_ts':mask_ts[:,:,None],
                    'mask_val':mask_valid[:,:,None]}
        elif 'sparse' in mode:
            mat_sp = dense_array_to_sparse(mat, expand_dims=True)
            mask_tr_sp = dense_array_to_sparse(mask_tr, expand_dims=True)
            mask_val_sp = dense_array_to_sparse(mask_valid, expand_dims=True)
            mask_ts_sp = dense_array_to_sparse(mask_ts, expand_dims=True)

            data = {'mat':mat[:,:,None],
                    'mask_tr':mask_tr[:,:,None],
                    'mask_val':mask_valid[:,:,None],
                    'mask_ts':mask_ts[:,:,None],
                    'mat_sp':mat_sp,
                    'mask_tr_sp':mask_tr_sp,
                    'mask_val_sp':mask_val_sp}
        # pdb.set_trace()
        return data   
        
    elif 'movielens-1M' in dataset:
        r_cols = ['user_id', None, 'movie_id', None, 'rating', None, 'unix_timestamp']
        path = os.path.join(data_folder, 'ml-1m/ratings.dat')

        ratings = pd.read_csv(path, sep=':', names=r_cols, encoding='latin-1')
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings = ratings[r_cols]
        # ratings = pd.read_csv(path, sep='::', names=r_cols, encoding='latin-1', engine='python') # pandas treats '::' as a regex and uses pytohn engine. Does it matter? 

        n_users = np.max(ratings.user_id)
        _, movies = np.unique(ratings.movie_id, return_inverse=True)
        n_movies = np.max(movies) + 1
        mat = np.zeros((n_users, n_movies), dtype=np.float32)
        mat[ratings.user_id-1, movies] = ratings.rating
        n_ratings = ratings.rating.shape[0]
        n_train = int(np.floor(n_ratings * train))
        n_test = n_train + int(np.floor(n_ratings * test))
        n_valid = n_test + int(np.floor(n_ratings * valid))
        rand_perm = np.random.permutation(n_ratings)
        mask_tr = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_ts = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_valid = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_tr[ratings.user_id[rand_perm[:n_train]]-1, movies[rand_perm[:n_train]]] = 1
        mask_ts[ratings.user_id[rand_perm[n_train:n_test]]-1,movies[rand_perm[n_train:n_test]]] = 1
        mask_valid[ratings.user_id[rand_perm[n_test:n_valid]]-1,movies[rand_perm[n_test:n_valid]]] = 1
        
        # data = {'mat':mat[:,:,None], 'mask_tr':mask_tr[:,:,None], 'mask_ts':mask_ts[:,:,None], 'mask_val':mask_valid[:,:,None]}
        if 'dense' in mode:
            data = {'mat':mat[:,:,None],
                    'mask_tr':mask_tr[:,:,None],
                    'mask_ts':mask_ts[:,:,None],
                    'mask_val':mask_valid[:,:,None]}
        elif 'sparse' in mode:
            mat_sp = dense_array_to_sparse(mat, expand_dims=True)
            mask_tr_sp = dense_array_to_sparse(mask_tr, expand_dims=True)
            mask_val_sp = dense_array_to_sparse(mask_valid, expand_dims=True)
            mask_ts_sp = dense_array_to_sparse(mask_ts, expand_dims=True)

            data = {'mat':mat[:,:,None],
                    'mask_tr':mask_tr[:,:,None],
                    'mask_val':mask_valid[:,:,None],
                    'mask_ts':mask_ts[:,:,None],
                    'mat_sp':mat_sp,
                    'mask_tr_sp':mask_tr_sp,
                    'mask_val_sp':mask_val_sp}

        # pdb.set_trace()
        return data
    else:
        raise Exception("unknown dataset")
    

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.name_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar



def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def logistic(x):
    return 1. / (1 + np.exp(-x))


def log_1_plus_exp(x):
    return np.log1p(x)


def sample_minibatch(mbsz, inps, targs=None, copy=False):
    for i in range(inps.shape[0] / mbsz):
        if targs is None:
            if not copy:
                yield inps[i * mbsz:(i + 1) * mbsz, ...]
            else:
                yield inps[i * mbsz:(i + 1) * mbsz, ...].copy()
        else:
            yield inps[i * mbsz:(i + 1) * mbsz, ...], targs[i * mbsz:(i + 1) * mbsz]


def tile_raster_images(X,
                       img_shape,
                       tile_shape,
                       tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4),
                dtype='uint8')
        else:
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4),
                dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(this_x.reshape(
                            img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs):tile_row * (H + Hs) + H, tile_col *
                        (W + Ws):tile_col * (W + Ws) + W] = this_img * c
        return out_array
 



def make_uniform(Y, #n x m: m is the number of features, we want to make each feature uniform
                 nbins=1000, #used for each feature in estimating the cdf 
):

    n, m = Y.shape
    Yuni = np.zeros_like(Y)
    f_to_uni = []
    f_from_uni = []
    for feature in range(m):
        hist, bin_edges = np.histogram(Y[:,feature], bins=nbins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        cdf = interpolate.interp1d(bin_edges, cum_values)
        inv_cdf = interpolate.interp1d(cum_values, bin_edges)
        f_to_uni.append(cdf)
        f_from_uni.append(inv_cdf)
        Yuni[:,feature] = cdf(Y[:,feature])
    return Yuni, f_to_uni, f_from_uni



def uniform_to_gaussian(Y,# nxm
                        nbins=1000,
):
    from scipy.stats import norm
    bin_edges = np.linspace(-8,8,nbins)
    #centers = (bin_edges[1:] + bin_edges[:-1])/2.
    cum_values = norm.cdf(bin_edges)
    cum_values[0] = 0
    cum_values[-1] = 1
    #hist /= np.sum(hist)
    #cum_values = np.zeros(bin_edges.shape)
    #cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    #cum_values[-1] = 1.
    cdf = interpolate.interp1d(bin_edges, cum_values)
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)    
    YGauss = np.zeros_like(Y)
    for feature in range(Y.shape[1]):
        YGauss[:,feature] = inv_cdf(Y[:,feature])
    return YGauss, cdf, inv_cdf


def to_normal(Y, nbins=1000):
    if Y.ndim == 1: Y = Y[:,None]    
    Yuni, f_to_uni, f_from_uni = make_uniform(Y, nbins)
    YGauss, cdf, inv_cdf = uniform_to_gaussian(Yuni)
    f_to_normal = []
    f_from_normal = []
    for feature in range(Y.shape[1]):
        f_to_normal_m = lambda x: inv_cdf(f_to_uni[feature](x))
        f_from_normal_m = lambda x: f_from_uni[feature](cdf(x))
        f_to_normal.append(f_to_normal_m)
        f_from_normal.append(f_from_normal_m)
    return YGauss, f_to_normal, f_from_normal
        

def to_cartesian(r, theta, phi):
    x = r* np.cos(theta)*np.sin(phi)
    y = r * np.sin(theta)*np.sin(phi)
    z = r * np.cos(phi)
    return (x, y, z)


def visualizeRGB(win, imSize=None, hidDims=None,
                 ordered=False, saveFile=None, normalize=False):
    w = win - np.min(win)
    w /= np.max(w)
    numVis, numHid = w.shape
    numVis /= 3

    if imSize is None:
        imSize = (int(np.sqrt(numVis)), int(np.sqrt(numVis)))
    assert(imSize[0] * imSize[1] == numVis)

    if hidDims is None:
        tmp = min(20, int(np.ceil(np.sqrt(numHid))))
        hidDims = (tmp, tmp)
    margin = 2
    img = np.zeros((hidDims[0] * (imSize[0] + margin),
                    hidDims[1] * (imSize[1] + margin), 3))

    if ordered:
        valList = []
        for h in range(numHid):
            wtmp = w[:, h] - np.mean(w[:, h])
            val = wtmp.dot(wtmp)
            valList.append(val)
        order = np.argsort(valList)[::-1]

    for h in range(min(hidDims[0] * hidDims[1], numHid)):
        i = h / hidDims[1]
        j = h % hidDims[1]
        if ordered:
            hshow = order[h]
        else:
            hshow = h
        for co in range(3):
            tmp = np.reshape(
                w[(numVis * co):(numVis * (co + 1)), hshow], imSize)
            if normalize:
                tmp -= tmp.min()
                tmp /= tmp.max()
            img[(i * (imSize[0] + margin)):(i * (imSize[0] + margin) + imSize[0]),
                (j * (imSize[1] + margin)):(j * (imSize[1] + margin) + imSize[1]), co] = tmp

    plt.axis('off')
    if saveFile is not None:
        plt.tight_layout()
        plt.savefig(
            './figures/' +
            saveFile +
            ".svg",
            bbox_inches='tight',
            dpi=2000)
    else:
        plt.imshow(img)

    plt.show()


def visualize(win, rgb=False, imSize=None, hidDims=None,
              ordered=False, saveFile=None, normalize=False):

    if rgb:
        visualizeRGB(
            win,
            imSize=imSize,
            hidDims=hidDims,
            saveFile=saveFile,
            normalize=normalize,
            ordered=ordered)
        return
    w = win - np.min(win)
    w /= np.max(w)
    numVis, numHid = w.shape
    if imSize is None:
        imSize = (int(np.sqrt(numVis)), int(np.sqrt(numVis)))
    assert(imSize[0] * imSize[1] == numVis)
    if hidDims is None:
        tmp = min(20, int(np.ceil(np.sqrt(numHid))))
        hidDims = (tmp, tmp)

    if ordered:
        valList = []
        for h in range(numHid):
            wtmp = w[:, h] - np.mean(w[:, h])
            val = wtmp.dot(wtmp)
            valList.append(val)
        order = np.argsort(valList)[::-1]

    margin = 1
    img = np.zeros((hidDims[0] * (imSize[0] + margin),
                    hidDims[1] * (imSize[1] + margin)))
    for h in range(min(hidDims[0] * hidDims[1], numHid)):
        i = h / hidDims[1]
        j = h % hidDims[1]
        if ordered:
            hshow = order[h]
        else:
            hshow = h
        content = (np.reshape(w[:, hshow], imSize))  # - np.mean(w[:,hshow]))
        img[(i * (imSize[0] + margin)):(i * (imSize[0] + margin) + imSize[0]),
            (j * (imSize[1] + margin)):(j * (imSize[1] + margin) + imSize[1])] = content
    plt.figure()
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.Greys_r, interpolation="nearest")
    if saveFile is not None:
        plt.tight_layout()
        plt.savefig(
            './figures/' +
            saveFile +
            ".svg",
            bbox_inches='tight',
            dpi=2000)

    plt.show()
