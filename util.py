from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pdb
import sys
import matplotlib as plt
import os
import scipy.interpolate as interpolate
import functools
from sparse_util import *

floatX = "float32"
data_folder = "data"


def to_indicator(mat):
    out = np.zeros((mat.shape[0], mat.shape[1], 5))
    for i in range(1, 6):
        out[:, :, i-1] = (1 * (mat == i)).reshape((mat.shape[0], mat.shape[1]))
    return out

def to_number(mat):
    out = (np.argmax(mat, axis=2).reshape((mat.shape[0], mat.shape[1], 1)))
    out[mat.sum(axis=2) > 0] += 1
    return np.array(out, dtype=floatX)

def get_ml100k(valid=0.1, rng=None, dense=False, fold=1):
    if rng is None:
        rng = np.random.RandomState()
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    path_tr = os.path.join(data_folder,'ml-100k/u%d.base' % fold)
    path_test = os.path.join(data_folder,'ml-100k/u%d.test' % fold)

    ratings_tr_val = pd.read_csv(path_tr, sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv(path_test, sep='\t', names=r_cols, encoding='latin-1')

    n = ratings_tr_val.shape[0]
    tr_val_split = np.concatenate((np.zeros(int(n * (1-valid)), np.int32), np.ones(int(n*valid), np.int32)))
    tr_val_split = rng.permutation(tr_val_split)
    ratings_all = pd.concat([ratings_tr_val, ratings_test], ignore_index=True)
    tr_val_test_split = np.concatenate((tr_val_split, 2 * np.ones(ratings_test.shape[0], np.int32)))

    ratings_all['tr_val_split'] = tr_val_test_split
    ratings_all = ratings_all.sort_values(by=['user_id', 'movie_id'])

    n_ratings = ratings_all.rating.shape[0]

    n_users = np.max(ratings_all.user_id)
    _, movies = np.unique(ratings_all.movie_id, return_inverse=True)
    n_movies = np.max(movies) + 1

    mat_values_all = np.array(ratings_all.rating)
    mat_values_tr = np.array(ratings_all.loc[ratings_all['tr_val_split']==0,:].rating)
    mat_values_val = np.array(ratings_all.loc[ratings_all['tr_val_split']==1,:].rating)
    mat_values_test = np.array(ratings_all.loc[ratings_all['tr_val_split']==2,:].rating)
    mat_values_tr_val = np.array(ratings_all.loc[ratings_all['tr_val_split']<=1,:].rating)

    mask_indices_all = np.array(list(zip(ratings_all.user_id-1, ratings_all.movie_id-1)))
    mask_indices_tr = np.array(list(zip(ratings_all.loc[ratings_all['tr_val_split']==0,:].user_id-1, 
                                        ratings_all.loc[ratings_all['tr_val_split']==0,:].movie_id-1)))
    mask_indices_val = np.array(list(zip(ratings_all.loc[ratings_all['tr_val_split']==1,:].user_id-1, 
                                        ratings_all.loc[ratings_all['tr_val_split']==1,:].movie_id-1)))
    mask_indices_test = np.array(list(zip(ratings_all.loc[ratings_all['tr_val_split']==2,:].user_id-1, 
                                        ratings_all.loc[ratings_all['tr_val_split']==2,:].movie_id-1)))
    mask_indices_tr_val = np.array(list(zip(ratings_all.loc[ratings_all['tr_val_split']<=1,:].user_id-1, 
                                        ratings_all.loc[ratings_all['tr_val_split']<=1,:].movie_id-1)))

    tr_val_split = np.array(ratings_all['tr_val_split'])

    # for dense...
    data = {'mat_values_all':mat_values_all,
            'mat_values_tr':mat_values_tr,
            'mat_values_val':mat_values_val,
            'mat_values_test':mat_values_test,
            'mask_indices_all':mask_indices_all,
            'mask_indices_tr':mask_indices_tr,
            'mask_indices_val':mask_indices_val,
            'mask_indices_test':mask_indices_test,                   
            'mat_shape':[n_users, n_movies, 1], 
            'mask_tr_val_split':tr_val_split}

    if dense:
        mat_tr_val = sparse_array_to_dense(mat_values_tr_val, mask_indices_tr_val, [n_users, n_movies, 1])

        mask_tr_val = np.zeros([n_users, n_movies])
        mask_tr_val[list(zip(*mask_indices_tr_val))] = 1

        mask_tr = np.zeros([n_users, n_movies])
        mask_tr[list(zip(*mask_indices_tr))] = 1

        mask_val = np.zeros([n_users, n_movies])
        mask_val[list(zip(*mask_indices_val))] = 1
        data.update({'mat_tr_val':mat_tr_val[:,:,None],
                'mask_tr_val':mask_tr_val[:,:,None],
                'mask_tr':mask_tr[:,:,None],
                'mask_val':mask_val[:,:,None]})
    return data

def get_data(dataset='movielens-small',
                 mode='sparse',#returned matrix: dense, sparse, table
                 train=.8,
                 test=.1,
                 valid=.1,
                 seed=1234,
                 **kwargs
                 ):
    rng = np.random.RandomState(seed)
    
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
            rand_perm = rng.permutation(n_ratings)
            mask_tr = np.zeros((n_users, n_movies), dtype=np.float32)
            mask_ts = np.zeros((n_users, n_movies), dtype=np.float32)
            mask_val = np.zeros((n_users, n_movies), dtype=np.float32)
            mask_tr[ratings.userId[rand_perm[:n_train]]-1, movies[rand_perm[:n_train]]] = 1
            mask_ts[ratings.userId[rand_perm[n_train:n_test]]-1,movies[rand_perm[n_train:n_test]]] = 1
            mask_val[ratings.userId[rand_perm[n_test:n_valid]]-1,movies[rand_perm[n_test:n_valid]]] = 1
            data = {'mat':mat[:,:,None], 'mask_tr':mask_tr[:,:,None],
                        'mask_ts':mask_ts[:,:,None], 'mask_val':mask_val[:,:,None]}
            return data

    elif 'movielens-TEST' in dataset:
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        path_tr = os.path.join(data_folder,'ml-TEST/u1.base')
        path_ts = os.path.join(data_folder,'ml-TEST/u1.test')
        ratings_tr_val = pd.read_csv(path_tr, sep='\t', names=r_cols, encoding='latin-1')
        ratings_ts = pd.read_csv(path_ts, sep='\t', names=r_cols, encoding='latin-1')
        ratings = pd.concat([ratings_tr_val, ratings_ts], ignore_index=True)
        n_ratings = ratings.rating.shape[0]
        n_ratings_tr_val = ratings_tr_val.rating.shape[0]
        n_ratings_ts = ratings_ts.rating.shape[0]
        n_users = np.max(ratings.user_id)
        _, movies = np.unique(ratings.movie_id, return_inverse=True)
        n_movies = np.max(movies) + 1
        _, movies_tr = np.unique(ratings_tr_val.movie_id, return_inverse=True)
        _, movies_ts = np.unique(ratings_ts.movie_id, return_inverse=True)
        mat_tr_val = np.zeros((n_users, n_movies), dtype=np.float32)
        mat_tr_val[ratings_tr_val.user_id-1, movies_tr] = ratings_tr_val.rating
        mat_ts = np.zeros((n_users, n_movies), dtype=np.float32)
        mat_ts[ratings_ts.user_id-1, movies_ts] = ratings_ts.rating
        rand_perm = rng.permutation(n_ratings_tr_val)
        n_train = int(n_ratings_tr_val * train)
        n_valid = n_ratings_tr_val - n_train
        mask_tr_val = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_tr_val[ratings_tr_val.user_id[rand_perm]-1, movies_tr[rand_perm]] = 1
        mask_indices_tr_val = get_mask_indices(mask_tr_val)
        p_train = train / (train + valid)
        p_valid = 1 - p_train
        mask_tr_val_split = rng.choice([0,1], size=n_ratings_tr_val, p=[p_train, p_valid])
        mask_indices_tr = mask_indices_tr_val[mask_tr_val_split == 0,:]
        mask_indices_val = mask_indices_tr_val[mask_tr_val_split == 1,:]
        mask_ts = np.zeros((n_users, n_movies), dtype=np.float32)
        mask_ts[ratings_ts.user_id-1, movies_ts] = 1
        mask_indices_ts = get_mask_indices(mask_ts)
        mask_tr = sparse_array_to_dense({'indices':mask_indices_tr, 'values':np.ones(shape=mask_indices_tr.shape[0]), 'dense_shape':[n_users, n_movies]})
        mask_val = sparse_array_to_dense({'indices':mask_indices_val, 'values':np.ones(shape=mask_indices_val.shape[0]), 'dense_shape':[n_users, n_movies]})
        mat_values_tr_val = dense_array_to_sparse_values(mat_tr_val[:,:,None], mask_indices_tr_val)
        mat_values_tr = dense_array_to_sparse_values(mat_tr_val[:,:,None], mask_indices_tr)
        mat_values_val = dense_array_to_sparse_values(mat_tr_val[:,:,None], mask_indices_val)
        mat_values_ts = dense_array_to_sparse_values(mat_ts[:,:,None], mask_indices_ts)

        mat_tr_val_ts = mat_tr_val + mat_ts
        mask_indices_tr_val_ts = get_mask_indices(mask_tr_val + mask_ts)
        mask_ts_split = np.zeros(mask_indices_tr_val_ts.shape[0])
        for i, row in enumerate(mask_indices_tr_val_ts):
            for row_ts in mask_indices_ts:
                if list(row) == list(row_ts):
                    mask_ts_split[i] = 1
        mat_values_tr_val_ts = dense_array_to_sparse_values(mat_tr_val_ts[:,:,None], mask_indices_tr_val_ts)

        data = {'mat_tr_val':mat_tr_val[:,:,None],
                'mask_tr_val':mask_tr_val[:,:,None],
                'mask_tr':mask_tr[:,:,None],
                'mask_val':mask_val[:,:,None],
                'mask_ts':mask_ts[:,:,None],
                
                'mat_values_tr_val':mat_values_tr_val,
                'mat_values_tr':mat_values_tr,
                'mat_values_val':mat_values_val,
                'mat_values_ts':mat_values_ts,
                'mask_indices_tr':mask_indices_tr,        
                'mask_indices_val':mask_indices_val,
                'mask_indices_ts':mask_indices_ts,
                'mask_indices_tr_val':mask_indices_tr_val,
                'mask_tr_val_split':mask_tr_val_split, 
                'mask_ts_split':mask_ts_split}

        # pdb.set_trace()
        return data

    elif 'movielens-100k' in dataset:
        return get_ml100k(valid, rng, mode=="dense", kwargs.get("fold", 1))

    elif 'movielens-1M' in dataset:
        r_cols = ['user_id', None, 'movie_id', None, 'rating', None, 'unix_timestamp']
        path = os.path.join(data_folder, 'ml-1m/ratings.dat')

        ratings = pd.read_csv(path, sep=':', names=r_cols, encoding='latin-1')
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

        n_ratings = ratings.rating.shape[0]
        n_users = np.max(ratings.user_id)

        _, movies = np.unique(ratings.movie_id, return_inverse=True)
        n_movies = np.max(movies) + 1



        split = np.random.choice([0,1], size=n_ratings, p=(train + valid, test))

        ratings_tr_val = ratings[split==0]
        ratings_ts = ratings[split==1]

        p_tr = train / (train + valid)
        p_val = valid / (train + valid)
        
        split_tr_val = np.random.choice([0,1], size=(n_ratings - len(ratings_ts)), p=(p_tr, p_val))

        ratings_tr = ratings_tr_val[split_tr_val==0]
        ratings_val = ratings_tr_val[split_tr_val==1]

        split_tr_val = np.concatenate((split_tr_val, 2 * np.ones(ratings_ts.shape[0], np.int32)))



        mat_values_all = np.array(ratings.rating)
        mat_values_tr_val = np.array(ratings_tr_val.rating)
        mat_values_tr = np.array(ratings_tr.rating)
        mat_values_val = np.array(ratings_val.rating)
        mat_values_ts = np.array(ratings_ts.rating)

        _, movies_tr = np.unique(ratings_tr.movie_id, return_inverse=True)
        _, movies_val = np.unique(ratings_val.movie_id, return_inverse=True)
        _, movies_ts = np.unique(ratings_ts.movie_id, return_inverse=True)
        _, movies_tr_val = np.unique(ratings_tr_val.movie_id, return_inverse=True)

        mask_indices_all = np.array(list(zip(ratings.user_id-1, movies)))
        mask_indices_tr_val = np.array(list(zip(ratings_tr_val.user_id-1, movies_tr_val)))
        mask_indices_tr = np.array(list(zip(ratings_tr.user_id-1, movies_tr)))
        mask_indices_val = np.array(list(zip(ratings_val.user_id-1, movies_val)))
        mask_indices_ts = np.array(list(zip(ratings_ts.user_id-1, movies_ts)))

        n_users_tr_val = np.max(mask_indices_tr_val[:,0]) + 1
        n_movies_tr_val = np.max(mask_indices_tr_val[:,1]) + 1

        data = {'mat_values_all':mat_values_all,
                'mask_indices_all':mask_indices_all,
                'mat_values_tr_val':mat_values_tr_val,
                'mask_indices_tr_val':mask_indices_tr_val,
                'mat_values_tr':mat_values_tr,
                'mask_indices_tr':mask_indices_tr,
                'mat_values_val':mat_values_val,                        
                'mask_indices_val':mask_indices_val, 
                'mat_values_test':mat_values_ts,
                'mask_indices_test':mask_indices_ts,
                'mat_shape':[n_users, n_movies, 1], 
                'mask_tr_val_split':split_tr_val
                }

        if mode=='dense':
            mat_tr_val = sparse_array_to_dense(mat_values_tr_val, mask_indices_tr_val, [n_users_tr_val, n_movies_tr_val, 1])
        
            mask_tr_val = np.zeros([n_users, n_movies])
            mask_tr_val[mask_indices_tr_val] = 1
        
            mask_tr = np.zeros([n_users, n_movies])
            mask_tr[mask_indices_tr] = 1
       
            mask_val = np.zeros([n_users, n_movies])
            mask_val[mask_indices_val] = 1
            data.update({'mat_tr_val':mat_tr_val,
                        'mask_tr_val':mask_tr_val,
                        'mask_tr':mask_tr,
                        'mask_val':mask_val})

        return data

    elif 'netflix' in dataset:

        print("--> loading ", dataset, " data")

        r_cols = ['user_id', 'movie_id', 'rating', 'date']
        path = os.path.join(data_folder, dataset, 'ratings.dat')

        print("     reading csv...")

        ratings = pd.read_csv(path, sep='\t', names=r_cols, encoding='latin-1')        
        ratings.rating = ratings.rating.astype(int)

        n_ratings = ratings.rating.shape[0]
        n_users = np.max(ratings.user_id)

        _, movies = np.unique(ratings.movie_id, return_inverse=True)
        n_movies = np.max(movies) + 1

        print("     loading", n_ratings, "ratings for", n_users, "users on", n_movies, "movies...")

        split = np.random.choice([0,1], size=n_ratings, p=(train + valid, test))

        ratings_tr_val = ratings[split==0]
        ratings_ts = ratings[split==1]

        p_tr = train / (train + valid)
        p_val = valid / (train + valid)
        
        split_tr_val = np.random.choice([0,1], size=(n_ratings - len(ratings_ts)), p=(p_tr, p_val))        

        print("     populating ratings arrays...")

        ratings_tr = ratings_tr_val[split_tr_val==0]
        ratings_val = ratings_tr_val[split_tr_val==1]

        split_tr_val = np.concatenate((split_tr_val, 2 * np.ones(ratings_ts.shape[0], np.int32)))

        mat_values_all = np.array(ratings.rating)
        mat_values_tr_val = np.array(ratings_tr_val.rating)
        mat_values_tr = np.array(ratings_tr.rating)
        mat_values_val = np.array(ratings_val.rating)
        mat_values_ts = np.array(ratings_ts.rating)

        print("     populating index arrays...")

        _, movies_tr_val = np.unique(ratings_tr_val.movie_id, return_inverse=True)
        _, movies_tr = np.unique(ratings_tr.movie_id, return_inverse=True)
        _, movies_val = np.unique(ratings_val.movie_id, return_inverse=True)
        _, movies_ts = np.unique(ratings_ts.movie_id, return_inverse=True)

        mask_indices_all = np.array(list(zip(ratings.user_id-1, movies)))
        mask_indices_tr_val = np.array(list(zip(ratings_tr_val.user_id-1, movies_tr_val)))
        mask_indices_tr = np.array(list(zip(ratings_tr.user_id-1, movies_tr)))
        mask_indices_val = np.array(list(zip(ratings_val.user_id-1, movies_val)))
        mask_indices_ts = np.array(list(zip(ratings_ts.user_id-1, movies_ts)))

        data = {'mat_values_all':mat_values_all,
                'mask_indices_all':mask_indices_all,
                'mat_values_tr_val':mat_values_tr_val,
                'mask_indices_tr_val':mask_indices_tr_val,
                'mat_values_tr':mat_values_tr,
                'mask_indices_tr':mask_indices_tr,
                'mat_values_val':mat_values_val,                        
                'mask_indices_val':mask_indices_val, 
                'mat_shape':[n_users, n_movies, 1], 
                'mask_tr_val_split':split_tr_val}


        # pdb.set_trace()

        print("--> netflix data loaded.")


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
