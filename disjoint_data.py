import pandas as pd
import numpy as np
import os

def random_split(x, p=0.5, rng=None):
    '''
    Split x vector in two with proporition p and (1-p) in each split
    '''
    if rng is None:
        rng = np.random.RandomState()
    n = x.shape[0]
    idx = rng.permutation(n)
    return x[idx[0:int(n*p)]], x[idx[int(n*p):n]]

def get_subset(df, users, movies):
    '''
    Get the subset of data points with ratings associated with given users & movies
    '''
    return df.loc[np.logical_and(np.isin(df["user_id"], users), 
                                 np.isin(df["movie_id"], movies)), :]

def get_n(df, n, rng=None):
    '''
    Get random `n` ratings from a dataframe. 
    Returns: mask, ratings, remaining dataframe (after `n` ratings are removed)
    '''
    if rng is None:
        rng = np.random.RandomState()
    df = np.array(df)
    idx = rng.permutation(np.arange(df.shape[0]))
    subset = df[idx[0:n], :]
    remainder = df[idx[n:], :]
    return subset[:, 0:2], subset[:, 2], remainder

def get_test(beta, n_train, ratings_obs_obs, ratings_obs_hid, ratings_hid_obs, ratings_hid_hid, rng=None):
    '''
    
    '''
    test_mask = np.zeros((0,2), dtype="int")
    test_values = np.zeros((0), dtype="float32")
    remainders = []
    for df, prop in zip([ratings_obs_obs, ratings_hid_obs, ratings_obs_hid, ratings_hid_hid],
                    [beta * beta, beta * (1-beta), (1-beta) * beta,  (1-beta) * (1-beta)]):
        out = get_n(df, int(n_train * prop), rng)
        test_mask = np.concatenate([test_mask, out[0]], axis=0)
        test_values = np.concatenate([test_values, out[1]], axis=0)
        remainders.append(out[2])
    return test_mask, test_values, remainders


def train_test_valid(df, train=0.8, test=0.2, valid=0., rng=None):
    assert (train + test + valid) == 1.
    test_mask, test_values, df = get_n(df, int(df.shape[0] * test),rng=rng)
    if valid > 0.:
        valid_mask, valid_values, df = get_n(df, int(df.shape[0] * valid),rng=rng)
    train_mask, train_values = (df[:,0:2], df[:,2])
    if valid > 0.:
        return (train_mask, train_values), (valid_mask, valid_values), (test_mask, test_values)
    else:
        return (train_mask, train_values), (test_mask, test_values)


def prep_data_dict(train, valid=None, test=None, n_users=None, n_movies=None):
    valid = valid if valid is not None else (np.zeros((0,2), dtype="int"), np.zeros((0), dtype="int"))
    test = test if test is not None else (np.zeros((0,2), dtype="int"), np.zeros((0), dtype="int"))
    
    return {'mat_values_tr_val':np.concatenate([np.array(i[1], dtype="int") for i in [train, valid, test]],axis=0),
            'mat_values_tr':np.array(train[1], dtype="int"),
            'mat_values_val':np.array(valid[1], dtype="int"),
            'mat_values_test':np.array(test[1], dtype="int"),
            'mask_indices_tr_val':np.concatenate([i[0] for i in [train, valid, test]], axis=0),
            'mask_indices_tr':train[0],
            'mask_indices_val':valid[0],
            'mask_indices_test':test[0],                   
            'mat_shape':[n_users, n_movies, 1], 
            'mask_tr_val_split':np.concatenate([i * np.ones_like(d[1]) for i, d in enumerate([train, valid, test])],axis=0)}

def load_ratings(seed=1234):
    rng = np.random.RandomState(seed)
    # read data
    r_cols = ['user_id', None, 'movie_id', None, 'rating', None, 'unix_timestamp']

    ratings = pd.read_csv("./data/ml-1m/ratings.dat", sep=':', names=r_cols, encoding='latin-1')
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    n_ratings = ratings.rating.shape[0]
    n_users = np.max(ratings.user_id)
    n_movies = np.max(ratings.movie_id)
    # get unique users / movies 
    movie_id = np.unique(ratings["movie_id"])
    user_id = np.unique(ratings["user_id"])

    # remove unneccessary columns
    ratings = ratings.loc[:, ["user_id", "movie_id", "rating"]]

    # split users into known (observed) and new (hidden) users
    observed_users, hidden_users = random_split(user_id, 0.3, rng=rng)
    observed_movies, hidden_movies = random_split(movie_id, 0.3, rng=rng)

    # get a subset of users and movies simmilar in size to ml-100k for training
    p = 1.
    # split user vec into 2 with `p` in the first hald and `1-p` in the second half (randomly permuted)
    training_users,_ = random_split(observed_users, p, rng=rng)
    training_movies,_ = random_split(observed_movies, p, rng=rng)
    # get subset of dataframe containing selected users / movies
    known_ratings = get_subset(ratings, training_users, training_movies)
    # construct new ids for the new data frame
    _, users = np.unique(known_ratings.user_id, return_inverse=True)
    known_ratings.loc[:,"user_id"] = users
    _, movies = np.unique(known_ratings.movie_id, return_inverse=True)
    known_ratings.loc[:,"movie_id"] = movies

    # split into train / valid / test as usual
    train, valid, test = train_test_valid(known_ratings, train=0.75, valid=0.05, test=0.2, rng=rng)
    # build data dictionary 
    known_dat = prep_data_dict(train, valid, test, n_users, n_movies) # users.max() + 1, movies.max() + 1)

    p = 0.5
    # get another subset of users and movies (distinct from previous users / movies) for evaluation.
    # the steps are the same except the users and movies are different
    new_users,_ = random_split(hidden_users, p, rng=rng)
    new_movies,_ = random_split(hidden_movies, p, rng=rng)
    new_ratings = get_subset(ratings, new_users, new_movies)
    _, users = np.unique(new_ratings.user_id, return_inverse=True)
    new_ratings.loc[:,"user_id"] = users
    _, movies = np.unique(new_ratings.movie_id, return_inverse=True)
    new_ratings.loc[:,"movie_id"] = movies

    new_obs, new_hidden = train_test_valid(new_ratings, train=0.8, valid=0., test=0.2, rng=rng)
    new_dat = prep_data_dict(new_obs, test=new_hidden, n_users=n_users, n_movies=n_movies)
    return known_dat, new_dat


