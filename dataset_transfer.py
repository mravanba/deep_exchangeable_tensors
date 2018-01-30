from __future__ import print_function
import h5py
from sparse_factorized_autoencoder_disjoint import main as train_model_with_opts
from sparse_factorized_autoencoder_disjoint import set_opts

def eval_dataset(fns, data, v=2):
    sess = fns["sess"]
    val_dict = {fns["mat_values_tr"]:sparse_factorized_autoencoder_disjoint.one_hot(data["mat_values_tr"]),
                fns["mask_indices_tr"]:data["mask_indices_tr"],
                fns["mat_values_val"]:sparse_factorized_autoencoder_disjoint.one_hot(data["mat_values_val"]),
                fns["mask_indices_val"]:data["mask_indices_val"],
                fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                fns["mask_split"]:(data["mask_split"] == v) * 1.
                }
    return np.sqrt(sess.run([fns["rec_loss_val"]], val_dict)[0])

def load_matlab_file(path_file, name_field):
    """
    Source: https://github.com/riannevdberg/gc-mc/blob/master/gcmc/preprocessing.py

    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out

def douban():
    path_dataset = "./data/douban/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    users, items = np.where(M)
    ratings = M[np.where(M)]
    data = {}
    data["mask_indices_tr_val"] = np.array([users, items], dtype="int").T
    data["mat_values_tr_val"] = np.array(ratings, dtype="int")
    data["mat_values_tr"] = np.array(ratings[Otraining[np.where(M)] == 1], dtype="int")
    data["mask_indices_tr"] = data["mask_indices_tr_val"][Otraining[np.where(M)] == 1, :]
    data["mask_indices_test"] = data["mask_indices_tr_val"][Otest[np.where(M)] == 1, :]
    data["mask_tr_val_split"] = np.array(Otest[np.where(M)] * 2, "int")
    return data, eval_dataset

def yahoo():
    path_dataset = "./data/yahoo_music/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    users, items = np.where(M)
    ratings = np.round(M[np.where(M)]/25.) + 1

    yahoo = {}
    yahoo["mask_indices_tr_val"] = np.array([users, items], dtype="int").T
    yahoo["mat_values_tr_val"] = np.array(ratings, dtype="int")
    yahoo["mat_values_tr"] = np.array(ratings[Otraining[np.where(M)] == 1], dtype="int")
    yahoo["mask_indices_tr"] = yahoo["mask_indices_tr_val"][Otraining[np.where(M)] == 1, :]
    yahoo["mask_indices_test"] = yahoo["mask_indices_tr_val"][Otest[np.where(M)] == 1, :]
    yahoo["mask_tr_val_split"] = np.array(Otest[np.where(M)] * 2, "int")
    
    def eval_yahoo(fns, data, v=2):
        '''
        custom eval function to map the 1-5 scale to 1-100 for yahoo.
        '''
        sess = fns["sess"]
        val_dict = {fns["mat_values_tr"]:sparse_factorized_autoencoder_disjoint.one_hot(data["mat_values_tr"]),
                    fns["mask_indices_tr"]:data["mask_indices_tr"],
                    fns["mat_values_val"]:sparse_factorized_autoencoder_disjoint.one_hot(data["mat_values_val"]),
                    fns["mask_indices_val"]:data["mask_indices_val"],
                    fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                    fns["mask_split"]:(data["mask_split"] == v) * 1.
                    }
        out = sess.run([fns["out_val"]], val_dict)[0]
        expval = (((softmax(out.reshape(-1,5))) * np.arange(1,6)[None, :]).sum(axis=1) - 0.5) * 25
        return np.sqrt(np.square(Otest[np.where(M)] *(expval- M[np.where(M)])).sum() / (Otest[np.where(M)]).sum())
    return yahoo, eval_yahoo

def flixter():
    path_dataset = "./data/flixster/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    users, items = np.where(M)
    ratings = np.round(M[np.where(M)]/25.) + 1

    flixter = {}
    flixter["mask_indices_tr_val"] = np.array([users, items], dtype="int").T
    flixter["mat_values_tr_val"] = np.array(ratings, dtype="int")
    flixter["mat_values_tr"] = np.array(ratings[Otraining[np.where(M)] == 1], dtype="int")
    flixter["mask_indices_tr"] = flixter["mask_indices_tr_val"][Otraining[np.where(M)] == 1, :]
    flixter["mask_indices_test"] = flixter["mask_indices_tr_val"][Otest[np.where(M)] == 1, :]
    flixter["mask_tr_val_split"] = np.array(Otest[np.where(M)] * 2, "int")
    
    def eval_flixter(fns, data, v=2):
        '''
        custom eval function to map the 1-5 scale to 1-100 for flixter.
        '''
        sess = fns["sess"]
        val_dict = {fns["mat_values_tr"]:sparse_factorized_autoencoder_disjoint.one_hot(data["mat_values_tr"]),
                    fns["mask_indices_tr"]:data["mask_indices_tr"],
                    fns["mat_values_val"]:sparse_factorized_autoencoder_disjoint.one_hot(data["mat_values_val"]),
                    fns["mask_indices_val"]:data["mask_indices_val"],
                    fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                    fns["mask_split"]:(data["mask_split"] == v) * 1.
                    }
        out = sess.run([fns["out_val"]], val_dict)[0]
        expval = (((softmax(out.reshape(-1,5))) * np.arange(1,6)[None, :]).sum(axis=1))
        return np.sqrt(np.square(Otest[np.where(M)] *(expval- M[np.where(M)])).sum() / (Otest[np.where(M)]).sum())
    return flixter, eval_flixter

def split_dataset_with_ratings(ratings, seed=1234, p_known=1., p_new=0.5, max_id_u=None, max_id_m=None):
    rng = np.random.RandomState(seed)
    n_ratings = ratings.rating.shape[0]
    n_users = np.max(ratings.user_id)
    n_movies = np.max(ratings.movie_id)
    # get unique users / movies 
    movie_id = np.unique(ratings["movie_id"])
    user_id = np.unique(ratings["user_id"])

    # split users into known (observed) and new (hidden) users (make 30% known)
    observed_users, hidden_users = random_split(user_id, 0.3, rng=rng)
    observed_movies, hidden_movies = random_split(movie_id, 0.3, rng=rng)

    # get a subset of users and movies simmilar in size to ml-100k for training
    p = p_known
    training_users,_ = random_split(observed_users, p, rng=rng)
    training_movies,_ = random_split(observed_movies, p, rng=rng)
    # get subset of dataframe containing selected users / movies
    known_ratings = get_subset(ratings, training_users, training_movies)
    print("Known movies / users size: %d" % known_ratings.shape[0])
    # construct new ids for the new data frame
    _, users = np.unique(known_ratings.user_id, return_inverse=True)

    known_ratings.loc[:,"user_id"] = users
    _, movies = np.unique(known_ratings.movie_id, return_inverse=True)
    known_ratings.loc[:,"movie_id"] = movies

    # split into train / valid / test as usual
    train, valid, test = train_test_valid(known_ratings, train=0.75, valid=0.05, test=0.2, rng=rng)
    # build data dictionary
    known_dat = prep_data_dict(train, valid, test, n_users, n_movies) # users.max() + 1, movies.max() + 1)

    p = p_new
    # get another subset of users and movies (distinct from previous users / movies) for evaluation.
    # the steps are the same except the users and movies are different
    new_users,_ = random_split(hidden_users, p, rng=rng)
    new_movies,_ = random_split(hidden_movies, p, rng=rng)
    new_ratings = get_subset(ratings, new_users, new_movies)
    
    _, users = np.unique(new_ratings.user_id, return_inverse=True)
    new_ratings.loc[:,"user_id"] = users
    _, movies = np.unique(new_ratings.movie_id, return_inverse=True)
    new_ratings.loc[:,"movie_id"] = movies
    if max_id_u is not None:
        new_ratings = new_ratings.loc[new_ratings.user_id < max_id_u,:]
    if max_id_m is not None:
        new_ratings = new_ratings.loc[new_ratings.movie_id < max_id_m,:]
    print("New movies / users size: %d" % new_ratings.shape[0])

    new_obs, new_hidden = train_test_valid(new_ratings, train=0.8, valid=0., test=0.2, rng=rng)
    new_dat = prep_data_dict(new_obs, test=new_hidden, n_users=n_users, n_movies=n_movies)
    return known_dat, new_dat, new_ratings

def netflix():
    '''
    TODO: finish netflix with neighhood sampling for evaluation.
    '''
    #ratings_df = pd.read_csv("./data/Netflix/NF_TRAIN/nf.train.txt", sep='\t', 
    #                  names=["user_id", "movie_id", "rating"], encoding='latin-1')
    #known_dat, new_dat, new_ratings = split_dataset_with_ratings(ratings_df, p_known=0., 
    #                                    p_new=0.07, max_id_m=10000, max_id_u=17770)
    return {}, lambda x: 0.

def main():
    print("Training")
    losses, fns = train_model_with_opts(set_opts(epochs=700))
    print("Training complete....")
    with open("results/tranfer_learning_results.log", "w") as save_file:
        print("model,rmse", file=save_file)
    for name, dataset in {"douban":douban, "flixter":flixter, "yahoo":yahoo, "netflix":netflix}.iteritems():
        data, eval_fn = dataset()
        rmse = eval_fn(fns, data)
        print(name, rmse)
        with open("results/tranfer_learning_results.log", "a") as save_file:
            print(name, rmse, sep=",", file=save_file)

if __name__ == '__main__':
    main()
