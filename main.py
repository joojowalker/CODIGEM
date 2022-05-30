from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import ddgm_model_rs as dg
import os
from scipy import sparse
#import evaluate_iop as ev


class DataLoader():
    '''
    Load Movielens-20m dataset
    '''

    def __init__(self, path):
        self.pro_dir = os.path.join(path, 'pro_sg')
        self.n_items = self.load_n_items()

    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(
                test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':

    path = "SPECIFY PATH TO DATA HERE"
    pro_dir = os.path.join(path, 'pro_sg')

    if os.path.exists(pro_dir):
        print("Data Already Preprocessed")
        loader = DataLoader(path)

        n_items = loader.load_n_items()
        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')
        test_data_tr, test_data_te = loader.load_data('test')

    else:
        print("Load and Preprocess Movielens-20m dataset")
        # Load Data
        DATA_DIR = path
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
        raw_data = raw_data[raw_data['rating'] > 3.5]

        # Filter Data
        raw_data, user_activity, item_popularity = filter_triplets(raw_data)

        # Shuffle User Indices
        unique_uid = user_activity.index
        np.random.seed(98765)
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]

        n_users = unique_uid.size
        n_heldout_users = 10000

        # Split Train/Validation/Test User Indices
        tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
        vd_users = unique_uid[(n_users - n_heldout_users * 2)                              : (n_users - n_heldout_users)]
        te_users = unique_uid[(n_users - n_heldout_users):]

        train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
        unique_sid = pd.unique(train_plays['movieId'])

        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        pro_dir = os.path.join(DATA_DIR, 'pro_sg')

        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)

        vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

        test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
        test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

        test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

        train_data = numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

        vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(
            pro_dir, 'validation_tr.csv'), index=False)

        vad_data_te = numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(
            pro_dir, 'validation_te.csv'), index=False)

        test_data_tr = numerize(test_plays_tr, profile2id, show2id)
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

        test_data_te = numerize(test_plays_te, profile2id, show2id)
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

        print("Done With Data Processing!")

    # Directories and Paths

    dataset_name = "ML20m"  # SPECIFY NAME OF DATA

    final_results_dir = "SPECIFY PATH TO WHERE RESULTS WOULD BE STORED"

    # Hyperparameters

    num = train_data.shape[0]

    D = n_items   # input dimension

    M = 200  # the number of neurons in scale (s) and translation (t) nets

    T = 3  # hyperparater to tune

    beta = 0.0001  # hyperparater to tune #Beta = 0.0001 is best so far

    lr = 1e-3  # learning rate
    num_epochs = 100  # max. number of epochs
    max_patience = 10  # an early stopping is used, if training doesn't improve for longer than 10 epochs, it is stopped
    patience = 0
    nll_val_list = []
    final_results = {}
    update_count = 0

    # Creating a folder for results

    name = 'ddgm_cf' + '_' + str(T) + '_' + str(beta)
    result_dir = 'results/' + name + '/'
    if not (os.path.exists(result_dir)):
        os.makedirs(result_dir)

    # Initializing the model

    p_dnns = nn.ModuleList([nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                          nn.Linear(M, M), nn.PReLU(),
                                          nn.Linear(M, M), nn.PReLU(),
                                          nn.Linear(M, M), nn.PReLU(),
                                          nn.Linear(M, M), nn.PReLU(),
                                          nn.Linear(M, 2*D)) for _ in range(T-1)])

    decoder_net = nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                nn.Linear(M, M), nn.PReLU(),
                                nn.Linear(M, M), nn.PReLU(),
                                nn.Linear(M, M), nn.PReLU(),
                                nn.Linear(M, M), nn.PReLU(),
                                nn.Linear(M, D), nn.Tanh())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Eventually, we initialize the full model
    model = dg.DDGM(p_dnns, decoder_net, beta=beta, T=T, D=D)
    model = model.to(device)

    # OPTIMIZER--Adamax is used
    optimizer = torch.optim.Adamax(
        [p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Training loop with early stopping
    for e in range(num_epochs):

        # Training and Validation procedure
        nll_val = dg.training(
            model=model, optimizer=optimizer, training_loader=train_data)

        nll_val_list.append(nll_val)  # save for plotting
        print("Loss at the training epoch {} is {}".format(e+1, nll_val))

        # Validation Procedure

        val_loss, n20, n50, n100, r20, r50, r100 = dg.evaluate(
            vad_data_tr, vad_data_te, model, num)

        print('| Results of training | Val loss {:4.4f} | n20 {:4.4f}| n50 {:4.4f}| n100 {:4.4f} | r20 {:4.4f} | '
              'r50 {:4.4f} | r100 {:4.4f} |'.format(val_loss, n20,  n50, n100, r20, r50, r100))

        metric_set = r50
        if e == 0:
            best_metric = metric_set
            print('saved!')
            torch.save({'state_dict': model.state_dict()},
                       result_dir + name + '_model.pth')

            final_results.update({'EpochofResults': e+1, 'val_loss': val_loss,
                                  'n20': n20, 'n50': n50, 'n100': n100, 'r20': r20, 'r50': r50, 'r100': r100})
        else:
            if metric_set > best_metric:
                best_metric = metric_set
                print('saved!')
                torch.save({'state_dict': model.state_dict()},
                           result_dir + name + '_model.pth')

                final_results.update({'EpochofResults': e+1, 'val_loss': val_loss,
                                      'n20': n20, 'n50': n50, 'n100': n100, 'r20': r20, 'r50': r50, 'r100': r100})
                # print(final_results)
                patience = 0

            else:
                patience = patience + 1

        if patience > max_patience:
            break

    # Final Evaluation Procedure

    # Load the best saved model.
    with open(result_dir + name + '_model.pth', 'rb') as f:
        saved_model = torch.load(f)
        model.load_state_dict(saved_model["state_dict"])

    test_loss, n20, n50, n100, r20, r50, r100 = dg.evaluate(
        test_data_tr, test_data_te, model, num)

    final_results.update({'test_loss': test_loss,
                          'n20': n20, 'n50': n50, 'n100': n100, 'r20': r20, 'r50': r50, 'r100': r100})

    quick_dir = os.path.join(final_results_dir, 'experiment_results')

    if not os.path.exists(quick_dir):
        os.makedirs(quick_dir)
    with open(os.path.join(quick_dir, dataset_name + name + "_experiments.txt"), 'w') as ff:
        ff.write('Number of Epochs: %d\t Epoch of Results: %d\t Validation loss: %4.4f\t Test loss: %4.4f\t n20: %4.4f\t n50: %4.4f\t n100: %4.4f\t r20: %4.4f\t r50: %4.4f\t r100: %4.4f\t' %
                 (e+1, final_results['EpochofResults'], final_results['val_loss'], final_results['test_loss'], final_results['n20'], final_results['n50'], final_results['n100'], final_results['r20'], final_results['r50'], final_results['r100']))

    print('=' * 154)
    print('| End of training | Validation loss {:4.4f} | Test loss {:4.4f} | n20 {:4.4f}| n50 {:4.4f}| n100 {:4.4f} | r20 {:4.4f} | '
          'r50 {:4.4f} | r100 {:4.4f} |'.format(final_results['val_loss'], final_results['test_loss'], final_results['n20'], final_results['n50'], final_results['n100'], final_results['r20'], final_results['r50'], final_results['r100']))
    print('=' * 154)
