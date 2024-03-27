from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import ddgm_model_rs as dg
import os
from scipy import sparse
from data_processing import DataLoader

# Directories and Paths

path = r"C:\Users\Walker\Code for RS\Datasets\ml-20m" # Path to the dataset

final_results_dir = r"C:\Users\Walker\Code for RS\CODIGEM-main\CODIGEM-main\Results" # Directory to save the results

pro_dir = os.path.join(path, 'pro_sg')

# General Parameters
dataset_name = "ml-20m" # Name of the dataset

# Load the data
if os.path.exists(pro_dir):
        print("Data Already Preprocessed")
        loader = DataLoader(path)
        n_items = loader.load_n_items()
        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')
        test_data_tr, test_data_te = loader.load_data('test')

else:
    print("Data Not Preprocessed")
    print("Preprocessing Data")
    os.system('python data_processing.py')
    loader = DataLoader(path)
    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')


# Parameters related to the model

num = train_data.shape[0]  # number of rows in the dataframe

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
