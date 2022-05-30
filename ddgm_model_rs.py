import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import ranking_metrics as metric


# Auxiliary functions, classes such as distributions


PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7


def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - \
        0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_min_exp(a, b, epsilon=1e-8):

    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y


def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
        F.logsigmoid((x + 0.5 - mean) / scale),
        F.logsigmoid((x - 0.5 - mean) / scale))

    return logp


def log_integer_probability_standard(x):
    logp = log_min_exp(
        F.logsigmoid(x + 0.5),
        F.logsigmoid(x - 0.5))

    return logp


# Definition of Collaborative Diffusion Generative Model (CODIGEM)


class DDGM(nn.Module):
    def __init__(self, p_dnns, decoder_net, beta, T, D):
        super(DDGM, self).__init__()

        print('CODIGEM in execution')

        self.p_dnns = p_dnns  # a list of sequentials

        self.decoder_net = decoder_net

        self.init_weights()

        # other params
        self.D = D

        self.T = T

        self.beta = torch.FloatTensor([beta]).to(device)

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.beta) * x + torch.sqrt(self.beta) * torch.randn_like(x)

    def forward(self, x, anneal, reduction='avg'):
        # =====
        # forward difussion
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        # =====
        # backward diffusion
        mus = []
        log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1):

            h = self.p_dnns[i](zs[i+1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            mus.append(mu_i)
            log_vars.append(log_var_i)

        mu_x = self.decoder_net(zs[0])

        # =====ELBO
        # RE

        # Normal RE
        RE = log_standard_normal(x - mu_x).sum(-1)

        # KL
        KL = (log_normal_diag(zs[-1], torch.sqrt(1. - self.beta) * zs[-1],
                              torch.log(self.beta)) - log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):
            KL_i = (log_normal_diag(zs[i], torch.sqrt(1. - self.beta) * zs[i], torch.log(
                self.beta)) - log_normal_diag(zs[i], mus[i], log_vars[i])).sum(-1)

            KL = KL + KL_i

        # Final ELBO
        anneal = 1
        if reduction == 'sum':
            loss = -(RE - anneal * KL).sum()
        else:
            loss = -(RE - anneal * KL).mean()

        return loss, mu_x

    def init_weights(self):
        for layer in self.p_dnns:
            # Xavier Initialization for weights
            size = layer[0].weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer[0].weight.data.normal_(0.0, 0.0001)

            # Normal Initialization for Biases
            layer[0].bias.data.normal_(0.0, 0.0001)

        for layer in self.decoder_net:
            # Xavier Initialization for weights
            if str(layer) == "Linear":
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, 0.0001)

                # Normal Initialization for Biases
                layer.bias.data.normal_(0.0, 0.0001)

# Training step


device = "cuda" if torch.cuda.is_available() else "cpu"
update_count = 0


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def training(model, optimizer, training_loader):

    N = training_loader.shape[0]
    global update_count
    batch_size = 200
    total_anneal_steps = 200000
    anneal_cap = 0.2

    idxlist = list(range(N))
    np.random.shuffle(idxlist)

    # TRAINING
    model.to(device)
    model.train()

    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(start_idx + batch_size, N)
        data = training_loader[idxlist[start_idx:end_idx]]

        data = naive_sparse2tensor(data)
        data = data.to(device)

        if total_anneal_steps > 0:
            anneal = min(anneal_cap,
                         1. * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap

        loss, recon = model.forward(data, anneal)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        update_count += 1

    loss_train = loss.item()

    return loss_train


def evaluate(data_tr, data_te, model, N):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    batch_size = 200
    total_anneal_steps = 200000
    anneal_cap = 0.2

    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    n20_list = []
    n50_list = []
    n100_list = []
    r20_list = []
    r50_list = []
    r100_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            if total_anneal_steps > 0:
                anneal = min(anneal_cap,
                             1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap

            loss, recon_batch = model.forward(data_tensor, anneal)

            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n20 = metric.NDCG_binary_at_k_batch(
                recon_batch, heldout_data, 20)
            n50 = metric.NDCG_binary_at_k_batch(
                recon_batch, heldout_data, 50)
            n100 = metric.NDCG_binary_at_k_batch(
                recon_batch, heldout_data, 100)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)
            r100 = metric.Recall_at_k_batch(recon_batch, heldout_data, 100)

            n20_list.append(n20)
            n50_list.append(n50)
            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)
            r100_list.append(r100)

    total_loss /= len(range(0, e_N, batch_size))
    n20_list = np.concatenate(n20_list)
    n50_list = np.concatenate(n50_list)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    r100_list = np.concatenate(r100_list)

    return total_loss, np.mean(n20_list), np.mean(n50_list), np.mean(n100_list), np.mean(r20_list), np.mean(r50_list), np.mean(r100_list)
