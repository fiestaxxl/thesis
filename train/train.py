import torch
import os
from tqdm import tqdm
import time
import numpy as np
import torch.nn as nn

def train_cvae(model, optimizer, iterations, data_train, data_test, num_epochs, save_iter, path, device):

    train_molecules_input = data_train['x']
    train_molecules_output = data_train['y']
    train_labels = data_train['c']
    vocab = data_train['v']

    test_molecules_input = data_test['x']
    test_molecules_output = data_test['y']
    test_labels = data_test['c']

    train_length = data_train['l']
    test_length = data_test['l']

    loss_dict_train = dict()
    loss_dict_train['recon'] = []
    loss_dict_train['klb'] = []
    loss_dict_train['fin'] = []

    loss_dict_test = dict()
    loss_dict_test['recon'] = []
    loss_dict_test['klb'] = []
    loss_dict_test['fin'] = []

    model.to(device)

    print("STARTING TRAINING \n\n")

    for epoch in tqdm(range(1,num_epochs+1)):
        recon_loss = 0
        klb_loss = 0
        final_loss = 0

        recon_loss_test = 0
        klb_loss_test = 0
        final_loss_test = 0
        for iteration in range(1,iterations+1):

            #train
            model.train()
            n = np.random.randint(len(train_molecules_input), size = 256)
            x = torch.tensor([train_molecules_input[i] for i in n], dtype=torch.int64).to(device)
            y = torch.tensor([train_molecules_output[i] for i in n], dtype=torch.int64).to(device)
            l = torch.tensor(np.array([train_length[i] for i in n]), dtype=torch.int64).to(device)
            c = nn.functional.normalize(torch.tensor(np.array([train_labels[i] for i in n]).astype(float),dtype=torch.float).unsqueeze(1),dim=-3).to(device)

            optimizer.zero_grad()
            y_hat_softmax, y_hat, mu, logvar = model(x,c,l)

            if (epoch>2) and (iteration>700):
                beta = 0.005
            else:
                beta = 0.0000001

            recon_loss_iter, klb_loss_iter, final_loss_iter = get_losses(y_hat, y, mu, logvar, kld_weight=beta)

            final_loss_iter.backward()
            optimizer.step()

            recon_loss += recon_loss_iter.cpu().detach().item()
            klb_loss += klb_loss_iter.cpu().detach().item()
            final_loss += final_loss_iter.cpu().detach().item()


            #test
            model.eval()
            with torch.no_grad():
                n = np.random.randint(len(test_molecules_input), size = 256)
                x = torch.tensor([test_molecules_input[i] for i in n], dtype=torch.int64).to(device)
                y = torch.tensor([test_molecules_output[i] for i in n], dtype=torch.int64).to(device)
                c = nn.functional.normalize(torch.tensor(np.array([test_labels[i] for i in n]).astype(float),dtype=torch.float).unsqueeze(1),dim=-3).to(device)
                l = torch.tensor(np.array([test_length[i] for i in n]), dtype=torch.int64).to(device)

                y_hat_softmax, y_hat, mu, logvar = model(x,c,l)

                recon_loss_iter, klb_loss_iter, final_loss_iter = get_losses(y_hat, y, mu, logvar, kld_weight=beta)

                recon_loss_test += recon_loss_iter.cpu().item()
                klb_loss_test += klb_loss_iter.cpu().item()
                final_loss_test += final_loss_iter.cpu().item()

            if iteration%100==0:
                print(f"End of {iteration} iteration,\n recon_loss_train: {recon_loss/iteration}, recon_loss_test: {recon_loss_test/iteration},\n klb_loss_train: {klb_loss/iteration}, klb_loss_test: {klb_loss_test/iteration},\n total_loss_train: {final_loss/iteration}, total_loss_test: {final_loss_test/iteration}\n")


        loss_dict_train['recon'].append(recon_loss/iterations)
        loss_dict_train['klb'].append(klb_loss/iterations)
        loss_dict_train['fin'].append(final_loss/iterations)

        loss_dict_test['recon'].append(recon_loss_test/iterations)
        loss_dict_test['klb'].append(klb_loss_test/iterations)
        loss_dict_test['fin'].append(final_loss_test/iterations)




        if epoch%save_iter==0:
            torch.save(model.state_dict(),os.path.join(path,f"CVAE_epoch{epoch}.pth"))

        print(f"\n\nEnd of epoch {epoch},\n recon_loss_train: {loss_dict_train['recon'][-1]}, recon_loss_test: {loss_dict_test['recon'][-1]},\n klb_loss_train: {loss_dict_train['klb'][-1]}, klb_loss_test: {loss_dict_test['klb'][-1]},\n total_loss_train: {loss_dict_train['fin'][-1]}, total_loss_test: {loss_dict_test['fin'][-1]}\n")

    return loss_dict_train, loss_dict_test

def sequence_mask(lengths, maxlen, dtype=torch.int32):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask

def get_losses(y_hat, y, mu, logvar, kld_weight=0.0000):
    #weight = sequence_mask(l,y.shape[1])
    #weight = torch.randint(0,1,(120,4))
    loss = nn.CrossEntropyLoss(ignore_index=0)
    #print(y_hat.shape, torch.permute(y_hat,(0,2,1)).shape, y.shape, weight.shape)
    recons_loss = loss(torch.permute(y_hat,(0,2,1)), y)
    kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )
    final_loss = recons_loss + kld_weight * kld_loss

    return recons_loss, kld_loss, final_loss
