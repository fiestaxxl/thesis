import torch
import os
from tqdm import tqdm
import time
import numpy as np
import torch.nn as nn

def train_cvae(model, optimizer, epochs, train_loader, test_loader, save_iter, path, device):


    print("STARTING TRAINING \n\n")
    model.train()
    cntr = len(train_loader)//len(test_loader)

    loss_dict_train, loss_dict_test = dict(), dict()

    loss_dict_train['recon'] = []
    loss_dict_train['klb'] = []
    loss_dict_train['fin'] = []

    loss_dict_test['recon'] = []
    loss_dict_test['klb'] = []
    loss_dict_test['fin'] = []

    for epoch in tqdm(range(1,epochs+1)):
        iteration = 0

        recon_loss = 0
        klb_loss = 0
        final_loss = 0

        recon_loss_test = 0
        klb_loss_test = 0
        final_loss_test = 0

        for (X,c), y in train_loader:

            iteration+=1

            #train
            X = X.to(device)
            c = c.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            _, preds, mu, logvar = model(X,c)

            if (epoch>2) and (iteration>700):
                beta = 0.005
            else:
                beta = 0.0000001

            recon_loss_iter, klb_loss_iter, final_loss_iter = get_losses(preds, y, mu, logvar, kld_weight=beta)

            final_loss_iter.backward()
            optimizer.step()

            recon_loss += recon_loss_iter.cpu().detach().item()
            klb_loss += klb_loss_iter.cpu().detach().item()
            final_loss += final_loss_iter.cpu().detach().item()

            if iteration%cntr==0:
                model.eval()
                with torch.no_grad():
                    (X,c), y = next(iter(train_loader))

                    X = X.to(device)
                    c = c.to(device)
                    y = y.to(device)

                    _, preds, mu, logvar = model(X,c)

                    recon_loss_iter, klb_loss_iter, final_loss_iter = get_losses(preds, y, mu, logvar, kld_weight=beta)

                    recon_loss_test += recon_loss_iter.cpu().item()
                    klb_loss_test += klb_loss_iter.cpu().item()
                    final_loss_test += final_loss_iter.cpu().item()

                    if iteration%(cntr*50)==0:
                        print(f"End of {iteration} iteration,\n recon_loss_train: {recon_loss/iteration}, recon_loss_test: {recon_loss_test/iteration*cntr},\n klb_loss_train: {klb_loss/iteration}, klb_loss_test: {klb_loss_test/iteration*cntr},\n total_loss_train: {final_loss/iteration}, total_loss_test: {final_loss_test/iteration*cntr}\n")
                model.train()


        #test
        '''
        model.eval()
        with torch.no_grad():
            for (X,c), y in train_loader:
                _, preds, mu, logvar = model(X,c)

                recon_loss_iter, klb_loss_iter, final_loss_iter = get_losses(preds, y, mu, logvar, kld_weight=beta)

                recon_loss_test += recon_loss_iter.cpu().item()
                klb_loss_test += klb_loss_iter.cpu().item()
                final_loss_test += final_loss_iter.cpu().item()

                iteration+=1

                if iteration%100==0:
                    print(f"End of {iteration} iteration,\n recon_loss_train: {recon_loss/iteration}, recon_loss_test: {recon_loss_test/iteration},\n klb_loss_train: {klb_loss/iteration}, klb_loss_test: {klb_loss_test/iteration},\n total_loss_train: {final_loss/iteration}, total_loss_test: {final_loss_test/iteration}\n")

        '''
        loss_dict_train['recon'].append(recon_loss/iteration)
        loss_dict_train['klb'].append(klb_loss/iteration)
        loss_dict_train['fin'].append(final_loss/iteration)

        loss_dict_test['recon'].append(recon_loss_test/iteration*cntr)
        loss_dict_test['klb'].append(klb_loss_test/iteration*cntr)
        loss_dict_test['fin'].append(final_loss_test/iteration*cntr)




        if epoch%save_iter==0:
            torch.save(model.state_dict(),os.path.join(path,f"CVAE_epoch{epoch}.pth"))

        print(f"\n\nEnd of epoch {epoch},\n recon_loss_train: {loss_dict_train['recon'][-1]}, recon_loss_test: {loss_dict_test['recon'][-1]},\n klb_loss_train: {loss_dict_train['klb'][-1]}, klb_loss_test: {loss_dict_test['klb'][-1]},\n total_loss_train: {loss_dict_train['fin'][-1]}, total_loss_test: {loss_dict_test['fin'][-1]}\n")

    return loss_dict_train, loss_dict_test


def get_losses(y_hat, y, mu, logvar, kld_weight=0.0000):

    loss = nn.CrossEntropyLoss(ignore_index=0)
    vocab_size = y_hat.shape[-1]
    #recons_loss = loss(y_hat.permute(0,2,1), y)
    recons_loss = loss(y_hat.view(-1,vocab_size), y.view(-1))
    kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )
    final_loss = recons_loss + kld_weight * kld_loss

    return recons_loss, kld_loss, final_loss
