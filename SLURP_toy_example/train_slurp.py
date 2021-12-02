import os
import h5py
import argparse
import numpy as np
from data_loader import DatasetFromFolder

import torch

from models import OneHiddenLayerNet, UncertaintyEstimator, weights_init_kaiming
from utils import single_draw_figs, single_draw_train_figs


# Dataset settings
def parse_arguments():
    parser = argparse.ArgumentParser(description='parameters for training main task estimator and uncertainty estimator.')
    parser.add_argument('--data_path', type = str, default = 'dataGP.hdf5', help='path of generated data.')
    parser.add_argument('--batch_size', type = int, default = 50, help='batch size for training.')
    parser.add_argument('--threads', type = int, default = 2, help='number of workers (threads) for training.')
    parser.add_argument('--nb_features', type = int, default = 3000, help='number of hidden units of one layer main task model.')
    parser.add_argument('--lr_main', type = float, default = 1e-2, help='learning rate for main task model.')
    parser.add_argument('--lr_uncer', type = float, default = 1e-3, help='learning rate for uncertainty estimator.')
    parser.add_argument('--wd_main', type = float, default = 1e-2, help='weight decay for main task model.')
    parser.add_argument('--wd_uncer', type = float, default = 1e-2, help='weight decay for uncertainty estimator.')
    parser.add_argument('--nb_epochs', type = int, default = 50, help='total epochs for training two tasks jointly.')
    parser.add_argument('--save_path', type = dir_path, default = './toy_slurp_models/', help='save path for models.')
    return parser
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        os.mkdir(path)
        return path

def main():
    parser = parse_arguments()
    args = parser.parse_args()
    
    print('----------------------------------------------------------------------------------------------------')
    print('1/ Loading the data')
    print('----------------------------------------------------------------------------------------------------')
    try:
        h5f = h5py.File(args.data_path, 'r')
        X_train = h5f['X_train'][:]
        y_train = h5f['y_train'][:]
        X_test = h5f['X_test'][:]
        y_test = h5f['y_test'][:]
        x_plot = h5f['x_plot'][:]
        y_plot = h5f['y_plot'][:]
        h5f.close()
    except IOError:
        print(args.data_path + " not accessible. Did you build the data? If not type first 'python save_data.py' . ")

    # Data transoform + Data lader
    batch_size = args.batch_size
    save_path = args.save_path

    normalize_data = True
    if normalize_data== False:
        std_X_train = np.ones(X_train.shape[1])
        mean_X_train = np.zeros(X_train.shape[1])
    else:
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)
    
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    print('CHECKING DIMENSION',np.shape(X_train),np.shape(y_train))#,np.shape(y_train))

    testloader_plot= torch.utils.data.DataLoader(DatasetFromFolder(x_plot,y_plot,mean_X_train,std_X_train,mean_y_train,std_y_train,phase='val'), batch_size=batch_size,
                                          shuffle=False, num_workers=args.threads)

    trainloader = torch.utils.data.DataLoader(DatasetFromFolder(X_train,y_train,mean_X_train,std_X_train,mean_y_train,std_y_train,phase='train'), batch_size=batch_size,
                                          shuffle=True, num_workers=args.threads)

    testloader = torch.utils.data.DataLoader(DatasetFromFolder(X_test,y_test,mean_X_train,std_X_train,mean_y_train,std_y_train,phase='val'), batch_size=batch_size,shuffle=False, num_workers=args.threads)

    # Models construction
    print('----------------------------------------------------------------------------------------------------')
    print('2/ Constructing the models')
    print('----------------------------------------------------------------------------------------------------')
    
    features = args.nb_features
    main_task = OneHiddenLayerNet(1, int(features), 1).cuda()
    feature_extractor_SLURP = OneHiddenLayerNet(D_in = 1, H = int(features), D_out = 1, is_extractor=True).cuda()
    feature_sizes = [features] *  1
    linear_SLURP = UncertaintyEstimator(feature_sizes, feature_sizes).cuda()
    linear_SLURP.apply(weights_init_kaiming)
    parameters = [{"params": main_task.parameters(), "lr": args.lr_main, "weight_decay": args.wd_main},
                {"params": feature_extractor_SLURP.parameters(), "lr": args.lr_main, "weight_decay": args.wd_main},
                {"params": linear_SLURP.parameters(), "lr": args.lr_uncer, "weight_decay": args.wd_uncer}]

    optimizer = torch.optim.Adam(parameters)

    # Training
    print('----------------------------------------------------------------------------------------------------')
    print('3/ Training the models')
    print('----------------------------------------------------------------------------------------------------')
    main_task.train()
    feature_extractor_SLURP.train()
    linear_SLURP.train()

    is_best_loss_on_test = True
    Best_loss_on_test = 1e10
    is_best_mse_on_test = True
    Best_mse_on_test = 1e10

    # preparation for ploting the values during training
    loss_nll_train = 0
    loss_mse_train = 0
    plot_epoch = []
    plot_mse_train = []
    plot_nll_train = []
    plot_mse_test = []
    plot_nll_test = []

    for epoch in range(args.nb_epochs):
        plot_epoch.append(epoch)
        loss_mse_train = 0
        loss_nll_train = 0
        # training
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            x, y = data
            x, y = x.cuda(), y.cuda()

            y_hat = main_task(x)
            y_hat_for_var = y_hat.detach().clone()
            with torch.no_grad():
                _,  h_x_for_var = main_task(x, True)
                h_x_for_var = h_x_for_var.detach().clone()

            h_y_hat = feature_extractor_SLURP(y_hat_for_var)
            log_var = linear_SLURP(h_x_for_var, h_y_hat)
            var = torch.exp(log_var + 1e-6)
            log_var = torch.log(var)

            optimizer.zero_grad()

            loss_nll = ((y_hat_for_var - y)**2/var + log_var).sum()
            loss_nll_train += loss_nll.detach()
            loss_mse = ((y_hat - y)**2).sum()
            loss_mse_train += loss_mse.detach()
            loss = loss_nll + loss_mse
            loss.backward()
            optimizer.step()

        loss_mse_train /= len(X_train)
        loss_nll_train /= len(X_train)
        plot_mse_train.append(loss_mse_train.item())
        plot_nll_train.append(loss_nll_train.item())

        loss_nll_test = 0
        loss_mse_test = 0
        main_task.eval()
        feature_extractor_SLURP.eval()
        linear_SLURP.eval()

        # validation
        with torch.no_grad():
            for data in testloader:
                x, y = data
                x, y = x.cuda(), y.cuda()
                y_hat, h_x = main_task(x, need_features=True)
                h_y_hat = feature_extractor_SLURP(y_hat)
                log_var = linear_SLURP(h_x, h_y_hat)
                var = torch.exp(log_var + 1e-6)
                log_var = torch.log(var)
                loss_nll_test += ((y_hat - y)**2/var + log_var).sum()
                loss_mse_test += ((y_hat - y)**2).sum()
            
            loss_nll_test = loss_nll_test/len(X_test)
            loss_mse_test = loss_mse_test/len(X_test)
            plot_nll_test.append(loss_nll_test.item())
            plot_mse_test.append(loss_mse_test.item())
        
            is_best_loss_on_test =  (loss_nll_test < Best_loss_on_test)
            is_best_mse_on_test =  (loss_mse_test < Best_mse_on_test)
            if is_best_loss_on_test:
                Best_loss_on_test = loss_nll_test
                torch.save(main_task, os.path.join(save_path, 'nll_main_task'))
                torch.save(feature_extractor_SLURP, os.path.join(save_path, 'nll_feature_extractor_SLURP'))
                torch.save(linear_SLURP, os.path.join(save_path, 'nll_linear_SLURP'))
                filename  = open(os.path.join(save_path, 'training_record.txt'), "a")
                filename.write('Best nll loss!! Epoch: {0}; MSE Loss: {1}; NLL Loss: {2} \n'.format(epoch, loss_mse_test.item(), loss_nll_test.item()))
                filename.close()
                print('When NLL is the best\non train set -- NLL Loss: {0:.4}; MSE Loss: {1:.4}.\non test set -- NLL Loss: {2:.4}; MSE Loss: {3:.4} '
                .format(loss_nll_train, loss_mse_train, loss_nll_test, loss_mse_test))
            if is_best_mse_on_test:
                Best_mse_on_test = loss_mse_test
                torch.save(main_task, os.path.join(save_path, 'mse_main_task'))
                torch.save(feature_extractor_SLURP, os.path.join(save_path, 'mse_feature_extractor_SLURP'))
                torch.save(linear_SLURP, os.path.join(save_path, 'mse_linear_SLURP'))
                filename  = open(os.path.join(save_path, 'training_record.txt'), "a")
                filename.write('Best mse loss!! Epoch: {0}; MSE Loss: {1}; NLL Loss: {2} \n'.format(epoch, loss_mse_test.item(), loss_nll_test.item()))
                filename.close()
                print('When MSE is the best\non train set -- NLL Loss: {0:.4}; MSE Loss: {1:.4}.\non test set -- NLL Loss: {2:.4}; MSE Loss: {3:.4} '
                .format(loss_nll_train, loss_mse_train, loss_nll_test, loss_mse_test))
        main_task.train()
        feature_extractor_SLURP.train()
        linear_SLURP.train()

    print('----------------------------------------------------------------------------------------------------')
    print('4/ Drawing the figures')
    print('----------------------------------------------------------------------------------------------------')
    
    single_draw_figs(save_path, 'nll', testloader_plot, std_y_train, mean_y_train, x_plot, y_plot)
    single_draw_figs(save_path, 'mse', testloader_plot, std_y_train, mean_y_train, x_plot, y_plot)
    single_draw_train_figs(plot_epoch, plot_mse_train, plot_mse_test, plot_nll_train, plot_nll_test, save_path)

    print('----------------------------------------------------------------------------------------------------')
    print('Done.')
    print('----------------------------------------------------------------------------------------------------')
    
if __name__ == '__main__':
    main()