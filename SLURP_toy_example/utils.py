import torch
from torch import Tensor
import os
import numpy as np
import matplotlib.pyplot as plt

def single_draw_figs(path, name, testloader_plot, std_y_train, mean_y_train, x_plot, y_plot):
    feature_extractor_Maintask = torch.load(os.path.join(path, name+'_main_task'))
    feature_extractor_SLURP = torch.load(os.path.join(path, name+'_feature_extractor_SLURP'))
    linear_SLURP = torch.load(os.path.join(path, name+'_linear_SLURP'))
    feature_extractor_Maintask.eval()
    feature_extractor_SLURP.eval()
    linear_SLURP.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader_plot, 0):
            inputs, target = data
            inputs, target = inputs.cuda(), target.cuda()
            y_pred, x_features = feature_extractor_Maintask(inputs, True)
            error = (y_pred - target)**2
            if len(x_features) == 1: x_features = x_features[0].detach().clone()
            else: x_features = x_features.detach().clone()
            y_pred_features = feature_extractor_SLURP(y_pred.detach().clone())
            y_uncer = linear_SLURP(x_features, y_pred_features)
            y_uncer = torch.sqrt(torch.exp(y_uncer))
            y_pred = y_pred * std_y_train + mean_y_train
            y_uncer = y_uncer * std_y_train + mean_y_train
            if batch_idx ==0:
                output_concat = y_pred.clone()
                uncer_concat = y_uncer.clone()
                error_concat = error.clone()
            else:
                output_concat=torch.cat((output_concat, y_pred), 0)
                uncer_concat=torch.cat((uncer_concat, y_uncer), 0)
                error_concat=torch.cat((error_concat, error), 0)

    y_preds=output_concat.clone().cpu().data.numpy()
    y_uncers=uncer_concat.clone().cpu().data.numpy()
    errors = np.round(error_concat.clone().cpu().data.numpy().mean(), 3)
    
    x_1=y_preds - np.abs(y_uncers)
    x_2=y_preds + np.abs(y_uncers)

    x_3=y_preds - 2*np.abs(y_uncers)
    x_4=y_preds + 2*np.abs(y_uncers)

    x_lin = np.linspace(-10, 10, 400)
    plt.figure(figsize=(8, 6))
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.plot(np.squeeze(x_plot), y_plot, alpha=0.5)
    plt.plot(x_lin, y_preds, alpha=0.5)
    plt.title('SLURP training - '+ name + ' MSE: ' + str(errors))
    plt.fill_between(np.squeeze(x_plot), np.squeeze(x_1), np.squeeze(x_2), color = 'orange', alpha = 0.4, label = 'Aleatoric')
    plt.fill_between(np.squeeze(x_plot), np.squeeze(x_3), np.squeeze(x_4), color = 'orange', alpha = 0.4, label = 'Aleatoric')
    plt.savefig(os.path.join(path, name+'.png'))


def single_draw_figs_main(path, name, testloader_plot, std_y_train, mean_y_train, x_plot, y_plot):
    feature_extractor_Maintask = torch.load(os.path.join(path, name+'_main_task'))
    feature_extractor_Maintask.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader_plot, 0):
            inputs, target = data
            inputs, target = inputs.cuda(), target.cuda()
            y_pred = feature_extractor_Maintask(inputs)
            y_pred = y_pred * std_y_train + mean_y_train
            if batch_idx ==0:
                output_concat = y_pred.clone()
            else:
                output_concat=torch.cat((output_concat, y_pred), 0)

    y_preds=output_concat.clone().cpu().data.numpy()

    x_lin = np.linspace(-10, 10, 400)
    plt.figure(figsize=(8, 6))
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.plot(np.squeeze(x_plot), y_plot, alpha=0.5)
    plt.plot(x_lin, y_preds, alpha=0.5)
    plt.title('SLURP training - '+ name)
    plt.savefig(os.path.join(path, name+'.png'))

def single_draw_train_figs(plot_epoch, plot_mse_train, plot_mse_test, plot_nll_train, plot_nll_test, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.squeeze(plot_epoch[5:]), plot_mse_train[5:], alpha=0.5, label='MSE train')
    plt.plot(np.squeeze(plot_epoch[5:]), plot_mse_test[5:], alpha=0.5, label='MSE test')
    plt.plot(np.squeeze(plot_epoch[5:]), plot_nll_train[5:], alpha=0.5, label='NLL train')
    plt.plot(np.squeeze(plot_epoch[5:]), plot_nll_test[5:], alpha=0.5, label='NLL test')
    plt.legend()
    plt.title('SLURP info - During training')
    plt.savefig(os.path.join(save_path, 'record.png'))

def tv_loss_1d(c):
    x = c[1:,:] - c[:-1,:]
    loss = torch.sum(torch.abs(x))/x.size()[0]
    return loss