# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This file contains code to train dropout networks on the UCI datasets using the following algorithm:
# 1. Create 20 random splits of the training-test dataset.
# 2. For each split:
# 3.   Create a validation (val) set taking 20% of the training set.
# 4.   Get best hyperparameters: dropout_rate and tau by training on (train-val) set and testing on val set.
# 5.   Train a network on the entire training set with the best pair of hyperparameters.
# 6.   Get the performance (MC RMSE and log-likelihood) on the test set.
# 7. Report the averaged performance (Monte Carlo RMSE and log-likelihood) on all 20 splits.

import numpy as np
import GPy
import matplotlib.pyplot as plt
import h5py
import argparse

# Dataset settings
parser = argparse.ArgumentParser(description='parameters for generating data from Gaussian process.')
parser.add_argument('--no_points', type = int, default = 1100, help='nb of training data')
parser.add_argument('--no_points_tot', type = int, default = 1500, help='nb of total data')
parser.add_argument('--lengthscale', type = float, default = 1.6, help='lengthscale')
parser.add_argument('--variance', type = float, default = 3.0, help='variance')
parser.add_argument('--sig_noise', type = float, default = 0.3, help='sig_noise')
parser.add_argument('--seed', type = int, default = 2, help='random seed')

print(' Building the training and test sets file')

def main():
     global args
     args = parser.parse_args()

     np.random.seed(args.seed)
     x1 = np.random.uniform(-7, 7, args.no_points)[:, None]
     x1.sort(axis=0)
     x_plot=np.linspace(-10, 10, num=400)[:, None]
     x_plot.sort(axis=0)
     x_tot=np.concatenate((x1, x_plot), axis=0)


     k = GPy.kern.RBF(input_dim=1, variance=args.variance, lengthscale=args.lengthscale)

     C = k.K(x_tot, x_tot) + np.eye(args.no_points_tot) * args.sig_noise ** 2


     y = np.random.multivariate_normal(np.zeros((args.no_points_tot)), C)[:, None]

     y = (y - y.mean())

     X_train=np.concatenate((x_tot[50:200], x_tot[225:350],x_tot[375:550], x_tot[600:725],x_tot[750:1050]), axis=0)
     y_train=np.concatenate((y[50:200],  y[225:350],y[375:550], y[600:725],y[750:1050]), axis=0)

     x_plot= x_tot[1100:1500]
     y_plot = y[1100:1500]

     X_test=np.concatenate((x_tot[0:50], x_tot[200:225], x_tot[350:375],x_tot[550:600],x_tot[725:750], x_tot[1050:1100]), axis=0)
     y_test=np.concatenate((y[0:50], y[200:225], y[350:375],y[550:600],y[725:750], y[1050:1100]), axis=0)

     print ("DATA Done.")


     c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


     plt.figure(figsize = (6, 5))
     plt.style.use('default')
     plt.scatter(X_train, y_train, s = 50, marker = 'x', color = 'blue', alpha = 0.5)
     plt.scatter(X_test, y_test, s = 50, marker = 'x', color = 'green', alpha = 0.5)
     #plt.fill_between(np.linspace(-5, 5, 200), np.squeeze(x_2), np.squeeze(x_2_1), color = c[0], alpha = 0.3, label = 'Epistemic + Aleatoric')
     #plt.fill_between(np.linspace(-5, 5, 200),np.squeeze(x_1_1), np.squeeze(x_1), color = c[0], alpha = 0.3)
     #plt.fill_between(np.linspace(-5, 5, 200), np.squeeze(x_1), np.squeeze(x_2), color = c[1], alpha = 0.4, label = 'Aleatoric')
     plt.plot(np.squeeze(x_plot), np.squeeze(y_plot), color = 'black', linewidth = 1)
     #plt.plot(x_plot, standard_pred_cpu, color = 'red', linewidth = 1)
     plt.xlim([-10, 10])
     plt.ylim([-5, 7])
     plt.xlabel('$x$', fontsize=15)
     plt.ylabel('$y$', fontsize=15)
     plt.title('SIMPLE DATA', fontsize=20)
     plt.tick_params(labelsize=15)
     plt.xticks(np.arange(-10, 10,3))
     plt.yticks(np.arange(-4, 7, 2))
     plt.gca().set_yticklabels([])
     plt.gca().yaxis.grid(alpha=0.3)
     plt.gca().xaxis.grid(alpha=0.3)
     plt.savefig('simple_data.png', bbox_inches = 'tight')

     h5f = h5py.File( 'dataGP.hdf5', 'w')
     h5f.create_dataset('X_train', data=X_train)
     h5f.create_dataset('y_train', data=y_train)
     h5f.create_dataset('X_test', data=X_test)
     h5f.create_dataset('y_test', data=y_test)
     h5f.create_dataset('x_plot', data=x_plot)
     h5f.create_dataset('y_plot', data=y_plot)
     h5f.close()

if __name__ == '__main__':
     main()