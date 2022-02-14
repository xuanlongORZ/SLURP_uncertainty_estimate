import numpy as np
from sklearn.metrics import roc_auc_score

"""Calculate the sparsification error.
    
    I simplified the code in https://github.com/fregu856/evaluating_bdl/tree/master.

    Calcualte the sparsification error for a given array according to a reference array.

    Args:
        unc_npy: Flatten estimated uncertainty numpy array. 
        err_npy: Flatten ground truth error numpy array (abs error). 
        nb_bins: Number of bins using for uncertainty estimation. Each time, 1/nb_bins * 100% items with highest value will be removed.
        return_hist: if return histograms for drawing the sparsification curve, otherwise, directly return the sum of sparsification error.
        gt_npy: if calculate the AUSE accroding to Absrel or RMSE. By default, None, which indicates we calculate AUSE_RMSE
        is_rmse: True if we calculate AUSE_RMSE. 
    Returns:
        By default, sum of the sparsification error after removing all the items in two given vectors given nb_bins.
        Given return_hist = True, three arrays corresponding to the components of sparsification curve.

"""
def sparsification_error(unc_npy, err_npy, nb_bins = 20, return_hist=False, is_epe = True):
    hist_pred = []
    hist_oracle = []
    nb_remain = []
    # Get the index of the sorted vector
    # From small to big
    argsorted_U = np.argsort(unc_npy)
    argsorted_E = np.argsort(err_npy)

    total_len = len(unc_npy)

    # Each time we calculate the remaining errors of the true error array,
    # according to the real order index and predicted uncertainty order index respectively.
    sigma_pred_curve = []
    error_curve = []
    # fractions = list(np.arange(start=0.0, stop=1.0 - 1/nb_bins, step=(1/nb_bins)))
    fractions = list(np.arange(start=0.0, stop=1.0, step=(1/nb_bins)))
    for fraction in fractions:
        # rmse (2ch uncertainty output) or epe (1ch uncertainty output)
        # in the paper we used 1ch output
        if is_epe:
            sigma_pred_point = np.mean( err_npy[argsorted_U[0:int((1.0-fraction)*total_len)]])
            error_point = np.mean( err_npy[argsorted_E[0:int((1.0-fraction)*total_len)]])
        else:
            sigma_pred_point = np.mean( err_npy[argsorted_U[0:int((1.0-fraction)*total_len)]]**2 )
            error_point = np.mean( err_npy[argsorted_E[0:int((1.0-fraction)*total_len)]]**2 )
            sigma_pred_point = np.sqrt(sigma_pred_point)
            error_point = np.sqrt(error_point)

        sigma_pred_curve.append(sigma_pred_point)
        error_curve.append(error_point)
        nb_remain.append(int((1.0-fraction)*total_len))
    
    # as a note, this kind of normalization cannot make sure that the curves stay below 1 all the time.
    hist_oracle = np.array(error_curve)/error_curve[0]
    hist_pred = np.array(sigma_pred_curve)/sigma_pred_curve[0]
    nb_remain = np.array(nb_remain)
    
    sparsification_errors_pred = abs(hist_pred - hist_oracle).sum()

    if return_hist:
        return hist_pred, hist_oracle, nb_remain, sparsification_errors_pred
    else:
        return sparsification_errors_pred


"""Calculate the AUROC.

    Args:
        unc_npy: Flatten estimated uncertainty numpy array. 
        gt_npy: Flatten ground truth numpy array. 
        pred_npy: Flatten predcition numpy array. 
        is_depth: True if it is a depth task (False if it is optical flow task)
    Returns:
        From time to time, it will have only one label after labeling. 
        Thus, we have two outputs, one is for roc_auc, another one is for counting the valid roc_auc
        The mean of the roc_auc will be averaged by taking only the valid cases.

"""
def AUROC(gt_npy, pred_npy, unc_npy, is_depth = True):
    if is_depth:
        thresh = np.maximum((gt_npy / pred_npy), (pred_npy / gt_npy))
        label = thresh < 1.25
    else:
        epe = (pred_npy - gt_npy)**2
        epe = np.sqrt(epe[:,:,0] + epe[:,:,1])
        label = epe < 2
    unc = (unc_npy - unc_npy.min())/(unc_npy.max() - unc_npy.min())
    unc = 1 - unc
    try:
        roc_auc = roc_auc_score(label, unc)
    except ValueError:
        print('ValueError')
        return 0, 0
    return roc_auc, 1