import torch.utils.data as data
import numpy as np
import torch


class DatasetFromFolder(data.Dataset):

    def __init__(self, data_feature, data_target,mean_X_train=0,std_X_train=0,mean_y_train=0,std_y_train=0,phase='train'):
        self.data_feature = data_feature
        self.data_target = data_target
        self.mean_X_train = mean_X_train
        self.std_X_train = std_X_train
        self.mean_y_train = mean_y_train
        self.std_y_train = std_y_train
        self.transformed_feature = self.transforms_feature()
        self.transformed_target = self.transforms_target()
        self.phase=phase

    def __len__(self):
        return len(self.data_feature)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        if self.phase=='train':
            data_feature_tmp = torch.from_numpy(self.transformed_feature[index]).float()
            data_target_tmp =  torch.from_numpy(self.transformed_target[index]).float()
        else:
            data_feature_tmp = torch.from_numpy(self.transformed_feature[index]).float()
            data_target_tmp = torch.from_numpy(self.data_target[index]).float()



        return data_feature_tmp,data_target_tmp


    def transforms_feature(self ):
        return (self.data_feature - np.full(self.data_feature.shape, self.mean_X_train)) /  np.full(self.data_feature.shape, self.std_X_train)


    def transforms_target(self ):
        y_train_normalized = (self.data_target  - self.mean_y_train) / self.std_y_train
        return np.array(y_train_normalized, ndmin = 2)#.T

