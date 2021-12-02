import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHiddenLayerNet(torch.nn.Module):
    """
    Here I code the network that the author of DE used for a simple task
    """
    """
    """
    def __init__(self, D_in, H, D_out = 1, p=0.3, is_extractor = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(OneHiddenLayerNet, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, H)
        self.is_extractor = is_extractor
        if not is_extractor:
            self.fc2 = torch.nn.Linear(H, D_out)
            torch.nn.init.kaiming_normal_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.fc2.bias,0)
        self.p = p
        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias,0)

    def forward(self, x, need_features = False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #h1 = F.dropout(x, p=self.p, training=self.training)
        hidden1 = F.relu(self.fc1(x))
        if not self.is_extractor:
            h_drop = F.dropout(hidden1, p=self.p, training=self.training)
            output_mu = self.fc2(h_drop)
            if need_features:
                return output_mu, hidden1
            else:
                return output_mu
        else:
            return hidden1

# # # # Uncertatinty Estimator # # # # 
def block_LinearBR(in_planes, out_planes, bias=False):
        return nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=bias),
            nn.BatchNorm1d(out_planes),
            nn.ReLU())

class FeatureFusionModule(nn.Module):
    def __init__(self, C1, C2, C_out):
        super(FeatureFusionModule, self).__init__()
        self.linear_rgb = block_LinearBR(C2, C_out)
        self.linear_pred = block_LinearBR(C1, C_out)
    def forward(self, pred, rgb):
        if pred.size()[1:] != rgb.size()[1:]:
            downsample = nn.Upsample(rgb.size()[1:])
            pred = downsample(pred)
        rgb = self.linear_rgb(rgb)
        pred = self.linear_pred(pred)
        return torch.cat([rgb, pred], dim=1)

class RefinementModule(nn.Module):
    def __init__(self, size0):
        super(RefinementModule, self).__init__()
        self.est = nn.Sequential(
            block_LinearBR(size0, 128),
        block_LinearBR(128, 64),
        block_LinearBR(64, 16))
        self.last = nn.Linear(16, 1, bias=True)
    def forward(self, x_y):
        feature = F.dropout(self.est(x_y), 0.4, training=self.training)
        return self.last(feature)


class FinalFusetModule(nn.Module):
    def __init__(self, size):
        super(FinalFusetModule, self).__init__()
        self.final = nn.Linear(size, 1, bias=True)
    def forward(self, x_y):
        return self.final(x_y)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
class UncertaintyEstimator(nn.Module):
    def __init__(self, pred_ch, rgb_ch):
        super(UncertaintyEstimator, self).__init__()

        self.FeatureFusionModules = nn.ModuleList(
            [FeatureFusionModule(pred_ch[i], rgb_ch[i], (pred_ch[i]+rgb_ch[i])//16) for i in range(len(pred_ch))])

        self.RefinementModules_1 = RefinementModule((pred_ch[0]+rgb_ch[0])//8)
        self.RefinementModules = nn.ModuleList(
            [RefinementModule((pred_ch[i+1]+rgb_ch[i+1]) + 1) for i in range(len(pred_ch)-1)])
        self.nb_ft = len(pred_ch)
        if len(pred_ch) > 1:
            self.final = FinalFusetModule(size = len(pred_ch), islast = False)

    def forward(self, xs, ys):
        fts = []
        if not isinstance(xs, list):
            xs = [xs]
            ys = [ys]
        for fuse_module, x, y  in zip(self.FeatureFusionModules, xs, ys):
            x = F.dropout(x, 0.4, training=self.training)
            y = F.dropout(y, 0.4, training=self.training)
            fts.append(fuse_module(x, y))

        fts[0] = self.RefinementModules_1(fts[0])
        for i, refine_module in enumerate(self.RefinementModules):
            fts[i+1] = refine_module(torch.cat([fts[i], fts[i+1]], dim=1))
        if self.nb_ft > 1:
            final = self.final(torch.cat(fts, dim=1), islast = False)
            return final
        else:
            return fts[0]

