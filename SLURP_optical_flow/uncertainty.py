import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

''' encoder class is from bts https://github.com/cogaplex-bts/bts/tree/master/pytorch
    We changed its input parameter to make it adaptable to more general cases.
'''
class encoder(nn.Module):
    def __init__(self, encoder_name):
        super(encoder, self).__init__()
        self.encoder_name = encoder_name
        import torchvision.models as models
        if isinstance(encoder_name, str):
            if 'densenet121' in encoder_name:
                self.base_model = models.densenet121(pretrained=True).features
                self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
                self.feat_out_channels = [64, 64, 128, 256, 1024]
            elif 'densenet161' in encoder_name:
                self.base_model = models.densenet161(pretrained=True).features
                self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
                self.feat_out_channels = [96, 96, 192, 384, 2208]
            elif 'resnet34' in encoder_name:
                self.base_model = models.resnet34(pretrained=True)
                self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
                self.feat_out_channels = [64, 64, 128, 256, 512]
            elif 'resnet50' in encoder_name:
                self.base_model = models.resnet50(pretrained=True)
                self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
                self.feat_out_channels = [64, 256, 512, 1024, 2048]
            elif 'resnet101' in encoder_name:
                self.base_model = models.resnet101(pretrained=True)
                self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
                self.feat_out_channels = [64, 256, 512, 1024, 2048]
            elif 'resnext50' in encoder_name:
                self.base_model = models.resnext50_32x4d(pretrained=True)
                self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
                self.feat_out_channels = [64, 256, 512, 1024, 2048]
            elif 'resnext101' in encoder_name:
                self.base_model = models.resnext101_32x8d(pretrained=True)
                self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
                self.feat_out_channels = [64, 256, 512, 1024, 2048]
            elif 'mobilenetv2' in encoder_name:
                self.base_model = models.mobilenet_v2(pretrained=True).features
                self.feat_inds = [2, 4, 7, 11, 19]
                self.feat_out_channels = [16, 24, 32, 64, 1280]
                self.feat_names = []
        elif isinstance(encoder_name, dict):
            # input should be a list if it's not one of the backbones listed above
            self.base_model = encoder_name['base_model']
            self.feat_out_channels = encoder_name['feat_out_channels']
        else:
            print('Not supported encoder: {}'.format(encoder_name))

    def forward(self, x):
        if isinstance(self.encoder_name, dict):
            skip_feat = self.base_model(x)
        else:
            feature = x
            skip_feat = []
            i = 1
            for k, v in self.base_model._modules.items():
                if 'fc' in k or 'avgpool' in k:
                    continue
                feature = v(feature)
                if 'mobilenetv2' in self.encoder_name:
                    if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                        skip_feat.append(feature)
                else:
                    if any(x in k for x in self.feat_names):
                        skip_feat.append(feature)
                i = i + 1
        return skip_feat

def rgb_encoder(rgb_encoder):
    if 'densenet121' in rgb_encoder:
        return [64, 64, 128, 256, 1024]
    elif 'densenet161' in rgb_encoder:
        return [96, 96, 192, 384, 2208]
        # return [96, 192, 384, 1056, 2208]
    elif 'resnet34' in rgb_encoder:
        return [64, 64, 128, 256, 512]
    elif 'resnet50' in rgb_encoder:
        return [64, 256, 512, 1024, 2048]
    elif 'resnet101' in rgb_encoder:
        return [64, 256, 512, 1024, 2048]
    elif 'resnext50' in rgb_encoder:
        return [64, 256, 512, 1024, 2048]
    elif 'resnext101' in rgb_encoder:
        return [64, 256, 512, 1024, 2048]
    elif 'mobilenetv2' in rgb_encoder:
        return [16, 24, 32, 64, 1280]
    # if the input is not a backbone name but a list of channel numbers
    # the special case if the main task uses a self-made feature extractor
    elif (isinstance(rgb_encoder, list)): 
        return rgb_encoder 
    else:
        print('Not supported encoder: {}'.format(rgb_encoder))
        return 0

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Pre_encoder(nn.Module):
    def __init__(self):
        super(Pre_encoder,self).__init__()
        self.to_3ch = nn.Conv2d(2, 3, 1)
    def forward(self, x):
        return self.to_3ch(x)


class UncerModel(nn.Module):
    def __init__(self, args):
        super(UncerModel, self).__init__()
        self.encoder_pred = encoder(args.encoder_U)

        if args.with_pre:
            self.pre_encoder = Pre_encoder()
            self.pre_encoder.apply(weights_init_kaiming)
        if args.with_en_rgb:
            self.encoder_rgb = encoder(args.encoder_U)
            self.decoder = UncertaintyEstimator(self.encoder_pred.feat_out_channels, self.encoder_rgb.feat_out_channels, args.uncer_ch)
        else:
            self.decoder = UncertaintyEstimator(self.encoder_pred.feat_out_channels, rgb_encoder(args.encoder_U), args.uncer_ch)
            
        self.decoder.apply(weights_init_kaiming)

    def forward(self, input_pred, input_rgb, size, args):
        if args.with_en_rgb:
            input_rgb = self.encoder_rgb(input_rgb)
        if args.with_pre:
            input_pred = self.pre_encoder(input_pred)
        final_depth_feat = self.encoder_pred(input_pred)
        output = self.decoder(final_depth_feat, input_rgb, size)
        return output

# loss
class logitbce_loss(nn.Module):
    def __init__(self):
        super(logitbce_loss, self).__init__()

    def forward(self, pred_uncer, target):
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(pred_uncer, target)
        return loss
    
def block_ConvBR(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, 
            stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

class FeatureFusionModule(nn.Module):
    def __init__(self, C1, C2, C_out):
        super(FeatureFusionModule, self).__init__()
        self.conv_pred = block_ConvBR(C1, C_out, 3, 1, 1)
        self.conv_rgb = block_ConvBR(C2, C_out, 3, 1, 1)
    def forward(self, pred, rgb):
        if pred.size()[2:] != rgb.size()[2:]:
            downsample = nn.UpsamplingBilinear2d(rgb.size()[2:])
            pred = downsample(pred)
        pred = self.conv_pred(F.dropout(pred, 0.4, training=self.training))        
        rgb = self.conv_rgb(F.dropout(rgb, 0.4, training=self.training))
        return torch.cat([rgb, pred], dim=1)


class RefinementModule(nn.Module):
    def __init__(self, C_in, C_out):
        super(RefinementModule, self).__init__()
        self.est = nn.Sequential(
            block_ConvBR(C_in, 128,  dilation=1), 
            block_ConvBR(128, 128, dilation=2), 
            block_ConvBR(128, 96, dilation=4), 
            block_ConvBR(96, 32, dilation=1))
        self.last = nn.Conv2d(32, C_out, 3, 1, 1, bias=True)
    def forward(self, x_y):
        feature = F.dropout(self.est(x_y), 0.2, training=self.training)
        return self.last(feature)

class FinalFuseModule(nn.Module):
    def __init__(self, C_in, C_out):
        super(FinalFuseModule, self).__init__()
        self.final = nn.Conv2d(C_in, C_out, 1, bias=True)
    def forward(self, x_y):
        return self.final(x_y)

class UncertaintyEstimator(nn.Module):
    def __init__(self, pred_ch, rgb_ch, uncer_ch = 1):
        super(UncertaintyEstimator, self).__init__()

        self.combine_5 = FeatureFusionModule(pred_ch[4], rgb_ch[4], (pred_ch[4]+rgb_ch[4])//16)
        self.combine_4 = FeatureFusionModule(pred_ch[3], rgb_ch[3], (pred_ch[3]+rgb_ch[3])//16)
        self.combine_3 = FeatureFusionModule(pred_ch[2], rgb_ch[2], (pred_ch[2]+rgb_ch[2])//16)
        self.combine_2 = FeatureFusionModule(pred_ch[1], rgb_ch[1], (pred_ch[1]+rgb_ch[1])//16)
        self.combine_1 = FeatureFusionModule(pred_ch[0], rgb_ch[0], (pred_ch[0]+rgb_ch[0])//16)

        self.fusion5 = RefinementModule((pred_ch[4]+rgb_ch[4])//8, uncer_ch)
        self.fusion4 = RefinementModule((pred_ch[3]+rgb_ch[3])//8 + uncer_ch, uncer_ch)
        self.fusion3 = RefinementModule((pred_ch[2]+rgb_ch[2])//8 + uncer_ch, uncer_ch)
        self.fusion2 = RefinementModule((pred_ch[1]+rgb_ch[1])//8 + uncer_ch, uncer_ch)
        self.fusion1 = RefinementModule((pred_ch[0]+rgb_ch[0])//8 + uncer_ch, uncer_ch)

        self.final = FinalFuseModule(uncer_ch*5, uncer_ch)

    def forward(self, x, y, size):
        ft_5= self.combine_5(x[4], y[4])
        ft_5 = self.fusion5(ft_5)

        ft_4 = self.combine_4(x[3], y[3])
        upsample = nn.UpsamplingBilinear2d(ft_4.size()[2:])
        ft_5 = upsample(ft_5)
        ft_4 = self.fusion4(torch.cat([ft_4, ft_5], dim=1))

        ft_3= self.combine_3(x[2], y[2])
        upsample = nn.UpsamplingBilinear2d(ft_3.size()[2:])
        ft_4 = upsample(ft_4)
        ft_3 = self.fusion3(torch.cat([ft_3, ft_4], dim=1))

        ft_2 = self.combine_2(x[1], y[1])
        upsample = nn.UpsamplingBilinear2d(ft_2.size()[2:])
        ft_3 = upsample(ft_3)
        ft_2 = self.fusion2(torch.cat([ft_2, ft_3], dim=1))

        ft_1 = self.combine_1(x[0], y[0])
        upsample = nn.UpsamplingBilinear2d(ft_1.size()[2:])
        ft_2 = upsample(ft_2)
        ft_1 = self.fusion1(torch.cat([ft_1, ft_2], dim=1))

        upsample = nn.UpsamplingBilinear2d(size)
        ft_1 = upsample(ft_1)
        ft_2 = upsample(ft_2)
        ft_3 = upsample(ft_3)
        ft_4 = upsample(ft_4)
        ft_5 = upsample(ft_5)
        
        final = self.final(torch.cat([ft_1, ft_2, ft_3, ft_4, ft_5], dim=1))

        return final


def get_scheduler(optimizer, arg):
    if arg.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - arg.niter) / float(arg.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif arg.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=arg.lr_decay_iters, gamma=0.1)
    elif arg.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, threshold=0.01, patience=20)
    elif arg.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.niter, eta_min=0)
    elif arg.lr_policy == 'milestones':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=arg.milestones, gamma=0.5)
    elif arg.lr_policy == None:
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', arg.lr_policy)
    return scheduler

'''with using plateau, lr should be updated only after obtaining new score on validation set,
so this bloc should be put after validation.
   with using milestones, (generally) this bloc should be put after every epoch (depending on how you define the milestones)
   with using the other lr updating approaches, this bloc could be put after every iteration.
'''
def update_learning_rate(scheduler, optimizer, global_step, num_total_steps, arg, avg_epe=0):
    if arg.lr_policy == 'plateau':
        scheduler.step(avg_epe)
        current_lr = optimizer.param_groups[0]['lr']
    elif scheduler == None:
        for param_group in optimizer.param_groups:
            current_lr = (arg.start_lr - arg.end_lr) * (1 - global_step / num_total_steps) ** 0.8 + arg.end_lr
            param_group['lr'] = current_lr
    else:
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
    return current_lr