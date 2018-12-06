import torch.nn as nn
import torch

class MaskAlexNet(nn.Module):
    # input size is 255
    def __init__(self, in_channels=1, num_classes=2, init_weights=True):
        super(MaskAlexNet, self).__init__()
        self.optics = nn.Conv2d(in_channels, 1, kernel_size=128, padding=0, bias=False)  # input size of 255, output size of 224
        self.classifier = alexnet_layers(in_channels, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.optics(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # initialize mask differently
        nn.init.uniform_(self.optics.weight, 0, 1)


# VGG stuff
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class MaskVGG16(nn.Module):

    def __init__(self, num_classes=2, init_weights=True, batch_norm=True, mask_size=128, mask_perc=0.5):
        super(MaskVGG16, self).__init__()
        self.optics = nn.Conv2d(1, 1, kernel_size=mask_size, padding=0, bias=False)
        self.features = make_layers(cfg['D'], batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights(mask_perc=mask_perc)

    def forward(self, x, opt):
        # break up channels of x
        batch_size = x.size(0)
        in_channels = x.size(1)
        x = x.view(-1, 1, x.size(2), x.size(3))
        # perform optics individually on each channel
        meas = self.optics(x)
        # put channels back together
        meas = meas.view(batch_size, in_channels, meas.size(2), meas.size(3))
        # normalize mask measurements
        if opt.normalize_feats:
            # 1: normalize to have max of 1
            # view is used to get max over the 3D image
            # unsqueeze is to get the dimensions correct for elementwise division
            meas = meas / meas.view(meas.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # 2: normalize [0, 1] to [-1, 1]
            mean = torch.Tensor([0.5, 0.5, 0.5]).cuda(opt.gpu)
            var = torch.Tensor([0.25, 0.25, 0.25]).cuda(opt.gpu)  # std squared
            meas = torch.batch_norm(meas, None, None, mean, var, False, 0, 0, torch.backends.cudnn.enabled)

        # forward through VGG net
        output = self.features(meas)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

    def _initialize_weights(self, mask_perc=0.5):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
            # initialize mask differently
            # nn.init.uniform_(self.optics.weight, 0, 1)  # uniform between 0 and 1
            with torch.no_grad():
                self.optics.weight.bernoulli_(mask_perc)  # bernoulli 0s and 1s


def alexnet_layers(in_channels=1, num_classes=2):
    layers = nn.Sequential(
        #  Feature extractor
        nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #  Classifier
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    return layers
