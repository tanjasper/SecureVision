import torch.nn as nn

class MaskAlexNet(nn.Module):
    # input size is 255
    def __init__(self, in_channels=1, num_classes=2, init_weights=True):
        super(MaskAlexNet, self).__init__()
        self.optics = nn.Conv2d(in_channels, 1, kernel_size=128, padding=48)  # input size of 255, output size of 224
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
