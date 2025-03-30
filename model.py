import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn1_enc = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn2_enc = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn3_enc = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn4_enc = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1_dec = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2_dec = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3_dec = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4_dec = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.bn1_enc(self.relu(self.conv1(x)))
        x2 = self.bn2_enc(self.relu(self.conv2(x1)))
        x3 = self.bn3_enc(self.relu(self.conv3(x2)))
        x4 = self.bn4_enc(self.relu(self.conv4(x3)))

        # Decoder
        y1 = self.bn1_dec(self.relu(self.deconv1(x4)))
        y2 = self.bn2_dec(self.relu(self.deconv2(y1)))
        y3 = self.bn3_dec(self.relu(self.deconv3(y2)))
        y4 = self.bn4_dec(self.relu(self.deconv4(y3)))

        # Classifier
        score = self.classifier(y4)
        return score