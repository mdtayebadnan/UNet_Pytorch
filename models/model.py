import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
     
        # Contracting path (encoder)
        self.conv1 = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features[0], features[0], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(features[0], features[1], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(features[1], features[1], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(features[1], features[2], kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(features[2], features[2], kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(features[2], features[3], kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(features[3], features[3], kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[3], features[3]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[3]*2, features[3]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Expanding path (decoder)
        self.upconv1 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(features[3]*2, features[3], kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(features[3], features[3], kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)
        
        self.upconv2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(features[3], features[2], kernel_size=3, padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(features[2], features[2], kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)


        self.upconv3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(features[2], features[1], kernel_size=3, padding=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(features[1], features[1], kernel_size=3, padding=1)
        self.relu14 = nn.ReLU(inplace=True)
        
        self.upconv4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(features[1], features[0], kernel_size=3, padding=1)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(features[0], features[0], kernel_size=3, padding=1)
        self.relu16 = nn.ReLU(inplace=True)
        
        self.conv17 = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Define forward pass through the encoder ...
        x1 = x
        x2 = self.relu2(self.conv2(self.relu1(self.conv1(x1))))
        x3 = self.relu4(self.conv4(self.relu3(self.conv3(self.pool1(x2)))))
        x4 = self.relu6(self.conv6(self.relu5(self.conv5(self.pool2(x3)))))
        x5 = self.relu8(self.conv8(self.relu7(self.conv7(self.pool3(x4)))))
      
        # Define forward pass through the bottleneck layer ...
        xn = self.bottleneck(self.pool4(x5))

        # Define forward pass through the decoder ...
        xup1 = self.upconv1(xn)
        xcat1 = torch.cat((xup1, x5), dim=1)
        xd1 = self.relu10(self.conv10(self.relu9(self.conv9(xcat1))))
        xup2 = self.upconv2(xd1)
        xcat2 = torch.cat((xup2, x4), dim=1)
        xd2 = self.relu12(self.conv12(self.relu11(self.conv11(xcat2))))
        xup3 = self.upconv3(xd2)
        xcat3 = torch.cat((xup3, x3), dim=1)
        xd3 = self.relu14(self.conv14(self.relu13(self.conv13(xcat3))))
        xup4 = self.upconv4(xd3)
        xcat4 = torch.cat((xup4, x2), dim =1)
        xd4 = self.relu16(self.conv16(self.relu15(self.conv15(xcat4))))
        output = self.conv17(xd4)

        return output
