
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Residual connection
        out = self.relu(out)
        return out


class FGENN(nn.Module):
    def __init__(self, input_channels=22, filter_sizes=None,channel_size=32, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [channel_size,  # First set of convs
                            channel_size*2,  # Second set of convs
                            channel_size*4,  # Dilated convs
                            channel_size*2]  # Transpose Convs

        if dilations is None:
            dilations = [2, 2]

        # 2 conv layers.
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        # 1 max pooling layers.
        self.maxp2d1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2 residual blocks.
        self.res1 = ResidualBlock(filter_sizes[1], filter_sizes[1], kernel_size=l3_k_size,stride=1, padding=2)
        self.res2 = ResidualBlock(filter_sizes[1], filter_sizes[1],kernel_size=l3_k_size,stride=1, padding=2)

        # 1 max pooling layers.
        self.maxp2d2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2 residual blocks.    
        self.res3 = ResidualBlock(filter_sizes[1], filter_sizes[1], kernel_size=l3_k_size,stride=1, padding=2)
        self.res4 = ResidualBlock(filter_sizes[1], filter_sizes[1], kernel_size=l3_k_size,stride=1, padding=2)

        # # 2 layers of UpsamplingBilinear2d.
        # self.conv3 = nn.ConvTranspose2d(filter_sizes[1], filter_sizes[2], kernel_size=4, stride=2, padding=1, output_padding=1)
        # self.bn1 = nn.BatchNorm2d(filter_sizes[2])
        # self.conv4 = nn.ConvTranspose2d(filter_sizes[2],filter_sizes[1], kernel_size=4, stride=2, padding=2, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(filter_sizes[1])

        # 2 layers of Dilated convolutions.
        self.conv5 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        # 2 layers of UpsamplingBilinear2d.
        self.ub2d1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res5 = ResidualBlock(filter_sizes[3], filter_sizes[3], kernel_size=3,stride=1, padding=1)
        self.ub2d2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res6 = ResidualBlock(filter_sizes[3], filter_sizes[3], kernel_size=3,stride=1, padding=1)

        # 4 layers of output layers.
        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
      
        x = self.maxp2d1(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.maxp2d2(x)
 
        x = self.res3(x)
        x = self.res4(x)

        # Dilated UpsamplingBilinear2d layers.
        # x = self.bn1(self.conv3(x))
        # print('conv3',x.shape)
        # x = self.bn2(self.conv4(x))
        # print('conv4',x.shape)

        # Dilated convolutions.
        x = self.relu3(self.conv5(x))
        x = self.relu4(self.conv6(x))

        x = self.ub2d1(x)
        x = self.res5(x)
        x = self.ub2d2(x)
        x = self.res6(x)
       
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        '''
        The key advantage of the smooth_l1_loss is 
        its robustness to outliers. Unlike the MSE loss, 
        which can be heavily influenced by large errors 
        (outliers), the smooth L1 loss assigns a smaller 
        penalty to large errors, making it less sensitive 
        to extreme data points.
        '''
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }