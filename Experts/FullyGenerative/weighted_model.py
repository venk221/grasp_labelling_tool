import torch.nn as nn
import torch
import torch.nn.functional as F


class FGWENN(nn.Module):
    def __init__(self, input_channels=4, expert_count=2, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        self.__input_channels = input_channels
        self.__expert_count = expert_count

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.image_features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1,
                      padding=(l3_k_size // 2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1,
                      padding=(l3_k_size // 2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Sequential(
            nn.Conv2d(filter_sizes[3], expert_count, kernel_size=1),
            nn.Softmax2d()
        )
        self.angle_output = nn.Sequential(
            nn.Conv2d(filter_sizes[3], expert_count, kernel_size=1),
            nn.Softmax2d()
        )
        self.width_output = nn.Sequential(
            nn.Conv2d(filter_sizes[3], expert_count, kernel_size=1),
            nn.Softmax2d()
        )

        # The masks for multiplication
        self.__p_mask = list(range(0, self.__expert_count * 3, 3))
        self.__a_mask = list(range(1, self.__expert_count * 3, 3))
        self.__w_mask = list(range(2, self.__expert_count * 3, 3))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

        print(f'Initialized FGWENN model with {self.__expert_count} experts and {self.__input_channels} input channels')

    def forward(self, x):
        # Compute the weights from the last 4 channels, which are the input RGBD image
        f = self.image_features(x[:, -self.__input_channels:])
        pos_weights = self.pos_output(f)
        angle_weights = self.angle_output(f)
        width_weights = self.width_output(f)

        # Compute the weighted pos, wid, and ang
        pos_output = torch.sum(pos_weights * x[:, self.__p_mask], dim=1)
        angle_output = torch.sum(angle_weights * x[:, self.__a_mask], dim=1)
        width_output = torch.sum(width_weights * x[:, self.__w_mask], dim=1)

        return pos_output, angle_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, angle_pred, width_pred = self(xc)
        cos_pred = torch.cos(angle_pred)
        sin_pred = torch.sin(angle_pred)

        p_loss = F.mse_loss(torch.unsqueeze(pos_pred, dim=1), y_pos)
        cos_loss = F.mse_loss(torch.unsqueeze(cos_pred, dim=1), y_cos)
        sin_loss = F.mse_loss(torch.unsqueeze(sin_pred, dim=1), y_sin)
        width_loss = F.mse_loss(torch.unsqueeze(width_pred, dim=1), y_width)

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
