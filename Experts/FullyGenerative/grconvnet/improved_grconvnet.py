import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from Datasets.dataset_processing import grasp
import numpy as np
from utils.common import post_process_output

class Swish(nn.Module):
      def forward(self, x):
            return x * torch.sigmoid(x)

class EnhancedResidualBlock(nn.Module):
      """
      Improved residual block with proper BatchNorm placement and enhanced training
      """
      def __init__(self, channels, kernel_size=3, dropout_prob=0.1):
            super(EnhancedResidualBlock, self).__init__()
            # Proper BN placement: after conv, before activation
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
            self.bn2 = nn.BatchNorm2d(channels)
            self.dropout = nn.Dropout2d(p=dropout_prob)
            self.swish = Swish()  # Use the module instead of lambda
    
      def forward(self, x_in):
            identity = x_in
            
            # First conv-bn-activation block
            x = self.conv1(x_in)
            x = self.bn1(x)
            x = self.swish(x)
            
            # Second conv-bn block (no activation before addition)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.bn2(x)
            
            # Residual connection
            x = x + identity
            
            # Activation after addition
            x = self.swish(x)
            
            return x

class ChannelAttention(nn.Module):
      def __init__(self, channels, reduction_ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            
            # Squeeze-Excitation block
            self.fc = nn.Sequential(
                  nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
            )
        
      def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = torch.sigmoid(avg_out + max_out)
            return out

class SpatialAttention(nn.Module):
      def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
      def forward(self, x):
            # Compute spatial attention map
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            y = torch.cat([avg_out, max_out], dim=1)
            y = torch.sigmoid(self.conv(y))
            return y

class EnhancedGenerativeResnet(nn.Module):
      def __init__(self, input_channels=4, output_channels=1, channel_size=32, 
                  dropout=True, dropout_prob=0.1):
            super(EnhancedGenerativeResnet, self).__init__()
        
            # Specialized RGB-D input processing
            if input_channels == 4:
                  self.process_rgb = nn.Conv2d(3, channel_size//2, kernel_size=7, stride=1, padding=3)
                  self.process_depth = nn.Conv2d(1, channel_size//2, kernel_size=7, stride=1, padding=3)
                  self.fuse_rgbd = nn.Conv2d(channel_size, channel_size, kernel_size=1)
                  self.bn_fuse = nn.BatchNorm2d(channel_size)
                  self.use_rgbd_process = True
            else:
                  self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
                  self.bn1 = nn.BatchNorm2d(channel_size)
                  self.use_rgbd_process = False
        
            # Encoder path
            self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(channel_size * 2)
            
            self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(channel_size * 4)
            
            # Enhanced residual blocks with more stable training
            self.res_blocks = nn.ModuleList([
                  EnhancedResidualBlock(channel_size * 4, dropout_prob=dropout_prob if dropout else 0) 
                  for _ in range(5)
            ])
        
            # Attention mechanisms at bottleneck
            self.channel_attention = ChannelAttention(channel_size * 4)
            self.spatial_attention = SpatialAttention()
            
            # Decoder path with skip connections
            self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, 
                                          kernel_size=4, stride=2, padding=1, output_padding=1)
            self.bn4 = nn.BatchNorm2d(channel_size * 2)
            
            # Skip connection processing
            self.skip1_process = nn.Conv2d(channel_size * 2, channel_size * 2, kernel_size=1)
            self.skip1_bn = nn.BatchNorm2d(channel_size * 2)
            
            self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, 
                                          kernel_size=4, stride=2, padding=2, output_padding=1)
            self.bn5 = nn.BatchNorm2d(channel_size)
            
            # Second skip connection
            self.skip2_process = nn.Conv2d(channel_size, channel_size, kernel_size=1)
            self.skip2_bn = nn.BatchNorm2d(channel_size)
            
            self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
            self.bn6 = nn.BatchNorm2d(channel_size)
        
            # Refined output heads - shared feature refinement followed by specialized branches
            self.common_output_refine = nn.Sequential(
                  nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1),
                  nn.BatchNorm2d(channel_size),
                  nn.ReLU(inplace=True)
            )
        
            # Output heads with dropout paths
            self.pos_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
            self.cos_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
            self.sin_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
            self.width_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
            
            # Dropout layers
            self.dropout = dropout
            self.dropout_pos = nn.Dropout2d(p=dropout_prob)
            self.dropout_cos = nn.Dropout2d(p=dropout_prob)
            self.dropout_sin = nn.Dropout2d(p=dropout_prob)
            self.dropout_wid = nn.Dropout2d(p=dropout_prob)
            
            # Activation function
            self.swish = Swish()

            
            # Better initialization
            self._initialize_weights()
        
      def _initialize_weights(self):
            for m in self.modules():
                  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                              nn.init.constant_(m.bias, 0)
                  elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                
      def forward(self, x_in):
            # Specialized input processing for RGB-D
            if self.use_rgbd_process:
                  # Split RGB and depth
                  depth = x_in[:, 0:1, :, :]
                  rgb = x_in[:, 1:4, :, :]
                  
                  # Process separately
                  depth_features = self.swish(self.process_depth(depth))
                  rgb_features = self.swish(self.process_rgb(rgb))
                  
                  # Combine features
                  x = torch.cat([depth_features, rgb_features], dim=1)
                  x = self.swish(self.bn_fuse(self.fuse_rgbd(x)))
                  skip1 = x  # Store for skip connection
            else:
                  # Standard processing
                  x = self.swish(self.bn1(self.conv1(x_in)))
                  skip1 = x  # Store for skip connection
        
            # Encoder path
            x = self.swish(self.bn2(self.conv2(x)))
            skip2 = x  # Store for skip connection
            
            x = self.swish(self.bn3(self.conv3(x)))
            
            # Residual blocks
            for res_block in self.res_blocks:
                  x = res_block(x)
        
            # Apply attention at bottleneck
            ca = self.channel_attention(x)
            x = x * ca  # Apply channel attention
            sa = self.spatial_attention(x)
            x = x * sa  # Apply spatial attention
            
            # Decoder path with skip connections
            x = self.swish(self.bn4(self.conv4(x)))
            
            # Add processed skip connection
            skip2_processed = self.swish(self.skip1_bn(self.skip1_process(skip2)))
            if x.size()[2:] == skip2_processed.size()[2:]:
                  x = x + skip2_processed
            else:
                  x = x + F.interpolate(skip2_processed, size=x.size()[2:])
        
            x = self.swish(self.bn5(self.conv5(x)))
            
            # Add second processed skip connection
            skip1_processed = self.swish(self.skip2_bn(self.skip2_process(skip1)))
            if x.size()[2:] == skip1_processed.size()[2:]:
                  x = x + skip1_processed
            else:
                  x = x + F.interpolate(skip1_processed, size=x.size()[2:])
        
            x = self.swish(self.bn6(self.conv6(x)))
            
            # Refined features for output
            x = self.common_output_refine(x)
            
            # Apply dropout if enabled
            if self.dropout:
                  pos_output = self.pos_output(self.dropout_pos(x))
                  cos_output = self.cos_output(self.dropout_cos(x))
                  sin_output = self.sin_output(self.dropout_sin(x))
                  width_output = self.width_output(self.dropout_wid(x))
            else:
                  pos_output = self.pos_output(x)
                  cos_output = self.cos_output(x)
                  sin_output = self.sin_output(x)
                  width_output = self.width_output(x)
        
            # Apply appropriate activations
            pos_output = pos_output
            cos_output = torch.tanh(cos_output)
            sin_output = torch.tanh(sin_output)
            width_output = torch.sigmoid(width_output)
            
            return pos_output, cos_output, sin_output, width_output

      def compute_loss(self, xc, yc):
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

      def predict(self, xc):
            pos_pred, cos_pred, sin_pred, width_pred = self(xc)
            return {
                  'pos': pos_pred,
                  'cos': cos_pred,
                  'sin': sin_pred,
                  'width': width_pred
      }