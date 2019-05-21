import torch
import torch.nn as nn
import torch.nn.functional as F
from filters import simple_filter

num_input_channels = 8

class HierarchicalModelImpl(nn.Module):
    def __init__(self):
        super(HierarchicalModelImpl, self).__init__()

        self.kernel_size = 3
        self.imgs_to_filter = 3
        self.kernel_total_weights = self.kernel_size*self.kernel_size*self.imgs_to_filter

        self.start = nn.Sequential(
                nn.Conv2d(in_channels=11, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.enc2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.enc3 = nn.Sequential(
                nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.enc4 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.enc5 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec5 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec4 = nn.Sequential(
                nn.Conv2d(in_channels=96+96, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec3 = nn.Sequential(
                nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec2 = nn.Sequential(
                nn.Conv2d(in_channels=48+48, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec1 = nn.Sequential(
                nn.Conv2d(in_channels=32+32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.albedo_kernel = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=self.kernel_size*self.kernel_size*2, kernel_size=3, stride=1, padding=1))

    def kernel_pred(self, img, weights, num_filtered=None):
        width, height = img.shape[-1], img.shape[-2]
        
        if num_filtered is None:
            num_filtered = self.imgs_to_filter

        assert(img.shape[1] == num_filtered and img.shape[2] == 3)

        dense_weights = F.softmax(weights[:, 0:self.kernel_total_weights, ...], dim=1)

        dense_weights = dense_weights.view(dense_weights.shape[0], num_filtered, self.kernel_size * self.kernel_size, dense_weights.shape[-2], dense_weights.shape[-1])

        padding = self.kernel_size // 2
        padded = F.pad(img, (padding, padding, padding, padding))

        shifted = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                shifted.append(padded[:, :, :, i:i + height, j:j + width])

        img_stack = torch.stack(shifted, dim=2)
        dense_output = torch.sum(dense_weights.unsqueeze(dim=3) * img_stack, dim=[1, 2])

        return dense_output

    def forward(self, color, normal, albedo, prev, upsampled, upsampled_albedo):
        full_input = torch.cat([color, normal, albedo, prev], dim=1)
        start = self.start(full_input)

        enc1_out = self.enc1(start)
        out = F.avg_pool2d(enc1_out, kernel_size=2, stride=2)

        enc2_out = self.enc2(out)
        out = F.avg_pool2d(enc2_out, kernel_size=2, stride=2)

        enc3_out = self.enc3(out)
        out = F.avg_pool2d(enc3_out, kernel_size=2, stride=2)

        enc4_out = self.enc4(out)
        out = F.avg_pool2d(enc4_out, kernel_size=2, stride=2)

        out = self.enc5(out)
        out = self.dec5(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc4_out], dim=1)
        out = self.dec4(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc3_out], dim=1)
        out = self.dec3(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc2_out], dim=1)
        out = self.dec2(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc1_out], dim=1)
        out = self.dec1(out)

        kernel_weights = self.kernel(out)
        filter_in = torch.stack([color, prev, upsampled], dim=1)
        filtered = self.kernel_pred(filter_in, kernel_weights)

        albedo_kernel_weights = self.albedo_kernel(out)
        albedo_filter_in = torch.stack([albedo, upsampled_albedo], dim=1)
        albedo_filtered = self.kernel_pred(albedo_filter_in, albedo_kernel_weights, 2)

        return filtered, albedo_filtered

class HierarchicalKernelModel(nn.Module):
    def __init__(self):
        super(HierarchicalKernelModel, self).__init__()

        self.model = HierarchicalModelImpl()

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
        
        self.model.apply(init_weights)
        self.register_buffer('gauss_weights', torch.tensor([1/8, 3/8, 3/8, 1/8]).float().expand(3, -1))

    def make_pyramid(self, tensor, depth=5, preblur=True):
        arr = [tensor]

        for i in range(depth-1):
            if preblur:
                tensor = F.pad(tensor, (2, 1, 2, 1), mode='replicate')
                tensor = F.conv2d(tensor, self.gauss_weights.view(3, 1, 1, 4), stride=(1, 2), groups=3)
                tensor = F.conv2d(tensor, self.gauss_weights.view(3, 1, 4, 1), stride=(2, 1), groups=3)
            else:
                tensor = F.avg_pool2d(tensor, kernel_size=2, stride=2)

            arr.append(tensor)

        return arr 

    def forward(self, color, normal, albedo, prev1, prev2):
        eps = 0.001
        color = color / (albedo + eps)

        mapped_color = torch.log1p(color)
        mapped_albedo = torch.log1p(albedo)
        mapped_prev = torch.log1p(prev1)

        irradiance_depth = 4

        color_pyramid = self.make_pyramid(mapped_color, depth=irradiance_depth)
        normal_pyramid = self.make_pyramid(normal, depth=irradiance_depth, preblur=False)
        albedo_pyramid = self.make_pyramid(mapped_albedo, depth=irradiance_depth)
        prev_pyramid = self.make_pyramid(mapped_prev, depth=irradiance_depth)

        irradiance_out = color_pyramid[irradiance_depth - 1]
        albedo_out = albedo_pyramid[irradiance_depth - 1]

        for i in reversed(range(irradiance_depth - 1)):
            cur_color, cur_normal, cur_albedo, cur_prev = color_pyramid[i], normal_pyramid[i], albedo_pyramid[i], prev_pyramid[i]

            irradiance_out, albedo_out = self.model(cur_color, cur_normal, cur_albedo, cur_prev,
                                                    F.interpolate(irradiance_out, scale_factor=2, mode='bilinear'),
                                                    F.interpolate(albedo_out, scale_factor=2, mode='bilinear'))

        irradiance_out = torch.expm1(irradiance_out)
        albedo_out = torch.expm1(albedo_out)

        return irradiance_out * (albedo_out + eps), irradiance_out, albedo_out

class TemporalHierarchicalKernelModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TemporalHierarchicalKernelModel, self).__init__()
        self.model = HierarchicalKernelModel(*args, **kwargs)

    def forward(self, color, normal, albedo):
        color = color.transpose(0, 1)
        normal = normal.transpose(0, 1)
        albedo = albedo.transpose(0, 1)

        all_outputs = []
        ei_outputs = []
        albedo_outputs = []

        prev1 = torch.zeros_like(color[0]);
        prev2 = torch.zeros_like(prev1);

        for i in range(color.shape[0]):
            output, ei, albedo_out = self.model(color[i], normal[i], albedo[i], prev1, prev2)
            all_outputs.append(output)
            ei_outputs.append(ei)
            albedo_outputs.append(albedo_out)

            prev2 = prev1
            prev1 = ei

        return torch.stack(all_outputs, dim=1), torch.stack(ei_outputs, dim=1), torch.stack(albedo_outputs, dim=1)
