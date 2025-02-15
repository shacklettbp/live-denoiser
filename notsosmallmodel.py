import torch
import torch.nn as nn
import torch.nn.functional as F
from filters import simple_filter

num_input_channels = 8

class ModelImpl(nn.Module):
    def __init__(self):
        super(ModelImpl, self).__init__()

        self.cur_start = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.prev_start = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.normal_start = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.cur_enc = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.normal_enc = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.prev_enc = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.flow = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.final = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    def make_pyramid(self, tensor, net):
        arr = []
        out = tensor

        for i in range(4):
            out = net(out)
            arr.append(out)

            if i < 3:
                out = F.avg_pool2d(out, kernel_size=2, stride=2)

        return arr 

    def warp(self, img, relative_flow):
        H, W = img.shape[2:]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)
        grid = torch.cat((xx, yy), 1).float().to(img.get_device())
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        #relative_flow = 16 * torch.stack([relative_flow[:, 0, ...] / (W - 1), relative_flow[:, 1, ...] / (H - 1)], dim=1)

        absolute_flow = relative_flow + grid

        absolute_flow = absolute_flow.permute(0, 2, 3, 1)

        return F.grid_sample(img, absolute_flow)

    def forward(self, cur, prev, normal):
        cur = self.cur_start(cur)
        prev = self.prev_start(prev)
        normal = self.normal_start(normal)

        cur_pyramid = self.make_pyramid(cur, self.cur_enc)
        prev_pyramid = self.make_pyramid(prev, self.prev_enc)
        normal_pyramid = self.make_pyramid(normal, self.normal_enc)
        
        out = self.bottleneck(torch.cat([cur_pyramid[-1], prev_pyramid[-1], normal_pyramid[-1]], dim=1))

        flow = torch.zeros_like(out[:, 0:2, ...])

        for cur, prev, normal in reversed(list(zip(cur_pyramid, prev_pyramid, normal_pyramid))[:-1]):
            out = F.interpolate(out, scale_factor=2, mode='bilinear')
            flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            flow = flow + self.flow(out)
            prev = self.warp(prev, flow)

            out = self.dec(torch.cat([out, prev, normal], dim=1))

        return self.final(out)

class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()

        self.model = ModelImpl()
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
        
        self.model.apply(init_weights)

    def forward(self, color, normal, albedo, prev1, prev2):
        eps = 0.001
        color = color / (albedo + eps)

        mapped_color = torch.log1p(color)
        mapped_prev1 = torch.log1p(prev1)

        out = self.model(mapped_color, mapped_prev1, normal)

        exp = torch.expm1(out)

        return exp * (albedo + eps), exp

class KernelModelImpl(nn.Module):
    def __init__(self):
        super(KernelModelImpl, self).__init__()

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

        self.enc6 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.dec6 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel6 = nn.Sequential(
                        nn.Conv2d(in_channels=128+6, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.flow6 = nn.Sequential(
                        nn.Conv2d(in_channels=128+3, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))

        self.dec5 = nn.Sequential(
                nn.Conv2d(in_channels=128+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel5 = nn.Sequential(
                        nn.Conv2d(in_channels=96+9, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.flow5 = nn.Sequential(
                        nn.Conv2d(in_channels=96+3, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))

        self.dec4 = nn.Sequential(
                nn.Conv2d(in_channels=96+96, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel4 = nn.Sequential(
                        nn.Conv2d(in_channels=64+9, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.flow4 = nn.Sequential(
                        nn.Conv2d(in_channels=64+3, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))

        self.dec3 = nn.Sequential(
                nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel3 = nn.Sequential(
                        nn.Conv2d(in_channels=48+9, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.flow3 = nn.Sequential(
                        nn.Conv2d(in_channels=48+3, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))

        self.dec2 = nn.Sequential(
                nn.Conv2d(in_channels=48+48, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel2 = nn.Sequential(
                        nn.Conv2d(in_channels=32+9, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.flow2 = nn.Sequential(
                        nn.Conv2d(in_channels=32+3, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))

        self.dec1 = nn.Sequential(
                nn.Conv2d(in_channels=32+32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU())

        self.kernel1 = nn.Sequential(
                        nn.Conv2d(in_channels=32+9, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=self.kernel_total_weights, kernel_size=3, stride=1, padding=1))

        self.flow1 = nn.Sequential(
                        nn.Conv2d(in_channels=32+3, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))

        self.register_buffer('gauss_weights', torch.tensor([1/8, 3/8, 3/8, 1/8]).float().expand(3, -1))

    def warp(self, img, relative_flow):
        H, W = img.shape[2:]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)
        grid = torch.cat((xx, yy), 1).float().to(img.get_device())
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        #relative_flow = 16 * torch.stack([relative_flow[:, 0, ...] / (W - 1), relative_flow[:, 1, ...] / (H - 1)], dim=1)

        absolute_flow = relative_flow + grid

        absolute_flow = absolute_flow.permute(0, 2, 3, 1)

        return F.grid_sample(img, absolute_flow)

    def kernel_pred(self, img, weights):
        width, height = img.shape[-1], img.shape[-2]

        assert(img.shape[1] == self.imgs_to_filter and img.shape[2] == 3)

        #dense_weights = F.softmax(weights[:, 0:self.kernel_total_weights, ...], dim=1)
        dense_weights = F.softmax(weights[:, 0:self.kernel_total_weights, ...], dim=1)

        dense_weights = dense_weights.view(dense_weights.shape[0], self.imgs_to_filter, self.kernel_size * self.kernel_size, dense_weights.shape[-2], dense_weights.shape[-1])

        padding = self.kernel_size // 2
        padded = F.pad(img, (padding, padding, padding, padding))

        shifted = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                shifted.append(padded[:, :, :, i:i + height, j:j + width])

        img_stack = torch.stack(shifted, dim=2)
        dense_output = torch.sum(dense_weights.unsqueeze(dim=3) * img_stack, dim=[1, 2])

        return dense_output


    def make_pyramid(self, tensor):
        arr = [tensor]

        for i in range(5):
            tensor = F.pad(tensor, (2, 1, 2, 1), mode='replicate')
            tensor = F.conv2d(tensor, self.gauss_weights.view(3, 1, 1, 4), stride=(1, 2), groups=3)
            tensor = F.conv2d(tensor, self.gauss_weights.view(3, 1, 4, 1), stride=(2, 1), groups=3)

            arr.append(tensor)

        return arr

    def make_laplacian_pyramid(self, tensor):
        arr = []

        for i in range(5):
            orig = tensor

            tensor = F.pad(tensor, (2, 1, 2, 1), mode='replicate')
            tensor = F.conv2d(tensor, self.gauss_weights.view(3, 1, 1, 4), stride=(1, 2), groups=3)
            tensor = F.conv2d(tensor, self.gauss_weights.view(3, 1, 4, 1), stride=(2, 1), groups=3)

            upsampled = F.interpolate(tensor, scale_factor=2, mode='bilinear')

            diff = orig - upsampled

            arr.append(diff)

        arr.append(tensor)


        return arr

    def forward(self, color, normal, albedo, prev):
        color_pyramid = self.make_pyramid(color)
        prev_pyramid = self.make_pyramid(prev)

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

        enc5_out = self.enc5(out)
        out = F.avg_pool2d(enc5_out, kernel_size=2, stride=2)

        out = self.enc6(out)
        out = self.dec6(out)

        flow = self.flow6(torch.cat([out, prev_pyramid[5]], dim=1))
        prev = self.warp(prev_pyramid[5], flow)
        filter_in = torch.stack([color_pyramid[5], prev, torch.zeros_like(out[:, 0:3, ...])], dim=1)
        kernel = self.kernel6(torch.cat([out, filter_in[:, 0, ...], filter_in[:, 1, ...]], dim=1))
        filtered = self.kernel_pred(filter_in, kernel)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        filtered = F.interpolate(filtered, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc5_out], dim=1)
        out = self.dec5(out)

        flow = self.flow5(torch.cat([out, F.interpolate(prev, scale_factor=2, mode='bilinear')], dim=1)) + F.interpolate(flow, scale_factor=2, mode='bilinear')
        prev = self.warp(prev_pyramid[4], flow)

        filter_in = torch.stack([color_pyramid[4], prev, filtered], dim=1)
        kernel = self.kernel5(torch.cat([out, filter_in[:, 0, ...], filter_in[:, 1, ...], filter_in[:, 2, ...]], dim=1))
        filtered = self.kernel_pred(filter_in, kernel)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        filtered = F.interpolate(filtered, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc4_out], dim=1)
        out = self.dec4(out)

        flow = self.flow4(torch.cat([out, F.interpolate(prev, scale_factor=2, mode='bilinear')], dim=1)) + F.interpolate(flow, scale_factor=2, mode='bilinear')
        prev = self.warp(prev_pyramid[3], flow)

        filter_in = torch.stack([color_pyramid[3], prev, filtered], dim=1)
        kernel = self.kernel4(torch.cat([out, filter_in[:, 0, ...], filter_in[:, 1, ...], filter_in[:, 2, ...]], dim=1))
        filtered = self.kernel_pred(filter_in, kernel)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        filtered = F.interpolate(filtered, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc3_out], dim=1)
        out = self.dec3(out)

        flow = self.flow3(torch.cat([out, F.interpolate(prev, scale_factor=2, mode='bilinear')], dim=1)) + F.interpolate(flow, scale_factor=2, mode='bilinear')
        prev = self.warp(prev_pyramid[2], flow)

        filter_in = torch.stack([color_pyramid[2], prev, filtered], dim=1)
        kernel = self.kernel3(torch.cat([out, filter_in[:, 0, ...], filter_in[:, 1, ...], filter_in[:, 2, ...]], dim=1))
        filtered = self.kernel_pred(filter_in, kernel)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        filtered = F.interpolate(filtered, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc2_out], dim=1)
        out = self.dec2(out)

        flow = self.flow2(torch.cat([out, F.interpolate(prev, scale_factor=2, mode='bilinear')], dim=1)) + F.interpolate(flow, scale_factor=2, mode='bilinear')
        prev = self.warp(prev_pyramid[1], flow)

        filter_in = torch.stack([color_pyramid[1], prev, filtered], dim=1)
        kernel = self.kernel2(torch.cat([out, filter_in[:, 0, ...], filter_in[:, 1, ...], filter_in[:, 2, ...]], dim=1))
        filtered = self.kernel_pred(filter_in, kernel)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        filtered = F.interpolate(filtered, scale_factor=2, mode='bilinear')

        out = torch.cat([out, enc1_out], dim=1)
        out = self.dec1(out)

        flow = self.flow1(torch.cat([out, F.interpolate(prev, scale_factor=2, mode='bilinear')], dim=1)) + F.interpolate(flow, scale_factor=2, mode='bilinear')
        prev = self.warp(prev_pyramid[0], flow)

        filter_in = torch.stack([color_pyramid[0], prev, filtered], dim=1)
        kernel = self.kernel1(torch.cat([out, filter_in[:, 0, ...], filter_in[:, 1, ...], filter_in[:, 2, ...]], dim=1))
        filtered = self.kernel_pred(filter_in, kernel)
        
        return filtered

class KernelModel(nn.Module):
    def __init__(self):
        super(KernelModel, self).__init__()

        self.model = KernelModelImpl()

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
        
        self.model.apply(init_weights)

    def forward(self, color, normal, albedo, prev1, prev2):
        color = color.squeeze(dim=1)
        normal = normal.squeeze(dim=1)
        albedo = albedo.squeeze(dim=1)

        eps = 0.001
        color = color / (albedo + eps)

        mapped_color = torch.log1p(color)
        mapped_albedo = torch.log1p(albedo)
        mapped_prev = torch.log1p(prev1)

        out = self.model(mapped_color, normal, mapped_albedo, mapped_prev)

        exp = torch.expm1(out)

        return exp * (albedo + eps), exp

class TemporalSmallModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TemporalSmallModel, self).__init__()
        self.model = KernelModel(*args, **kwargs)

    def forward(self, color, normal, albedo):
        color = color.transpose(0, 1)
        normal = normal.transpose(0, 1)
        albedo = albedo.transpose(0, 1)

        all_outputs = []
        ei_outputs = []

        prev1 = torch.zeros_like(color[0]);
        prev2 = torch.zeros_like(prev1);

        for i in range(color.shape[0]):
            output, ei = self.model(color[i], normal[i], albedo[i], prev1, prev2)
            all_outputs.append(output)
            ei_outputs.append(ei)

            prev2 = prev1
            prev1 = ei

        return torch.stack(all_outputs, dim=1), torch.stack(ei_outputs, dim=1)


class SmallModelWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SmallModelWrapper, self).__init__()
        self.model = SmallModel(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
