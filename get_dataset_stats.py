from dataset import get_files
from data_loading import load_exr, dump_raw
from utils import tonemap
import sys
import os
import torch
from itertools import chain

data_dir = sys.argv[1]
files = get_files(data_dir, 'exr', int(sys.argv[2]))

color_mean = torch.zeros(3).cuda()
normal_mean = torch.zeros(3).cuda()
albedo_mean = torch.zeros(3).cuda()

color_var = torch.zeros(3).cuda()
normal_var = torch.zeros(3).cuda()
albedo_var = torch.zeros(3).cuda()

n = 0

for idx, (hdr, normal, albedo) in enumerate(files):
    n += 1
    color, normal, albedo = load_exr(hdr).cuda(), load_exr(normal).cuda(), load_exr(albedo).cuda()
    color = torch.log1p(color)
    albedo = torch.log1p(albedo)
    color, normal, albedo = color.view(3, -1), normal.view(3, -1), albedo.view(3, -1)

    prev_color_mean = color_mean
    prev_normal_mean = normal_mean
    prev_albedo_mean = albedo_mean

    cur_color_mean = color.mean(dim=1)
    cur_normal_mean = normal.mean(dim=1)
    cur_albedo_mean = albedo.mean(dim=1)

    color_mean = 1/n*(cur_color_mean + (n-1)*color_mean)
    normal_mean = 1/n*(cur_normal_mean + (n-1)*normal_mean)
    albedo_mean = 1/n*(cur_albedo_mean + (n-1)*albedo_mean)

    color_var = color_var + (cur_color_mean - prev_color_mean)*(cur_color_mean - color_mean)
    normal_var = normal_var + (cur_normal_mean - prev_normal_mean)*(cur_normal_mean - normal_mean)
    albedo_var = albedo_var + (cur_albedo_mean - prev_albedo_mean)*(cur_albedo_mean - albedo_mean)

color_std = (color_var / n)**(1/2)
normal_std = (normal_var / n)**(1/2)
albedo_std = (albedo_var / n)**(1/2)

print("{} {} {} {} {} {}".format(color_mean.cpu(), normal_mean.cpu(), albedo_mean.cpu(), color_var.cpu(), normal_var.cpu(), albedo_var.cpu()))
