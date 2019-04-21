from smallmodel import KernelModelImpl
from data_loading import load_exr, save_exr, pad_data
import torch.nn.functional as F

tensor = load_exr("/home/bps/rendering/data-fast/finals/1spp/bistro/trainpath/hdr_200.exr").unsqueeze(dim=0)

tensor = pad_data(tensor).cuda()

save_exr(tensor.squeeze(dim=0), "/tmp/t.exr")

m = KernelModelImpl().cuda()
print(m.gauss_weights)

pyramid = m.make_laplacian_pyramid(tensor)

for n, i in enumerate(pyramid):
    if n > 0:
        i = F.interpolate(i, scale_factor=2**n, mode='bilinear')
    save_exr(i.squeeze(dim=0).cpu(), "/tmp/t{}.exr".format(n))
