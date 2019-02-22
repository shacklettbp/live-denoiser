import os
from torch.utils.cpp_extension import load

os.environ['CC'] = 'gcc-7'
os.environ['CXX'] = 'g++-7'

lltm = load(name='lltm', sources=['denoiser_model.cpp', 'denoiser_model.cu'])
