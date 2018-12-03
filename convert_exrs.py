from dataset import get_files
from data_loading import load_exr, dump_raw
import sys
import os

data_dir = sys.argv[1]
files = get_files(data_dir, 'exr', int(sys.argv[2]))

for idx, (hdr, normal, albedo) in enumerate(files):
    dump_raw(load_exr(hdr), os.path.join(data_dir, 'hdr_{}.dmp'.format(idx)))
    dump_raw(load_exr(normal), os.path.join(data_dir, 'normal_{}.dmp'.format(idx)))
    dump_raw(load_exr(albedo), os.path.join(data_dir, 'albedo_{}.dmp'.format(idx)))
