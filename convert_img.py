import sys
import os
from data_loading import load_raw, dump_raw, load_exr, save_exr, save_png

a = sys.argv[1]
b = sys.argv[2]

a_ext = os.path.splitext(a)[1][1:]
b_ext = os.path.splitext(b)[1][1:]

if a_ext == 'png':
    print("Cannot load pngs", file=sys.stderr)
    sys.exit(1)

if a_ext == 'exr':
    tensor = load_exr(a)
elif a_ext == 'dmp':
    dim_x = int(sys.argv[3])
    dim_y = int(sys.argv[4])
    fullshape = (3, dim_y, dim_x)
    tensor = load_raw(a, fullshape)

if b_ext == 'exr':
    save_exr(tensor, b)
elif b_ext == 'png':
    save_png(tensor, b)
elif b_ext == 'dmp':
    dump_raw(tensor, b)
