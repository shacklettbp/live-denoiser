import torchvision
from utils import tonemap
from data_loading import load_raw
import sys

src = sys.argv[1]
dst = sys.argv[2]
dim_x = int(sys.argv[3])
dim_y = int(sys.argv[4])

tensor = load_raw(src, (3, dim_y, dim_x))
img = torchvision.transforms.ToPILImage()(tonemap(tensor))
img.save(dst)
