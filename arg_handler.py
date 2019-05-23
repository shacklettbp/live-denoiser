import argparse

def common_args(parser):
    parser.add_argument('--img-height', type=int, default=None)
    parser.add_argument('--img-width', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vanilla-net', default=False, action='store_true')
    parser.add_argument('--disable-recurrence', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0003)

def parse_train_args():
    parser = argparse.ArgumentParser()
    common_args(parser)

    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--training-set', type=str, required=True)
    parser.add_argument('--reference-set', type=str, default=None)
    parser.add_argument('--validation-set', type=str, required=True)
    parser.add_argument('--num-pairs', type=int, default=None)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--name', type=str, default='run')

    return parser.parse_args()

def parse_infer_args(multi_inputs = False):
    parser = argparse.ArgumentParser()
    common_args(parser)

    parser.add_argument('--weights', type=str, required=False)
    if multi_inputs:
        parser.add_argument('--inputs', type=str, nargs="+", required=True)        
    else:
        parser.add_argument('--inputs', type=str, required=True)
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--num-imgs', type=int, default=None)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--loss-check', type=str, default=None)

    return parser.parse_args()
