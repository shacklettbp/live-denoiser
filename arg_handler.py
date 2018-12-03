import argparse

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--training-set', type=str, required=True)
    parser.add_argument('--reference-set', type=str, default=None)
    parser.add_argument('--num-pairs', type=int, default=None)
    parser.add_argument('--img-height', type=int, default=None)
    parser.add_argument('--img-width', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--restore', type=str, default=None)

    return parser.parse_args()

def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--inputs', type=str, required=True)
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--num-imgs', type=int, default=None)
    parser.add_argument('--img-height', type=int, default=None)
    parser.add_argument('--img-width', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start-frame', type=int, default=0)

    return parser.parse_args()
