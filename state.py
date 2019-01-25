import os
import pathlib
from datetime import datetime
import torch
import glob
import re
import pygit2
import os
from collections import OrderedDict

def convert_to_cpu(state):
    if isinstance(state, torch.Tensor):
        return state.cpu()
    elif isinstance(state, float) or isinstance(state, int):
        return state
    elif isinstance(state, dict):
        cpu_state_dict = OrderedDict()
        for key in state.keys():
            cpu_state_dict[key] = convert_to_cpu(state[key])
        return cpu_state_dict
    elif isinstance(state, list):
        return [convert_to_cpu(elem) for elem in state]
    elif isinstance(state, tuple):
        return tuple(convert_to_cpu(elem) for elem in state)
    else:
        print(type(state))
        assert(False)

def save_with_cpu(state_dict, path):
    torch.save(convert_to_cpu(state_dict), path)

class StateManager:
    def __init__(self, args, model, optimizer, device):
        if args.restore is None:
            timestr = datetime.now().strftime('%H-%M-%S-%m-%d-%Y')
            self.weights_dir = os.path.join('weights', timestr)
            path = pathlib.Path(self.weights_dir).mkdir(parents=True, exist_ok=True)

            self.start_epoch = 0

            with open(os.path.join(self.weights_dir, 'setup'), 'w') as f:
                print(vars(args), file=f)
                try:
                    repo_path = os.path.dirname(os.path.realpath(__file__))
                except:
                    repo_path = '.'
                repo = pygit2.Repository(repo_path)
                print("Commit: {}".format(repo.head.get_object().short_id), file=f)
        else:
            self.weights_dir = args.restore
            weights = glob.iglob(os.path.join(self.weights_dir, '*pth'))
            latest_weights = max(weights, key=os.path.getctime)
            m = re.search('weights_(\d+).pth', latest_weights)

            self.start_epoch = int(m.group(1))

            optim_state_path = os.path.join(self.weights_dir,
                                            'optim_{}'.format(self.start_epoch))

            model.load_state_dict(torch.load(latest_weights, map_location='cpu'))
            optimizer.load_state_dict(torch.load(optim_state_path, map_location='cpu'))

    def save_state(self, model, optimizer, epoch):
        save_with_cpu(model.state_dict(),
                      os.path.join(self.weights_dir,
                                   "weights_{}.pth".format(epoch)))
        save_with_cpu(optimizer.state_dict(),
                      os.path.join(self.weights_dir,
                                   "optim_{}".format(epoch)))
                                        

    def get_start_epoch(self):
        return self.start_epoch
