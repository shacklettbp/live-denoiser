import os
import pathlib
from datetime import datetime
import torch
import glob
import re
import pygit2
import os

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
        torch.save(model.state_dict(),
                   os.path.join(self.weights_dir,
                                "weights_{}.pth".format(epoch)))
        torch.save(optimizer.state_dict(),
                   os.path.join(self.weights_dir,
                                "optim_{}".format(epoch)))
                                        

    def get_start_epoch(self):
        return self.start_epoch
