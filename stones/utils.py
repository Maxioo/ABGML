import imp
import torch
import socket
import random
import datetime

import numpy as np
import torch.nn as nn

def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, non_blocking=True)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    if args.deterministic:
        """Switch pytorch into a deterministic computation mode."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    # set_random_seed(args.seed)
        
    return setup

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def set_random_seed(seed=666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)