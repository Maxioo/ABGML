import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import stones
from data import Data
from model import Model
from torch_geometric.seed import seed_everything

if __name__ == '__main__':
    seed_everything(2022)
    args = stones.options().parse_args()
    setup = stones.utils.system_startup(args)

    experiment = None

    data = Data(args, setup=setup)
    model = Model(args, setup=setup, data=data)

    model.train()