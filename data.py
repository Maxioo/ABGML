import torch
import stones
import numpy as np
from torch_geometric.utils.loop import add_self_loops

class Data:

    def __init__(self, args, setup=dict(device=torch.device('cpu'))) -> None:
        self.args, self.setup = args, setup

        if self.args.dataset != "wiki-cs":
            self.dataset = stones.get_dataset(self.args.dataset_dir, self.args.dataset)
            self.num_eval_splits = self.args.num_eval_splits
        else:
            self.dataset, self.train_masks, self.val_masks, self.test_mask = stones.get_wiki_cs(args.dataset_dir)
            self.num_eval_splits = self.train_masks.shape[1]
        self.graph = self.dataset[0]
        self.graph.edge_index, self.graph.edge_attr = add_self_loops(self.graph.edge_index, self.graph.edge_attr)

        if self.args.method == 'BGRL':
            self.transform_1, self.transform_2 = self.get_two_transform()
        self.transforms = self.get_transforms()
        if self.args.dataset != 'wiki-cs':
            self.create_masks()
        
        self.graph = self.graph.to(**setup)


    def get_transforms(self):
        transform_1 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_1, drop_feat_p=self.args.drop_feat_p_1)
        transform_2 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2, drop_feat_p=self.args.drop_feat_p_2)
        # transform_2 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.2, drop_feat_p=self.args.drop_feat_p_2+0.1)
        # transform_3 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.1, drop_feat_p=self.args.drop_feat_p_2+0.1)
        # transform_6 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.1, drop_feat_p=self.args.drop_feat_p_2+0.2)
        # transform_7 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.2, drop_feat_p=self.args.drop_feat_p_2+0.2)
        # transform_8 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.3, drop_feat_p=self.args.drop_feat_p_2+0.2)
        # transform_9 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.2, drop_feat_p=self.args.drop_feat_p_2+0.3)
        # transform_10 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.3, drop_feat_p=self.args.drop_feat_p_2+0.3)
        # transform_11 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.4, drop_feat_p=self.args.drop_feat_p_2+0.2)
        # transform_12 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2+0.2, drop_feat_p=self.args.drop_feat_p_2+0.4)
        transform_4 = stones.get_graph_drop_transform(drop_edge_p=0, drop_feat_p=self.args.drop_feat_p_2)
        transform_5 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2, drop_feat_p=0)
        # transform_6 = stones.get_graph_drop_transform(drop_edge_p=0, drop_feat_p=self.args.drop_feat_p_1)
        # transform_7 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_1, drop_feat_p=0)
        # transform_6 = stones.get_graph_transform(0, 0, 0.3)
        transforms = [transform_1, transform_2, transform_4, transform_5]
        # transforms = [transform_1, transform_2, transform_3, transform_4, transform_5, transform_6, transform_7, transform_8, transform_9, transform_10, transform_11, transform_12]
        return transforms

    # def get_transforms(self):
    #     transform_1 = stones.get_graph_transform(0.2, 0.2, 0.2)
    #     transform_2 = stones.get_graph_transform(0.2, 0.2, 0)
    #     transform_3 = stones.get_graph_transform(0, 0.2, 0.2)
    #     transform_4 = stones.get_graph_transform(0.2, 0, 0.2)
    #     transform_5 = stones.get_graph_transform(0.2, 0, 0)
    #     transform_6 = stones.get_graph_transform(0, 0.2, 0)
    #     transform_7 = stones.get_graph_transform(0, 0, 0.2)
    #     transform_8 = stones.get_graph_transform(0, 0, 0)
    #     transforms = [transform_1, transform_2, transform_3, transform_4, transform_5, transform_6, transform_7, transform_8]
    #     return transforms
    
    def get_two_transform(self):
        transform_1 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_1, drop_feat_p=self.args.drop_feat_p_1)
        transform_2 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2, drop_feat_p=self.args.drop_feat_p_2)
        # transform_1 = stones.get_graph_drop_transform(drop_edge_p=0, drop_feat_p=self.args.drop_feat_p_2)
        # transform_2 = stones.get_graph_drop_transform(drop_edge_p=self.args.drop_edge_p_2, drop_feat_p=0)
        return transform_1, transform_2

    def create_masks(self):
        """
        Splits data into training, validation, and test splits in a stratified manner if
        it is not already splitted. Each split is associated with a mask vector, which
        specifies the indices for that split. The data will be modified in-place
        :param data: Data object
        :return: The modified data
        """
        if not hasattr(self.graph, "val_mask"):

            self.graph.train_mask = 1
            self.graph.dev_mask = 1
            self.graph.test_mask = 1

            for i in range(self.args.num_eval_splits):
                labels = self.graph.y.numpy()
                dev_size = int(labels.shape[0] * 0.1)
                test_size = int(labels.shape[0] * 0.8)

                perm = np.random.permutation(labels.shape[0])
                test_index = perm[:test_size]
                dev_index = perm[test_size:test_size + dev_size]

                data_index = np.arange(labels.shape[0])
                test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
                dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
                train_mask = ~(dev_mask + test_mask)
                test_mask = test_mask.reshape(1, -1)
                dev_mask = dev_mask.reshape(1, -1)
                train_mask = train_mask.reshape(1, -1)

                if type(self.graph.train_mask) is int:
                    self.graph.train_mask = train_mask
                    self.graph.val_mask = dev_mask
                    self.graph.test_mask = test_mask
                else:
                    self.graph.train_mask = torch.cat((self.graph.train_mask, train_mask), dim=0)
                    self.graph.val_mask = torch.cat((self.graph.val_mask, dev_mask), dim=0)
                    self.graph.test_mask = torch.cat((self.graph.test_mask, test_mask), dim=0)

        else:  # in the case of WikiCS
            self.graph.train_mask = self.graph.train_mask.T
            self.graph.val_mask = self.graph.val_mask.T