import copy
from setuptools import setup
import torch
import torch.nn as nn
import numpy as np

from ray import tune
from tqdm import tqdm
from typing import OrderedDict
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cosine_similarity
from stones.models import GCN, BGRL, MetaGCN, MetaMLP_Predictor, MetaBGRL, compute_representations, MetaCLASSBGRL
from stones.scheduler import CosineDecayScheduler
from stones.logistic_regression_eval import evaluate_node, evaluate_node_wikics


class Model:

    def __init__(self, args, setup=dict(device=torch.device('cpu')), data=None):
        self.args, self.setup, self.data = args, setup, data

        input_size, representation_size = self.data.graph.x.size(1), self.args.graph_encoder_layer[-1]
        print("_____________________________")
        print(self.args.graph_encoder_layer)

        self.encoder = MetaGCN([input_size] + self.args.graph_encoder_layer, batchnorm=True)
        self.predictor = MetaMLP_Predictor(representation_size, representation_size, hidden_size=self.args.predictor_hidden_size)
        self.classifier = nn.Linear(representation_size * 2, 2 * len(self.data.transforms))
        self.model = MetaCLASSBGRL(self.encoder, self.predictor, self.classifier).to(**setup)
        self.bce = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(self.model.trainable_parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Scheduler
        self.lr_scheduler = CosineDecayScheduler(self.args.lr, self.args.lr_warmup_epochs, self.args.epochs)
        self.mm_scheduler = CosineDecayScheduler(1 - self.args.m, 0, self.args.epochs)

        # setup tensorboard and make custom layout
        self.writer = SummaryWriter("./logs/" + args.logdir)
        layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(self.data.num_eval_splits)]]}}
        self.writer.add_custom_scalars(layout)

        self.best_val = 0


    def train(self):
        print("Start training....")
        self.start_epoch = 1
        self.eval(0)
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.meta_classbgrl_step(epoch)
            
            if epoch % self.args.eval_epochs == 0:
                self.eval(epoch)

    def meta_classbgrl_step(self, step):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        aux = 2
        node_update = 1
        node_loss = []
        cont_losses = []
        bce_losses = []

        lr = self.lr_scheduler.get(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        mm = 1 - self.mm_scheduler.get(step)
        node_size = self.data.graph.x.size(0)
        test_size = int(node_size * self.args.meta_p)
        perm = np.random.permutation(node_size)
        test_index = perm[:test_size]
        data_index = np.arange(node_size)
        test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
        train_mask = ~test_mask

        for _ in range(aux):
            idx1, idx2 = np.random.choice(np.arange(len(self.data.transforms)), size=2, replace=True)
            class_label = torch.full([node_size], idx1+idx2).to(**self.setup)
            train_label = class_label[train_mask]
            test_label = class_label[test_mask]

            transform1= self.data.transforms[idx1]
            transform2 = self.data.transforms[idx2]

            fast_weights = OrderedDict(self.model.named_parameters())
            x1, x2 = transform1(self.data.graph), transform2(self.data.graph)

            for i in range(node_update):

                c1, q1, y2 = self.model(x1, x2, fast_weights)
                c2, q2, y1 = self.model(x2, x1, fast_weights)
                q1, y2 = q1[train_mask], y2[train_mask]
                q2, y1 = q2[train_mask], y1[train_mask]
                c1, c2 = c1[train_mask], c2[train_mask]

                x = torch.cat([c1, c2], dim=1)
                logits = self.model.linear(x, fast_weights)
                bce_loss = self.bce(logits, train_label)

                cont_loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
                loss = self.args.task1_p * cont_loss + (1 - self.args.task1_p) * bce_loss
                fast_weights1 = OrderedDict((a,b) for a,b in self.model.named_parameters() if b.requires_grad)
                gradients = torch.autograd.grad(loss, fast_weights1.values(), create_graph=False, allow_unused=True)
                f2 = OrderedDict((name, param) for name, param in fast_weights.items() if "target_encoder" in name)
                fast_weights = OrderedDict(
                    (name, param - lr * grad)
                    for ((name, param), grad) in zip(fast_weights1.items(), gradients)
                )
                fast_weights.update(f2)
                
                fast_weights = self.model.update_target_network_meta(mm, fast_weights)


            c1, q1, y2 = self.model(x1, x2, fast_weights)
            c2, q2, y1 = self.model(x2, x1, fast_weights)
            q1, y2 = q1[test_mask], y2[test_mask]
            q2, y1 = q2[test_mask], y1[test_mask]
            c1, c2 = c1[test_mask], c2[test_mask]
            x = torch.cat([c1, c2], dim=1)
            logits = self.model.linear(x, fast_weights)
            bce_loss = self.bce(logits, test_label)
            cont_loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            loss = self.args.task1_p * cont_loss + (1 - self.args.task1_p) * bce_loss

            node_loss.append(loss)
            cont_losses.append(cont_loss)
            bce_losses.append(bce_loss)

        self.optimizer.zero_grad()
        node_loss = torch.stack(node_loss).mean()
        bce_losses = torch.stack(bce_losses).mean()
        cont_losses = torch.stack(cont_losses).mean()

        node_loss.backward()
        self.optimizer.step()
        self.model.update_target_network(mm)

        self.writer.add_scalar('params/lr', lr, step)
        self.writer.add_scalar('params/mm', mm, step)
        self.writer.add_scalar('train/node_loss', node_loss, step)
        self.writer.add_scalar('train/bce_losses', bce_losses, step)
        self.writer.add_scalar('train/cont_losses', cont_losses, step)
        
    def eval(self, epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(self.model.online_encoder)
        representations, labels = compute_representations(tmp_encoder, self.data.dataset, self.setup['device'])

        if self.args.dataset != 'wiki-cs':
            dev_score, dev_std, test_score, test_std = evaluate_node(representations, labels, self.data.graph, self.args.dataset, repeat=self.data.num_eval_splits, device=self.setup['device'])
        else:
            dev_score, dev_std, test_score, test_std = evaluate_node_wikics(representations, labels, self.data.num_eval_splits,
                                                           self.data.train_masks, self.data.val_masks, self.data.test_mask, device=self.setup['device'])

        self.writer.add_scalar(f'accuracy/dev_score', dev_score, epoch)
        self.writer.add_scalar(f'accuracy/dev_std', dev_std, epoch)
        self.writer.add_scalar(f'accuracy/test_score', test_score, epoch)
        self.writer.add_scalar(f'accuracy/test_std', test_std, epoch)
        tune.report(dev_score=dev_score, dev_std=dev_std, test_score=test_score, test_std=test_std)
        if dev_score > self.best_val:
            self.writer.add_scalar(f'best_accuracy/dev_score', dev_score, epoch)
            self.writer.add_scalar(f'best_accuracy/dev_std', dev_std, epoch)

            self.writer.add_scalar(f'best_accuracy/test_score', test_score, epoch)
            self.writer.add_scalar(f'best_accuracy/test_std', test_std, epoch)
            self.best_val = dev_score