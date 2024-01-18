import argparse

def options():
    
    parser = argparse.ArgumentParser(description="Meta-Cs")

    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--dataset", default="amazon-photos", type=str,\
        help="['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs']")
    parser.add_argument("--dataset_dir", default="data", type=str)
    parser.add_argument("--logdir", default="", type=str)
    parser.add_argument("--num_eval_splits", default=20, type=int)

    parser.add_argument("--graph_encoder_layer", default=[512], type=int)
    parser.add_argument("--predictor_hidden_size", default=512, type=int)

    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--eval_epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--m", default=0.99, type=float)
    parser.add_argument("--meta_p", default=0.2, type=float)
    parser.add_argument("--task1_p", default=0.9, type=float)
    parser.add_argument("--lr_warmup_epochs", default=1000, type=int)
    

    parser.add_argument("--drop_edge_p_1", default=0.3, type=float)
    parser.add_argument("--drop_feat_p_1", default=0.3, type=float)
    parser.add_argument("--drop_edge_p_2", default=0.2, type=float)
    parser.add_argument("--drop_feat_p_2", default=0.4, type=float)

    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')
    parser.add_argument("--gpu_index", default=1, type=int)

    return parser