import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize


def fit_logistic_regression(X, y, data_random_seed=1, repeat=1):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    accuracies = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    print(np.mean(accuracies))
    return accuracies

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss

def evaluate_node(X, y, dataset, name, repeat=1, device=None):

    emb_dim, num_class = X.shape[1], y.unique().shape[0]
    dev_accs, test_accs = [], []
    for i in range(repeat):
        # different random split after each repeat
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        train_mask = dataset.train_mask[i]
        dev_mask = dataset.val_mask[i]
        test_mask = dataset.test_mask[i]

        classifier = LogisticRegression(emb_dim, num_class)
        classifier = classifier.to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

        for _ in range(100):
            classifier.train()
            logits, loss = classifier(X[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        dev_logits, _ = classifier(X[dev_mask], y[dev_mask])
        test_logits, _ = classifier(X[test_mask], y[test_mask])
        dev_preds = torch.argmax(dev_logits, dim=1)
        test_preds = torch.argmax(test_logits, dim=1)

        dev_acc = (torch.sum(dev_preds == y[dev_mask]).float() /
                    y[dev_mask].shape[0]).detach().cpu().numpy()
        test_acc = (torch.sum(test_preds == y[test_mask]).float() / 
                    y[test_mask].shape[0]).detach().cpu().numpy()  

        dev_accs.append(dev_acc)
        test_accs.append(test_acc)

    dev_accs = np.stack(dev_accs)
    test_accs = np.stack(test_accs)

    dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
    test_acc, test_std = test_accs.mean(), test_accs.std()

    # print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(dev_acc, dev_std, test_acc, test_std))

    # dev_accs = np.stack(dev_accs)
    # test_accs = np.stack(test_accs)
    # print(dev_accs)
    # indexs = np.argmax(dev_accs, axis=1)
    # accuracies = [test_accs[i][val] for i, val in enumerate(indexs)]
    # test_accs[:,indexs]
    return dev_acc, dev_std, test_acc, test_std

def evaluate_node_wikics(X, y, repeat, train_masks, val_masks, test_mask, device=None):

    emb_dim, num_class = X.shape[1], y.unique().shape[0]
    dev_accs, test_accs = [], []
    for i in range(repeat):

        train_mask = train_masks[:, i]
        dev_mask = val_masks[:, i]
        test_mask = test_mask

        classifier = LogisticRegression(emb_dim, num_class)
        classifier = classifier.to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

        for _ in range(100):
            classifier.train()
            logits, loss = classifier(X[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        dev_logits, _ = classifier(X[dev_mask], y[dev_mask])
        test_logits, _ = classifier(X[test_mask], y[test_mask])
        dev_preds = torch.argmax(dev_logits, dim=1)
        test_preds = torch.argmax(test_logits, dim=1)

        dev_acc = (torch.sum(dev_preds == y[dev_mask]).float() /
                    y[dev_mask].shape[0]).detach().cpu().numpy()
        test_acc = (torch.sum(test_preds == y[test_mask]).float() / 
                    y[test_mask].shape[0]).detach().cpu().numpy()  

        dev_accs.append(dev_acc)
        test_accs.append(test_acc)

    dev_accs = np.stack(dev_accs)
    test_accs = np.stack(test_accs)

    dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
    test_acc, test_std = test_accs.mean(), test_accs.std()

    # print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(dev_acc, dev_std, test_acc, test_std))

    # dev_accs = np.stack(dev_accs)
    # test_accs = np.stack(test_accs)
    # print(dev_accs)
    # indexs = np.argmax(dev_accs, axis=1)
    # accuracies = [test_accs[i][val] for i, val in enumerate(indexs)]
    # test_accs[:,indexs]
    return dev_acc, dev_std, test_acc, test_std