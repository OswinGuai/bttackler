from bttackler.bridger.optuna_pruner import BTTPruner
import optuna
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from bttackler.api.btwatcher import BTWatcher

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

seed = 529
params = {
    "conv1_k_num": 8,
    "conv2_k_num": 32,
    "conv3_k_num": 64,
    "conv4_k_num": 64,
    "mlp_f_num": 200,
    "conv_k_size": 2,
    "lr": 0.1,
    "weight_decay": 0.1,
    "act": 0,
    "opt": 0,
    "drop_rate": 0.5,
    "batch_norm": 1,
    "batch_size": 90
}
manager = None


def set_seed():
    print("seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def choose_act():
    act_func = params["act"]
    if act_func == 0:
        return nn.ReLU()
    elif act_func == 1:
        return nn.Tanh()
    elif act_func == 2:
        return nn.Sigmoid()
    elif act_func == 3:
        return nn.ELU()
    elif act_func == 4:
        return nn.LeakyReLU()


def choose_opt():
    opt_func = params["opt"]
    if opt_func == 0:
        return optim.SGD
    elif opt_func == 1:
        return optim.Adam
    elif opt_func == 2:
        return optim.RMSprop
    elif opt_func == 3:
        return optim.Adagrad
    elif opt_func == 4:
        return optim.Adadelta


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, params["conv1_k_num"], params["conv_k_size"])
        self.conv2 = nn.Conv2d(params["conv1_k_num"], params["conv2_k_num"], params["conv_k_size"])
        self.conv3 = nn.Conv2d(params["conv2_k_num"], params["conv3_k_num"], params["conv_k_size"])
        self.conv4 = nn.Conv2d(params["conv3_k_num"], params["conv4_k_num"], params["conv_k_size"])
        # an affine operation: y = Wx + b
        self.feature_num = self.num_flat_features_()
        self.fc1 = nn.Linear(self.feature_num, params["mlp_f_num"])  # 5*5 from image dimension
        self.fc2 = nn.Linear(params["mlp_f_num"], params["mlp_f_num"])
        self.fc3 = nn.Linear(params["mlp_f_num"], 10)
        self.drop = nn.Dropout(params["drop_rate"])
        self.act = choose_act()
        self.bn1 = nn.BatchNorm2d(params["conv1_k_num"])
        self.bn2 = nn.BatchNorm2d(params["conv2_k_num"])
        self.bn3 = nn.BatchNorm2d(params["conv3_k_num"])
        self.bn4 = nn.BatchNorm2d(params["conv4_k_num"])

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.bn1(x) if params["batch_norm"] == 1 else x
        x = F.max_pool2d(self.act(self.conv2(x)), 2)
        x = self.bn2(x) if params["batch_norm"] == 1 else x
        x = self.act(self.conv3(x))
        x = self.bn3(x) if params["batch_norm"] == 1 else x
        x = F.max_pool2d(self.act(self.conv4(x)), 2)
        x = self.bn4(x) if params["batch_norm"] == 1 else x
        x = x.view(-1, self.feature_num)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

    def num_flat_features_(self):
        r = 32  # cifar10
        r = r - params["conv_k_size"] + 1
        r = r - params["conv_k_size"] + 1
        r = (r - 2) // 2 + 1
        r = r - params["conv_k_size"] + 1
        r = r - params["conv_k_size"] + 1
        r = (r - 2) // 2 + 1
        return r * r * params["conv4_k_num"]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct = 0
    loss_sum = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_sum += float(loss.item())
        optimizer.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()
        manager.collect_per_batch(model)
        optimizer.step()
    acc = correct / size
    loss_ave = loss_sum / num_batches
    return acc, loss_ave


def standard_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    acc = correct / size
    return loss, acc


def test(dataloader, model, loss_fn):
    loss, acc = standard_test(dataloader, model, loss_fn)
    manager.collect_after_testing(acc, loss)
    return acc, loss


def validate(dataloader, model, loss_fn):
    loss, acc = standard_test(dataloader, model, loss_fn)
    return acc, loss


def make_trial(manager, params, trial, max_epoch):
    print("params: ", params)
    train_kwargs = {'batch_size': params["batch_size"]}
    test_kwargs = {'batch_size': params["batch_size"]}
    if device == "cuda":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_data, validate_data = torch.utils.data.random_split(train_data, [40000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = CNN().to(device)

    optimizer = choose_opt()(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss()
    epochs = max_epoch

    manager.init_basic(model, train_dataloader)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        manager.refresh_before_epoch_start()

        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        manager.collect_after_training(train_acc, train_loss)
        valid_acc, valid_loss = validate(validate_dataloader, model, loss_fn)
        manager.collect_after_validating(valid_acc, valid_loss)
        manager.report_intermediate_result(trial.number)
        scheduler.step()

        if trial.should_prune():
            valid_acc, _ = validate(validate_dataloader, model, loss_fn)
            manager.report_final_result(trial.number)
            print(f"pruning!!! ")
            raise optuna.TrialPruned()
    print(f"Finish one trial.")
    valid_acc, _ = validate(validate_dataloader, model, loss_fn)
    manager.report_final_result(trial.number)
    return valid_acc


if __name__ == '__main__':
    set_seed()

    sampler = optuna.samplers.TPESampler(seed=1)
    diagnose = {
        'p_eg1': 10000,
        'p_eg2': 1000,
        'p_vg1': 0.00001,
        'p_vg2': 0.0001,
        'p_dr1': 0.5,
        'p_sc1': 0.0001,
        'p_sc2': 0.5,
        'p_ho1': 0.01,
        'p_ho2': 0.2,
        'p_of': 0,
        'wd_ho': 6,
        'wd_nmg': 6
    }
    max_epoch = 2
      
    pruner = BTTPruner(
        max_epoch=max_epoch,
        quick_calc=True,
        symptom_name_list=["accuracy", "loss"],
        cmp_percent=0.1,
        min_cmp_num=5,
        diagnose=diagnose,
        seed=1,
    )
    study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="minimize", sampler=sampler, pruner=pruner)
    study.enqueue_trial({"T0": 1.0, "alpha": 2.0, "patience": 50})  # default params
    manager = BTWatcher(max_epoch=max_epoch, quick_calc=True, intermediate_default='val_acc', final_default='val_acc', seed=seed)
    def objective(trial):
        global count
        count += 1
        print('make a trial %s' % trial.number)
        params = {
            "conv1_k_num": trial.suggest_int("conv1_k_num", 8, 32),
            "conv2_k_num": trial.suggest_int("conv2_k_num", 32, 64),
            "conv3_k_num": trial.suggest_int("conv3_k_num", 64, 128),
            "conv4_k_num": trial.suggest_int("conv4_k_num", 64, 128),
            "mlp_f_num": trial.suggest_int("mlp_f_num", 10, 1000),
            "conv_k_size": trial.suggest_categorical("conv_k_size", [2, 3]),
            "lr": trial.suggest_loguniform("lr", 0.0001, 0.4),
            "weight_decay": trial.suggest_loguniform("weight_decay", 0.00001, 0.001),
            "act": trial.suggest_categorical("act", [0, 1, 2, 3, 4]),
            "opt": trial.suggest_categorical("opt", [0, 1, 2, 3, 4]),
            "batch_size": trial.suggest_categorical("batch_size", [45, 90, 180, 360, 720]),
            "batch_norm": trial.suggest_categorical("batch_norm", [0, 1]),
            "drop_rate": trial.suggest_uniform("drop_rate", 0.0, 0.9)
        }
        return make_trial(manager, params, trial, max_epoch)

    N_TRIALS = 2
    count = 0
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"The number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value} (params: {study.best_params})")




