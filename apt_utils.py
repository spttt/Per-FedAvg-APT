import torch
import random
import numpy as np
from typing import Iterator, Tuple, Union
from argparse import ArgumentParser
from sklearn.metrics import classification_report
import os
import json
import time

def get_args_apt():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="Non-zero value for using gpu, 0 for using cpu",
    )
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument(
        "--hf",
        type=int,
        default=1,
        help="0 for performing Per-FedAvg(FO), others for Per-FedAvg(HF)",
    )
    parser.add_argument("--client_num_per_round", type=int, default=10) # 一轮选10个
    parser.add_argument(
        "--eval_while_training",
        type=int,
        default=1,
        help="Non-zero value for performing local evaluation before and after local training",
    )

    
    parser.add_argument("--alpha", type=float, default=1e-2) # 本地学习率
    parser.add_argument("--beta", type=float, default=1e-2) # 全局学习率
    parser.add_argument("--global_epochs_begin", type=int, default=0)
    parser.add_argument("--global_epochs_end", type=int, default=2000)
    parser.add_argument("--local_epochs", type=int, default=4) # 训练客户端本地更新
    parser.add_argument( # 测试客户端本地更新
        "--pers_epochs",
        type=int,
        default=100,
        help="Indicate how many data batches would be used for personalization. Negatives means that equal to train phase.",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["CICIDS2017", "dapt2020"], default="CICIDS2017"
    )
    parser.add_argument(
        "--model", type=str, choices=["MLP", "resnet"], default="MLP"
    )
    parser.add_argument("--save_model_root", type=str, default=None)
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument(
        "--pers_freq",
        type=int,
        default=10000,
        help="Indicate how many global epochs would be performed before personalization.",
    )


    return parser.parse_args()



@torch.no_grad()
def eval(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Union[torch.nn.MSELoss, torch.nn.CrossEntropyLoss],
    device=torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0
    num_samples = 0
    acc = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        # total_loss += criterion(logit, y) / y.size(-1)
        total_loss += criterion(logit, y)
        pred = torch.softmax(logit, -1).argmax(-1)
        acc += torch.eq(pred, y).int().sum()
        num_samples += y.size(-1)

        import warnings
        warnings.filterwarnings('ignore')
        report = classification_report(y.cpu(), pred.cpu(), output_dict=True)
    model.train()
    return total_loss, acc / num_samples, report['macro avg']['f1-score'], report


def get_data_batch(
    dataloader: torch.utils.data.DataLoader,
    iterator: Iterator,
    device=torch.device("cpu"),
):
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        x, y = next(iterator)

    return x.to(device), y.to(device)


def fix_random_seed(seed: int):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_args(args):
    args_save_name = "args" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".json"
    with open(os.path.join(args.save_model_root, args_save_name), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
