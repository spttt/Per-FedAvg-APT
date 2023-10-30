import rich
import torch
import apt_utils
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
from apt_data_utils import get_dataloader
from fedlab.utils.serialization import SerializationTool

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd


class PerFedAvgClient:
    def __init__(
        self,
        client_id: int,
        alpha: float,
        beta: float,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss], # criterion=torch.nn.CrossEntropyLoss(),
        # batch_size: int,
        dataset_root: str,
        clients_4_training_num: int,
        local_epochs: int,
        # valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
    ):
        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger = logger

        self.local_epochs = local_epochs
        self.criterion = criterion
        self.id = client_id
        self.model = deepcopy(global_model)
        self.alpha = alpha
        self.beta = beta
        self.trainloader, self.valloader = get_dataloader(client_id, clients_4_training_num, dataset_root)
        
        self.iter_trainloader = iter(self.trainloader)

    def train(
        self,
        global_epoch :int,
        save_log_root :str,
        global_model: torch.nn.Module,
        hessian_free=False,
        eval_while_training=False,
    ):
        self.model.load_state_dict(global_model.state_dict()) # 5.本地参数 = 全局参数 
        if eval_while_training: 
            loss_before, acc_before, macro_f1_before, report = apt_utils.eval(self.model, self.valloader, self.criterion, self.device)
        
        self._train(hessian_free) # 6.

        if eval_while_training:
            loss_after, acc_after, macro_f1_after, report = apt_utils.eval(self.model, self.valloader, self.criterion, self.device)
            self.logger.log(
                "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%   [blue]macro_f1: {:.2f} -> {:.2f}".format(
                    self.id,
                    loss_before, loss_after,
                    acc_before * 100.0, acc_after * 100.0,
                    macro_f1_before, macro_f1_after
                )
            )

        # 存logtrain
        # global_epoch, self.id, loss_before, loss_after, acc_before, acc_after, macro_f1_before, macro_f1_after
        df = pd.DataFrame([[global_epoch+1, self.id, loss_before.cpu().numpy(), loss_after.cpu().numpy(), acc_before.cpu().numpy(), acc_after.cpu().numpy(), macro_f1_before, macro_f1_after]])
        df.to_csv(save_log_root+'train_log.csv', mode='a', header=False, index=None)

        return SerializationTool.serialize_model(self.model) # 11.Agent i sends w^i_k+1, back to server;

    def _train(self, hessian_free=False):
        if hessian_free:  # Per-FedAvg(HF) # hessian_free = true # 复杂的梯度更新
            # 2 ）HF（ Hessian-vector MAML）：考虑使用一阶偏导的差（二阶导的定义）代替二阶偏导。
            for _ in range(self.local_epochs): # 4轮
                temp_model = deepcopy(self.model) # 克隆模型

                data_batch_1 = apt_utils.get_data_batch(self.trainloader, self.iter_trainloader, self.device) 
                # data_batch_1[0].size(),data_batch_1[1].size() # (torch.Size([40, 1, 28, 28]), torch.Size([40]))
                grads = self.compute_grad(temp_model, data_batch_1) # 7.
                for param, grad in zip(temp_model.parameters(), grads): # 8.更新cloned模型
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = apt_utils.get_data_batch(self.trainloader, self.iter_trainloader, self.device)
                grads_1st = self.compute_grad(temp_model, data_batch_2)
                data_batch_3 = apt_utils.get_data_batch(self.trainloader, self.iter_trainloader, self.device)
                grads_2nd = self.compute_grad(self.model, data_batch_3, v=grads_1st, second_order_grads=True)
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(self.model.parameters(), grads_1st, grads_2nd): # 9.更新本地模型
                    param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2)
            return

        else:  # Per-FedAvg(FO) # hessian_free = false # 看代码像是原始MAML的梯度更新
            # 1 ）FO（ First-Order MAML）：直接考虑使用一阶偏导代替二阶导；
            for _ in range(self.local_epochs):
                # ========================== FedAvg ==========================
                # NOTE: You can uncomment those codes for running FedAvg. 
                #       When you're trying to run FedAvg, comment other codes in this branch.

                # data_batch = utils.get_data_batch(
                #     self.trainloader, self.iter_trainloader, self.device
                # )
                # grads = self.compute_grad(self.model, data_batch) # 不拷贝模型，直接在原模型上算
                # for param, grad in zip(self.model.parameters(), grads):
                #     param.data.sub_(self.beta * grad)

                # ============================================================

                temp_model = deepcopy(self.model)
                data_batch_1 = apt_utils.get_data_batch(self.trainloader, self.iter_trainloader, self.device)
                grads = self.compute_grad(temp_model, data_batch_1)

                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = apt_utils.get_data_batch(self.trainloader, self.iter_trainloader, self.device)
                grads = self.compute_grad(temp_model, data_batch_2)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.beta * grad)

    def compute_grad(
        self,
        model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
    ):
        x, y = data_batch 
        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)  # 用的是新一批数据，论文中写的D''
            # loss_1 = self.criterion(logit_1, y) / y.size(-1)
            loss_1 = self.criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.criterion(logit_2, y)
            # loss_2 = self.criterion(logit_2, y) / y.size(-1)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params) # model没有变，只算出了grads

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads

        else:
            logit = model(x)
            # loss = self.criterion(logit, y) / y.size(-1)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads


    @torch.no_grad()
    def macro_f1_print(self):
        self.model.eval()
        for x, y in self.valloader:
            x, y = x.to(self.device), y.to(self.device)
            logit = self.model(x)
            pred = torch.softmax(logit, -1).argmax(-1)
            conf_mat = confusion_matrix(y.data.cpu().numpy(), pred.data.cpu().numpy())
        print(conf_mat)
        self.model.train()
        return


    def pers_N_eval(self, global_model: torch.nn.Module, pers_epochs: int, global_epoch: int, save_log_path: str, finetune_lr=None):
        self.model.load_state_dict(global_model.state_dict())

        loss_acc_f1=[]
        loss_before, acc_before, macro_f1_before, report = apt_utils.eval(self.model, self.valloader, self.criterion, self.device)
        loss_acc_f1.append([0, loss_before.data.cpu().numpy(), acc_before.data.cpu().numpy(), macro_f1_before, report])

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)

        if finetune_lr:
            params_notfc = [param for name, param in self.model.named_parameters() if name not in ["fc.weight", "fc.bias"]]
            optimizer = torch.optim.Adam([{'params': params_notfc}, {'params': self.model.fc.parameters(), 'lr': finetune_lr}], lr=0)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

        for epoch in range(pers_epochs):
            x, y = apt_utils.get_data_batch(self.trainloader, self.iter_trainloader, self.device)
            logit = self.model(x)
            # loss = self.criterion(logit, y) / y.size(-1)
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_after, acc_after, macro_f1_after, report = apt_utils.eval(self.model, self.valloader, self.criterion, self.device)
            loss_acc_f1.append([epoch+1, loss_after.data.cpu().numpy(), acc_after.data.cpu().numpy(), macro_f1_after, report])

        df = pd.DataFrame(loss_acc_f1, columns=['epoch', 'loss', 'acc', 'f1', 'report'])
        df['client_id'] = self.id
        df['global_epoch'] = global_epoch+1
        df.to_csv(save_log_path, mode='a', header=False, index=None)

        self.macro_f1_print()

        self.logger.log(
            "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%   [blue]macro_f1: {:.2f} -> {:.2f}".format(
                self.id,
                loss_before, loss_after,
                acc_before * 100.0, acc_after * 100.0,
                macro_f1_before, macro_f1_after
            )
        )
        return {
            "loss_before": loss_before,
            "acc_before": acc_before,
            "loss_after": loss_after,
            "acc_after": acc_after,
            "macro_f1_before": macro_f1_before,
            "macro_f1_after": macro_f1_after
        }


