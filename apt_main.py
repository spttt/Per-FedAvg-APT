import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console
from rich.progress import track
from apt_utils import get_args_apt, fix_random_seed, save_args
from apt_model import get_model
from apt_perfedavg import PerFedAvgClient
from apt_data_utils import get_dataset_path


args = get_args_apt()
fix_random_seed(args.seed)
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("[!] device = torch.device(\"cpu\")")

args_dataset_root = get_dataset_path(args.dataset)

if args.save_model_root is None:
    import time
    args.save_model_root = "log/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "/"
    os.makedirs(args.save_model_root)
save_args(args)

global_model = get_model(args.model, device)
if args.global_epochs_begin > 0:
    state_dict_root = os.path.join(args.save_model_root, f'epoch{args.global_epochs_begin}.pt')
    global_model.load_state_dict(torch.load(state_dict_root))
    print("[print] load_state_dict: ", state_dict_root)
print("[print] global_model device: ", next(global_model.parameters()).device)

logger = Console(record=args.log)

# logger.log(f"Arguments:", dict(args._get_kwargs()))

clients_4_training_num = 400
clients_4_testing_num = 50 if args.dataset == "CICIDS2017" else 10
clients_4_training, clients_4_eval, client_num_in_total = list(range(clients_4_training_num)), list(range(
    clients_4_training_num, clients_4_training_num+clients_4_testing_num)), clients_4_training_num+clients_4_testing_num



# init clients
print("[print] init clients ing...")
clients = [
    PerFedAvgClient(
        client_id=client_id,  # 每次client_id参数改变，遍历0到client_num_in_total-1
        alpha=args.alpha,
        beta=args.beta,
        global_model=global_model,
        criterion=torch.nn.CrossEntropyLoss(),
        # batch_size=args_batch_size, #####
        dataset_root=args_dataset_root,
        clients_4_training_num=clients_4_training_num,
        local_epochs=args.local_epochs,
        # valset_ratio=args_valset_ratio, #####
        logger=logger,
        gpu=args.gpu,
    )
    for client_id in range(client_num_in_total)
]


def eval_fine_tune(global_epoch, final_personalise=False):
    logger.log("=" * 20, "EVALUATION", "=" * 20, style="bold blue")
    loss_before = []
    loss_after = []
    acc_before = []
    acc_after = []
    macro_f1_before = []
    macro_f1_after = []

    save_log_path = os.path.join(args.save_model_root, "test_log.csv")
    if(final_personalise):
        save_log_path = os.path.join(args.save_model_root, f"test_log_{args.global_epochs_begin}_finetune_lr_{args.finetune_lr}_alpha_{args.alpha}.csv")

    for client_id in track(clients_4_eval, "Evaluating...", console=logger, disable=args.log):  # 10个测试clint
        stats = clients[client_id].pers_N_eval(
            global_model=global_model, 
            pers_epochs=args.pers_epochs, 
            global_epoch=global_epoch, 
            save_log_path=save_log_path,
            finetune_lr=args.finetune_lr,
        )
        loss_before.append(stats["loss_before"])
        loss_after.append(stats["loss_after"])
        acc_before.append(stats["acc_before"])
        acc_after.append(stats["acc_after"])
        macro_f1_before.append(stats["macro_f1_before"])
        macro_f1_after.append(stats["macro_f1_after"])

    logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
    logger.log(f"loss_before_pers: {(sum(loss_before) / len(loss_before)):.4f}")
    logger.log(f"loss_after_pers: {(sum(loss_after) / len(loss_after)):.4f}")

    logger.log(f"acc_before_pers: {(sum(acc_before) * 100.0 / len(acc_before)):.2f}%")
    logger.log(f"acc_after_pers: {(sum(acc_after) * 100.0 / len(acc_after)):.2f}%")

    logger.log(f"macro_f1_before_pers: {(sum(macro_f1_before) / len(macro_f1_before)):.2f}")
    logger.log(f"macro_f1_after_pers: {(sum(macro_f1_after) / len(macro_f1_after)):.2f}")


def training():
    logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red")
    for epoch in track(range(args.global_epochs_begin, args.global_epochs_end), "Training...", console=logger, disable=args.log):  # 1-14步. 200轮
        # selected_clients是list  2.选A_k
        selected_clients = random.sample(
            clients_4_training, args.client_num_per_round)

        model_params_cache = []
        # client local training
        for client_id in selected_clients:  # 4-12. for all i \in A_k do
            serialized_model_params = clients[client_id].train(
                global_model=global_model,  # 3.Server sends w_k to all users in A_k
                hessian_free=args.hf,
                eval_while_training=args.eval_while_training,
                global_epoch=epoch, 
                save_log_root=args.save_model_root
            )
            model_params_cache.append(serialized_model_params)  # 11.

        # aggregate model parameters
        aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)  # 13. 算10个clinet返回的参数均值
        SerializationTool.deserialize_model(global_model, aggregated_model_params)  # 更新global_model
        logger.log("=" * 60, 'epoch ', epoch, ' done')

        if ((epoch+1) % 200 == 0):
            torch.save(global_model.state_dict(), args.save_model_root + 'epoch{}.pt'.format(epoch+1))

        if ((epoch+1) % args.pers_freq == 0):
            eval_fine_tune(epoch)

        print("next epoch :{}".format(epoch+1))


training()
if(args.global_epochs_begin == args.global_epochs_end):
    eval_fine_tune(args.global_epochs_end, final_personalise=True)
