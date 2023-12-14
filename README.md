# Privacy-preserving Few-shot Traffic Detection against Advanced Persistent Threats via Federated Meta Learning [[TNSE 2023]](https://ieeexplore.ieee.org/document/10214668)

PyTorch implementation of the paper "Privacy-preserving Few-shot Traffic Detection against Advanced Persistent Threats via Federated Meta Learning [[TNSE 2023]](https://ieeexplore.ieee.org/document/10214668)"

Code is modified from KarhouTam's [Per-FedAvg](https://github.com/KarhouTam/Per-FedAvg) project.


## Requirements
Some major dependent library versions:
```
torch~=1.13.1
numpy~=1.21.5
fedlab~=1.2.1
rich~=12.6.0
scikit-learn~=1.0.2
pandas~=1.3.5
```
## Preprocess dataset

### Download dataset
CICIDS-2017 download [link](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/). Download GeneratedLabelledFlows.zip and place it in the directory "data_raw/CICIDS2017".

DAPT2020 download [link](https://gitlab.com/asu22/dapt2020). The directory structure is "data_raw/dapt2020/csv/enp0s3-monday-pvt.pcap_Flow.csv".

### Preprocessing dataset

Run [data_prepare/load_CICIDS2017.ipynb](data_prepare/load_CICIDS2017.ipynb) and [data_prepare/load_dapt2020.ipynb](data_prepare/load_dapt2020.ipynb) to load the dataset and divide the data by client.


## Run the experiment

Run FML on the CICIDS2017 dataset using the MLP classification model to obtain optimal model initialization parameters:
```
python apt_main.py --dataset CICIDS2017 --model MLP
// python apt_main.py --dataset CICIDS2017 --model resnet
// python apt_main.py --dataset dapt2020 --model MLP
// python apt_main.py --dataset dapt2020 --model resnet
```
Load the initialization parameters and run the client personalization step:
```
// Meta init, normal training
python apt_main.py --dataset CICIDS2017 --model MLP --save_model_root [Training log path] --global_epochs_begin 2000 --global_epochs_end 2000 --alpha 0.01

// Default init, normal training
python apt_main.py --dataset CICIDS2017 --model MLP --save_model_root [Training log path] --global_epochs_begin 0 --global_epochs_end 0 --alpha 0.01

// Meta init, fine-tuning
python apt_main.py --dataset CICIDS2017 --model MLP --save_model_root [Training log path] --global_epochs_begin 2000 --global_epochs_end 2000 --finetune_lr 0.1

```

Fill in the corresponding log path in [log/plt_log_comparison.ipynb](log/plt_log_comparison.ipynb), and run the notebook to compare different training methods.


## Citation

```
@article{hu2023privacy,
  title={Privacy-preserving Few-shot Traffic Detection against Advanced Persistent Threats via Federated Meta Learning},
  author={Hu, Yilun and Wu, Jun and Li, Gaolei and Li, Jianhua and Cheng, Jinke},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2023},
  publisher={IEEE}
}
```