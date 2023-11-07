import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def pcap_2_csv(pcap_path):
    csv_path = pcap_path+".csv"
    if not os.path.exists(csv_path):
        os.system(f"cicflowmeter -f {pcap_path} -c {csv_path}")
    return csv_path


def pcap_2_predict_df(pcap_path, model_path='/r2_fc_8967_0.004.jit.pt'):
    """
    pcap_path: pcap文件路径
    model_path: 模型路径
    return: 一个pd.DataFrame, 各列为["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp", "Stage"]
    """

    csv_path = pcap_2_csv(pcap_path)

    X_all = pd.read_csv(csv_path)
    X_all.columns = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Packet Length Max", "Packet Length Min", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "Fwd Header Length", "Bwd Header Length", "Fwd Seg Size Min", "Fwd Act Data Pkts", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Flow IAT Std", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Min", "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Total", "Bwd IAT Max", "Bwd IAT Min", "Bwd IAT Mean", "Bwd IAT Std", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "FWD Init Win Bytes", "Bwd Init Win Bytes", "Active Max", "Active Min", "Active Mean", "Active Std", "Idle Max", "Idle Min", "Idle Mean", "Idle Std", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bulk Rate Avg", "Fwd Segment Size Avg", "Bwd Segment Size Avg", "CWR Flag Count", "Subflow Fwd Packets", "Subflow Bwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Bytes"]
    order = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
    X_all = X_all[order]
    X_all_copy = X_all
    X_all = X_all.drop(labels=['Src IP', 'Src Port','Dst IP', 'Dst Port', 'Timestamp'], axis=1)
    X_all = X_all.drop(labels=['Fwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Count', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'FWD Init Win Bytes', 'Fwd Seg Size Min', ], 
                    axis=1)

    # 归一化
    scaler = MinMaxScaler()
    X_all_01 = scaler.fit_transform(X_all)
    X_all_01 = pd.DataFrame(X_all_01, columns=X_all.columns)

    X_all_01_float32 = torch.tensor(np.array(X_all_01), dtype=torch.float32)

    net = torch.jit.load(model_path)
    netout = net(X_all_01_float32)

    predictions = netout.argmax(dim=1) # .view(label.shape)
    data_array = predictions.numpy()
    label = pd.DataFrame({"Stage":data_array})
    X_all_copy_label = pd.concat([X_all_copy, label], axis=1)
    XY_all_output = X_all_copy_label[["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp", "Stage"]]
    
    return XY_all_output

