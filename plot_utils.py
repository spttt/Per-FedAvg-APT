import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["font.family"] = "Times New Roman"


def groupby_epoch100_plt(test_log_in, label, strlist):

    log_global_epoch_x_acc = test_log_in.groupby('epoch')['acc'].mean()
    log_global_epoch_x_f1 = test_log_in.groupby('epoch')['f1'].mean()

    epoch = log_global_epoch_x_acc.size - 1
    for str_name in strlist:
        plt.plot(range(epoch+1), test_log_in.groupby('epoch')
                 [str_name].mean(), label=label)  # label=label+str_name)

    print(f'acc:{log_global_epoch_x_acc[epoch]:.4f}, f1:{log_global_epoch_x_f1[epoch]:.4f}, ({label})')
    # print(f'{log_global_epoch_x_acc[epoch]:.4f}, {log_global_epoch_x_f1[epoch]:.4f}',end=', ')


def read_csv_and_rename_columns(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['epoch', 'loss', 'acc', 'f1',
                  'report', 'client_id', 'global_epoch']

    report = [eval(di) for di in df['report']]
    report_f1_score0 = [report_i['0']['f1-score'] for report_i in report]
    report_f1_score1 = [report_i['1']['f1-score'] for report_i in report]
    report_f1_score2 = [report_i['2']['f1-score'] for report_i in report]
    report_f1_score3 = [report_i['3']['f1-score'] for report_i in report]
    report_f1_score4 = [report_i['4']['f1-score'] for report_i in report]
    f1_0 = pd.DataFrame(report_f1_score0, columns=['f1_0'])
    f1_1 = pd.DataFrame(report_f1_score1, columns=['f1_1'])
    f1_2 = pd.DataFrame(report_f1_score2, columns=['f1_2'])
    f1_3 = pd.DataFrame(report_f1_score3, columns=['f1_3'])
    f1_4 = pd.DataFrame(report_f1_score4, columns=['f1_4'])
    pd_f101234 = pd.concat([df, f1_0, f1_1, f1_2, f1_3, f1_4], axis=1)
    pd_f101234 = pd_f101234.drop(labels=['report'], axis=1)

    return pd_f101234


def name_or_path_read_and_separate_CICIDS2017(log_path, name_or_path):
    csv_path = name_or_path if (os.path.exists(name_or_path)) else os.path.join(log_path, name_or_path)
    test_log = read_csv_and_rename_columns(csv_path)
    unknown = []
    unknown.append(test_log.loc[(test_log['client_id'] >= 400) & (test_log['client_id'] < 410)])
    unknown.append(test_log.loc[(test_log['client_id'] >= 410) & (test_log['client_id'] < 420)])
    unknown.append(test_log.loc[(test_log['client_id'] >= 420) & (test_log['client_id'] < 430)])
    unknown.append(test_log.loc[(test_log['client_id'] >= 430) & (test_log['client_id'] < 440)])
    unknown.append(test_log.loc[(test_log['client_id'] >= 440) & (test_log['client_id'] < 450)])
    return unknown


def plot5figure_CICIDS2017(log_path=None, csv_list=[], strlist=['acc'], unknow_nums=[0, 1, 2, 3, 4],
                           title=None,
                           csv_legend_list=[],
                           ylim=(0.21, 0.95)
                           ):

    csv_pd_list = [name_or_path_read_and_separate_CICIDS2017(log_path, name_or_path) for name_or_path in csv_list]

    sub_fig_num = len(unknow_nums)
    if sub_fig_num > 1:
        plt.figure(figsize=(6*len(unknow_nums), 5))

    for i, unknow_num in enumerate(unknow_nums):
        if sub_fig_num > 1:
            plt.subplot(1, len(unknow_nums), i+1)
        print(f"\nunknow_num = {unknow_num}")
        for csv_pd_index, unknow_num01234 in enumerate(csv_pd_list):
            groupby_epoch100_plt(unknow_num01234[i], csv_legend_list[csv_pd_index], strlist)

        plt.ylim(ylim)
        plt.legend(loc='lower right')
        plt.xlabel('local epoch (n)')
        plt.ylabel('Acc')
        plt.title(title if (title)
                  else f'Meta test (unknow_class_num={unknow_num})')
        # plt.title(f'Meta test on the test client after federated meta-learning (unknow_class_num={unknow_num})')


def plot1figure_dapt2020(log_path, D_NT=None, M_NT=None, M_FT=None, strlist=['acc']):

    test_log_D_NT = read_csv_and_rename_columns(os.path.join(log_path, D_NT))
    test_log_M_NT = read_csv_and_rename_columns(os.path.join(log_path, M_NT))
    test_log_M_FT = read_csv_and_rename_columns(os.path.join(log_path, M_FT))

    plt.subplots()
    groupby_epoch100_plt(test_log_M_NT, 'Meta init, normal training ', strlist)
    groupby_epoch100_plt(test_log_D_NT, 'Default init, normal training ', strlist)
    groupby_epoch100_plt(test_log_M_FT, 'Meta init, fine-tuning ', strlist)

    plt.legend()
    plt.xlabel('local epoch (n)')
    plt.ylabel('Acc')
    plt.title(f'Meta test (Classification of five attack stages)')


def groupby_global_epoch2000_plt(test_log_loc_epoch_100, name):

    test_log_loc_epoch_100_while2000_acc = test_log_loc_epoch_100.groupby('global_epoch')['acc'].mean()
    test_log_loc_epoch_100_while2000_f1 = test_log_loc_epoch_100.groupby('global_epoch')['f1'].mean()

    epoch = test_log_loc_epoch_100_while2000_acc.index.max()
    plt.plot(test_log_loc_epoch_100_while2000_acc.index, test_log_loc_epoch_100_while2000_acc)
    plt.plot(test_log_loc_epoch_100_while2000_f1.index, test_log_loc_epoch_100_while2000_f1)

    plt.legend(labels=['Acc', 'F1'], loc='lower right')
    plt.xlabel('global epoch (n)')
    plt.ylabel('acc and f1')
    plt.title(name)
