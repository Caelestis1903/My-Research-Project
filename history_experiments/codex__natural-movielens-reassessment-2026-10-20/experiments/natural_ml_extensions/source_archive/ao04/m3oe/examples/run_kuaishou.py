import sys
import os

sys.path.append("..")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import random, uuid
from MDMTRec.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from MDMTRec.trainers import CTRTrainer
from MDMTRec.utils.data import DataGenerator
from MDMTRec.models.multi_domain import *
import warnings
from collections import Counter

# Filter UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_kuaishou_data_rank_multidomain(task, domain, data_path="D:/CODE/data/kuaishou"):
    data = pd.read_csv(os.path.join(data_path, "kuai_rand.csv"))

    # 修复：x_used_cols 只包含 CSV 中真实存在的原始列（不含标签列和派生列 domain_indicator），避免后续索引错位
    x_used_cols = ['tab', 'user_active_degree', 'is_live_streamer', 'is_video_author',
                   'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range',
                   'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4', 'onehot_feat5',
                   'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
                   'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17',
                   'video_type', 'upload_dt', 'upload_type', 'server_width', 'server_height', 'music_type', 'tag',
                   'counts']

    tab = [0, 1, 4]
    domain_indicator = {'0': 0, '1': 1, '4': 2}
    domain_num = 3

    # 只保留 x_used_cols 中的列
    data = data[x_used_cols + ['is_click', 'long_view']]

    mask_ = data['tab'].apply(lambda x: x in tab)
    data = data[mask_]

    data["domain_indicator"] = data["tab"].apply(lambda x: domain_indicator[str(x)])
    del data['tab']

    tab_c = Counter(data['domain_indicator'])
    print(tab_c)
    # choose domain: 0, 1, 2, all
    if domain != 'all':
        data = data[data['domain_indicator'] == int(domain)]
        assert len(data) != 0, 'wrong domain indicator'
    else:
        data = data

    useless_features = []
    dense_features = []
    sparse_features = ['user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                       'fans_user_num_range', 'friend_user_num_range', 'register_days_range', 'onehot_feat0',
                       'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
                       'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
                       'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16',
                       'onehot_feat17', 'video_type', 'upload_dt', 'upload_type', 'server_width', 'server_height',
                       'music_type', 'tag', 'counts'] + ['domain_indicator']

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])

    for feature in useless_features:
        del data[feature]
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1

    # 修复：domain_indicator 重新编码为从 0 开始的连续值。
    # 训练器按 domain_id == d (d ∈ [0, domain_num)) 做掩码，+1 后的 1/2/3 会导致域 0 为空。
    # 单域运行时也能保证唯一取值被映射为 0。
    data['domain_indicator'] = LabelEncoder().fit_transform(data['domain_indicator'])

    dense_feas = [DenseFeature(x_used_cols.index(feature_name)) for feature_name in dense_features]
    # 修复：vocab_size +1 避免 LabelEncoder+1 后的最大索引越界
    sparse_feas = [
        SparseFeature(sparse_features.index(feature_name), vocab_size=data[feature_name].nunique() + 1, embed_dim=16) for
        feature_name in sparse_features]

    # task: click, long_view, all
    data.rename(columns={'is_click': 'click', 'long_view': 'long'}, inplace=True)
    if task == 'all':
        y = data[['click', 'long']]
    else:
        y = data[task]
    del data['click']
    del data['long']

    # 修复：删除标签列后，确保 data 的列顺序与 sparse_features 严格一致
    # （特征名即列索引，'tab' 已删除、'domain_indicator' 为派生列）
    data = data[sparse_features]

    x, y = data.values, y.values

    if task != 'all':
        y = np.expand_dims(y, axis=1)

    train_idx, val_idx = int(len(data) * 0.8), int(len(data) * 0.9)
    x_train, y_train = x[:train_idx, :], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx, :], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:, :], y[val_idx:]
    print(f'train: {(x_train.shape)}, {(y_train.shape)}')
    print(f'val: {(x_val.shape)}, {(y_val.shape)}')
    print(f'test: {(x_test.shape)}, {(y_test.shape)}')
    return dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num


def convert_target(val, target):
    v = int(val)
    if target == 'click':
        thre = 3
    elif target == 'like':
        thre = 4
    else:
        assert 0, 'wrong target'
    if v > thre:
        return int(1)
    else:
        return int(0)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(args.seed)
    dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num = get_kuaishou_data_rank_multidomain(
        args.task, args.domain, args.dataset_path)
    if args.domain != 'all':
        domain_num = 1
    task_num = 2 if args.task == 'all' else 1
    fcn_dims = [512, 256, 256, 64]

    dg = DataGenerator(x_train, y_train)
    # Windows spawn cannot reliably pickle the ~700MB dataset object to
    # num_workers>0 child processes (OSError Errno 22 during reduction.dump).
    # num_workers=0 loads in-process: deterministic and fast enough here.
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test,
                                                                               y_test=y_test,
                                                                               batch_size=args.batch_size,
                                                                               num_workers=0)

    model = MDMTRec(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims,
                    expert_num=args.expert_num, exp_d=args.exp_d, exp_t=args.exp_t, bal_d=args.bal_d, bal_t=args.bal_t)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {trainable_params}')

    ctr_trainer = CTRTrainer(
        model,
        optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        optimizer_params_darts={"lr": args.learning_rate_darts, "weight_decay": args.weight_decay},
        n_epoch=args.epoch,
        earlystop_patience=10,
        device=args.device,
        model_path=args.save_dir + str(uuid.uuid4()),
        scheduler_params={"step_size": 4, "gamma": 0.85}
    )

    ctr_trainer.fit(
        train_dataloader,
        val_dataloader=val_dataloader,
        exp_d=args.exp_d,
        exp_t=args.exp_t,
        bal_d=args.bal_d,
        bal_t=args.bal_t,
        domain_num=domain_num,
        task_num=task_num
    )

    logloss_list, auc_list, total_logloss, total_auc = ctr_trainer.evaluate_multi_domain(
        ctr_trainer.model, test_dataloader, domain_num, task_num=task_num)

    print(f'test auc: {total_auc} | test logloss: {total_logloss}')
    assert len(logloss_list) == domain_num * task_num
    col_name_ = list()
    auc_row_, log_row_ = list(), list()
    for d in range(domain_num):
        for t in range(task_num):
            auc_ = auc_list[d + t * domain_num]
            log_ = logloss_list[d + t * domain_num]
            print(f'domain: {d}, task: {t}, auc: {auc_} | test logloss: {log_}')
            col_name_.append(f'd{d}t{t}')
            auc_row_.append(auc_)
            log_row_.append(log_)
    col_name_ = [f'auc_{i}' for i in col_name_] + [f'log_{i}' for i in col_name_] + [
        'auc', 'log', 'seed', 'exp_d', 'exp_t', 'bal_d', 'bal_t']

    import csv
    file_name_ = f'output/{args.model_name}_kuaishou_domain_{args.domain}_task_{args.task}_exp_{args.expert_num}_{args.learning_rate}.csv'
    print_round_ = 4
    file_exists_ = os.path.isfile(file_name_)
    mode_ = 'a' if file_exists_ else 'w'
    with open(file_name_, mode_) as f:
        writer = csv.writer(f)
        if not file_exists_:
            writer.writerow(col_name_)
        writer.writerow([round(auc_, print_round_) for auc_ in auc_row_] +
                        [round(log_, print_round_) for log_ in log_row_] +
                        [round(total_auc, print_round_), round(total_logloss, print_round_),
                         args.seed, args.exp_d, args.exp_t, args.bal_d, args.bal_t]
                       )
    print(f'wrote to file: {file_name_}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=r"D:\CODE\data\kuaishou")
    parser.add_argument('--model_name', default='MDMTRec')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_darts', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096 * 10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./model_para/')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--expert_num', type=int, default=4)
    parser.add_argument('--exp_d', type=float, default=1, help='weight of domain expert')
    parser.add_argument('--exp_t', type=float, default=1, help='weight of task expert')
    parser.add_argument('--bal_d', type=float, default=1, help='weight of domain expert d, others are 1-bal_d')
    parser.add_argument('--bal_t', type=float, default=1, help='weight of task expert t, others are 1-bal_t')
    parser.add_argument('--task', type=str, default='all', choices=['click', 'long', 'all'])
    parser.add_argument('--domain', type=str, default='all', choices=['0', '1', '2', 'all'])

    args = parser.parse_args()
    print(args)
    main(args)