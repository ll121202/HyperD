import numpy as np
import json
import argparse

def statistical_prior_initialization(dataset_name, num_nodes, shape, daily_len, weekly_len, train_val_test_ratio):
    data = np.memmap(data_file_path, dtype='float32', mode='r', shape=shape)

    total_len = len(data)
    valid_len = int(total_len * train_val_test_ratio[1])
    test_len = int(total_len * train_val_test_ratio[2])
    train_len = total_len - valid_len - test_len

    train_data = data[:train_len]
    flow_data = train_data[:, :, 0]  # (L, N)
    train_mean = np.mean(flow_data)
    train_std = np.std(flow_data)
    flow_data = (flow_data - train_mean) / train_std

    time_of_day = train_data[:, 0, 1] * daily_len
    time_of_week = train_data[:, 0, 1] * daily_len + train_data[:, 0, 2] * weekly_len

    daily_init = np.zeros((daily_len, num_nodes))
    for t in range(daily_len):
        idx = (time_of_day == t)
        daily_init[t] = flow_data[idx].mean(axis=0)
    np.save(f'datasets/{dataset_name}/daily_init.npy', daily_init)

    weekly_init = np.zeros((weekly_len, num_nodes))
    for t in range(weekly_len):
        idx = (time_of_week == t)
        weekly_init[t] = flow_data[idx].mean(axis=0)
    np.save(f'datasets/{dataset_name}/weekly_init.npy', weekly_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="statistical prior initialization for learnable periodic embeddings")
    parser.add_argument("-d", "--dataset_name", type=str, default='PEMS07')
    parser.add_argument("--daily_len", type=int, default=288)
    parser.add_argument("--weekly_len", type=int, default=288*7)
    args = parser.parse_args()

    data_file_path = f'datasets/{args.dataset_name}/data.dat'
    description_file_path = f'datasets/{args.dataset_name}/desc.json'

    with open(description_file_path, 'r') as f:
        description = json.load(f)

    train_val_test_ratio = description["regular_settings"]["TRAIN_VAL_TEST_RATIO"]

    statistical_prior_initialization(args.dataset_name, description["num_nodes"], tuple(description["shape"]),
                                     args.daily_len, args.weekly_len, train_val_test_ratio)