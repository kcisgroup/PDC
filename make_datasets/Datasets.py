import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

        x_data = raw_data[:-1]
        data = x_data
        data = self.normalize(data)
        data = torch.tensor(data, dtype=torch.float32)
        self.x = self.process(data)

    def __len__(self):
        return len(self.x)

    def process(self, data):
        x_arr = []
        node_num, total_len = data.shape

        for i in range(total_len):
            ft = data[:, i]
            x_arr.append(ft)

        x = torch.stack(x_arr).contiguous()
        return x

    def __getitem__(self, idx):
        feature = self.x[idx].float()
        return feature

    def normalize(self, data):
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        return data



class Seg_Datasets(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

        data = raw_data[:-1]
        labels = raw_data[-1]

        data = self.normalize(data)
        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()
        self.x, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr = []
        labels_arr = []
        node_num, total_time_len = data.shape

        for i in range(0, total_time_len):
            ft = data[:, i]
            x_arr.append(ft)
            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()
        return x, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        label = self.labels[idx].double()
        return feature, label

    def normalize(self, data):
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        return data

