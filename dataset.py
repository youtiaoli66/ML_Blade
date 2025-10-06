import os
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from tqdm import tqdm,trange
import numpy as np

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8', newline='') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            row_list = []
            for i in row:
                row_list.append(float(i))
            data.append(np.array(row_list, dtype=np.float32))
    return np.array(data, dtype=np.float32)

class Normalize(torch.nn.Module):
    def __init__(self, mean=None, std=None):
        super(Normalize, self).__init__()
        self.mean = torch.FloatTensor(mean) if mean is not None else None
        self.std = torch.FloatTensor(std) if std is not None else None

    def forward(self, x):
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        else:
            return x

    def denorm(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

class Blade_Dataset(Dataset):
    def __init__(self, path, flatten=False, transform_data=None, transform_targets=None):
        super(Blade_Dataset, self).__init__()
        self.data_path = path
        self.flatten = flatten
        self.filenames = os.listdir(path)
        self.transform_data = transform_data
        self.transform_targets = transform_targets
        self.data, self.targets, self.mask = self.flatten_dataset()


        if isinstance(transform_data, Normalize) and (
            transform_data.mean is None or transform_data.std is None):
            mean_inputs, std_inputs, mean_targets, std_targets = self.normalize()
            self.transform_data.mean = mean_inputs
            self.transform_data.std = std_inputs
            self.transform_targets.mean = mean_targets
            self.transform_targets.std = std_targets

    def __len__(self):
        return len(self.filenames)

    def flatten_dataset(self):
        inputs_collection = []
        targets_collection = []
        mask_collection = []
        for idx in trange(self.__len__(), desc="prepare data..."):
            inlet_total_pressure = pd.read_csv(os.path.join(self.data_path, self.filenames[idx], 'STARCCM_3/total_pressure_inputs.csv'), header=None).values
            total_temperature = pd.read_csv(os.path.join(self.data_path, self.filenames[idx], 'STARCCM_3/total_temperature_inputs.csv'), header=None).values
            outlet_pressure = pd.read_csv(os.path.join(self.data_path, self.filenames[idx], 'STARCCM_3/pressure_inputs.csv'), header=None).values
            mach = pd.read_csv(os.path.join(self.data_path, self.filenames[idx], 'STARCCM_3/mach_targets.csv'), header=None).values
            pressure = pd.read_csv(os.path.join(self.data_path, self.filenames[idx], 'STARCCM_3/pressure_targets.csv'), header=None).values
            mask = pd.read_csv(os.path.join(self.data_path, self.filenames[idx], 'STARCCM_3/mask.csv'), header=None).values

            data = torch.from_numpy(np.stack((mask, total_temperature, outlet_pressure), -1)).float()
            targets = torch.from_numpy(np.stack((mach, pressure), -1)).float()
            mask = torch.from_numpy(mask).float()

            inputs_collection.append(data)
            targets_collection.append(targets)
            mask_collection.append(mask)

            inputs = torch.stack(inputs_collection)
            targets = torch.stack(targets_collection)
            mask = torch.stack(mask_collection)
        return inputs, targets, mask

    def normalize(self):
        inputs, targets, mask = self.data, self.targets, self.mask
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3).permute(3, 0, 1, 2)
        #inputs = inputs.permute(0, 3, 1, 2).flatten(-2)[mask.flatten(-2).bool()].reshape(inputs.shape[-1], -1)


        inputs = inputs.permute(3, 0, 1, 2).flatten(-2)[mask.flatten(-2).bool()].reshape(inputs.shape[-1], -1)


        targets = targets.permute(3, 0, 1, 2).flatten(-2)[mask[:2,:,:,:].flatten(-2).bool()].reshape(targets.shape[-1], -1)


        mean_inputs = torch.mean(inputs, dim=1)
        std_inputs = torch.std(inputs, dim=1)
        std_inputs[std_inputs == 0] = 1e-6
        mean_targets = torch.mean(targets, dim=1)
        std_targets = torch.std(targets, dim=1)


        with open("C:/Users/86176/Desktop/python/AICFD/data/mean_std.csv", "w", encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            header = ['mean_inputs', 'std_inputs', 'mean_targets', 'std_targets']
            csv_writer.writerow(header)
            csv_writer.writerow(mean_inputs.numpy().tolist())
            csv_writer.writerow(std_inputs.numpy().tolist())
            csv_writer.writerow(mean_targets.numpy().tolist())
            csv_writer.writerow(std_targets.numpy().tolist())

        return mean_inputs, std_inputs, mean_targets, std_targets

    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        mask = self.mask[idx]
        if self.transform_data is not None:
            data = self.transform_data(data)
            data = data * mask.unsqueeze(-1).repeat(1, 1, 3)
        if self.transform_targets is not None:
            targets = self.transform_targets(targets)
            targets = targets * mask.unsqueeze(-1).repeat(1, 1, 2)
        return data.permute(2, 0, 1), targets.permute(2, 0, 1),mask



def read_mean_std(path):
    with open(path, "r", encoding="utf-8", newline='') as file:
        data = []
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            row_list = []
            for i in row:
                row_list.append(float(i))
            data.append(row_list)
    return data[0], data[1], data[2], data[3]

def get_dataloader(path, train_size, batch_size=16, num_workers=8):
    mean_inputs, std_inputs, mean_targets, std_targets = read_mean_std("C:/Users/86176/Desktop/python/AICFD/data/mean_std.csv")
    dataset_data = Blade_Dataset(path, transform_data=Normalize(mean_inputs, std_inputs), transform_targets=Normalize(mean_targets, std_targets))
    train_size = int(train_size * len(dataset_data))
    test_size = int((len(dataset_data) - train_size) * 2 / 3)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, train_size + test_size))
    verify_indices = list(range(train_size + test_size, len(dataset_data)))

    train_data = Subset(dataset_data, train_indices)
    test_data = Subset(dataset_data, test_indices)
    verify_data = Subset(dataset_data, verify_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    verify_loader = DataLoader(verify_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, verify_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needs_path = "C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0"
    dataset = Blade_Dataset(needs_path, transform_data=Normalize(), transform_targets=Normalize())
    for i,name in tqdm(enumerate(dataset),'dataset'):
        data_loader = DataLoader(dataset,batch_size=8,num_workers=8)