from torch.utils.data import Dataset


class ImageDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length

