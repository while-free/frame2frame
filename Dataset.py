from torch.utils.data import Dataset
import torch
import cv2
from IPython import display
from matplotlib import pyplot as plt
import os
import pickle


def CreateDataset(datafile):
    frames = []
    len_frames = [0]
    width, height = 512, 512
    for root, dirs, files in os.walk(datafile):
        print(root)
        for file in files:
            path = os.path.join(root, file)
            print(path)

            # 打开mp4文件
            video = cv2.VideoCapture(path)

            # 读取每一帧图像并将其转换为PyTorch张量
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB模式
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # 转换为PyTorch张量
                frames.append(frame)

            len_frames.append(len(frames))

    # 将张量列表堆叠成一个张量
    tensor = torch.stack(frames)

    dataset = MP4Dataset(tensor, len_frames)
    print(f'length of dataset: {len(dataset)}')

    return dataset, torch.stack([tensor[0]])


class MP4Dataset(Dataset):
    def __init__(self, data, len_frames):
        self.datas = []
        for i in range(len(len_frames) - 1):
            for j in range(len_frames[i], len_frames[i+1] - 1):
                self.datas.append((data[j], data[j+1]))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item]


def save_dataset(myDataset, path):
    """
    path: end with '.pkl'
    """
    with open(path, "wb") as f:
        pickle.dump(myDataset, f)


def read_dataset(path, myDataset=None):
    with open("example.pkl", "wb") as f:
        pickle.dump(myDataset, f)
    return myDataset


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize