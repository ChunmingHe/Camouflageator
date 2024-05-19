import os
import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LiverDataset(Dataset):
    def __init__(self, imgs_dir, other_dir='', scale_seg=False):
        self.imgs_dir = imgs_dir
        self.other_dir = other_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-99.5946,-99.5946), (126.5396,126.5396))
        ])
        self.scale_seg = scale_seg

        with open(self.imgs_dir, 'r') as fp:
            img_list = fp.readlines()
        for i in range(len(img_list)):
            img_list[i] = img_list[i].strip()
        if self.other_dir != '':
            with open(self.other_dir, 'r') as fp:
                other_list = fp.readlines()
            for i in range(len(other_list)):
                other_list[i] = other_list[i].strip()
            self.img_list = img_list + other_list
        else:
            self.img_list = img_list
        # self.idx = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = np.load(self.img_list[idx]).item()
        case_name = (self.img_list[idx]).split('/')[-2]
        slice_idx = os.path.basename(self.img_list[idx])[:6]
        img = data['liver']
        seg = data['seg_label']
        body_mask = np.where(img <= 200, 0, 1)
        liver_mask = np.where(seg > 0, 1, 0)

        seg_tumor = np.where(seg == 2, 1, 0)  # 只留肿瘤标签
        seg_tumor_np = seg_tumor.astype(np.uint8)

        img = torch.tensor(img)
        seg = torch.tensor(seg)
        seg_tumor = torch.tensor(seg_tumor)

        return_data = [img, seg, seg_tumor, torch.tensor(body_mask), case_name, slice_idx]

        if self.scale_seg:
            kernel = np.ones((self.scale_seg * 2, self.scale_seg * 2), np.uint8)
            seg_scale = cv2.dilate(seg_tumor_np, kernel, iterations=1)
            return_data.append(torch.tensor(seg_scale * liver_mask))

        return return_data


class BrainDataset(Dataset):
    def __init__(self, imgs_dir, has_mean=False, percent=1.0):
        '''
        has_mean: bool, if your npy has key of 'mean'/'std'
        scale_seg: int, pixle you want to scale your seg
        '''
        self.has_mean = has_mean
        self.imgs_dir = imgs_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-99.5946,-99.5946), (126.5396,126.5396))
        ])
        self.percent = percent
        self.img_list = []

        with open(self.imgs_dir, 'r') as fp:
            img_list = fp.readlines()
        for i in range(int(len(img_list) * self.percent)):
            self.img_list.append(img_list[i].strip())
        # self.idx = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = np.load(self.img_list[idx],allow_pickle=True).item()
        case_name = (self.img_list[idx]).split('/')[-2]
        slice_idx = os.path.basename(self.img_list[idx])[:3]
        img = data['brain']
        seg = data['seg_label']
        seg_tumor = np.where(seg > 0, 1, 0)  # 只留肿瘤标签
        brain_mask = np.where(img != 0, 1, 0)
        # seg_tumor_np = seg_tumor.astype(np.uint8)

        img = torch.tensor(img)
        img = ((img - img.min()) / (img.max() - img.min()) - 0.5) * 2
        seg = torch.tensor(seg)
        seg_tumor = torch.tensor(seg_tumor)

        return_data = [img, seg, seg_tumor, torch.tensor(brain_mask), case_name, slice_idx]

        if self.has_mean:
            std = data['std']
            mean = data['mean']
            mean = mean * brain_mask
            std = std * brain_mask
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            return_data += [mean, std]

        return return_data

# if __name__ == "__main__":
#     train = BrainDataset('data/train_brats.txt')
#     train_loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
#     for i,(img, seg, seg_tumor, [name,slice_idx], [mean,std]) in enumerate(train_loader):
#         img = img.numpy().astype(np.int32)
#         img_min = img.min(-1).min(-1)
#         img_max = img.max(-1).max(-1)
#         # ones = torch.ones((512,512),dtype=torch.uint8)
#         for i in range(len(img_min)):
#             tmp = (img[i] + abs(img_min[i])) / (img_max[i] + abs(img_min[i])) * 255
#             tmp = tmp.astype(np.uint8)
#             tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2RGB)
#             cv2.circle(tmp, tuple(center[i]), 1, (0,0,255), 4)
#             for j in range(36):
#                 x = (center[i][0] + dists[i][-j] * np.cos((j * 10) * np.pi/180)).astype(np.int16)
#                 y = (center[i][1] - dists[i][-j] * np.sin((j * 10) * np.pi/180)).astype(np.int16)
#                 cv2.circle(tmp, (x,y), 1, (0,0,255), 4)
#             cv2.imwrite('test/test%d.png'%(i),tmp)
#
#             # cv2.circle(img[i], tuple(center[i]), 1, 0, 4)
#         pass
