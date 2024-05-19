import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
import cv2


# several data augumentation strategies
def cv_random_flip(img, label, edge, dis_mask, dilate, dilate_gaussian):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        dis_mask = dis_mask.transpose(Image.FLIP_LEFT_RIGHT)
        dilate = dilate.transpose(Image.FLIP_LEFT_RIGHT)
        dilate_gaussian = dilate_gaussian.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge, dis_mask, dilate, dilate_gaussian


def randomCrop(image, label, edge, dis_mask, dilate, dilate_gaussian):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region), dis_mask.crop(random_region) \
        , dilate.crop(random_region), dilate_gaussian.crop(random_region)


def randomRotation(image, label, edge, dis_mask, dilate, dilate_gaussian):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
        dis_mask = dis_mask.rotate(random_angle, mode)
        dilate = dilate.rotate(random_angle, mode)
        dilate_gaussian = dilate_gaussian.rotate(random_angle, mode)
    return image, label, edge, dis_mask, dilate, dilate_gaussian


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, dis_mask_root, dilate_root, dilate_gaussian_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.dis_masks = [dis_mask_root + f for f in os.listdir(dis_mask_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.dilates = [dilate_root + f for f in os.listdir(dilate_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.dilate_gaussians = [dilate_gaussian_root + f for f in os.listdir(dilate_gaussian_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.dis_masks = sorted(self.dis_masks)
        self.dilates = sorted(self.dilates)
        self.dilate_gaussians = sorted(self.dilate_gaussians)
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.dis_mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.dilate_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.dilate_gaussian_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.kernel = np.ones((3, 3), np.uint8)
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        dis_mask = self.binary_loader(self.dis_masks[index])
        dilate = self.binary_loader(self.dilates[index])
        dilate_gaussian = self.binary_loader(self.dilate_gaussians[index])
        # dis_mask = cv2.imread(self.dis_masks[index], cv2.IMREAD_GRAYSCALE)
        # dis_mask = Image.fromarray(dis_mask)
        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)

        image, gt, edge, dis_mask, dilate, dilate_gaussian = cv_random_flip(image, gt, edge, dis_mask, dilate, dilate_gaussian)
        image, gt, edge, dis_mask, dilate, dilate_gaussian = randomCrop(image, gt, edge, dis_mask, dilate, dilate_gaussian)
        image, gt, edge, dis_mask, dilate, dilate_gaussian = randomRotation(image, gt, edge, dis_mask, dilate, dilate_gaussian)

        image = colorEnhance(image)
        gt = randomPeper(gt)
        edge = randomPeper(edge)
        dis_mask = randomPeper(dis_mask)
        dilate = randomPeper(dilate)
        dilate_gaussian = randomPeper(dilate_gaussian)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)
        dis_mask = self.dis_mask_transform(dis_mask)
        dilate = self.dilate_transform(dilate)
        dilate_gaussian = self.dilate_gaussian_transform(dilate_gaussian)
        # edge_small = self.Threshold_process(edge)
        return image, gt, edge, dis_mask, dilate, dilate_gaussian

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.dis_masks) == len(self.images) \
               and len(self.dis_masks) == len(self.gts) and len(self.dis_masks) == len(self.edges) \
               and len(self.dilates) == len(self.gts) and len(self.dilate_gaussians) == len(self.gts)
        images = []
        gts = []
        edges = []
        dis_masks = []
        dilates = []
        dilate_gaussians = []
        for img_path, gt_path, edge_path, dis_mask_path, dilate_path, dilate_gaussian_path in zip(self.images, self.gts, self.edges, self.dis_masks,
                                                                                                  self.dilates, self.dilate_gaussians):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            dis_mask = Image.open(dis_mask_path)
            dilate = Image.open(dilate_path)
            dilate_gaussian = Image.open(dilate_gaussian_path)

            if img.size == gt.size and img.size == dis_mask.size and img.size == edge.size and dilate.size == gt.size and dilate_gaussian.size == gt.size:
                # if img.size == gt.size and img.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                dis_masks.append(dis_mask_path)
                edges.append(edge_path)
                dilates.append(dilate_path)
                dilate_gaussians.append(dilate_gaussian_path)

        self.images = images
        self.gts = gts
        self.dis_masks = dis_masks
        self.edges = edges
        self.dilates = dilates
        self.dilate_gaussians = dilate_gaussians

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size


# solve dataloader random bug
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


# dataloader for training
def get_loader(image_root, gt_root, edge_root, dis_mask_root, dilate_root, dilate_gaussian_root, batchsize, trainsize, shuffle=True, num_workers=12,
               pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, edge_root, dis_mask_root, dilate_root, dilate_gaussian_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  worker_init_fn=seed_worker)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.tif') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


if __name__ == '__main__':
    train_root = r'G:\software_package\TrainDataset/'
    test_root = r"G:\software_package\TestDataset/CHAMELEON/"
    batchsize = 6
    trainsize = 384
    # val_loader = test_dataset(image_root=test_root + 'Imgs/',
    #                           gt_root=test_root + 'GT/',
    #                           testsize=trainsize)
    # for i in range(val_loader.size):
    #     image, gt, name = val_loader.load_data()
    #     image = np.asarray(image, np.float32)
    #     gt = np.asarray(gt, np.float32)

    train_loader = get_loader(image_root=train_root + 'Imgs/',
                              gt_root=train_root + 'GT/',
                              edge_root=train_root + 'Edge/',
                              dis_mask_root=train_root + 'dis_mask/',
                              dilate_root=train_root + 'dilate/',
                              dilate_gaussian_root=train_root + 'dilate_gaussian/',
                              batchsize=batchsize,
                              trainsize=trainsize,
                              num_workers=0,
                              shuffle=False)
    for i, (images, gts, edges, dis_masks, dilates, dilate_gaussians) in enumerate(train_loader, start=1):
        # gt = gts[0].data.cpu().numpy().squeeze()
        # edge = edges[0].data.cpu().numpy().squeeze()
        # dis_mask = dis_masks[0].data.cpu().numpy().squeeze()
        counts = []
        for i in gts:
            number = torch.nonzero(i).shape[0]
            counts.append([number])
        count = torch.tensor(counts)
        count = count.unsqueeze(-1).unsqueeze(-1)
        z = []
        for m in range(images.shape[0]):
            y = []
            for n in images[m]:
                x = (n * gts[m]).mean()
                y.append([x])
            z.append(y)
        z = torch.tensor(z).unsqueeze(-1)

        brain_avg_dis = torch.ones_like(images) * z * gts.shape[2] * gts.shape[3] / count
        unloader = transforms.ToPILImage()
        dilates_new = unloader(dilates[0])
        dilate_gaussians_new = unloader(dilate_gaussians[0])
        images_new = unloader(images[0])
        brain_avg_dis_new = unloader(brain_avg_dis[0])
        gts_new = unloader(gts[0])
        dilates_new.show()
        dilate_gaussians_new.show()
        gts_new.show()
        images_new.show()
        brain_avg_dis_new.show()
        print()
        # cv2.imshow('ceshi_gt.png', gt)
        # cv2.imshow('ceshi_edge.png', edge)
        # cv2.imshow('ceshi_dis_masks.png', dis_mask)
        # cv2.waitKey(0)
