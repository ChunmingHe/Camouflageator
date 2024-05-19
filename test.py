import argparse
import os
import time

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image

from torchvision.utils import save_image
from unet.networks import define_G
from unet.unet_model import *
from utils.dataset import BrainDataset
from utils.init_logging import init_logging
from utils.data_val import get_loader, test_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, default='/data0/hcm/GVS-V5/checkpoints/generator3_2/2022-12-07_11-32-46_31.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('--output', '-o', metavar='OUTPUT', type=str, default='./Results/',
                        help='Filenames of output images', dest='output')
    parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int, help='witch gpu to use')
    parser.add_argument('-t', '--train_root', dest='train_root', default=r"/data0/hcm/dataset/COD/TrainDataset/", type=str,
                        help='training dataset path')
    parser.add_argument('--val_root', type=str, default=r"/data0/hcm/dataset/COD/TestDataset/COD10K/",
                        help='the test rgb images root')
    return parser.parse_args()


def predict(args, logger=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    model_pth = args.model
    output = args.output

    output_img = output + '{}/{}'.format(model_pth.split('/')[-2],model_pth.split('_')[-1].split('.')[0])

    if not os.path.exists(output):
        os.mkdir(output)
    if not os.path.exists(output_img):
        os.makedirs(output_img)
    print(output_img)

    # load model
    reconstucter = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='instance')
    reconstucter.to(device=device)
    logger.info("Loading model {}".format(model_pth))
    model_dict = torch.load(model_pth, map_location=device)['reconstucter']
    reconstucter.load_state_dict({k.replace('module.',''):v for k,v in model_dict.items()})
    logger.info("Model loaded !")

    reconstucter.eval()

    val_loader = test_dataset(image_root=args.train_root + 'Imgs/',
                                gt_root=args.train_root + 'GT/',
                                testsize=384)

    # bar = tqdm(enumerate(val_loader), total=n_test, desc='TEST ROUND', unit='batch', ncols=120, ascii=True)
    # for i, data in bar:
    for i in tqdm(range(val_loader.size)):
        image, gt, name= val_loader.load_data()
        # img, gts, edges, dis_masks = data
        gt = np.asarray(gt, np.float32)
        img = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            img_no_tumor = reconstucter(img)  # shape(batch,1,512,512)
            img_no_tumor = F.upsample(img_no_tumor, size=gt.shape, mode='bilinear', align_corners=True)
            res = img_no_tumor[0].clone()
            res = res.sigmoid().data.squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            save_image(res,'{}/{}_reco.jpg'.format(output_img,name.split(".")[0]))
            # cv2.imwrite('{}/{}.jpg'.format(output_img,i), res*255)


if __name__ == "__main__":
    args = get_args()
    t = time.localtime()
    starttime = time.strftime("%Y-%m-%d_%H-%M-%S", t)
    starttime_r = time.strftime("%Y/%m/%d %H:%M:%S", t)  # readable time
    logger = init_logging(starttime, log_file=False)

    try:
        predict(args, logger=logger)
    except KeyboardInterrupt:
        logger.error('User canceled, start on %s, Saved interrupt.' % (starttime_r))
