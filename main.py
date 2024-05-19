import argparse
import os
import time
import traceback

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet.networks import define_G
from unet.unet_model import *
from utils.dataset import BrainDataset
from utils.init_logging import init_logging

dir_checkpoint = 'checkpoints/'
train_list = 'data/train_brats_only_tumor.txt'
test_list = 'data/test_brats.txt'


def train_net(reconstucter, segmenter, device, epochs=5, batch_size=1, lr=0.001, percent=0.1, save_cp=True, ):
    train_set = BrainDataset(test_list, percent=percent)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    n_train = len(train_loader)

    logger.info(f'''Starting training:
        File names        main.py
        Starting time     {starttime_r}
        Epochs:           {epochs}
        Batch size:       {batch_size}
        Learning rate:    {lr}
        λ:                {lambd}
        Training size:    {n_train}
        Checkpoints:      {save_cp}
        Device:           {device.type}
        Training percent: {percent}
    ''')

    optimizer_R = optim.Adam(reconstucter.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer_S = optim.Adam(segmenter.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler_R = optim.lr_scheduler.StepLR(optimizer_R, step_size=int(0.8 * epochs))
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=int(0.8 * epochs))

    ce_loss = nn.CrossEntropyLoss()
    ce_loss1 = nn.CrossEntropyLoss(reduce=False)
    mse_loss = nn.MSELoss(reduce=False)
    for epoch in range(epochs):
        reconstucter.train()
        segmenter.train()
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        logger.info("Epoch {:2d} learning(S) rate: {:.2e}, learning(R) rate: {:.2e}, ".format(epoch, optimizer_S.param_groups[0]['lr'],
                                                                                              optimizer_R.param_groups[0]['lr']))
        bar = tqdm(enumerate(train_loader), total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', ncols=120, ascii=True)
        for i, data in bar:
            [img, seg, seg_tumor, brain_mask, _, _] = data[0:6]
            img = img.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            seg_tumor = seg_tumor.to(device=device, dtype=torch.long)
            brain_mask = brain_mask.to(device=device, dtype=torch.float32)

            seg_no_tumor = torch.zeros_like(seg).to(device=device, dtype=torch.long)
            seg_tumor_neg = 1 - seg_tumor
            mask = (seg_tumor_neg * brain_mask).unsqueeze(dim=1)
            brain_avg = torch.ones_like(img) * img[mask == 1].mean()

            # train step 1:
            optimizer_R.zero_grad()
            img_pred = reconstucter(img)
            seg_pred = segmenter(img_pred)

            loss_seg2 = lambd[1] * ce_loss(seg_pred, seg_no_tumor)
            loss_mse = lambd[2] * (mse_loss(img_pred, img) * seg_tumor_neg).mean()
            loss_var = lambd[3] * (mse_loss(img_pred, brain_avg.to(dtype=torch.float)).squeeze(1) * seg_tumor).mean()
            loss_recer = loss_seg2 + loss_mse + loss_var
            loss_recer.backward()
            optimizer_R.step()

            # train step 2:
            optimizer_S.zero_grad()
            seg_pred = segmenter(img_pred.detach())

            # Calculate weight
            diff = torch.abs(img_pred.detach() - img).squeeze()
            diff_norm = diff / diff.view(batch_size, -1).max(dim=-1)[0].view(batch_size, 1, 1)
            weightt = 1 - diff_norm
            weightt = torch.where(weightt < 0.1, 0.1 * torch.ones_like(weightt), weightt)

            loss_seg1 = lambd[0] * (ce_loss1(seg_pred, seg_tumor) * weightt).mean()
            loss_seg1.backward()
            optimizer_S.step()

            bar.set_postfix_str(
                'loss(batch):{:>.2e},{:>.2e},{:>.2e},{:.2e}'.format(loss_seg1.item(), loss_seg2.item(), loss_mse.item(), loss_var.item()))
            loss1 += loss_seg1
            loss2 += loss_seg2
            loss3 += loss_mse
            loss4 += loss_var

        logger.info('Epoch loss: {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(loss1 / i, loss2 / i, loss3 / i, loss4 / i))
        scheduler_S.step()
        scheduler_R.step()

        if save_cp and (epoch + 1) % 5 == 0:
            torch.save({'segmenter': segmenter.state_dict(),
                        'optimizer_S': optimizer_S.state_dict(),
                        'scheduler_S': scheduler_S.state_dict(),
                        'reconstucter': reconstucter.state_dict(),
                        'optimizer_R': optimizer_R.state_dict(),
                        'scheduler_R': scheduler_R.state_dict(),
                        'epoch': epoch},
                       dir_checkpoint + '%s_%s.pth' % (starttime, epoch + 1))
            logger.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int, help='witch gpu to use')
    parser.add_argument('--percent', '-p', metavar='PERCENT', dest='percent', type=float, default=10.0,
                        help='Percent of the data that is used as training (0-100)')
    parser.add_argument('--lambd', '-d', type=str, default='[1,1,10.0,5.0]', dest='lambd',
                        metavar='FILE', help="param of 3 losses")

    return parser.parse_args()


if __name__ == '__main__':
    t = time.localtime()
    starttime = time.strftime("%Y-%m-%d_%H-%M-%S", t)
    starttime_r = time.strftime("%Y/%m/%d %H:%M:%S", t)  # readable time

    logger = init_logging(starttime)
    args = get_args()
    lambd = eval(args.lambd)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    reconstucter = define_G(input_nc=1, output_nc=1, ngf=64, netG='resnet_9blocks', norm='instance')
    segmenter = Segmenter(n_channels=1, n_classes=2, bilinear=True)

    # if args.load:
    #     segmenter.load_state_dict(
    #         torch.load(args.load, map_location=device)
    #     )
    #     logger.info(f'Model loaded from {args.load}')

    reconstucter.to(device=device)
    segmenter.to(device=device)

    try:
        train_net(reconstucter=reconstucter,
                  segmenter=segmenter,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  percent=args.percent / 100)
        logger.info('Program done. Start on %s' % (starttime_r))
    except KeyboardInterrupt:
        torch.save({'reconstucter': reconstucter.state_dict(),
                    'segmenter': segmenter.state_dict()}, './checkpoints/INTERRUPTED.pth')
        logger.error('User canceled, start on %s, Saved interrupt.' % (starttime_r))
        os.remove(f'./log/{starttime[:10]}/{starttime}.txt')
    except:
        logger.error('\n' + (traceback.format_exc()))
        logger.info('Error! Start on %s' % (starttime_r))
