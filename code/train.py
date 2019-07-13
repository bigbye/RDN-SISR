import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import RDN
from custom_datasets import TrainDataset
from utils import AverageMeter, convert_rgb_to_y, calc_psnr, denormalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data param
    parser.add_argument('--lr_train_path', type=str, required=True)
    parser.add_argument('--hr_train_path', type=str, required=True)
    parser.add_argument('--lr_eval_path', type=str, required=True)
    parser.add_argument('--hr_eval_path', type=str, required=True)
    parser.add_argument('--outputs_path', type=str, default='./saved_weights')
    parser.add_argument('--weights_file', type=str)
    parser.add_argument('--is_select', type=bool, default=False)
    parser.add_argument('--n_frames', type=int, default=5)

    # model param
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    # train param
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpus', type=int, default=4, help='number of gpus')

    args = parser.parse_args()
    args.outputs_path = os.path.join(args.outputs_path, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_path):
        os.makedirs(args.outputs_path)

    gpus_list = range(args.gpus)
    cudnn.benchmark = True

    torch.manual_seed(args.seed)

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers,
                ).cuda(gpus_list[0])

    model = torch.nn.DataParallel(model, device_ids=list(gpus_list))

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.L1Loss().cuda(gpus_list[0])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = TrainDataset(args.lr_train_path, args.hr_train_path, args.patch_size, args.scale, True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    eval_dataset = TrainDataset(args.lr_eval_path, args.hr_eval_path, args.patch_size, args.scale, False)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.train_batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.cuda(gpus_list[0])
                labels = labels.cuda(gpus_list[0])

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_path, 'epoch_{}.pth'.format(epoch)))
        # 验证非常耗时间，故选择每十个epoch验证一次。
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.cuda(gpus_list[0])
            labels = labels.cuda(gpus_list[0])

            with torch.no_grad():
                preds = model(inputs)

            for i in range(args.eval_batch_size):
                # squeeze(0) 把shape为1的第0维去掉
                preds = convert_rgb_to_y(denormalize(preds[i].squeeze(0)), dim_order='chw')
                labels = convert_rgb_to_y(denormalize(labels[i].squeeze(0)), dim_order='chw')

                preds = preds[args.scale:-args.scale, args.scale:-args.scale]
                labels = labels[args.scale:-args.scale, args.scale:-args.scale]

                epoch_psnr.update(calc_psnr(preds, labels), len(preds))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_path, 'best.pth'))
