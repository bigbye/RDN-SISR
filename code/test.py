import argparse
import os
import torch

import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import PIL.Image as Image

from models import RDN
from custom_datasets import TestDataset
from utils import denormalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data param
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./test_result')
    # model param
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)

    # test param
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=4, help='number of gpus')

    args = parser.parse_args()

    cudnn.benchmark = True
    gpus_list = range(args.gpus)

    test_dataset = TestDataset(args.test_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers,
                ).cuda(gpus_list[0])
    model = torch.nn.DataParallel(model, device_ids=list(gpus_list))

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    for (idx, input, input_path) in test_dataloader:
        input = input.cuda(gpus_list[0])
        with torch.no_grad():
            preds = model(input)

        for i in range(args.batch_size):

            # permute 交换维数
            output = Image.fromarray(denormalize(preds[i]).permute(1, 2, 0).byte().cpu().numpy())

            input_dir, input_filename = os.path.split(input_path[i])

            output_dir = args.output_path

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output.save(os.path.join(output_dir, input_filename))

        print('saved/total:{}/{}'.format((idx + 1), test_dataset.__len__()))
