# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import os
import torch

import helpers.misc as misc
import torch.backends.cudnn as cudnn

from pathlib import Path
from utils import load_model
from torch.utils.data import SequentialSampler, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop


def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate a pretrained model on image recognition", add_help=False)

    # Model parameters
    parser.add_argument('--model_name', default='vit_hvm1_none', type=str, help="Model identifier")
    parser.add_argument("--img_size", default=224, type=int, help="Images spatial size")
    parser.add_argument("--num_frames", default=16, type=int, help="Repeat image this many times to match model input size")

    # Bookkeeping
    parser.add_argument("--device", default="cuda", help="Device to use for testing")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of data loading workers")

    # Data parameters
    parser.add_argument("--val_dir", default="", help="Path to val data")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")

    return parser

def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # validation transforms
    val_transform = Compose([
        Resize(args.img_size + 32, interpolation=3),
        CenterCrop(args.img_size),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # set up and load model
    model = load_model(args.model_name)
    model.to(device)  # move model to device
    print(f"Model = {model}")

    # evaluate model and print results
    test_stats = evaluate(val_loader, model, device)
    print("==========================================")
    print(f"Number of test images: {len(val_dataset)}")
    print(f"Acc@1: {test_stats['acc1']:.1f}%") 
    print(f"Acc@5: {test_stats['acc5']:.1f}%")
    print(f"Loss: {test_stats['loss']:.2f}")


@torch.no_grad()
def evaluate(data_loader, model, device, num_frames):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    num_logs_per_epoch = 1

    # switch to evaluation mode
    model.eval()

    for _, (images, target) in enumerate(metric_logger.log_every(data_loader, len(data_loader) // num_logs_per_epoch, header)):

        images = images.unsqueeze(2).repeat((1, 1, num_frames, 1, 1))

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # evaluate
    main(args)