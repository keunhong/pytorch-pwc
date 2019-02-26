from pathlib import Path
import imageio
import numpy as np
import torch
import argparse

import visdom


from pwcnet.visualize import flow_to_color

from .model import PWCNet


vis = visdom.Visdom(env='pwcnet')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_image(path):
    image = imageio.imread(str(path))

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if image.dtype == np.float64:
        image = image.astype(np.float32)

    # To BGR.
    image = image[:, :, [2, 0, 1]]
    return torch.from_numpy(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument(dest='first', type=Path)
    parser.add_argument(dest='second', type=Path)
    args = parser.parse_args()

    first = read_image(args.first)
    second = read_image(args.second)

    model = PWCNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    flow = model.estimate_flow(first, second)
    vis.image(flow_to_color(flow.permute(1, 2, 0).numpy())
              .transpose((2, 0, 1)))


if __name__ == '__main__':
    main()
