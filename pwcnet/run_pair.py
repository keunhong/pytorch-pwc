from pathlib import Path
import imageio
import torch
import argparse

import visdom

from pwcnet.visualize import flow_to_color
from pwcnet.model import PWCNet
from pwcnet.utils import image_to_tensor


vis = visdom.Visdom(env='pwcnet')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_image(path):
    image = imageio.imread(path)
    return image_to_tensor(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument(dest='first', type=Path)
    parser.add_argument(dest='second', type=Path)
    args = parser.parse_args()

    first = load_image(args.first)
    second = load_image(args.second)

    model = PWCNet(flow_shape=(436, 1024)).to(device)
    model.load_pretrained()

    flow = model.estimate_flow(first, second)
    vis.image(flow_to_color(flow.permute(1, 2, 0).numpy())
              .transpose((2, 0, 1)), win='flow')


if __name__ == '__main__':
    main()
