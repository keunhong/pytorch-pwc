import argparse
import functools
import multiprocessing
import time
from pathlib import Path

import imageio
import torch
import visdom
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pwcnet.model import PWCNet
from pwcnet.utils import image_to_tensor, save_flow_as_png
from pwcnet.visualize import flow_to_color

vis = visdom.Visdom(env='pwcnet')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FloWrita(multiprocessing.Process):

    def __init__(self, out_dir, batch_size, queue):
        super().__init__()
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.queue = queue
        self.running = False

    def run(self):
        while True:
            batch_idx, flows = self.queue.get()
            if batch_idx is None:
                print("Done, exiting!")
                break

            for i, flow in enumerate(flows):
                idx = self.batch_size * batch_idx + i + 1
                out_path = self.out_dir / f"{idx:06d}.png"
                save_flow_as_png(out_path, flow)


class VideoDataset(Dataset):

    def __init__(self, path, model):
        self.paths = sorted(path.glob('*.jpg'))
        self.model = model

    def __len__(self):
        return len(self.paths) - 1

    @functools.lru_cache(maxsize=10)
    def get_frame(self, index):
        image = imageio.imread(self.paths[index])
        return self.model.process_input(
            image_to_tensor(image).unsqueeze(0)).squeeze()

    def __getitem__(self, index):
        first = self.get_frame(index)
        second = self.get_frame(index + 1)
        return first, second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument(dest='in_path', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--batch-size', default=16, type=int)
    args = parser.parse_args()

    model = PWCNet(flow_shape=(255, 456)).to(device)
    model.load_pretrained()

    dataset = VideoDataset(args.in_path, model)
    loader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=2)

    if args.out:
        out_dir = args.out
    else:
        out_dir = args.in_path.parent / f'{args.in_path.stem}.flow'

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    queue = multiprocessing.Queue()
    writer = FloWrita(out_dir, args.batch_size, queue)
    writer.start()

    tic = time.time()
    pbar = tqdm(loader)
    for batch_idx, (first, second) in enumerate(pbar):
        flows = model.estimate_flow(first, second, out_shape=(256, 256),
                                    preprocessed=True).cpu()

        queue.put_nowait((batch_idx, flows.numpy()))

        if args.show:
            for flow in flows:
                vis.image(flow_to_color(flow.permute(1, 2, 0).numpy())
                          .transpose((2, 0, 1)), win='flow')

        total_time = time.time() - tic
        fps = 1 / total_time * args.batch_size
        pbar.set_description(f"fps={fps:.1f}")
        tic = time.time()

    queue.put_nowait((None, None))

    queue.close()
    queue.join_thread()
    # writer.terminate()


if __name__ == '__main__':
    main()
