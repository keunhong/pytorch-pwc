import abc
import os
import typing
from pathlib import Path

import numpy
import torch
import numpy as np
import imageio
from numba import jit


def save_flow_as_png(path: Path, flow: typing.Union[np.array, torch.Tensor]):
    assert path.suffix == '.png'
    image = flow_to_png_format_16bits(flow)
    imageio.imwrite(path, image)


def load_flow_from_png(path: Path):
    assert path.suffix == '.png'
    image = imageio.imread(path)
    return png_to_flow_format_16bits(image)


@jit(nopython=True)
def _flow_to_png_format_32bits(flow):
    flow_y = np.zeros((*flow.shape[1:], 4), dtype=np.uint8)
    flow_x = np.zeros((*flow.shape[1:], 4), dtype=np.uint8)

    for png_channel, shift in enumerate([0, 8, 16, 24]):
        flow_y[..., png_channel] = (flow[0] >> shift) & 0xff

    for png_channel, shift in enumerate([0, 8, 16, 24]):
        flow_x[..., png_channel] = (flow[1] >> shift) & 0xff

    return np.hstack((flow_y, flow_x))


def flow_to_png_format_32bits(flow: typing.Union[np.array, torch.Tensor]):
    if torch.is_tensor(flow):
        flow = flow.numpy()

    assert flow.dtype == np.float32

    flow = np.frombuffer(flow.data, dtype=np.uint32).reshape(flow.shape)

    return _flow_to_png_format_32bits(flow)


def png_to_flow_format_32bits(png_array: np.array):
    assert png_array.dtype == np.uint8
    png_array = png_array.astype(np.uint32)
    height = png_array.shape[0]
    width = png_array.shape[1] // 2

    flow = np.zeros((2, height, width), dtype=np.uint32)

    channel_data = [
        png_array[:, :width, :],
        png_array[:, width:, :],
    ]

    for flow_channel in range(2):
        data = channel_data[flow_channel]
        for png_channel, shift in enumerate([0, 8, 16, 24]):
            flow[flow_channel] |= data[..., png_channel] << shift

    flow = np.frombuffer(flow.data, dtype=np.float32).reshape(flow.shape)

    return torch.from_numpy(flow)


@jit(nopython=True)
def _flow_to_png_format_16bits(flow):
    png_array = np.zeros((*flow.shape[1:], 4), dtype=np.uint8)

    for png_channel, shift in enumerate([0, 8]):
        png_array[..., png_channel] = (flow[0] >> shift) & 0xff

    for png_channel, shift in enumerate([0, 8]):
        png_array[..., png_channel + 2] = (flow[1] >> shift) & 0xff

    return png_array


def flow_to_png_format_16bits(flow: typing.Union[np.array, torch.Tensor]):
    if torch.is_tensor(flow):
        flow = flow.numpy()

    assert flow.dtype == np.float32

    flow = flow.astype(np.float16)
    flow = np.frombuffer(flow.data, dtype=np.uint16).reshape(flow.shape)

    return _flow_to_png_format_16bits(flow)


def png_to_flow_format_16bits(png_array: np.array):
    assert png_array.dtype == np.uint8
    png_array = png_array.astype(np.uint16)

    flow = np.zeros((2, *png_array.shape[:2]), dtype=np.uint16)

    for png_channel, shift in enumerate([0, 8]):
        flow[0][:, :] |= png_array[..., png_channel] << shift

    for png_channel, shift in enumerate([0, 8]):
        flow[1][:, :] |= png_array[..., png_channel + 2] << shift

    flow = np.frombuffer(flow.data, dtype=np.float16).reshape(flow.shape)

    return torch.from_numpy(flow.astype(np.float32))


def image_to_tensor(image):
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if image.dtype == np.float64:
        image = image.astype(np.float32)

    return torch.from_numpy(image).permute(2, 0, 1)


class PretrainedMixIn(abc.ABC):
    RESOURCE_DIR = Path(os.environ['HOME'], '.local1/share/pwcnet')
    WEIGHTS_NAME = ''
    WEIGHTS_ROOT = 'https://github.com/keunhong/pytorch-pwc'

    @classmethod
    def maybe_download_weights(cls, path=None):
        if path is None:
            path = cls.default_weights_path()

        if path.exists():
            return path

        import requests
        r = requests.get(cls.default_weights_url())
        r.raise_for_status()
        if not path.parent.exists():
            print(f"Creating directory {path.parent!s}")
            path.parent.mkdir(parents=True)

        print(f"Downloading {cls.__qualname__} weights to {path!s}")
        with path.open('wb') as f:
            for block in r.iter_content(1024):
                f.write(block)

        return path

    @classmethod
    @abc.abstractmethod
    def default_weights_name(cls):
        pass

    @classmethod
    def default_weights_path(cls):
        return Path(cls.RESOURCE_DIR, cls.default_weights_name())

    @classmethod
    def default_weights_url(cls):
        return (f'{cls.WEIGHTS_ROOT}/releases/download/0.1'
                f'/{cls.default_weights_name()}')

    def load_pretrained(self, path=None):
        if path is None:
            path = self.maybe_download_weights()
        print(f"Loading {self.__class__.__qualname__} weights from {path!s}")
        self.load_state_dict(torch.load(path))

    @abc.abstractmethod
    def load_state_dict(self, param):
        pass
