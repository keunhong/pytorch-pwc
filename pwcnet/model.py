import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from pwcnet import utils
from . import correlation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Backward(object):
    _grid = {}
    _partial = {}

    def __call__(self, input: torch.Tensor, flow: torch.Tensor):
        if str(flow.size()) not in self._grid:
            horiz = (torch.linspace(-1.0, 1.0, flow.size(3))
                     .view(1, 1, 1, flow.size(3))
                     .expand(flow.size(0), -1, flow.size(2), -1))
            vert = (torch.linspace(-1.0, 1.0, flow.size(2))
                    .view(1, 1, flow.size(2), 1)
                    .expand(flow.size(0), -1, -1, flow.size(3)))

            self._grid[str(flow.size())] = torch.cat(
                (horiz, vert), dim=1).to(device)

        if str(flow.size()) not in self._partial:
            self._partial[str(flow.size())] = flow.new_ones(
                [flow.size(0), 1, flow.size(2), flow.size(3)])

        flow = torch.cat(
            (flow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0),
             flow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0)), 1)
        input = torch.cat(
            (input, self._partial[str(flow.size())]), 1)

        out = F.grid_sample(
            input=input,
            grid=(self._grid[str(flow.size())] + flow).permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='zeros')

        mask = out[:, -1:, :, :];
        mask[mask > 0.999] = 1.0;
        mask[mask < 1.0] = 0.0

        return out[:, :-1, :, :] * mask


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=196,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, input):
        x1 = self.moduleOne(input)
        x2 = self.moduleTwo(x1)
        x3 = self.moduleThr(x2)
        x4 = self.moduleFou(x3)
        x5 = self.moduleFiv(x4)
        x6 = self.moduleSix(x5)

        return [x1, x2, x3, x4, x5, x6]


class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()

        prev = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                81 + 96 + 2 + 2,
                81 + 128 + 2 + 2, 81, None][level + 1]
        current = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                   81 + 96 + 2 + 2,
                   81 + 128 + 2 + 2, 81, None][level + 0]

        if level < 6:
            self.moduleUpflow = nn.ConvTranspose2d(
                in_channels=2, out_channels=2, kernel_size=4, stride=2,
                padding=1)
            self.moduleUpfeat = nn.ConvTranspose2d(
                in_channels=prev + 128 + 128 + 96 + 64 + 32,
                out_channels=2, kernel_size=4, stride=2, padding=1)
            self.backward_scales = \
                [None, None, None, 5.0, 2.5, 1.25, 0.625, None][level + 1]

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=current, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=current + 128,
                      out_channels=128, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=current + 128 + 128,
                      out_channels=96, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=current + 128 + 128 + 96,
                      out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(
                in_channels=current + 128 + 128 + 96 + 64,
                out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(
                in_channels=current + 128 + 128 + 96 + 64 + 32,
                out_channels=2, kernel_size=3, stride=1, padding=1)
        )

        self.backward = Backward()

    def forward(self, first, second, prev_obj):
        if prev_obj is None:
            volume = nn.functional.leaky_relu(
                input=correlation.correlation(
                    first=first, second=second),
                negative_slope=0.1, inplace=False)
            feat = volume
        else:
            flow = self.moduleUpflow(prev_obj['tensorFlow'])
            feat = self.moduleUpfeat(prev_obj['tensorFeat'])

            volume = F.leaky_relu(
                input=correlation.correlation(
                    first=first,
                    second=self.backward(input=second,
                                         flow=flow * self.backward_scales)),
                negative_slope=0.1, inplace=False)

            feat = torch.cat((volume, first, flow, feat), 1)

        feat = torch.cat((self.moduleOne(feat), feat), 1)
        feat = torch.cat((self.moduleTwo(feat), feat), 1)
        feat = torch.cat((self.moduleThr(feat), feat), 1)
        feat = torch.cat((self.moduleFou(feat), feat), 1)
        feat = torch.cat((self.moduleFiv(feat), feat), 1)

        flow = self.moduleSix(feat)

        return {
            'tensorFlow': flow,
            'tensorFeat': feat
        }


class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.moduleMain = nn.Sequential(
            nn.Conv2d(
                in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                out_channels=128, kernel_size=3, stride=1, padding=1,
                dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=4,
                      dilation=4),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96,
                      kernel_size=3, stride=1, padding=8,
                      dilation=8),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64,
                      kernel_size=3, stride=1, padding=16,
                      dilation=16),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1,
                      dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2,
                      kernel_size=3, stride=1, padding=1,
                      dilation=1)
        )

    def forward(self, x):
        return self.moduleMain(x)


class PWCNet(nn.Module, utils.PretrainedMixIn):

    @classmethod
    def default_weights_name(cls):
        return 'pwcnet-default.pth'

    def __init__(self, flow_shape=(436, 1024)):
        super(PWCNet, self).__init__()

        self.flow_shape = (
            int(math.floor(math.ceil(flow_shape[0] / 64.0) * 64.0)),
            int(math.floor(math.ceil(flow_shape[1] / 64.0) * 64.0)))

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

    def process_input(self, image: torch.Tensor):
        """
        Convert image to BGR and resize to input shape.
        """
        image = image[:, [2, 1, 0], :, :]
        image = F.interpolate(
            input=image,
            size=self.flow_shape,
            mode='bilinear',
            align_corners=False)
        return image

    def _process_flow_output(self, flow, shape):
        """
        Resize to original image dimensions.
        """
        height, width = shape
        flow = 20.0 * F.interpolate(
            input=flow.clone(),
            size=(height, width), mode='bilinear', align_corners=False)
        flow = flow.squeeze()

        flow[0, :, :] *= float(width) / float(self.flow_shape[1])
        flow[1, :, :] *= float(height) / float(self.flow_shape[0])
        return flow

    def forward(self, first, second):
        first = self.moduleExtractor.forward(first)
        second = self.moduleExtractor.forward(second)

        x = self.moduleSix.forward(first[-1], second[-1], None)
        x = self.moduleFiv.forward(first[-2], second[-2], x)
        x = self.moduleFou.forward(first[-3], second[-3], x)
        x = self.moduleThr.forward(first[-4], second[-4], x)
        x = self.moduleTwo.forward(first[-5], second[-5], x)

        return x['tensorFlow'] + self.moduleRefiner.forward(x['tensorFeat'])

    def estimate_flow(self, first, second, out_shape=None,
                      preprocessed=False):
        self.eval()

        width = first[0].size(2)
        height = first[0].size(1)
        shape = out_shape if out_shape is not None else (height, width)

        if not preprocessed:
            first = self.process_input(first)
            second = self.process_input(second)

        flow = self.forward(first.to(device), second.to(device))
        flow = flow.detach()
        flow = self._process_flow_output(flow, shape=shape)
        flow = flow.cpu()

        return flow
