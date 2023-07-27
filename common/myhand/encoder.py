import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from common.myhand.model_zoo import get_hrnet, conv1x1, conv3x3, deconv3x3, weights_init, GCN_vert_convert, build_fc_layer, Bottleneck

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNetSimple_decoder(nn.Module):
    def __init__(self, expansion=4,
                 fDim=[256, 256, 256, 256], direction=['flat', 'up', 'up', 'up'],
                 out_dim=3):
        super(ResNetSimple_decoder, self).__init__()
        self.models = nn.ModuleList()
        fDim = [512 * expansion] + fDim
        for i in range(len(direction)):
            kernel_size = 1 if direction[i] == 'flat' else 3
            self.models.append(self.make_layer(fDim[i], fDim[i + 1], direction[i], kernel_size=kernel_size))

        self.final_layer = nn.Conv2d(
            in_channels=fDim[-1],
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def make_layer(self, in_dim, out_dim,
                   direction, kernel_size=3, relu=True, bn=True):
        assert direction in ['flat', 'up']
        assert kernel_size in [1, 3]
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0

        layers = []
        if direction == 'up':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        fmaps = []
        for i in range(len(self.models)):
            x = self.models[i](x)
            fmaps.append(x)
        x = self.final_layer(x)
        return x, fmaps


class ResNetSimple(nn.Module):
    def __init__(self, model_type='resnet50',
                 pretrained=False,
                 fmapDim=[256, 256, 256, 256],
                 handNum=2,
                 heatmapDim=21):
        super(ResNetSimple, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet101':
            self.resnet = resnet101(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet152':
            self.resnet = resnet152(pretrained=pretrained)
            self.expansion = 4

        self.hms_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                                fDim=fmapDim,
                                                direction=['flat', 'up', 'up', 'up'],
                                                out_dim=heatmapDim * handNum)
        for m in self.hms_decoder.modules():
            weights_init(m)

        self.dp_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                               fDim=fmapDim,
                                               direction=['flat', 'up', 'up', 'up'],
                                               out_dim=handNum + 3 * 1) ### change llj  handNum + 3 * handNum)   handNum + 3 * 1
        self.handNum = handNum

        for m in self.dp_decoder.modules():
            weights_init(m)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x4 = self.resnet.layer1(x)
        x3 = self.resnet.layer2(x4)
        x2 = self.resnet.layer3(x3)
        x1 = self.resnet.layer4(x2)

        img_fmaps = [x1, x2, x3, x4]

        hms, hms_fmaps = self.hms_decoder(x1)
        out, dp_fmaps = self.dp_decoder(x1)
        mask = out[:, :self.handNum]
        dp = out[:, self.handNum:]

        return hms, mask, dp, \
            img_fmaps, hms_fmaps, dp_fmaps


class resnet_mid(nn.Module):
    def __init__(self,
                 model_type='resnet50',
                 in_fmapDim=[256, 256, 256, 256],
                 out_fmapDim=[256, 256, 256, 256]):
        super(resnet_mid, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18' or model_type == 'resnet34':
            self.expansion = 1
        elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
            self.expansion = 4

        self.img_fmaps_dim = [512 * self.expansion, 256 * self.expansion,
                              128 * self.expansion, 64 * self.expansion]
        self.dp_fmaps_dim = in_fmapDim
        self.hms_fmaps_dim = in_fmapDim

        self.convs = nn.ModuleList()
        for i in range(len(out_fmapDim)):
            inDim = self.dp_fmaps_dim[i] + self.hms_fmaps_dim[i]
            if i > 0:
                inDim = inDim + self.img_fmaps_dim[i]
            self.convs.append(conv1x1(inDim, out_fmapDim[i]))

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.global_feature_dim = 512 * self.expansion
        self.fmaps_dim = out_fmapDim

    def get_info(self):
        return {'global_feature_dim': self.global_feature_dim,
                'fmaps_dim': self.fmaps_dim}

    def forward(self, img_fmaps, hms_fmaps, dp_fmaps):
        global_feature = self.output_layer(img_fmaps[0])
        fmaps = []
        for i in range(len(self.convs)):
            x = torch.cat((hms_fmaps[i], dp_fmaps[i]), dim=1)
            if i > 0:
                x = torch.cat((x, img_fmaps[i]), dim=1)
            fmaps.append(self.convs[i](x))
        return global_feature, fmaps


class HRnet_encoder(nn.Module):
    def __init__(self, model_type, pretrained='', handNum=2, heatmapDim=21):
        super(HRnet_encoder, self).__init__()
        name = 'w' + model_type[model_type.find('hrnet') + 5:]
        assert name in ['w18', 'w18_small_v1', 'w18_small_v2', 'w30', 'w32', 'w40', 'w44', 'w48', 'w64']

        self.hrnet = get_hrnet(name=name,
                               in_channels=3,
                               head_type='none',
                               pretrained='')

        if os.path.isfile(pretrained):
            print('load pretrained params: {}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = self.hrnet.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and k.find('classifier') == -1}
            model_dict.update(pretrained_dict)
            self.hrnet.load_state_dict(model_dict)

        self.fmaps_dim = list(self.hrnet.stage4_cfg['NUM_CHANNELS'])
        self.fmaps_dim.reverse()

        self.hms_decoder = self.mask_decoder(outDim=heatmapDim * handNum)
        for m in self.hms_decoder.modules():
            weights_init(m)

        self.dp_decoder = self.mask_decoder(outDim=1 + 3 * handNum)
        for m in self.dp_decoder.modules():
            weights_init(m)

    def mask_decoder(self, outDim=3):
        last_inp_channels = 0
        for temp in self.fmaps_dim:
            last_inp_channels += temp

        return nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels, out_channels=last_inp_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels, out_channels=outDim,
                kernel_size=1, stride=1, padding=0)
        )

    def forward(self, img):
        ylist = self.hrnet(img)

        # Upsampling
        x0_h, x0_w = ylist[0].size(2), ylist[0].size(3)
        x1 = F.interpolate(ylist[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(ylist[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(ylist[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([ylist[0], x1, x2, x3], 1)

        hms = self.hms_decoder(x)
        out = self.dp_decoder(x)
        mask = out[:, 0]
        dp = out[:, 1:]

        ylist.reverse()
        return hms, mask, dp, \
            ylist, None, None


class hrnet_mid(nn.Module):
    def __init__(self,
                 model_type,
                 in_fmapDim=[256, 256, 256, 256],
                 out_fmapDim=[256, 256, 256, 256]):
        super(hrnet_mid, self).__init__()
        name = 'w' + model_type[model_type.find('hrnet') + 5:]
        assert name in ['w18', 'w18_small_v1', 'w18_small_v2', 'w30', 'w32', 'w40', 'w44', 'w48', 'w64']

        self.convs = nn.ModuleList()
        for i in range(len(out_fmapDim)):
            self.convs.append(conv1x1(in_fmapDim[i], out_fmapDim[i]))

        self.global_feature_dim = 2048
        self.fmaps_dim = out_fmapDim

        in_fmapDim.reverse()
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(in_fmapDim)

    def get_info(self):
        return {'global_feature_dim': self.global_feature_dim,
                'fmaps_dim': self.fmaps_dim}

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, img_fmaps, hms_fmaps=None, dp_fmaps=None):
        fmaps = []
        for i in range(len(self.convs)):
            fmaps.append(self.convs[i](img_fmaps[i]))

        img_fmaps.reverse()
        y = self.incre_modules[0](img_fmaps[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](img_fmaps[i + 1]) + \
                self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                             [2:]).view(y.size(0), -1)

        return y, fmaps


def load_encoder(cfg):
    if cfg.MODEL.ENCODER_TYPE.find('resnet') != -1:
        encoder = ResNetSimple(model_type=cfg.MODEL.ENCODER_TYPE,
                               pretrained=True,
                               fmapDim=[128, 128, 128, 128],
                               handNum=2,
                               heatmapDim=21)
        mid_model = resnet_mid(model_type=cfg.MODEL.ENCODER_TYPE,
                               in_fmapDim=[128, 128, 128, 128],
                               out_fmapDim=cfg.MODEL.DECONV_DIMS)
    if cfg.MODEL.ENCODER_TYPE.find('hrnet') != -1:
        encoder = HRnet_encoder(model_type=cfg.MODEL.ENCODER_TYPE,
                                pretrained=cfg.MODEL.ENCODER_PRETRAIN_PATH,
                                handNum=2,
                                heatmapDim=21)
        mid_model = hrnet_mid(model_type=cfg.MODEL.ENCODER_TYPE,
                              in_fmapDim=encoder.fmaps_dim,
                              out_fmapDim=cfg.MODEL.DECONV_DIMS)

    return encoder, mid_model
