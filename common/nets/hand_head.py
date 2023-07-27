import torch
from torch import nn
import torch.nn.functional as F

class hand_regHead(nn.Module):
    ## roi res 32 -28 ### llj change h // 8
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1):
        """
        Args:
            inr_res: input image size
            joint_nb: hand joint num
        """
        super(hand_regHead, self).__init__()

        # hand head
        self.out_res = roi_res
        self.joint_nb = joint_nb

        self.channels = channels
        self.blocks = blocks
        self.stacks = stacks

        self.betas = nn.Parameter(torch.ones((self.joint_nb, 1), dtype=torch.float32))

        center_offset = 0.5
        vv, uu = torch.meshgrid(torch.arange(self.out_res).float(), torch.arange(self.out_res).float())
        uu, vv = uu + center_offset, vv + center_offset
        self.register_buffer("uu", uu / self.out_res)
        self.register_buffer("vv", vv / self.out_res)

        self.softmax = nn.Softmax(dim=2)
        block = Bottleneck
        self.features = self.channels // block.expansion

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.stacks):
            hg.append(Hourglass(block, self.blocks, self.features, 4))
            res.append(self.make_residual(block, self.channels, self.features, self.blocks))
            fc.append(BasicBlock(self.channels, self.channels, kernel_size=1))
            score.append(nn.Conv2d(self.channels, self.joint_nb, kernel_size=1, bias=True))
            if i < self.stacks - 1:
                fc_.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(self.joint_nb, self.channels, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def make_residual(self, block, inplanes, planes, blocks, stride=1):
        skip = None
        if stride != 1 or inplanes != planes * block.expansion:
            skip = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True))
        layers = []
        layers.append(block(inplanes, planes, stride, skip))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def spatial_softmax(self, latents):
        latents = latents.view((-1, self.joint_nb, self.out_res ** 2))
        latents = latents * self.betas
        heatmaps = self.softmax(latents)
        heatmaps = heatmaps.view(-1, self.joint_nb, self.out_res, self.out_res)
        return heatmaps

    def generate_output(self, heatmaps):
        predictions = torch.stack((
            torch.sum(torch.sum(heatmaps * self.uu, dim=2), dim=2),
            torch.sum(torch.sum(heatmaps * self.vv, dim=2), dim=2)), dim=2)
        return predictions

    def forward(self, x):
        out, encoding, preds = [], [], []
        for i in range(self.stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            latents = self.score[i](y)
            heatmaps= self.spatial_softmax(latents)
            out.append(heatmaps)
            predictions = self.generate_output(heatmaps)
            preds.append(predictions)
            if i < self.stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](heatmaps)
                x = x + fc_ + score_
                encoding.append(x)
            else:
                encoding.append(y)
        return out, encoding, preds


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,groups=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=((kernel_size - 1) // 2),
                      groups=groups,bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, skip=None, groups=1):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True, groups=groups)
        self.leakyrelu = nn.LeakyReLU(inplace=True)  # negative_slope=0.01
        self.skip = skip
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.skip is not None:
            residual = self.skip(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):

        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):

        layers = []
        for i in range(0, num_blocks):
            # channel changes: planes*block.expansion->planes->2*planes
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                # 3 residual modules composed of a residual unit
                # <2*planes><2*planes>
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                # i=0 in a recursive construction build the basic network path
                # see: low2 = self.hg[n-1][3](low1)
                # <2*planes><2*planes>
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)  # skip branches
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)  # only for depth=1 basic path of the hourglass network
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2) ### change llj for 28 resolution size=up1.shape[-2:] # scale_factor=2 should be consistent with F.max_pool2d(2,stride=2)
        out = up1 + up2
        return out

    def forward(self, x):
        # depth: order of the hourglass network
        # do network forward recursively
        return self._hour_glass_forward(self.depth, x)


class hand_Encoder(nn.Module):
    def __init__(self, num_heatmap_chan=21, num_feat_chan=256, size_input_feature=(32, 32), ### ### change llj h // 8
                 nRegBlock=4, nRegModules=2):
        super(hand_Encoder, self).__init__()

        self.num_heatmap_chan = num_heatmap_chan
        self.num_feat_chan = num_feat_chan
        self.size_input_feature = size_input_feature

        self.nRegBlock = nRegBlock
        self.nRegModules = nRegModules

        self.heatmap_conv = nn.Conv2d(self.num_heatmap_chan, self.num_feat_chan,
                                      bias=True, kernel_size=1, stride=1)
        self.encoding_conv = nn.Conv2d(self.num_feat_chan, self.num_feat_chan,
                                       bias=True, kernel_size=1, stride=1)

        reg = []
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                reg.append(Residual(self.num_feat_chan, self.num_feat_chan))

        self.reg = nn.ModuleList(reg)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_scale = 2 ** self.nRegBlock

        # fc layers
        self.num_feat_out = self.num_feat_chan * (size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2))

    def forward(self, hm_list, encoding_list):
        x = self.heatmap_conv(hm_list[-1]) + self.encoding_conv(encoding_list[-1])
        if len(encoding_list) > 1:
            x = x + encoding_list[-2]

        # x: B x num_feat_chan x 32 x 32
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            # if i != self.nRegBlock - 1: ### change llj ##
                ### x = self.maxpool(x)
            x = self.maxpool(x)

        # x: B x num_feat_chan x 2 x 2
        out = x.view(x.size(0), -1)

        return out
