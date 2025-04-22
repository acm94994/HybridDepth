import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from huggingface_hub import PyTorchModelHubMixin
from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super().__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1)
            for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels),
                    nn.GELU()
                ) for _ in range(len(out_channels))
            ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features, nclass, kernel_size=1)
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.ReLU(True),
                nn.Identity()
            )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        l1 = self.scratch.layer1_rn(layer_1)
        l2 = self.scratch.layer2_rn(layer_2)
        l3 = self.scratch.layer3_rn(layer_3)
        l4 = self.scratch.layer4_rn(layer_4)

        p4 = self.scratch.refinenet4(l4, size=l3.shape[2:])
        p3 = self.scratch.refinenet3(p4, l3, size=l2.shape[2:])
        p2 = self.scratch.refinenet2(p3, l2, size=l1.shape[2:])
        p1 = self.scratch.refinenet1(p2, l1)

        out = self.scratch.output_conv1(p1)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']

        if encoder == 'vits':
            model_name = 'vit_small_patch14_dino'
        elif encoder == 'vitb':
            model_name = 'vit_base_patch14_dino'
        elif encoder == 'vitl':
            model_name = 'vit_large_patch14_dino'

        self.pretrained = timm.create_model(model_name, pretrained=False)
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        # Add get_intermediate_layers function using hooks
        def get_intermediate_layers(x, n=4, return_class_token=True):
            outputs = []
            hooks = []

            def save_output(module, input, output):
                outputs.append(output)

            for blk in self.pretrained.blocks[-n:]:
                hooks.append(blk.register_forward_hook(save_output))

            _ = self.pretrained.forward_features(x)
            for h in hooks:
                h.remove()

            if return_class_token:
                return [(o[:, 1:], o[:, 0]) for o in outputs]
            else:
                return [(o[:, 1:],) for o in outputs]

        self.pretrained.get_intermediate_layers = get_intermediate_layers
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels, use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        return F.relu(depth).squeeze(1)


class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="vits", type=str, choices=["vits", "vitb", "vitl"])
    args = parser.parse_args()

    model = DepthAnything({
        "encoder": args.encoder,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "use_bn": False,
        "use_clstoken": False,
    })
    print(model)
