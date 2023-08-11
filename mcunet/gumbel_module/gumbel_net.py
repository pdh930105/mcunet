from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ..tinynas.nn.modules import MBInvertedConvLayer
from ..tinynas.nn.networks import MobileInvertedResidualBlock

from ..utils import MyModule, MyNetwork, SEModule, build_activation, get_same_padding, sub_filter_start_end
from ..tinynas.nn.modules import ZeroLayer, set_layer_from_config



def get_deep_attr(obj, attrs):
    for attr in attrs.split("."):
        obj = getattr(obj, attr)
    return obj

def has_deep_attr(obj, attrs):
    try:
        get_deep_attr(obj, attrs)
        return True
    except AttributeError:
        return False

def set_deep_attr(obj, attrs, value):
    for attr in attrs.split(".")[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs.split(".")[-1], value)
    



class MBGumbelInvertedConvLayer(MyModule):
    global_kernel_size_list = [3,5,7]
    global_expand_ratio_list = [1,3,4,5,6]
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False, **kwargs):
        super(MBGumbelInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_kernel_size = kernel_size
        self.kernel_size_list = []
        self.stride = stride
        self.max_expand_ratio = expand_ratio
        self.expand_ratio_list = []
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        
        
        if self.max_kernel_size in self.global_kernel_size_list:
            for kernel in sorted(self.global_kernel_size_list):
                if kernel == self.max_kernel_size:
                    self.kernel_size_list.append(kernel)
                    break
                self.kernel_size_list.append(kernel)
            
            self.kernel_size_list.reverse() # sorted in descending order
        
        else:
            self.kernel_size_list = [self.max_kernel_size]
        
        if self.max_expand_ratio in self.global_expand_ratio_list:        
            for expand in sorted(self.global_expand_ratio_list):
                if expand == self.max_expand_ratio:
                    self.expand_ratio_list.append(expand)
                    break
                self.expand_ratio_list.append(expand)
        
        else:
            self.expand_ratio_list = [self.max_expand_ratio]
        

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.max_expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.max_expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        pad = get_same_padding(self.max_kernel_size)
        depth_conv_modules = [
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True))
        ]
        if self.use_se:
            depth_conv_modules.append(('se', SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

        self.kernel_transform_linear_list = nn.ModuleList()
        
        for i, kernel in enumerate(self.kernel_size_list[1:]):
            kernel_linear = nn.Linear(kernel*kernel, kernel*kernel)
            self.kernel_transform_linear_list.append(kernel_linear)

    def forward(self, x, gumbel=None):
        """
        gumbel: [batch_size, len(self.expand_ratio_list) + len(self.kernel_size_list)]
        """
        if gumbel==None:
            if self.inverted_bottleneck:
                x = self.inverted_bottleneck(x)
            x = self.depth_conv(x)
            x = self.point_linear(x)
            return x
        else:
            if len(self.expand_ratio_list) == 1: ## 
                if len(self.kernel_size_list) == 1:
                    if self.inverted_bottleneck:
                        x = self.inverted_bottleneck(x)
                    x = self.depth_conv(x)
                    x = self.point_linear(x)
                    return x
                else:
                    # expand_ratio only one, multiple kernel size, so gumbel length is multple of kernel size
                    assert len(gumbel[0]) == len(self.kernel_size_list), "gumbel size is not match with kernel_size_list"
                    if self.inverted_bottleneck:
                        x = self.inverted_bottleneck(x)
                    depth_weight = self.depth_conv.conv.weight
                    pad = get_same_padding(self.max_kernel_size)
                    kernel_max_out = F.conv2d(x, depth_weight, stride=self.stride, padding=pad, groups=x.size(1))
                    kernel_max_out = self.depth_conv.bn(kernel_max_out)
                    kernel_max_out = self.depth_conv.act(kernel_max_out)
                    kernel_max_out *= gumbel[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    for i, active_kernel_size in enumerate(self.kernel_size_list[1:]):
                        start, end = sub_filter_start_end(self.kernel_size_list[i], active_kernel_size)
                        kernel_weight = depth_weight[:, :, start:end, start:end].contiguous()
                        kernel_weight = kernel_weight.view(kernel_weight.size(0), kernel_weight.size(1), -1)
                        kernel_weight = self.kernel_transform_linear_list[i](kernel_weight)
                        kernel_weight = kernel_weight.view(kernel_weight.size(0), kernel_weight.size(1), active_kernel_size, active_kernel_size)
                        pad = get_same_padding(active_kernel_size)
                        kernel_out = F.conv2d(x, kernel_weight, stride=self.stride, padding=pad, groups=x.size(1))
                        kernel_out = self.depth_conv.bn(kernel_out)
                        kernel_out = self.depth_conv.act(kernel_out)
                        kernel_out *= gumbel[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        kernel_max_out += kernel_out
                    x = kernel_max_out
                    if self.use_se:
                        x = self.depth_conv.se(x)
                    # 3. pointwise convolution weights (out_channels)
                    x = self.point_linear(x)
                    return x
            
            elif len(self.kernel_size_list) == 1:
                # kernel size only one, multiple expand ratio, so gumbel length is multple of expand ratio
                assert len(gumbel[0]) == len(self.expand_ratio_list), "gumbel size is not match with expand_ratio_list"
                
                if self.inverted_bottleneck:
                    # 1. inverted bottleneck weights (max_expand_ratio)
                    expand_weight = self.inverted_bottleneck.conv.weight
                    expand_max_out = F.conv2d(x, expand_weight, stride=1, padding=0)
                    expand_max_out = self.inverted_bottleneck.bn(expand_max_out)
                    expand_max_out = self.inverted_bottleneck.act(expand_max_out)
                    expand_max_out *= gumbel[:, -1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    for i, expand_ratio in enumerate(self.expand_ratio_list[:-1]):
                        out = F.conv2d(x, expand_weight[:expand_ratio*self.in_channels, :, :, :], stride=1, padding=0)
                        out = F.batch_norm(out, self.inverted_bottleneck.bn.running_mean[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.running_var[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.weight[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.bias[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.training, self.inverted_bottleneck.bn.momentum, self.inverted_bottleneck.bn.eps)
                        out = self.inverted_bottleneck.act(out)
                        out *= gumbel[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        out = F.pad(out, [0, 0, 0, 0, 0, expand_max_out.size(1) - out.size(1)], mode='constant', value=0) # zero pad
                        expand_max_out += out
                    x = expand_max_out
                x = self.depth_conv(x)
                x = self.point_linear(x)
                return x
                
            elif len(gumbel[0]) == len(self.expand_ratio_list) + len(self.kernel_size_list):
                if self.inverted_bottleneck:
                    # 1. inverted bottleneck weights (max_expand_ratio)
                    expand_weight = self.inverted_bottleneck.conv.weight
                    expand_max_out = F.conv2d(x, expand_weight, stride=1, padding=0)
                    expand_max_out = self.inverted_bottleneck.bn(expand_max_out)
                    expand_max_out = self.inverted_bottleneck.act(expand_max_out)
                    expand_max_out *= gumbel[:, len(self.expand_ratio_list)-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    for i, expand_ratio in enumerate(self.expand_ratio_list[:-1]):
                        out = F.conv2d(x, expand_weight[:expand_ratio*self.in_channels, :, :, :], stride=1, padding=0)
                        out = F.batch_norm(out, self.inverted_bottleneck.bn.running_mean[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.running_var[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.weight[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.bias[:expand_ratio*self.in_channels], self.inverted_bottleneck.bn.training, self.inverted_bottleneck.bn.momentum, self.inverted_bottleneck.bn.eps)
                        out = self.inverted_bottleneck.act(out)
                        out *= gumbel[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        out = F.pad(out, [0, 0, 0, 0, 0, expand_max_out.size(1) - out.size(1)], mode='constant', value=0) # zero pad
                        expand_max_out += out
                    x = expand_max_out
                # 2. depthwise convolution weights (max_kernel_size)
                depth_weight = self.depth_conv.conv.weight
                pad = get_same_padding(self.max_kernel_size)
                kernel_max_out = F.conv2d(x, depth_weight, stride=self.stride, padding=pad, groups=x.size(1))
                kernel_max_out = self.depth_conv.bn(kernel_max_out)
                kernel_max_out = self.depth_conv.act(kernel_max_out)
                kernel_max_out *= gumbel[:, len(self.expand_ratio_list)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                for i, active_kernel_size in enumerate(self.kernel_size_list[1:]):
                    start, end = sub_filter_start_end(self.kernel_size_list[i], active_kernel_size)
                    kernel_weight = depth_weight[:, :, start:end, start:end].contiguous()
                    kernel_weight = kernel_weight.view(kernel_weight.size(0), kernel_weight.size(1), -1)
                    kernel_weight = self.kernel_transform_linear_list[i](kernel_weight)
                    kernel_weight = kernel_weight.view(kernel_weight.size(0), kernel_weight.size(1), active_kernel_size, active_kernel_size)
                    pad = get_same_padding(active_kernel_size)
                    kernel_out = F.conv2d(x, kernel_weight, stride=self.stride, padding=pad, groups=x.size(1))
                    kernel_out = self.depth_conv.bn(kernel_out)
                    kernel_out = self.depth_conv.act(kernel_out)
                    kernel_out *= gumbel[:, len(self.expand_ratio_list) + i+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    kernel_max_out += kernel_out
                x = kernel_max_out
                if self.use_se:
                    x = self.depth_conv.se(x)
                # 3. pointwise convolution weights (out_channels)
                x = self.point_linear(x)
                return x
            else:
                assert False, "gumbel size is not match with expand_ratio_list and kernel_size_list"
            
    
    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.max_expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = '%dx%d_GumbelMBConv%d_%s' % (self.max_kernel_size, self.max_kernel_size, expand_ratio, self.act_func.upper())
        if self.use_se:
            layer_str = 'SE_' + layer_str
        layer_str += '_O%d' % self.out_channels
        return layer_str

    @property
    def config(self):
        return {
            'name': MBGumbelInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.max_kernel_size,
            'kernel_size_list': self.kernel_size_list,
            'stride': self.stride,
            'expand_ratio': self.max_expand_ratio,
            'expand_ratio_list': self.expand_ratio_list,
            'mid_channels': self.mid_channels,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return MBGumbelInvertedConvLayer(**config)
    
    #@staticmethod
    #def build_from_module(module: MBInvertedConvLayer):
    #    mbgumbel = MBGumbelInvertedConvLayer.build_from_config(module.config)
    #    for n, m in module.named_parameters():
            


class MobileGumbelInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileGumbelInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x, gumbel_idx=None):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer) and gumbel_idx == None:
            res = self.mobile_inverted_conv(x)
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer) and gumbel_idx != None:
            res = self.mobile_inverted_conv(x, gumbel_idx)
        elif self.shortcut is not None and gumbel_idx == None:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        else:
            res = self.mobile_inverted_conv(x, gumbel_idx) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileGumbelInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = MBGumbelInvertedConvLayer.build_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileGumbelInvertedResidualBlock(mobile_inverted_conv, shortcut)

    @staticmethod
    def build_from_module(module):
        if isinstance(module, MobileGumbelInvertedResidualBlock):
            print("build from gumbel module")
            return module
        elif isinstance(module, MobileInvertedResidualBlock):
            print("build from normal MobileInvertedResidualBlock module")
            mobile_inverted_conv = module.mobile_inverted_conv
            shortcut = module.shortcut
            return MobileGumbelInvertedResidualBlock(module.mobile_inverted_conv, module.shortcut)
        
        
class GumbelMCUNets(MyNetwork):
    def __init__(self, first_conv, blocks, feature_mix_layer, classifier, gumbel_feature_extract_block):
        super(GumbelMCUNets, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.gumbel_feature_extract_block = gumbel_feature_extract_block
        
        self.gumbel_index_list = []
        for i, block in enumerate(self.blocks):
            if i < self.gumbel_feature_extract_block:
                continue
            if len(block.mobile_inverted_conv.expand_ratio_list) > 1:
                self.gumbel_index_list.append(len(block.mobile_inverted_conv.expand_ratio_list))
                
            if len(block.mobile_inverted_conv.kernel_size_list) > 1:
                self.gumbel_index_list.append(len(block.mobile_inverted_conv.kernel_size_list))
        
        
        self.gumbel_input_channel = blocks[gumbel_feature_extract_block].mobile_inverted_conv.out_channels
        
        self.avgpool_policy = nn.AdaptiveAvgPool2d((8, 8))
        self.gumbel_features_flatten = nn.Flatten()
        self.gumbel_fc1 = nn.Linear(self.gumbel_input_channel*8*8, 256)
        self.dropout = nn.Dropout(0.2)
        self.gumbel_fc2 = nn.Linear(256, sum(self.gumbel_index_list))
        
        self.gumbel_block = nn.Sequential(self.avgpool_policy, 
                                           self.gumbel_features_flatten, 
                                           self.gumbel_fc1, 
                                           self.dropout, 
                                           self.gumbel_fc2)
        
        
    def forward(self, x):
        x = self.first_conv(x)
        for i, block in enumerate(self.blocks):            
            if i == self.gumbel_feature_extract_block:
                # feautre map and gumbel output extract
                gumbel_output = self.gumbel_block(x)
                gumbel_output = gumbel_output.view(-1, sum(self.gumbel_index_list))
                break
            x = block(x)

        gumbel_index = 0
        gumbel_one_hot_list = []
        for j, block in enumerate(self.blocks[self.gumbel_feature_extract_block:]):
            expand_index, kernel_index = len(block.mobile_inverted_conv.expand_ratio_list), len(block.mobile_inverted_conv.kernel_size_list)
            
            if self.training:            
                if expand_index > 1 and kernel_index > 1:
                    gumbel_input = gumbel_output[:, gumbel_index: gumbel_index + expand_index  + kernel_index]
                    gumbel_one_hot = F.gumbel_softmax(gumbel_input, tau=1, hard=True)
                    gumbel_index += expand_index + kernel_index
                    
                elif expand_index > 1:
                    gumbel_input = gumbel_output[:, gumbel_index: gumbel_index + expand_index]
                    gumbel_one_hot = F.gumbel_softmax(gumbel_input, tau=1, hard=True)
                    gumbel_index += expand_index
                elif kernel_index >1:
                    gumbel_input = gumbel_output[:, gumbel_index: gumbel_index + kernel_index]
                    gumbel_one_hot = F.gumbel_softmax(gumbel_input, tau=1, hard=True)
                    gumbel_index += kernel_index
                else:
                    gumbel_one_hot = None
                x = block(x, gumbel_one_hot)    
        
            else:
                if expand_index > 1 and kernel_index > 1:
                    gumbel_input = gumbel_output[:, gumbel_index: gumbel_index + expand_index  + kernel_index]
                    index = gumbel_input.max(dim=-1, keepdim=True)[1]
                    gumbel_one_hot = torch.zeros_like(gumbel_input, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
                    gumbel_index += expand_index + kernel_index
                    
                elif expand_index > 1:
                    gumbel_input = gumbel_output[:, gumbel_index: gumbel_index + expand_index]
                    index = gumbel_input.max(dim=-1, keepdim=True)[1]
                    gumbel_one_hot = torch.zeros_like(gumbel_input, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
                    gumbel_index += expand_index
                elif kernel_index >1:
                    gumbel_input = gumbel_output[:, gumbel_index: gumbel_index + kernel_index]
                    index = gumbel_input.max(dim=-1, keepdim=True)[1]
                    gumbel_one_hot = torch.zeros_like(gumbel_input, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
                    gumbel_index += kernel_index
                else:
                    gumbel_one_hot = None
                x = block(x, gumbel_one_hot)
            
            gumbel_one_hot_list.append(gumbel_one_hot)    
                
        
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
        
    def forward_original(self, x):
        x = self.first_conv(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
    
    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str
        
    @property
    def config(self):
        return {
            'name': GumbelMCUNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }
    
    
    @staticmethod
    def build_from_config(net_config, gumbel_config):
        MBGumbelInvertedConvLayer.global_expand_ratio_list = gumbel_config['global_expand_ratio_list']
        MBGumbelInvertedConvLayer.global_kernel_size_list = gumbel_config['global_kernel_size_list']
        gumbel_feature_extract_block = gumbel_config['gumbel_feature_extract_block']
        
        first_conv = set_layer_from_config(net_config['first_conv'])
        feature_mix_layer = set_layer_from_config(net_config['feature_mix_layer'])
        classifier = set_layer_from_config(net_config['classifier'])
        
        blocks = []
        
        for i, block_config in enumerate(net_config['blocks']):
            if i < gumbel_feature_extract_block:
                print(i, block_config)
                blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
            else:
                blocks.append(MobileGumbelInvertedResidualBlock.build_from_config(block_config))
        
        net = GumbelMCUNets(first_conv, blocks, feature_mix_layer, classifier, gumbel_feature_extract_block)
        
        if 'bn' in net_config:
            net.set_bn_param(**net_config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        
        return net
    
    def load_pretrained_mcunet_param(self, mcunet):
        
        for n, p in self.named_parameters():
            if has_deep_attr(mcunet, n):
                print("load {} params ({})".format(n, p.shape))
                set_deep_attr(self, n, get_deep_attr(mcunet, n))
        