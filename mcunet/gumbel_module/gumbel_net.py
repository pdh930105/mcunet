from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from fvcore.nn import FlopCountAnalysis


from ..tinynas.nn.modules import MBInvertedConvLayer
from ..tinynas.nn.networks import MobileInvertedResidualBlock

from .gumbel_layer import MBGumbelInvertedConvLayer, MobileGumbelInvertedResidualBlock

from ..utils import MyModule, MyNetwork, SEModule, build_activation, get_same_padding, sub_filter_start_end, rm_bn_from_net, has_deep_attr, get_deep_attr, set_deep_attr
from ..tinynas.nn.modules import ZeroLayer, set_layer_from_config

        
        
class GumbelMCUNet(MyNetwork):
    def __init__(self, first_conv, blocks, feature_mix_layer, classifier, gumbel_feature_extract_block_idx):
        super(GumbelMCUNet, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.gumbel_feature_extract_block_idx = gumbel_feature_extract_block_idx
        
        self.gumbel_index_list = []
        for i, block in enumerate(self.blocks):
            if i < self.gumbel_feature_extract_block_idx:
                continue
            if len(block.mobile_inverted_conv.expand_ratio_list) > 1:
                self.gumbel_index_list.append(len(block.mobile_inverted_conv.expand_ratio_list))
                
            if len(block.mobile_inverted_conv.kernel_size_list) > 1:
                self.gumbel_index_list.append(len(block.mobile_inverted_conv.kernel_size_list))
        
        
        self.gumbel_input_channel = blocks[gumbel_feature_extract_block_idx].mobile_inverted_conv.out_channels
        
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
            if i == self.gumbel_feature_extract_block_idx:
                # feautre map and gumbel output extract
                gumbel_output = self.gumbel_block(x)
                gumbel_output = gumbel_output.view(-1, sum(self.gumbel_index_list))
                break
            x = block(x)

        gumbel_index = 0
        gumbel_one_hot_list = []
        for j, block in enumerate(self.blocks[self.gumbel_feature_extract_block_idx:]):
            expand_index, kernel_index = len(block.mobile_inverted_conv.expand_ratio_list), len(block.mobile_inverted_conv.kernel_size_list)
            if self.training:            
                if expand_index > 1 and kernel_index > 1:
                    gumbel_input_expand = gumbel_output[:, gumbel_index: gumbel_index + expand_index]
                    gumbel_one_hot_expand = F.gumbel_softmax(gumbel_input_expand, tau=1, hard=True)
                    gumbel_input_kernel = gumbel_output[:, gumbel_index + expand_index: gumbel_index + expand_index + kernel_index]
                    gumbel_one_hot_kernel = F.gumbel_softmax(gumbel_input_kernel, tau=1, hard=True)
                    gumbel_one_hot = torch.cat([gumbel_one_hot_expand, gumbel_one_hot_kernel], dim=-1)
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
                    gumbel_input_expand = gumbel_output[:, gumbel_index: gumbel_index + expand_index]
                    gumbel_one_hot_expand = gumbel_input_expand.max(dim=-1, keepdim=True)[1]
                    gumbel_one_hot_expand = torch.zeros_like(gumbel_input_expand, memory_format=torch.legacy_contiguous_format).scatter_(-1, gumbel_one_hot_expand, 1.0)
                    gumbel_input_kernel = gumbel_output[:, gumbel_index + expand_index: gumbel_index + expand_index + kernel_index]
                    gumbel_one_hot_kernel = gumbel_input_kernel.max(dim=-1, keepdim=True)[1]
                    gumbel_one_hot_kernel = torch.zeros_like(gumbel_input_kernel, memory_format=torch.legacy_contiguous_format).scatter_(-1, gumbel_one_hot_kernel, 1.0)
                    gumbel_one_hot = torch.cat([gumbel_one_hot_expand, gumbel_one_hot_kernel], dim=-1)                    
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
                #print("check gumbel one hot :", gumbel_one_hot)
                x = block(x, gumbel_one_hot)
            
            gumbel_one_hot_list.append(gumbel_one_hot)    
                
        
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x, gumbel_one_hot_list
        
    def forward_original(self, x):
        x = self.first_conv(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
    
    def set_static_flops(self, x):
        flops = 0
        flops += FlopCountAnalysis(self.first_conv, x).total()
        x = self.first_conv(x)
        for i, block in enumerate(self.blocks):
            if i == self.gumbel_feature_extract_block_idx:
                # feautre map and gumbel output extract
                gumbel_output = self.gumbel_block(x)
                gumbel_output = gumbel_output.view(-1, sum(self.gumbel_index_list))
                flops += FlopCountAnalysis(self.gumbel_block, x).total()
                break
            
            rm_bn_block = copy.deepcopy(block)
            rm_bn_from_net(rm_bn_block)
            flops += FlopCountAnalysis(rm_bn_block, x).total()
            x = block(x)
            del rm_bn_block
        self.static_flops = flops
        self.dynamic_flops = 0
        flops = 0
        for j, block in enumerate(self.blocks[self.gumbel_feature_extract_block_idx:]):
            rm_bn_block = copy.deepcopy(block)
            rm_bn_from_net(rm_bn_block)
            flops += FlopCountAnalysis(rm_bn_block, x).total()
            del rm_bn_block
            x = block(x)
            
        self.dynamic_flops += flops
        flops = 0
        if self.feature_mix_layer:
            flops += FlopCountAnalysis(self.feature_mix_layer, x).total()
            x = self.feature_mix_layer(x)
            
        x = x.mean(3).mean(2)
        flops = FlopCountAnalysis(self.classifier, x).total()
        x = self.classifier(x)
        self.static_flops += flops
        print(f"Success Log Static & Dynamic Flops : {self.static_flops}, {self.dynamic_flops}")
        
    def compute_flops(self, gumbel_one_hot_list):
        flops = 0
        for j, block in enumerate(self.blocks[self.gumbel_feature_extract_block_idx:]):
            expand_index, kernel_index = len(block.mobile_inverted_conv.expand_ratio_list), len(block.mobile_inverted_conv.kernel_size_list)
            if expand_index > 1 and kernel_index > 1:
                flops += block.count_flops(gumbel_one_hot_list[j])
            elif expand_index > 1:
                flops += block.count_flops(gumbel_one_hot_list[j])
            elif kernel_index >1:
                flops += block.count_flops(gumbel_one_hot_list[j])
            #else:
            #    flops += block.compute_flops(x)
        flops = flops * 1e-6 # MFLOPs
        return flops
    
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
            'name': GumbelMCUNet.__name__,
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
        gumbel_feature_extract_block_idx = gumbel_config['gumbel_feature_extract_block_idx']
        
        first_conv = set_layer_from_config(net_config['first_conv'])
        feature_mix_layer = set_layer_from_config(net_config['feature_mix_layer'])
        classifier = set_layer_from_config(net_config['classifier'])
        
        blocks = []
        
        for i, block_config in enumerate(net_config['blocks']):
            if i < gumbel_feature_extract_block_idx:
                blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
            else:
                blocks.append(MobileGumbelInvertedResidualBlock.build_from_config(block_config))
        
        net = GumbelMCUNet(first_conv, blocks, feature_mix_layer, classifier, gumbel_feature_extract_block_idx)
        
        if 'bn' in net_config:
            net.set_bn_param(**net_config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        
        return net
    
    def load_pretrained_mcunet_param(self, mcunet):
        
        for n, p in self.named_parameters():
            if has_deep_attr(mcunet, n):
                set_deep_attr(self, n, get_deep_attr(mcunet, n))
        
        for n, p in self.named_buffers():
            if has_deep_attr(mcunet, n):
                set_deep_attr(self, n, get_deep_attr(mcunet, n))
        
        print(f"load pretrained mcumodel to gumbel net")
        