from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ..tinynas.nn.modules import MBInvertedConvLayer
from ..tinynas.nn.networks import MobileInvertedResidualBlock

from ..tinynas.nn.modules import ZeroLayer, set_layer_from_config
from ..utils import MyModule, MyNetwork, SEModule, build_activation, get_same_padding, sub_filter_start_end


def count_conv_gumbel_flops(weight_shape, width, height, batch=1, stride=1):
    out_c, in_c , k_w, k_h = weight_shape
    stride = stride
    if isinstance(stride, tuple) or isinstance(stride, list):
        stride = stride[0]
    flops = batch * out_c * width * height * in_c * k_w * k_h / stride / stride
    return flops

class DynamicGumbelBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim, **kwargs):
        super(DynamicGumbelBatchNorm2d, self).__init__()
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)
        
    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicGumbelBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0
            
            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                if bn.momentum is None: # use cumulative moving average
                    exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                else: # use exponential moving average
                    exponential_average_factor = bn.momentum
            return F.batch_norm(x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim], bn.bias[:feature_dim], 
                                bn.training or not bn.track_running_stats, exponential_average_factor, bn.eps)
    
    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y

class MBGumbelInvertedConvLayer(MyModule):
    global_kernel_size_list = [3,5,7]
    global_expand_ratio_list = [1,3,4,5,6]
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False, **kwargs):
        super(MBGumbelInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_kernel_size = kernel_size
        self.stride = stride
        self.max_expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.initialize_flops = False
        self.kernel_transform_linear_list = nn.ModuleList()

        kernel_size_list = []
        expand_ratio_list = []
        
        if self.max_kernel_size in self.global_kernel_size_list:
            for kernel in sorted(self.global_kernel_size_list):
                if kernel == self.max_kernel_size:
                    kernel_size_list.append(kernel)
                    break
                kernel_size_list.append(kernel)
            
            kernel_size_list.reverse() # sorted in descending order
        
        else:
            kernel_size_list = [self.max_kernel_size]
                
        if self.max_expand_ratio in self.global_expand_ratio_list:        
            for expand in sorted(self.global_expand_ratio_list):
                if expand == self.max_expand_ratio:
                    expand_ratio_list.append(expand)
                    break
                expand_ratio_list.append(expand)
        
        else:
            expand_ratio_list = [self.max_expand_ratio]

        for i, kernel in enumerate(kernel_size_list[1:]):
            kernel_linear = nn.Linear(kernel*kernel, kernel*kernel)
            self.kernel_transform_linear_list.append(kernel_linear)

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

        #self.kernel_size_list = kernel_size_list
        #self.expand_ratio_list =expand_ratio_list
        self.register_buffer('kernel_size_list', torch.tensor(kernel_size_list, dtype=torch.long))
        self.register_buffer('expand_ratio_list', torch.tensor(expand_ratio_list, dtype=torch.long))
        

    def forward(self, x, gumbel=None):
        """
        gumbel: [batch_size, len(self.expand_ratio_list) + len(self.kernel_size_list)]
        """
        self.input_width, self.input_height = x.size(2), x.size(3)
        
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


    def compute_flops(self):
        assert hasattr(self, 'input_width'), "please run forward at least once"
        flops = 0
        self.inverted_flops = count_conv_gumbel_flops(self.inverted_bottleneck.conv.weight.shape, self.input_width, self.input_height)
        self.dw_flops = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, stride = self.depth_conv.conv.stride[0])
        self.pw_flops = count_conv_gumbel_flops(self.point_linear.conv.weight.shape, self.input_width / self.depth_conv.conv.stride[0], self.input_height / self.depth_conv.conv.stride[0])
        
        return self.inverted_flops + self.dw_flops + self.pw_flops
    
    def compute_gumbel_flops(self, gumbel_idx=None):
        
        assert hasattr(self, 'input_width'), "please run forward at least once"
        
        if self.initialize_flops == False:
            if (len(self.expand_ratio_list) == 1 and len(self.kernel_size_list) == 1) or gumbel_idx==None:
                print("expand ratio list = 1, kernel size list = 1")
                inverted_flops = 0
                if self.inverted_bottleneck:
                    inverted_flops += count_conv_gumbel_flops(self.inverted_bottleneck.conv.weight.shape, self.input_width, self.input_height)
                    print("inverted blocks flops :", inverted_flops)
                dw_flops = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, stride = self.depth_conv.conv.stride[0])
                print("depth conv flops :", dw_flops)
                pw_flops = count_conv_gumbel_flops(self.point_linear.conv.weight.shape, self.input_width / self.depth_conv.conv.stride[0], self.input_height / self.depth_conv.conv.stride[0])
                print("pw flops :", pw_flops)
                self.inverted_flops = inverted_flops
                self.dw_flops = dw_flops
                self.pw_flops = pw_flops
                
            elif len(self.expand_ratio_list) == 1 and len(self.kernel_size_list) > 1:
                print("expand ratio list = 1, kernel size list > 1")
                
                inverted_flops = 0
                if self.inverted_bottleneck:
                    inverted_flops = count_conv_gumbel_flops(self.inverted_bottleneck.conv.weight.shape, self.input_width, self.input_height)
                    print("inverted blocks flops :", inverted_flops)
                dw_flops_tensor = torch.zeros(len(self.kernel_size_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, kernel in enumerate(self.kernel_size_list):
                    if kernel == self.max_kernel_size:
                        dw_flops_tensor[i] = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                    else:
                        out_c, in_c, h, w = self.depth_conv.conv.weight.shape
                        convert_flops = self.convert_kernel_flops(kernel)
                        compute_flops = count_conv_gumbel_flops((out_c, in_c, kernel, kernel), self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                        dw_flops_tensor[i] = convert_flops + compute_flops
                
                print("dw blocks flops :", dw_flops_tensor)
                pw_flops = count_conv_gumbel_flops(self.point_linear.conv.weight.shape, self.input_width / self.depth_conv.conv.stride[0], self.input_height / self.depth_conv.conv.stride[0])
                print("pw flops :", pw_flops)
                
                self.inverted_flops = inverted_flops
                self.dw_flops = dw_flops_tensor
                self.pw_flops = pw_flops
            
            elif len(self.expand_ratio_list) > 1 and len(self.kernel_size_list) == 1:
                print("expand ratio list > 1, kernel size list = 1")
                inverted_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, in_c, h, w = self.inverted_bottleneck.conv.weight.shape
                    inverted_flops_tensor[i] = count_conv_gumbel_flops((in_c*expand, in_c, h, w), self.input_width, self.input_height)
                print("inverted blocks flops :", inverted_flops_tensor)                   
                self.inverted_flops = inverted_flops_tensor
                dw_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    _, _, h, w = self.depth_conv.conv.weight.shape
                    dw_flops = count_conv_gumbel_flops((in_c*expand, 1, h, w), self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                    dw_flops_tensor[i] += dw_flops
                print("dw blocks flops :", dw_flops_tensor)                   
                self.dw_flops = dw_flops_tensor
                pw_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, _, h, w = self.point_linear.conv.weight.shape
                    pw_flops = count_conv_gumbel_flops((out_c, in_c*expand, h, w), self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                    pw_flops_tensor[i] += pw_flops
                print("pw blocks flops :", pw_flops_tensor)
                self.pw_flops = pw_flops_tensor
            
            else: # expand_ratio_list > 1 and kernel_size_list > 1
                """
                print("^^"*20)
                print(f"expand ratio list > 1 ({len(self.expand_ratio_list)}), kernel size list > 1 ({len(self.kernel_size_list)})")
                print(f"module config")
                print(f"input h/w : {self.input_height} / {self.input_width}")
                print(f"stride : {self.depth_conv.conv.stride[0]}")
                print(f"inverted conv input shape : {self.inverted_bottleneck.conv.weight.shape}")
                print(f"depth conv weight shape : {self.depth_conv.conv.weight.shape}")
                print(f"point conv weight shape : {self.point_linear.conv.weight.shape}")
                print(f"max expand size : {self.max_expand_ratio}")
                print(f"max kernel size : {self.max_kernel_size}")
                print(f"expand ratio check :", self.max_expand_ratio * self.inverted_bottleneck.conv.weight.shape[1] == self.depth_conv.conv.weight.shape[0])
                print("^^"*20)
                """
                inverted_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, in_c, h, w = self.inverted_bottleneck.conv.weight.shape
                    inverted_flops_tensor[i] = count_conv_gumbel_flops((in_c*expand, in_c, h, w), self.input_width, self.input_height)
                #print("inverted blocks flops :", inverted_flops_tensor)                   
                self.inverted_flops = inverted_flops_tensor
                dw_flops_tensor = torch.zeros(len(self.kernel_size_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                convert_flops_tensor = torch.zeros(len(self.kernel_size_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, kernel in enumerate(self.kernel_size_list):
                    if kernel == self.max_kernel_size:
                        dw_flops_tensor[i] = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, stride=self.depth_conv.conv.stride[0])
                        convert_flops_tensor[i] = 0
                    else:
                        out_c, _, h, w = self.depth_conv.conv.weight.shape
                        convert_flops = self.convert_kernel_flops(kernel)
                        compute_flops = count_conv_gumbel_flops((out_c, 1, kernel, kernel), self.input_width, self.input_height, stride= self.depth_conv.conv.stride[0])
                        convert_flops_tensor[i] = convert_flops
                        dw_flops_tensor[i] = compute_flops
                        
                #print("dw convert flops :, ", convert_flops_tensor)                
                #print("dw blocks flops :", dw_flops_tensor)
                self.convert_flops = convert_flops_tensor
                self.dw_flops = dw_flops_tensor
                pw_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, _, h, w = self.point_linear.conv.weight.shape
                    pw_flops = count_conv_gumbel_flops((out_c, in_c*expand, h, w), self.input_width, self.input_height, stride=self.depth_conv.conv.stride[0])
                    pw_flops_tensor[i] += pw_flops
                #print("pw blocks flops :", pw_flops_tensor)
                self.pw_flops = pw_flops_tensor
            
            self.initialize_flops = True
        
        # 2. compute gumbel value return
        if len(self.expand_ratio_list) == 1 and len(self.kernel_size_list) == 1:
            return (self.inverted_flops + self.dw_flops + self.pw_flops).sum(dim=1)
        elif len(self.expand_ratio_list) > 1 and len(self.kernel_size_list) == 1:
            return (gumbel_idx * (self.inverted_flops + self.dw_flops + self.pw_flops)).sum(dim=1)
        elif len(self.expand_ratio_list) == 1 and len(self.kernel_size_list) > 1:
            return (gumbel_idx * (self.inverted_flops + self.dw_flops + self.pw_flops)).sum(dim=1)
        else:
            gumbel_expand, gumbel_kernel = gumbel_idx[:, :len(self.expand_ratio_list)], gumbel_idx[:, len(self.expand_ratio_list):]
            expand_flops = gumbel_expand * (self.inverted_flops + self.pw_flops)
            expand_kernel_flops = (gumbel_expand * self.expand_ratio_list / self.max_expand_ratio).sum(dim=1) 
            expand_kernel_flops = expand_kernel_flops.unsqueeze(1) * (gumbel_kernel * (self.dw_flops)) + gumbel_kernel * (self.convert_flops) 
            expand_flops = expand_flops.sum(dim=1)
            expand_kernel_flops = expand_kernel_flops.sum(dim=1)
            return expand_flops + expand_kernel_flops
        

    """    
    def compute_count_flops(self, gumbel_idx=None):
        assert hasattr(self, 'input_width'), "please run forward at least once"
        if gumbel_idx == None:
            inverted_flops = 0
            if self.inverted_bottleneck:
                inverted_flops += count_conv_gumbel_flops(self.inverted_bottleneck.conv.weight.shape, self.input_width, self.input_height)
                print("inverted blocks flops :", inverted_flops)
            dw_flops = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, stride = self.depth_conv.conv.stride[0])
            print("depth conv flops :", dw_flops)
            pw_flops = count_conv_gumbel_flops(self.point_linear.conv.weight.shape, self.input_width / self.depth_conv.conv.stride[0], self.input_height / self.depth_conv.conv.stride[0])
            print("pw flops :", pw_flops)
            return inverted_flops + dw_flops + pw_flops
        
        else:
            if len(self.expand_ratio_list) == 1 and len(self.kernel_size_list) == 1:
                print("expand ratio list = 1, kernel size list = 1")
                
                inverted_flops = 0
                if self.inverted_bottleneck:
                    inverted_flops += count_conv_gumbel_flops(self.inverted_bottleneck.conv.weight.shape, self.input_width, self.input_height)
                    print("inverted blocks flops :", inverted_flops)
                dw_flops = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, stride = self.depth_conv.conv.stride[0])
                print("depth conv flops :", dw_flops)
                pw_flops = count_conv_gumbel_flops(self.point_linear.conv.weight.shape, self.input_width / self.depth_conv.conv.stride[0], self.input_height / self.depth_conv.conv.stride[0])
                print("pw flops :", pw_flops)
                return inverted_flops + dw_flops + pw_flops
            
            elif len(self.expand_ratio_list) == 1 and len(self.kernel_size_list) > 1:
                print("expand ratio list = 1, kernel size list > 1")
                
                
                
                inverted_flops = 0
                if self.inverted_bottleneck:
                    inverted_flops = count_conv_gumbel_flops(self.inverted_bottleneck.conv.weight.shape, self.input_width, self.input_height)
                    print("inverted blocks flops :", inverted_flops)
                dw_flops_tensor = torch.zeros(len(self.kernel_size_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, kernel in enumerate(self.kernel_size_list):
                    if kernel == self.max_kernel_size:
                        dw_flops_tensor[i] = count_conv_gumbel_flops(self.depth_conv.conv.weight.shape, self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                    else:
                        out_c, in_c, h, w = self.depth_conv.conv.weight.shape
                        convert_flops = self.convert_kernel_flops(kernel)
                        compute_flops = count_conv_gumbel_flops((out_c, in_c, kernel, kernel), self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                        dw_flops_tensor[i] = convert_flops + compute_flops
                
                print("dw flops :", dw_flops_tensor)
                pw_flops = count_conv_gumbel_flops(self.point_linear.conv.weight.shape, self.input_width / self.depth_conv.conv.stride[0], self.input_height / self.depth_conv.conv.stride[0])
                print("pw flops :", pw_flops)
                
                gumbel_flops = gumbel_idx * (dw_flops_tensor + pw_flops + inverted_flops)
                
                return gumbel_flops
            
            elif len(self.expand_ratio_list) > 1 and len(self.kernel_size_list) == 1:
                print("expand ratio list > 1, kernel size list = 1")
                inverted_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, in_c, h, w = self.inverted_bottleneck.conv.weight.shape
                    inverted_flops_tensor[i] = count_conv_gumbel_flops((in_c*expand, in_c, h, w), self.input_width, self.input_height)
                print("inverted blocks flops :", inverted_flops_tensor)                   
                
                dw_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, in_c, h, w = self.depth_conv.conv.weight.shape
                    dw_flops = count_conv_gumbel_flops((in_c*expand, in_c, h, w), self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                    dw_flops_tensor[i] += dw_flops
                print("dw blocks flops :", dw_flops_tensor)                   
                
                pw_flops_tensor = torch.zeros(len(self.expand_ratio_list), dtype=torch.float32).to(self.depth_conv.conv.weight.device)
                for i, expand in enumerate(self.expand_ratio_list):
                    out_c, in_c, h, w = self.point_linear.conv.weight.shape
                    pw_flops = count_conv_gumbel_flops((out_c, in_c*expand, h, w), self.input_width, self.input_height, self.depth_conv.conv.stride[0])
                    pw_flops_tensor[i] += pw_flops
                print("total blocks flops :", inverted_flops_tensor + dw_flops_tensor + pw_flops_tensor)
                gumbel_flops = gumbel_idx * (inverted_flops_tensor + dw_flops_tensor + pw_flops_tensor)
                
                return gumbel_flops
        """
                
    def convert_kernel_flops(self, target_kernel):
        flops = 0
        for i, kernel in enumerate(self.kernel_size_list):
            if kernel == self.max_kernel_size:
                # do not convert kernel size
                flops += 0
            else:
                out_c, in_c, h, w = self.depth_conv.conv.weight.shape
                flops += out_c * in_c * kernel * kernel * kernel * kernel
            
            if kernel == target_kernel:
                return flops
            
    
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
        elif (self.shortcut is None or isinstance(self.shortcut, ZeroLayer)) and gumbel_idx == None:
            res = self.mobile_inverted_conv(x)
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer) and gumbel_idx != None:
            res = self.mobile_inverted_conv(x, gumbel_idx)
        elif self.shortcut is not None and gumbel_idx == None:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        else:
            res = self.mobile_inverted_conv(x, gumbel_idx) + self.shortcut(x)
        return res
    
    def count_flops(self, gumbel_idx=None):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            return 0
        else:
            return self.mobile_inverted_conv.compute_gumbel_flops(gumbel_idx)

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