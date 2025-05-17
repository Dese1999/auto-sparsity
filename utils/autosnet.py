import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# 
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        b = self.bias_mask * self.bias if self.bias is not None else None
        return F.linear(input, W, b)

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        b = self.bias_mask * self.bias if self.bias is not None else None
        return self._conv_forward(input, W, b)

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        W = self.weight_mask * self.weight if self.affine else self.weight
        b = self.bias_mask * self.bias if self.affine else self.bias
        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        W = self.weight_mask * self.weight if self.affine else self.weight
        b = self.bias_mask * self.bias if self.affine else self.bias
        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps)

class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W

# تعریف BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        self.residual_function = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = Identity2d(in_channels)
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# تعریف ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_block, base_width, num_classes=1, dense_classifier=False):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, base_width)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, base_width)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, base_width)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, base_width)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes) if not dense_classifier else nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_channels, out_channels, stride, base_width))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layer_list)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

# تعریف تابع resnet18
def resnet18(input_shape=(3, 32, 32), num_classes=1, dense_classifier=False, pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/resnet18-cifar{}.pt'.format(num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

# تعریف MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            Linear(2, 16),
            BatchNorm1d(16),
            nn.ReLU(),
            Linear(16, 64),
            BatchNorm1d(64),
            nn.ReLU(),
            Linear(64, 1024),
            BatchNorm1d(1024),
            nn.ReLU(),
            Linear(1024, 1024),
            BatchNorm1d(1024),
            nn.ReLU(),
            Linear(1024, 64),
            BatchNorm1d(64),
            nn.ReLU(),
            Linear(64, 1)
        )
        for layer in self.model:
            if isinstance(layer, Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        output = self.model(combined)
        return output

# تعریف ResNet18 برای AutoS
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.mlp = nn.Sequential(
            Linear(2, 64),
            BatchNorm1d(64),
            nn.ReLU(),
            Linear(64, 64*16*3),  # خروجی به 3072 برای سازگاری با CIFAR10
            BatchNorm1d(64*16*3),
            nn.ReLU(),
        )
        self.resnet18 = resnet18(input_shape=(3, 32, 32), num_classes=1, dense_classifier=False, pretrained=False)

    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        mlp_output = self.mlp(combined).reshape(-1, 3, 32, 32)
        return self.resnet18(mlp_output)
