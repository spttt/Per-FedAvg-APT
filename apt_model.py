import torch
import torch.nn as nn
import torch.nn.functional as F



class fully_connected(nn.Module):
    def __init__(self):
        super(fully_connected, self).__init__()
        self.fc = nn.Sequential( 
            nn.Linear(65, 128), 
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.ReLU(),
        )

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1, end_dim=-1)
        outputs = self.fc(inputs)
        return outputs


class fully_connected_fc(nn.Module): # 最后一层单列
    def __init__(self):
        super(fully_connected_fc, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(65, 128) 
        self.fc2 = nn.Linear(128, 32) 
        self.fc = nn.Linear(32, 5) 

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
        return x


class Residual_apt(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=(1,3), padding=(0,1), stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv: 
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet_apt(nn.Module): # 用if构建的
    def __init__(self):
        super(ResNet_apt, self).__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                nn.BatchNorm2d(2),
                                # nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),

                                Residual_apt(2, 2),
                                Residual_apt(2, 2),
                                Residual_apt(2, 4, use_1x1conv=True, strides=2),
                                Residual_apt(4, 4),
                                Residual_apt(4, 8, use_1x1conv=True, strides=2),
                                Residual_apt(8, 8),
                                Residual_apt(8, 16, use_1x1conv=True, strides=2),
                                Residual_apt(16, 16),

                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(), nn.Linear(16, 5)
                            )

    def forward(self, inputs):
        inputs_reshape = torch.reshape(inputs,(inputs.size()[0],1,1,inputs.size()[-1]))
        outputs = self.f(inputs_reshape)
        return outputs


class ResNet_apt_4(nn.Module): # 最开始1变4通道的ResNet
    def __init__(self):
        super(ResNet_apt_4, self).__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                nn.BatchNorm2d(4),
                                # nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),

                                Residual_apt(4, 4),
                                Residual_apt(4, 4),
                                Residual_apt(4, 8, use_1x1conv=True, strides=2),
                                Residual_apt(8, 8),
                                Residual_apt(8, 16, use_1x1conv=True, strides=2),
                                Residual_apt(16, 16),
                                Residual_apt(16, 32, use_1x1conv=True, strides=2),
                                Residual_apt(32, 32),

                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(), nn.Linear(32, 5)
                            )

    def forward(self, inputs):
        inputs_reshape = torch.reshape(inputs,(inputs.size()[0],1,1,inputs.size()[-1]))
        outputs = self.f(inputs_reshape)
        return outputs


class ResNet_apt_fc_if(nn.Module): # 用if构建的 最后一层单列
    def __init__(self):
        super(ResNet_apt_fc, self).__init__()
        
        self.layer_seq = nn.Sequential(nn.Conv2d(1, 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                nn.BatchNorm2d(2),
                                # nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),

                                Residual_apt(2, 2),
                                Residual_apt(2, 2),
                                Residual_apt(2, 4, use_1x1conv=True, strides=2),
                                Residual_apt(4, 4),
                                Residual_apt(4, 8, use_1x1conv=True, strides=2),
                                Residual_apt(8, 8),
                                Residual_apt(8, 16, use_1x1conv=True, strides=2),
                                Residual_apt(16, 16),

                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten()
                            )
        self.fc = nn.Linear(16, 5)

    def forward(self, inputs):
        x = torch.reshape(inputs,(inputs.size()[0],1,1,inputs.size()[-1]))
        x = self.layer_seq(x)
        x = self.fc(x)
        return x


class Residual_apt_use_1x1conv(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=(1,3), padding=(0,1), stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        X = self.conv3(X)
        Y += X
        return F.relu(Y)


class Residual_apt_unuse_1x1conv(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=(1,3), padding=(0,1), stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = None
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += X
        return F.relu(Y)


class ResNet_apt_fc(nn.Module): # 不使用if构建，以支持jit编译
    def __init__(self, output_class = 5):
        super(ResNet_apt_fc, self).__init__()
        
        self.layer_seq = nn.Sequential(nn.Conv2d(1, 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                nn.BatchNorm2d(2),
                                # nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),

                                Residual_apt_unuse_1x1conv(2, 2),
                                Residual_apt_unuse_1x1conv(2, 2),
                                Residual_apt_use_1x1conv(2, 4, use_1x1conv=True, strides=2),
                                Residual_apt_unuse_1x1conv(4, 4),
                                Residual_apt_use_1x1conv(4, 8, use_1x1conv=True, strides=2),
                                Residual_apt_unuse_1x1conv(8, 8),
                                Residual_apt_use_1x1conv(8, 16, use_1x1conv=True, strides=2),
                                Residual_apt_unuse_1x1conv(16, 16),

                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten()
                            )
        self.fc = nn.Linear(16, output_class)

    def forward(self, inputs):
        x = torch.reshape(inputs,(inputs.size()[0],1,1,inputs.size()[-1]))
        x = self.layer_seq(x)
        x = self.fc(x)
        return x
    


MODEL_DICT = {"MLP": fully_connected_fc, "resnet": ResNet_apt_fc}


def get_model(model_neme, device=torch.device("cuda")):
    return MODEL_DICT[model_neme]().to(device)