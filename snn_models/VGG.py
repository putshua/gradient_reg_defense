
from snn_models.layers import *

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ],
    'vggdvs': [
        [64, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A'],
        []
    ],
    'vgggesture': [
        [16, 32, 'A'],
        [32, 32, 'A'],
        [],
        [],
        []
    ]
}

class VGG(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, tau=1., init_c=3):
        super(VGG, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = None
            # TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.tau = tau
        self.init_channels = init_c

        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            if init_c == 2:
                self.W = 36
            else:
                self.W = 16
        else:
            self.W = 1
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        self.classifier = self._make_classifier(num_class)
        
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.encode = Poisson()
        self.poisson = False
        self.hooked = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def hook(self, model, gradin, gradout):
        self.hooked = gradin[0]

    def act_hook(self, model, input, output):
        x = input[0]
        if self.T > 0:
            x = self.expand(x)
            x = x.mean(0)
        self.hooked = x

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(T=self.T, tau=self.tau))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [
        nn.Flatten(),
        nn.Linear(512*self.W, 4096),
        LIFSpike(self.T, tau=self.tau),
        nn.Linear(4096, 4096),
        LIFSpike(self.T, tau=self.tau), 
        nn.Linear(4096, num_class)
        ]
        return nn.Sequential(*layer)
    
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    
    def set_attack_mode(self, mode):
        for module in self.modules():
            if isinstance(module, (LIFSpike)):
                module.mode = mode[0]
                module.gama = mode[1]
                module.set_surrogate_gradient(mode[2])
        return
    
    def _rate(self, ra):
        for name, module in self.named_modules():
            if isinstance(module, LIFSpike):
                ra[name] += module.r.mean().item()
        return ra

    def _grad(self, gd):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                grad = module.weight.grad.data
                gd[name] += (1-((grad>0)*(grad<1e-6).float()).mean())
        return gd
    
    def init_dic(self):
        vdic = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                vdic[name] = 0
            if isinstance(module, LIFSpike):
                vdic[name] = 0
        return vdic
        
    def forward(self, input):
        if not self.poisson and self.norm:
            input = self.norm(input)
        if self.T > 0:
            if self.W != 36:
                input = add_dimention(input, self.T)
            if self.poisson:
                input = self.encode(input)
            input = self.merge(input)
        # print(input.shape)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out.mean(0)
        


    