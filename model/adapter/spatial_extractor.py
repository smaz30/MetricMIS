import torch
import torch.nn as nn
import torch.nn.functional as F
class ScaleShiftFocalHead(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dim=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # or just .mean(dim=1)
        self.mlp_scale = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # outputs [scale, shift]
            #nn.ReLU()
            nn.Identity()
        )
        self.mlp_shift = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # outputs [scale, shift]
        )

        self.mlp_fxfy = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # outputs [scale, shift]
            # nn.ReLU()
            nn.Identity()
        )
        self.init_weights()
        nn.init.normal_(self.mlp_scale[-2].weight, std=0.001)
        nn.init.zeros_(self.mlp_scale[-2].bias)
        nn.init.normal_(self.mlp_shift[-1].weight, std=0.001)
        nn.init.zeros_(self.mlp_shift[-1].bias)
        # nn.init.normal_(self.mlp_fxfy[-2].weight, std=0.001)
        # nn.init.normal_(self.mlp_fxfy[-2].bias, std=0.2)
    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = F.dropout(x, p=0.2, training=self.training)
        # x: [B, patch_h*patch_w, embed_dim]
        x = x.mean(dim=1)  # [B, embed_dim]
        #scale = self.mlp_scale(x) +1e-5
        scale = torch.exp(self.mlp_scale(x))
        shift = self.mlp_shift(x) 

        focal = self.mlp_fxfy(x)

        fx = focal[:, 0]+ 1.0
        fy = focal[:, 1] +1.0
        #print('here', fx[0,...], fy[0,...])
        return  scale,shift, fx, fy# [B, 2]

    
class LayerNorm2d(nn.Module):
    """
    Applies LayerNorm over the channel dimension of 4D image tensors (B, C, H, W)
    by permuting to (B, H, W, C), normalizing, then permuting back.
    """
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)       # -> (B, H, W, C)
        x = self.norm(x)               # LayerNorm over C
        x = x.permute(0, 3, 1, 2)      # -> (B, C, H, W)
        return x

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, patch_size = 14, out_ch_decoder = None):
        super().__init__()
        self.with_cp = with_cp
        
        in_dim = 3

        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, inplanes, kernel_size=4, stride=4),
            # nn.BatchNorm2d(inplanes),
            LayerNorm2d(inplanes),
            )
        self.conv_1 = nn.Sequential(*[
            nn.Conv2d(inplanes,  inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d( inplanes),
            LayerNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(2 * inplanes),
            LayerNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(4 * inplanes),
            LayerNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(4 * inplanes),
            LayerNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        ])

     
        self.reshape1 = nn.Conv2d(inplanes,embed_dim,kernel_size=1, bias=False)
        self.reshape2 = nn.Conv2d(2 * inplanes,embed_dim,kernel_size=1, bias=False)
        self.reshape3 =nn.Conv2d(4 * inplanes,embed_dim,kernel_size=1, bias=False)
        self.reshape4 = nn.Conv2d(4 * inplanes,embed_dim,kernel_size=1, bias=False)

        self.norm = LayerNorm2d(embed_dim)
        self.patch_size = patch_size
        self.init_weights()
    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):

        # def _inner_forward(x):
        c1 = self.stem(x)
        c1 = self.conv_1(c1)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)


        r1 = nn.functional.interpolate(c1, (x.shape[-2]//self.patch_size, x.shape[-1] //self.patch_size),align_corners=False, mode='bilinear')
        r1 = self.reshape1(r1)
        r1 = self.norm(r1)

        r2 = nn.functional.interpolate(c2, (x.shape[-2]//self.patch_size, x.shape[-1] //self.patch_size),align_corners=False, mode='bilinear')
        r2 = self.reshape2(r2)
        r2 = self.norm(r2)

        r3 = nn.functional.interpolate(c3, (x.shape[-2]//self.patch_size, x.shape[-1] //self.patch_size),align_corners=False, mode='bilinear')
        r3 = self.reshape3(r3)
        r3 = self.norm(r3)

        r4 = nn.functional.interpolate(c4, (x.shape[-2]//self.patch_size, x.shape[-1] //self.patch_size),align_corners=False, mode='bilinear')
        r4 = self.reshape4(r4)
        r4 = self.norm(r4)


        out2 = r1, r2, r3, r4
        return  out2
 


    def forward(self, x):

        f_list = self.backbone(x)

        
        r1 = self.reshape1(f_list[0])

        # r1 = F.dropout(r1,p=0.1,training=self.training)

        r2 = self.reshape2(f_list[1])

        # r2 = F.dropout(r2,p=0.1,training=self.training)

        r3 = self.reshape3(f_list[2])

        # r3 = F.dropout(r3,p=0.1,training=self.training)

        r4 = self.reshape4(f_list[3])

        # low --> high semantic
        f_out = [r1,r2,r3,r4]
        return f_out

if __name__ == "__main__":
    dummy_in = torch.randn([2,3,224,224])
    model = SpatialFeatureExtractor()
    out = model(dummy_in)
    for o in out:
        print(o.shape)
    