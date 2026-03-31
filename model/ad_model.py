
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2, DPTHead
from model.adapter.adapterconv import  AdapterWSpatial
from model.adapter.spatial_extractor import ScaleShiftFocalHead, SpatialPriorModule
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }




class MetricMIS(nn.Module):
    def __init__(self,
                mode_finet :str = 'only_head',
                size : str = 'vitb',

                
                ):
        super().__init__()
        self.mode_finet = mode_finet
        self.size = size
        self.init_dav()
        self.patch_size = 14
        self.spatial_extractor_1 = SpatialPriorModule(embed_dim= self.embed_dim, patch_size=self.patch_size, out_ch_decoder = self.out_channels)
        self.spatial_extractor_2 = SpatialPriorModule(embed_dim= self.embed_dim, patch_size=self.patch_size, out_ch_decoder = self.out_channels)
        
        self.number_of_blocks = self.dav2.pretrained.n_blocks
        self.injection_layers = [1,4,7,10]
        self.extraction_layers = [2,5,8,11]

        
        self.adapters_domain = nn.ModuleList([AdapterWSpatial(embed_dim=self.embed_dim) for _ in range(4)]) 
        self.adapters_scale = nn.ModuleList([AdapterWSpatial(embed_dim=self.embed_dim, zero_init=True) for _ in range(4)]) 
        self.layer_norm_parallel= nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(4)]) 
        self.layer_norm_ss = nn.LayerNorm(self.embed_dim)
        self.scale_and_shift_head = ScaleShiftFocalHead(self.embed_dim, hidden_dim=128)

        self.segmentation_extractor = SpatialPriorModule(embed_dim= self.embed_dim, patch_size=self.patch_size, out_ch_decoder = self.out_channels)
        self.segmentation_adapters = nn.ModuleList([AdapterWSpatial(embed_dim=self.embed_dim) for _ in range(4)]) 
        self.segmentation_layer_norm= nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(4)]) 
        self.segmentation_head = DPTHead(self.dav2.pretrained.embed_dim, 64, False, out_channels=[48, 96, 192, 384], use_clstoken=False)
    def init_dav(self):
        self.out_channels = model_configs[self.size]['out_channels']
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11], 
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.f_list = [2, 5, 8, 11]
        self.dav2 =  DepthAnythingV2(**{**model_configs[self.size]})
        self.embed_dim = self.dav2.pretrained.embed_dim 
        try:
            self.dav2.load_state_dict(torch.load(rf'C:\Users\Utente\Desktop\project\git\consistent_vo_mis\checkpoints_dv2/depth_anything_v2_{self.size}.pth', map_location='cpu'),strict=False)
        except:
            pass

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)
        for param in self.dav2.pretrained.parameters():
            param.requires_grad = False

        if self.mode_finet in ['only_head']:
            target_submodule = "scratch.output_conv"  

            for name, param in self.dav2.depth_head.named_parameters():

                if name.startswith(target_submodule):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # for param in self.dav2.parameters():
            #     param.requires_grad = False
        elif self.mode_finet in ['none']:
            for name, param in self.dav2.depth_head.named_parameters():
                param.requires_grad = False
    def froze_for_segmentation(self):
        for name, param in self.dav2.depth_head.named_parameters():
                target_submodule = 'segmentation'
                if name.startswith(target_submodule):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    def forward_geom(self, x):
        # Normalize img
        x = (x - self._resnet_mean) / self._resnet_std
        
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        spatial_features_domain = self.spatial_extractor_1(x)
        spatial_features_scale = self.spatial_extractor_2(x)

        features = self.dav2.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.size],  return_class_token=True)

    
        list_of_f_adapted = []
        B,_,_,_ = x.shape
        out_adapter_ss = None
        for i, (f, s_d, s_s) in enumerate(zip(features, spatial_features_domain, spatial_features_scale )):
            
            if i == 0:
                in_adapter = f.detach()
                s_in_d = s_d
                s_in_s = s_s
            else:
                in_adapter = f.detach() #+x
                x_img_d = x_d[:,1:,:]
                x_img_d = x_img_d.reshape(B,patch_h,patch_w,-1).permute(0,3,1,2)

                x_img_s = x_s[:,1:,:]
                x_img_s = x_img_s.reshape(B,patch_h,patch_w,-1).permute(0,3,1,2)
                s_in_d = s_d + x_img_d 
                s_in_s = s_s + x_img_s  
            x_d = self.adapters_domain[i](in_adapter, s_in_d, patch_h, patch_w)
            x_s = self.adapters_scale[i](in_adapter, s_in_s, patch_h, patch_w)
        
            fn = self.layer_norm_parallel[i](f.detach()+ x_d)

            cls = fn[:, 0] 
            y = fn[:,1:] 
            list_of_f_adapted.append((y,cls))
            out_adapter_ss = f.detach() + x_s


        disp = self.dav2.depth_head.forward(list_of_f_adapted, patch_h, patch_w)
        disp = F.relu(disp) +1e-6
        scale, shift, fx, fy = self.scale_and_shift_head(self.layer_norm_ss(out_adapter_ss[:,1:,:]))
        return disp, scale, shift, fx, fy
    

    def forward_segmentation(self, x):
        # Normalize img
        x = (x - self._resnet_mean) / self._resnet_std
        
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        spatial_features_domain = self.segmentation_extractor(x)
        

        features = self.dav2.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.size],  return_class_token=True)
        list_of_f_adapted = []
        B,_,_,_ = x.shape
        out_adapter_ss = None
        for i, (f, s_d) in enumerate(zip(features, spatial_features_domain )):
            if i == 0:
                in_adapter = f.detach()
                s_in_d = s_d

            else:
                in_adapter = f.detach() 
                x_img_d = x_d[:,1:,:]
                x_img_d = x_img_d.reshape(B,patch_h,patch_w,-1).permute(0,3,1,2)
                s_in_d = s_d + x_img_d 

            x_d = self.segmentation_adapters[i](in_adapter, s_in_d, patch_h, patch_w)

        
            fn = self.segmentation_layer_norm[i](f+ x_d)
            cls = fn[:, 0] 
            y = fn[:,1:] 
            list_of_f_adapted.append((y,cls))


        logits = self.segmentation_head(list_of_f_adapted, patch_h, patch_w)

        return logits
    
    def forward(self,x):
        # Norm Img
        x = (x - self._resnet_mean) / self._resnet_std
        
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        spatial_features_domain = self.spatial_extractor_1(x)
        spatial_features_scale = self.spatial_extractor_2(x)
        spatial_features_segmentation =  self.segmentation_extractor(x)
        features = self.dav2.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.size],  return_class_token=True)
    
        list_of_f_adapted = []
        list_of_f_segmentation = []
        B,_,_,_ = x.shape
        out_adapter_ss = None
        fssss = []
        for i, (f, s_d, s_s, f_seg) in enumerate(zip(features, spatial_features_domain, spatial_features_scale, spatial_features_segmentation )):
            if i == 0:
                in_adapter = f.detach()
                s_in_d = s_d
                s_in_s = s_s
                f_seg_in = f_seg
            else:
                in_adapter = f.detach() #+x
                x_img_d = x_d[:,1:,:]
                x_img_d = x_img_d.reshape(B,patch_h,patch_w,-1).permute(0,3,1,2)

                x_img_s = x_s[:,1:,:]
                x_img_s = x_img_s.reshape(B,patch_h,patch_w,-1).permute(0,3,1,2)

                x_img_seg = x_seg[:,1:,:]
                x_img_seg = x_img_seg.reshape(B,patch_h,patch_w,-1).permute(0,3,1,2)
                s_in_d = s_d + x_img_d 
                s_in_s = s_s + x_img_s  
                f_seg_in = f_seg + x_img_seg
            x_d = self.adapters_domain[i](in_adapter, s_in_d, patch_h, patch_w)
            x_s = self.adapters_scale[i](in_adapter, s_in_s, patch_h, patch_w)
            x_seg = self.segmentation_adapters[i](in_adapter, f_seg_in, patch_h, patch_w)
            fn = self.layer_norm_parallel[i](f+ x_d)
            cls = fn[:, 0] 
            y = fn[:,1:] 
            list_of_f_adapted.append((y,cls))
            out_adapter_ss = f + x_s

            fn_seg = self.segmentation_layer_norm[i](f + x_seg)
            cls_seg = fn_seg[:,0]
            y_seg  = fn_seg[:,1:]
            list_of_f_segmentation.append((y_seg, cls_seg))



        disp = self.dav2.depth_head.forward(list_of_f_adapted, patch_h, patch_w)
        logits = self.segmentation_head(list_of_f_segmentation, patch_h, patch_w)

        disp = F.relu(disp) +1e-6
        scale, shift, fx, fy = self.scale_and_shift_head(self.layer_norm_ss(out_adapter_ss[:,1:,:]))
        return disp, scale, shift, fx, fy, logits
    

if __name__ == "__main__":
    pass