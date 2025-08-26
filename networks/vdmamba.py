from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from networks.SoftMedian import softMedian
from networks.LiteFlowNet_muli import backwarp as flow_warping
from networks.LiteFlowNet_muli import LightFlowNet
from .s3ml import S3ML
from .tsml import TSML
from .rtrwnet import Restormer
from networks.LiteFlowNet import LightFlowNet
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class ChannelNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(ChannelNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class Norm(nn.Module):
    def __init__(self, dim, Norm_type):
        super(Norm, self).__init__()
        if Norm_type == 'Group':
#            self.body = BiasFree_LayerNorm(dim)
            pass
        elif Norm_type == 'Channel':
            self.body = ChannelNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
class MambaBlock_spa(nn.Module):
    def __init__(self, dim, Norm_type):
        super(MambaBlock_spa, self).__init__()

        self.norm = Norm(dim, Norm_type)
        self.s3ml_spatial = S3ML(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(self, x):
        x = spatial_states + self.norm(self.s3ml_spatial(x))
        return x


##########################################################################
class MambaBlock(nn.Module):
    def __init__(self, dim, Norm_type, if_dsf=False):
        super(MambaBlock, self).__init__()
        self.if_dsf = if_dsf
        self.norm1 = Norm(dim, Norm_type)
        self.s3ml = S3ML(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.norm2 = Norm(dim, Norm_type)
        self.tsml = TSML(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            if_dsf=if_dsf,
        )

    def forward(self, x, spatial_states=None):
        assert (spatial_states is not None) == self.if_dsf, "{}, {}".format(spatial_states is not None, self.if_dsf)
        x = x + self.norm1(self.s3ml(x))
        x = x + self.norm2(self.tsml(x, spatial_states))
        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = mySequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = mySequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))


    def forward(self, x):
        return self.body(x)

#############################################################################
class VDMamba_spa(nn.Module):
    def __init__(self,
        inp_channels=21,
        out_channels=3,
        dim = 24,
        num_blocks = [1,1,1,1],  # 1,2,2,4
        num_refinement_blocks = 1,
        bias = False,
        Norm_type = 'Channel',   ## Other options 'Group', 'Instance', 'Layer'
    ):

        super(VDMamba_spa, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = mySequential(*[MambaBlock(dim=dim, Norm_type=Norm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = mySequential(*[MambaBlock(dim=int(dim*2**1), Norm_type=Norm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = mySequential(*[MambaBlock(dim=int(dim*2**2), Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = mySequential(*[MambaBlock(dim=int(dim*2**3), Norm_type=Norm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = mySequential(*[MambaBlock(dim=int(dim*2**2), Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = mySequential(*[MambaBlock(dim=int(dim*2**1), Norm_type=Norm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*2**0), kernel_size=1, bias=bias)
        self.decoder_level1 = mySequential(*[MambaBlock(dim=int(dim*2**0), Norm_type=Norm_type) for i in range(num_blocks[0])])

        self.refinement = mySequential(*[MambaBlock(dim=int(dim*2**0), Norm_type=Norm_type) for i in range(num_refinement_blocks)])
        ###########################
        self.output = nn.Conv2d(int(dim*2**0), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dim = dim

    def forward(self, frame):
        inp_img = frame
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1( inp_dec_level1 )


        out = self.refinement(out_dec_level1)

        out = self.output(out)

        return out, [out_enc_level1, out_enc_level2, out_enc_level3, latent, out_dec_level3, out_dec_level2, out_dec_level1]


#############################################################################
class VDMamba_flow(nn.Module):
    def __init__(self,
        inp_channels=21,
        out_channels=3,
        dim = 24,
        num_blocks = [1,1,1,1],  # 1,2,2,4
        num_refinement_blocks = 1,
        bias = False,
        Norm_type = 'Channel',   ## Other options 'Group', 'Instance', 'Layer'
    ):

        super(VDMamba_flow, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = mySequential(*[MambaBlock(dim=dim, Norm_type=Norm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = mySequential(*[MambaBlock(dim=int(dim*2**1), Norm_type=Norm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = mySequential(*[MambaBlock(dim=int(dim*2**2), Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = mySequential(*[MambaBlock(dim=int(dim*2**3), Norm_type=Norm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = mySequential(*[MambaBlock(dim=int(dim*2**2), Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = mySequential(*[MambaBlock(dim=int(dim*2**1), Norm_type=Norm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*2**0), kernel_size=1, bias=bias)
        self.decoder_level1 = mySequential(*[MambaBlock(dim=int(dim*2**0), Norm_type=Norm_type) for i in range(num_blocks[0])])

        self.refinement = mySequential(*[MambaBlock(dim=int(dim*2**0), Norm_type=Norm_type) for i in range(num_refinement_blocks)])
        ###########################
        self.output = nn.Conv2d(int(dim*2**0), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dim = dim

    def forward(self, frame):
        inp_img = frame
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1( inp_dec_level1 )

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1

#############################################################################
class VDMamba(nn.Module):
    def __init__(self, 
        inp_channels=21, 
        out_channels=3, 
        dim = 24,
        num_blocks = [1,1,1,1],  # 1,2,2,4
        num_refinement_blocks = 1,
        bias = False,
        Norm_type = 'Channel',   ## Other options 'Group', 'Instance', 'Layer'
    ):

        super(VDMamba, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = mySequential(*[MambaBlock(dim=dim, Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = mySequential(*[MambaBlock(dim=int(dim*2**1), Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = mySequential(*[MambaBlock(dim=int(dim*2**2), Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = mySequential(*[MambaBlock(dim=int(dim*2**3), Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = mySequential(*[MambaBlock(dim=int(dim*2**2), Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = mySequential(*[MambaBlock(dim=int(dim*2**1), Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*2**0), kernel_size=1, bias=bias)
        self.decoder_level1 = mySequential(*[MambaBlock(dim=int(dim*2**0), Norm_type=Norm_type, if_dsf=(True and i==0)) for i in range(num_blocks[0])])
        
        self.refinement = mySequential(*[MambaBlock(dim=int(dim*2**0), Norm_type=Norm_type) for i in range(num_refinement_blocks)])
        
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**0), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.dim = dim
        self.frame_restorer_spa = VDMamba_spa(inp_channels=3, out_channels=3, dim=12, num_blocks=[1,1,1,1], num_refinement_blocks=2)
        
    def forward(self, frame, frame_center):

        out_spa, out_levels_spa = self.frame_restorer_spa(frame_center)
        inp_img = frame
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, out_levels_spa[0])
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, out_levels_spa[1])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, out_levels_spa[2]) 

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, out_levels_spa[3])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, out_levels_spa[4])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, out_levels_spa[5])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, out_levels_spa[6])

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1, out_spa

class Model(nn.Module):
    def __init__(self, opts, scales=[1]):
        super(Model, self).__init__()
        self.prev_frame_num = 2
        self.scales = scales
        self.ifAlign = opts.ifAlign
        self.ifAggregate = opts.ifAggregate
        self.frame_restorer = VDMamba(inp_channels=39 + self.prev_frame_num*3, out_channels=3, num_blocks=[1,1,1,1], num_refinement_blocks=2)
        if self.ifAlign:
            self.flow_estimator = VDMamba_flow(inp_channels=21 + self.prev_frame_num*3, out_channels=12, num_blocks=[1,1,1,1], num_refinement_blocks=2)
            if self.ifAggregate:
                self.frame_restorer = VDMamba(inp_channels=39 + self.prev_frame_num*3, out_channels=3, num_blocks=[1,1,1,1], num_refinement_blocks=3)
        else:
            self.flow_estimator = mySequential()

        self.FlowNet = LightFlowNet()
        self.prev_frames = []
    
    def set_new_video(self,):
        self.prev_frames = []

    def update_prev_frames(self, frame):
        self.prev_frames = [frame.detach(),] + self.prev_frames
        self.prev_frames = self.prev_frames[:self.prev_frame_num]

    def get_prev_frames(self, frames, selected_index=None):
        bn, sq = frames.shape[:2]
        if selected_index is None:
            selected_index = list(range(bn))
        prev_frames = [self.prev_frames[i][selected_index] if i < len(self.prev_frames) else frames[:, sq//2 - i - 1] for i in range(self.prev_frame_num) ]

        return [prev_frame.unsqueeze(1) for prev_frame in prev_frames]

    def predict_flow(self, input_, scales):
        flow_list = []
        bn, sq, ch_f, h, w = input_.shape
        restored_prev_frames = self.get_prev_frames(input_)
        for j, scale in enumerate(scales):
            input_ = torch.cat(restored_prev_frames + [input_,], 1).contiguous().view(bn, -1, h, w)
            input_scaled = F.interpolate(input_, scale_factor=1./scales[j], mode="bilinear", align_corners=False)
            flow_inward = self.flow_estimator(input_scaled)
            flow_list.append(flow_inward)
        return flow_list

    def forward(self, inputs, ifInferece=True, ):
        bn, sq, ch_f, h, w = inputs.shape
        flow_list = []
        frame_center = inputs[:,sq//2]
        restored_prev_frames = self.get_prev_frames(inputs)
        assert len(restored_prev_frames) == self.prev_frame_num, "{}".format(len(restored_prev_frames))
        if self.ifAlign:
            seq_around = inputs[:,[i for i in range(sq) if i != sq//2]]

            input_ = torch.cat(restored_prev_frames + [inputs,], 1).contiguous().view(bn, -1, h, w)
            
            flow_inward = self.flow_estimator(input_)
            flow_list.append(flow_inward)

            flow_inward = flow_inward.contiguous().view(bn, sq-1, 2, h, w)
            warp_inward = [flow_warping(seq_around[:,i], flow_inward[:,i]) for i in range(sq-1)]
            warpped_prev_frames = [flow_warping(restored_prev_frames[i][:,0], flow_inward[:,sq//2-1-i]) for i in range(self.prev_frame_num)]

            seq_input = torch.cat(warpped_prev_frames+
                                  [seq_around[:,i] for i in range(3)]+
                                  [frame for frame in warp_inward[:3]]
                                   +[frame_center]+
                                  [frame for frame in warp_inward[3:]]
                                  +[seq_around[:,3+i] for i in range(3)]
                                  ,1).detach()

        
        out, out_spa = self.frame_restorer(seq_input, frame_center)
        out = out + frame_center
        out_spa = out_spa + frame_center
        self.update_prev_frames(out)
        return out, out_spa, flow_list, seq_input


