from __future__ import absolute_import, division, print_function
from layers import *
from torch.nn.init import trunc_normal_
from networks.update import BasicUpdateBlock
from networks.triplet_attention import attentionUpdate1,attentionUpdate2,SepConvGRU


class DepthDecoder(nn.Module):
    def __init__(self,num_ch_enc,iters, refinement_kernel_size=5):
        super().__init__()

        self.num_ch_enc = num_ch_enc

        self.iters = iters
        
        self.convX31 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),torch.nn.Conv2d(in_channels=self.num_ch_enc[2], out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),torch.nn.Tanh())
        self.convX41 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),torch.nn.Conv2d(in_channels=self.num_ch_enc[3], out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),torch.nn.Tanh())
        
        self.convX32 = torch.nn.Sequential(
            nn.ReflectionPad2d(2),torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0, groups=128, bias=True),torch.nn.Tanh())
        self.convX42 = torch.nn.Sequential(
            nn.ReflectionPad2d(2),torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0, groups=128, bias=True),torch.nn.Tanh())

        if refinement_kernel_size == 1:
            # Older two-iteration checkpoints save unused pointwise refinements.
            self.convX32 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.Tanh())
            self.convX42 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.Tanh())
        elif refinement_kernel_size != 5:
            raise ValueError("refinement_kernel_size must be 1 or 5")
        
        #self.gruc = SepConvGRU()
        #self.gruc = attentionUpdate1()
        self.gruc = attentionUpdate2()
        self.update_block = BasicUpdateBlock()
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    @classmethod
    def from_checkpoint(cls, num_ch_enc, iters, state_dict):
        """Load both published refinement layouts with strict key/shape checks."""
        legacy = "convX32.0.weight" in state_dict
        if legacy and iters != 2:
            raise ValueError("Legacy pointwise checkpoints require iters=2")
        decoder = cls(num_ch_enc, iters, refinement_kernel_size=1 if legacy else 5)
        decoder.load_state_dict(state_dict, strict=True)
        return decoder

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def upsample_depth(self, depth, mask):

        N, C, H, W = depth.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_depth = nn.functional.unfold(depth, [3,3], padding=1)
        up_depth = up_depth.view(N, C, 9, 1, 1, H, W)

        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(N, C, 8*H, 8*W)


    def forward(self, input_features,iters=4):
        self.outputs = {}
        iters = self.iters

        x1,x2,x3,x4= input_features

        b, c, h, w = x3.shape
        dispFea = torch.zeros([b, 1, h, w], requires_grad=True).to(x3.device)
        net = torch.zeros([b, 256, h, w], requires_grad=True).to(x3.device)

        for itr in range(iters):
            if itr in [0]:
                corr = self.convX31(x3)
            elif itr in [1]:
                corrh = corr
                if iters == 2:
                    # The two-step model uses both encoder scales. It is not a
                    # truncated four-step model (whose second step refines x3).
                    corr = upsample(self.convX41(x4))
                else:
                    corr = self.convX32(corr)
                corr = self.gruc(corrh, corr)
            elif itr in [2]:
                corrh = corr
                corr = upsample(self.convX41(x4))
                corr = self.gruc(corrh, corr)
            elif itr in [3]:
                corrh = corr
                corr = self.convX42(corr)
                corr = self.gruc(corrh, corr)

            net, up_mask, delta_disp = self.update_block(net, corr, dispFea)
            dispFea = dispFea + delta_disp
            disp = self.sigmoid(dispFea)

            if self.training:
                disp_up = self.upsample_depth(disp, up_mask)
                self.outputs[("disp_up", itr)] = disp_up
            else:
                if (iters-1)==itr:
                    disp_up = self.upsample_depth(disp, up_mask)
                    self.outputs[("disp_up", itr)] = disp_up

        return self.outputs

'''
        for itr in range(iters):
            if itr in [0]:
                corr = self.convX31(x3)
            elif itr in [1]:
                corrh = corr
                corr = upsample(self.convX41(x4))
                corr = self.gruc(corrh, corr)

                
        for itr in range(iters):
            if itr in [0]:
                corr = self.convX31(x3)
            elif itr in [1]:
                corrh = corr
                corr = self.convX32(corr)
                corr = self.gruc(corrh, corr)
            elif itr in [2]:
                corrh = corr
                corr = upsample(self.convX41(x4))
                corr = self.gruc(corrh, corr)
            elif itr in [3]:
                corrh = corr
                corr = self.convX42(corr)
                corr = self.gruc(corrh, corr)
'''
