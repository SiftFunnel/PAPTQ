import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from typing import Optional, Tuple, List


# ---------------- Patch Embedding ----------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# ---------------- Attention ----------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------- Transformer Block ----------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------- Standard ViT ----------------
class ViT(nn.Module):
    MODEL_CONFIGS = {
        'tiny':  {'embed_dim': 192,  'depth': 12, 'num_heads': 3,  'mlp_ratio': 4},
        'small': {'embed_dim': 384,  'depth': 12, 'num_heads': 6,  'mlp_ratio': 4},
        'base':  {'embed_dim': 768,  'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4},
    }

    def __init__(self, model_size='base', img_size=224, patch_size=16, in_chans=1, num_classes=1000):
        super().__init__()
        assert model_size in self.MODEL_CONFIGS, f"Unknown model_size {model_size}"
        cfg = self.MODEL_CONFIGS[model_size]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, cfg['embed_dim'])
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,cfg['embed_dim']))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, cfg['embed_dim']))
        self.pos_drop = nn.Dropout(0.)

        self.blocks = nn.ModuleList([
            Block(cfg['embed_dim'], cfg['num_heads'], cfg['mlp_ratio'])
            for _ in range(cfg['depth'])
        ])
        self.norm = nn.LayerNorm(cfg['embed_dim'])
        self.head = nn.Linear(cfg['embed_dim'], num_classes)

    def forward(self, x, release=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x[:,0])

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)



# -------------------------
# helpers
# -------------------------
class DropPath(nn.Module):
    """Stochastic Depth (aka DropPath)"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: (B, H, W, C)
    returns windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    windows: (num_windows*B, window_size, window_size, C)
    returns x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

# -------------------------
# Patch Embedding
# -------------------------
class PatchEmbed_swin(nn.Module):
    """Split image into non-overlapping patches and embed."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size
        self.W = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,int,int]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1,2)  # (B, HW, C)
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

# -------------------------
# Window Attention
# -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: Tuple[int,int], num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        Wh, Ww = window_size
        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*Wh-1)*(2*Ww-1), num_heads)
        )  # (2*Wh-1 * 2*Ww-1, nH)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:,:,0] += Wh - 1
        relative_coords[:,:,1] += Ww - 1
        relative_coords[:,:,0] *= 2*Ww - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask: Optional[torch.Tensor]=None):
        """
        x: (num_windows*B, N, C) where N = Wh*Ww
        mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))  # (B_, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # N, N, nH
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()  # nH, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: (num_windows, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -------------------------
# MLP
# -------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -------------------------
# Swin Transformer Block
# -------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution: Tuple[int,int], num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            # window_partition requires H/W divisible by window_size. For small
            # inputs such as 64x64 with patch_size=4, the first Swin resolution
            # is 16x16, which is not divisible by the default window_size=7.
            # We therefore build the shifted-window attention mask on the padded
            # resolution, and crop back to the original resolution in forward().
            Hp = int(math.ceil(H / window_size)) * window_size
            Wp = int(math.ceil(W / window_size)) * window_size
            img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)  # nW, ws, ws, 1
            mask_windows = mask_windows.view(-1, window_size*window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, N, N
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to multiples of window_size before window partition.
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)  # nW*B, N, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, N, C

        # reverse windows on padded resolution
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B Hp Wp C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x

        # crop back to original resolution
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# -------------------------
# Patch Merging (downsample)
# -------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int,int], dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, HW, C
        """
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        # padding if odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0,0,0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4*C)  # B, H/2*W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B, H/2*W/2, 2*C
        return x

# -------------------------
# Basic Layer (several blocks + downsample)
# -------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution: Tuple[int,int], depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., downsample=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(dim=dim,
                                     input_resolution=input_resolution,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
            )
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample:
            x = self.downsample(x, H, W)
            H, W = (H+1)//2, (W+1)//2
        return x, H, W

# -------------------------
# Swin Transformer (backbone + head)
# -------------------------
class SwinTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=128,            # Swin-B default
                 depths=[2,2,18,2],        # Swin-B
                 num_heads=[4,8,16,32],    # Swin-B
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ape=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.window_size = window_size

        # patch embedding
        self.patch_embed = PatchEmbed_swin(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # absolute position embed (optional)
        if self.ape:
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # per-block drop path rates
        cur = 0

        # build layers
        self.layers = nn.ModuleList()
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        resolutions = []
        # compute resolution after patch embed
        patches_resolution = (img_size // patch_size, img_size // patch_size)
        H, W = patches_resolution

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=dims[i_layer],
                               input_resolution=(H, W),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[cur:cur+depths[i_layer]],
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               norm_layer=norm_layer)
            self.layers.append(layer)
            cur += depths[i_layer]
            # update H,W for next layer
            if i_layer < self.num_layers - 1:
                H = (H + 1) // 2
                W = (W + 1) // 2

        self.norm = norm_layer(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # initialization
        self._init_weights()

    def _init_weights(self):
        # init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, release=False):
        """
        x: (B, 3, H, W)
        returns logits (B, num_classes)
        """
        x, H, W = self.patch_embed(x)  # B, L, C
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # B, L, C
        x = x.mean(dim=1)  # global average pooling
        x = self.head(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)

# -------------------------
# simple factory function for Swin-B
# -------------------------
def swin_b(num_classes=1000, img_size=224, **kwargs):
    return SwinTransformer(img_size=img_size,
                           patch_size=4,
                           in_chans=1,
                           num_classes=num_classes,
                           embed_dim=128,
                           depths=[2,2,18,2],
                           num_heads=[4,8,16,32],
                           window_size=7,
                           mlp_ratio=4.,
                           qkv_bias=True,
                           drop_rate=0.0,
                           attn_drop_rate=0.0,
                           drop_path_rate=0.3,
                           norm_layer=nn.LayerNorm,
                           patch_norm=True,
                           ape=False,
                           **kwargs)

# -------------------------
# quick local test
# -------------------------
if __name__ == "__main__":
    model = swin_b(num_classes=1000, img_size=224)
    print("Params:", sum(p.numel() for p in model.parameters())/1e6, "M")
    x = torch.randn(2,3,224,224)
    y = model(x)
    print(y.shape)


# ======================================================================
# PTQ / mixed-precision friendly ViT & Swin wrappers
# Added to make ViT/Swin expose a ResNet50_PTQ-like interface:
#   - apply_weight_ptq(...)
#   - set_all_act_bits(...)
#   - set_calib_mode()/finalize_act_calib()
#   - set_ptq_scale_optim_mode()
#   - set_inference_mode()
#   - forward(..., return_feats=True)
# These classes use fake/dequantized PTQ weights and activation quantizers so
# checkpoints stay portable as normal PyTorch state_dict files.
# ======================================================================

class ActQuantTokenwise(nn.Module):
    """
    Per-channel activation quantizer for Transformer/CNN representations.

    Supported input layouts:
      - [B, C]         : channel dimension = 1
      - [B, N, C]      : token/channel dimension = last dim
      - [B, C, H, W]   : channel dimension = 1

    quant_mode:
      - "none"    : no quantization
      - "calib"   : collect per-channel min/max, return x unchanged
      - "fake"    : STE fake quantization q*s
      - "defense" : q*s + detach, numerically same as PTQ inference
    """
    def __init__(
        self,
        num_channels: int,
        init_bits: int = 8,
        init_scale: float = 1.0,
        scale_min: float = 1e-4,
        scale_max: float = 10.0,
        unsigned: bool = False,
        cl: bool = False,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.bits = int(init_bits)
        self.unsigned = bool(unsigned)
        self.cl = bool(cl)

        init = torch.full((1, self.num_channels), float(init_scale), dtype=torch.float32)
        self.log_scale = nn.Parameter(init.log())
        self.register_buffer("base_log_scale", self.log_scale.data.clone())
        self.register_buffer("scale_min", torch.tensor(float(scale_min)))
        self.register_buffer("scale_max", torch.tensor(float(scale_max)))
        self.register_buffer("calib_min", torch.full((1, self.num_channels), float("inf")))
        self.register_buffer("calib_max", torch.full((1, self.num_channels), float("-inf")))
        self.quant_mode = "none"

    def set_quant_mode(self, mode: str):
        assert mode in ("none", "calib", "fake", "defense")
        self.quant_mode = mode

    def set_bits(self, bits: int):
        self.bits = int(bits)

    def reset_calib(self):
        self.calib_min.fill_(float("inf"))
        self.calib_max.fill_(float("-inf"))

    def _get_qmin_qmax(self):
        b = int(self.bits)
        if self.unsigned:
            return 0.0, float(2 ** b - 1)
        qmax = float(2 ** (b - 1) - 1)
        qmin = float(-2 ** (b - 1))
        return qmin, qmax

    def _scale_for_x(self, x: torch.Tensor):
        s = F.softplus(self.log_scale)
        s = torch.clamp(s, min=self.scale_min.item(), max=self.scale_max.item())
        if x.dim() == 4:
            return s.view(1, self.num_channels, 1, 1)
        if x.dim() == 3:
            return s.view(1, 1, self.num_channels)
        if x.dim() == 2:
            return s.view(1, self.num_channels)
        # fallback: broadcast along last dimension
        return s.view(*([1] * (x.dim() - 1)), self.num_channels)

    def _channel_minmax(self, x: torch.Tensor):
        if x.dim() == 4:
            cur_min = x.amin(dim=(0, 2, 3)).view(1, self.num_channels)
            cur_max = x.amax(dim=(0, 2, 3)).view(1, self.num_channels)
        elif x.dim() == 3:
            cur_min = x.amin(dim=(0, 1)).view(1, self.num_channels)
            cur_max = x.amax(dim=(0, 1)).view(1, self.num_channels)
        elif x.dim() == 2:
            cur_min = x.amin(dim=0).view(1, self.num_channels)
            cur_max = x.amax(dim=0).view(1, self.num_channels)
        else:
            x_flat = x.reshape(-1, self.num_channels)
            cur_min = x_flat.amin(dim=0).view(1, self.num_channels)
            cur_max = x_flat.amax(dim=0).view(1, self.num_channels)
        return cur_min, cur_max

    def finalize_calib_scale(self):
        qmin, qmax = self._get_qmin_qmax()
        with torch.no_grad():
            if self.unsigned:
                min_val = torch.zeros_like(self.calib_min)
                max_val = torch.clamp(self.calib_max, min=1e-8)
                scale = (max_val - min_val) / max(qmax - qmin, 1.0)
            else:
                a = torch.maximum(self.calib_min.abs(), self.calib_max.abs())
                a = torch.clamp(a, min=1e-8)
                scale = a / max(qmax, 1.0)
            scale = torch.clamp(scale, min=self.scale_min.item(), max=self.scale_max.item())
            self.log_scale.copy_(scale.log())
            self.base_log_scale.copy_(self.log_scale.data)

    def _quant_dequant(self, x: torch.Tensor, cl: bool = False):
        if self.bits is None:
            return x
        qmin, qmax = self._get_qmin_qmax()
        s = self._scale_for_x(x)
        q = torch.round(x / s)
        q = torch.clamp(q, qmin, qmax)
        return q if cl else q * s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.quant_mode
        if mode == "none":
            return x
        if mode == "calib":
            with torch.no_grad():
                cur_min, cur_max = self._channel_minmax(x.detach())
                self.calib_min = torch.minimum(self.calib_min, cur_min)
                self.calib_max = torch.maximum(self.calib_max, cur_max)
            return x
        if mode == "defense":
            x_q = self._quant_dequant(x.detach(), cl=self.cl)
            return x_q.detach()
        if mode == "fake":
            x_detach = x.detach()
            x_q = self._quant_dequant(x_detach, cl=False)
            return x + (x_q - x_detach)
        return x


class TransformerPTQMixin:
    """Shared ResNet50_PTQ-like helpers for ViT/Swin PTQ models."""

    def _iter_weight_modules(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                yield name, m

    def _iter_act_layers(self):
        for m in self.modules():
            if isinstance(m, ActQuantTokenwise):
                yield m

    def act_scale_parameters(self):
        params = []
        for m in self._iter_act_layers():
            params.append(m.log_scale)
        return params

    def _set_all_act_quant_mode(self, mode: str):
        for m in self._iter_act_layers():
            m.set_quant_mode(mode)

    def set_all_act_bits(self, bits: int):
        for m in self._iter_act_layers():
            m.set_bits(bits)

    def set_fp32_train_mode(self):
        self._set_all_act_quant_mode("none")
        for p in self.parameters():
            p.requires_grad = True

    def set_calib_mode(self):
        self._set_all_act_quant_mode("calib")

    def reset_act_calib(self):
        for m in self._iter_act_layers():
            m.reset_calib()

    def finalize_act_calib(self):
        for m in self._iter_act_layers():
            m.finalize_calib_scale()

    def set_ptq_scale_optim_mode(self):
        self._set_all_act_quant_mode("fake")
        for p in self.parameters():
            p.requires_grad = False
        for m in self._iter_act_layers():
            m.log_scale.requires_grad = True

    def set_inference_mode(self, bit_cfg: dict = None, default_bits: int = 8):
        self._set_all_act_quant_mode("defense")
        for p in self.parameters():
            p.requires_grad = False

    def fuse_model(self):
        # Transformer models do not have Conv+BN+ReLU fusion in this implementation.
        return self

    def apply_weight_ptq(self, bit_cfg: dict = None, default_bits: int = 8):
        """
        Per-output-channel symmetric fake PTQ for Conv2d/Linear weights.
        Weights are dequantized back to floating point q*s and stored in-place.
        """
        for name, m in self._iter_weight_modules():
            bits = int(bit_cfg.get(name, default_bits)) if bit_cfg is not None else int(default_bits)
            if bits is None or bits <= 0:
                continue
            w = m.weight.data
            C_out = w.size(0)
            w_view = w.reshape(C_out, -1)
            qmax = float(2 ** (bits - 1) - 1)
            qmin = float(-2 ** (bits - 1))
            max_abs = w_view.abs().amax(dim=1, keepdim=True)
            scale = max_abs / max(qmax, 1.0)
            scale[scale < 1e-8] = 1.0
            q = torch.round(w_view / scale)
            q = torch.clamp(q, qmin, qmax)
            w_deq = (q * scale).reshape_as(w)
            m.weight.data.copy_(w_deq)
            if not hasattr(m, "weight_scale"):
                m.register_buffer("weight_scale", torch.ones(C_out, device=w.device))
            m.weight_scale.copy_(scale.squeeze(1).to(w.device))

    def harden_weights_to_int(self, bit_cfg: dict = None, default_bits: int = 8):
        """
        Optional: convert current q*s weights to integer q stored as float.
        Not used by default because it changes numerical scale; kept for parity.
        """
        for name, m in self._iter_weight_modules():
            if not hasattr(m, "weight_scale"):
                continue
            bits = int(bit_cfg.get(name, default_bits)) if bit_cfg is not None else int(default_bits)
            qmax = float(2 ** (bits - 1) - 1)
            qmin = float(-2 ** (bits - 1))
            w = m.weight.data
            C_out = w.size(0)
            scale = m.weight_scale.view(C_out, 1)
            q = torch.round(w.reshape(C_out, -1) / scale).clamp(qmin, qmax)
            m.weight.data.copy_(q.reshape_as(w))

    def assign_act_bits_from_weight_cfg(self, bit_cfg: dict, default_bits: int = 8):
        """
        Generic fallback: set all activation quantizers to default_bits.
        Model-specific classes below override this to map nearby weight layers.
        """
        self.set_all_act_bits(default_bits)


class MLPPTQ(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.act_fc1 = ActQuantTokenwise(hidden_features, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act_fc2 = ActQuantTokenwise(out_features, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, feat_list=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_fc1(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act_fc2(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.drop(x)
        return x


class AttentionPTQ(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.act_qkv = ActQuantTokenwise(dim * 3, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act_attn_out = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.proj = nn.Linear(dim, dim)
        self.act_proj = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, feat_list=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = self.act_qkv(qkv)
        if feat_list is not None:
            feat_list.append(qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.act_attn_out(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.proj(x)
        x = self.act_proj(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.proj_drop(x)
        return x


class BlockPTQ(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionPTQ(dim, num_heads, qkv_bias, attn_drop, drop, act_bits, act_scale_min, act_scale_max)
        self.act_res1 = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPPTQ(dim, int(dim * mlp_ratio), dim, drop, act_bits, act_scale_min, act_scale_max)
        self.act_res2 = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)

    def forward(self, x, feat_list=None):
        x = x + self.attn(self.norm1(x), feat_list=feat_list)
        x = self.act_res1(x)
        if feat_list is not None:
            feat_list.append(x)
        x = x + self.mlp(self.norm2(x), feat_list=feat_list)
        x = self.act_res2(x)
        if feat_list is not None:
            feat_list.append(x)
        return x


class ViT_PTQ(TransformerPTQMixin, nn.Module):
    MODEL_CONFIGS = ViT.MODEL_CONFIGS

    def __init__(
        self,
        model_size='base',
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=1000,
        default_act_bits: int = 8,
        act_scale_min: float = 1e-4,
        act_scale_max: float = 10.0,
    ):
        super().__init__()
        assert model_size in self.MODEL_CONFIGS, f"Unknown model_size {model_size}"
        cfg = self.MODEL_CONFIGS[model_size]
        self.model_size = model_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, cfg['embed_dim'])
        self.act_patch = ActQuantTokenwise(cfg['embed_dim'], default_act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg['embed_dim']))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg['embed_dim']))
        self.pos_drop = nn.Dropout(0.)
        self.blocks = nn.ModuleList([
            BlockPTQ(cfg['embed_dim'], cfg['num_heads'], cfg['mlp_ratio'],
                     act_bits=default_act_bits, act_scale_min=act_scale_min, act_scale_max=act_scale_max)
            for _ in range(cfg['depth'])
        ])
        self.norm = nn.LayerNorm(cfg['embed_dim'])
        self.act_pre_head = ActQuantTokenwise(cfg['embed_dim'], default_act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.head = nn.Linear(cfg['embed_dim'], num_classes)

    def assign_act_bits_from_weight_cfg(self, bit_cfg: dict, default_bits: int = 8):
        self.act_patch.set_bits(bit_cfg.get("patch_embed.proj", default_bits))
        self.act_pre_head.set_bits(bit_cfg.get("head", default_bits))
        for i, blk in enumerate(self.blocks):
            prefix = f"blocks.{i}"
            blk.attn.act_qkv.set_bits(bit_cfg.get(f"{prefix}.attn.qkv", default_bits))
            blk.attn.act_attn_out.set_bits(bit_cfg.get(f"{prefix}.attn.qkv", default_bits))
            blk.attn.act_proj.set_bits(bit_cfg.get(f"{prefix}.attn.proj", default_bits))
            blk.act_res1.set_bits(bit_cfg.get(f"{prefix}.attn.proj", default_bits))
            blk.mlp.act_fc1.set_bits(bit_cfg.get(f"{prefix}.mlp.fc1", default_bits))
            blk.mlp.act_fc2.set_bits(bit_cfg.get(f"{prefix}.mlp.fc2", default_bits))
            blk.act_res2.set_bits(bit_cfg.get(f"{prefix}.mlp.fc2", default_bits))

    def forward(self, x, release: bool = False, return_feats: bool = False):
        feats = [] if return_feats else None
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.act_patch(x)
        if feats is not None:
            feats.append(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, feat_list=feats)
        x = self.norm(x)
        if feats is not None:
            feats.append(x)
        x = self.act_pre_head(x[:, 0])
        if feats is not None:
            feats.append(x)
        x = self.head(x)
        out = F.softmax(x, dim=1) if release else F.log_softmax(x, dim=1)
        if return_feats:
            return feats, out
        return out


# ----- Swin PTQ modules -----

class MlpPTQ(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.act_fc1 = ActQuantTokenwise(hidden_features, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act_fc2 = ActQuantTokenwise(out_features, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, feat_list=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_fc1(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act_fc2(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.drop(x)
        return x


class WindowAttentionPTQ(nn.Module):
    def __init__(self, dim, window_size: Tuple[int, int], num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads))
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.act_qkv = ActQuantTokenwise(dim * 3, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act_attn_out = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.proj = nn.Linear(dim, dim)
        self.act_proj = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask: Optional[torch.Tensor] = None, feat_list=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = self.act_qkv(qkv)
        if feat_list is not None:
            feat_list.append(qkv)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.act_attn_out(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.proj(x)
        x = self.act_proj(x)
        if feat_list is not None:
            feat_list.append(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlockPTQ(nn.Module):
    def __init__(self, dim, input_resolution: Tuple[int,int], num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionPTQ(dim, window_size=(window_size, window_size), num_heads=num_heads,
                                       qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                       act_bits=act_bits, act_scale_min=act_scale_min, act_scale_max=act_scale_max)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act_res1 = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.norm2 = norm_layer(dim)
        self.mlp = MlpPTQ(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop,
                          act_bits=act_bits, act_scale_min=act_scale_min, act_scale_max=act_scale_max)
        self.act_res2 = ActQuantTokenwise(dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)

        if self.shift_size > 0:
            H, W = self.input_resolution
            # window_partition requires H/W divisible by window_size. For small
            # inputs such as 64x64 with patch_size=4, the first Swin resolution
            # is 16x16, which is not divisible by the default window_size=7.
            # Build the attention mask on the padded resolution; forward() pads
            # feature maps the same way and crops back afterwards.
            Hp = int(math.ceil(H / window_size)) * window_size
            Wp = int(math.ceil(W / window_size)) * window_size
            img_mask = torch.zeros((1, Hp, Wp, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, H, W, feat_list=None):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # Pad to multiples of window_size before window partition.
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask, feat_list=feat_list)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = self.act_res1(x)
        if feat_list is not None:
            feat_list.append(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), feat_list=feat_list))
        x = self.act_res2(x)
        if feat_list is not None:
            feat_list.append(x)
        return x


class PatchMergingPTQ(nn.Module):
    def __init__(self, input_resolution: Tuple[int,int], dim, norm_layer=nn.LayerNorm,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.act_out = ActQuantTokenwise(2 * dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)

    def forward(self, x, H, W, feat_list=None):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        x = self.act_out(x)
        if feat_list is not None:
            feat_list.append(x)
        return x


class BasicLayerPTQ(nn.Module):
    def __init__(self, dim, input_resolution: Tuple[int,int], depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None, norm_layer=nn.LayerNorm,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SwinTransformerBlockPTQ(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                act_bits=act_bits,
                act_scale_min=act_scale_min,
                act_scale_max=act_scale_max,
            ))
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                     act_bits=act_bits, act_scale_min=act_scale_min, act_scale_max=act_scale_max) if downsample else None

    def forward(self, x, H, W, feat_list=None):
        for blk in self.blocks:
            x = blk(x, H, W, feat_list=feat_list)
        if self.downsample:
            x = self.downsample(x, H, W, feat_list=feat_list)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W


class PatchEmbedSwinPTQ(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None,
                 act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size
        self.W = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None
        self.act_out = ActQuantTokenwise(embed_dim, act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)

    def forward(self, x, feat_list=None):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act_out(x)
        if feat_list is not None:
            feat_list.append(x)
        return x, H, W


class SwinTransformer_PTQ(TransformerPTQMixin, nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=1,
                 num_classes=1000,
                 embed_dim=128,
                 depths=[2,2,18,2],
                 num_heads=[4,8,16,32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ape=False,
                 default_act_bits: int = 8,
                 act_scale_min: float = 1e-4,
                 act_scale_max: float = 10.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.window_size = window_size
        self.patch_embed = PatchEmbedSwinPTQ(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                             norm_layer=norm_layer if patch_norm else None,
                                             act_bits=default_act_bits, act_scale_min=act_scale_min, act_scale_max=act_scale_max)
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.ape:
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.layers = nn.ModuleList()
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        H = W = img_size // patch_size
        for i_layer in range(self.num_layers):
            layer = BasicLayerPTQ(
                dim=dims[i_layer],
                input_resolution=(H, W),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur + depths[i_layer]],
                downsample=PatchMergingPTQ if (i_layer < self.num_layers - 1) else None,
                norm_layer=norm_layer,
                act_bits=default_act_bits,
                act_scale_min=act_scale_min,
                act_scale_max=act_scale_max,
            )
            self.layers.append(layer)
            cur += depths[i_layer]
            if i_layer < self.num_layers - 1:
                H = (H + 1) // 2
                W = (W + 1) // 2

        self.norm = norm_layer(dims[-1])
        self.act_pre_head = ActQuantTokenwise(dims[-1], default_act_bits, scale_min=act_scale_min, scale_max=act_scale_max, unsigned=False)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def assign_act_bits_from_weight_cfg(self, bit_cfg: dict, default_bits: int = 8):
        self.patch_embed.act_out.set_bits(bit_cfg.get("patch_embed.proj", default_bits))
        self.act_pre_head.set_bits(bit_cfg.get("head", default_bits))
        for li, layer in enumerate(self.layers):
            for bi, blk in enumerate(layer.blocks):
                prefix = f"layers.{li}.blocks.{bi}"
                blk.attn.act_qkv.set_bits(bit_cfg.get(f"{prefix}.attn.qkv", default_bits))
                blk.attn.act_attn_out.set_bits(bit_cfg.get(f"{prefix}.attn.qkv", default_bits))
                blk.attn.act_proj.set_bits(bit_cfg.get(f"{prefix}.attn.proj", default_bits))
                blk.act_res1.set_bits(bit_cfg.get(f"{prefix}.attn.proj", default_bits))
                blk.mlp.act_fc1.set_bits(bit_cfg.get(f"{prefix}.mlp.fc1", default_bits))
                blk.mlp.act_fc2.set_bits(bit_cfg.get(f"{prefix}.mlp.fc2", default_bits))
                blk.act_res2.set_bits(bit_cfg.get(f"{prefix}.mlp.fc2", default_bits))
            if layer.downsample is not None:
                layer.downsample.act_out.set_bits(bit_cfg.get(f"layers.{li}.downsample.reduction", default_bits))

    def forward(self, x: torch.Tensor, release: bool = False, return_feats: bool = False):
        feats = [] if return_feats else None
        x, H, W = self.patch_embed(x, feat_list=feats)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W, feat_list=feats)
        x = self.norm(x)
        if feats is not None:
            feats.append(x)
        x = x.mean(dim=1)
        x = self.act_pre_head(x)
        if feats is not None:
            feats.append(x)
        x = self.head(x)
        out = F.softmax(x, dim=1) if release else F.log_softmax(x, dim=1)
        if return_feats:
            return feats, out
        return out


def vit_ptq(model_size='base', img_size=224, patch_size=16, in_chans=1, num_classes=1000,
            default_act_bits: int = 8, act_scale_min: float = 1e-4, act_scale_max: float = 10.0):
    return ViT_PTQ(
        model_size=model_size,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        default_act_bits=default_act_bits,
        act_scale_min=act_scale_min,
        act_scale_max=act_scale_max,
    )


def swin_b_ptq(num_classes=1000, img_size=224, in_chans=1, default_act_bits: int = 8,
               act_scale_min: float = 1e-4, act_scale_max: float = 10.0, **kwargs):
    return SwinTransformer_PTQ(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=128,
        depths=[2,2,18,2],
        num_heads=[4,8,16,32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        ape=False,
        default_act_bits=default_act_bits,
        act_scale_min=act_scale_min,
        act_scale_max=act_scale_max,
        **kwargs,
    )
