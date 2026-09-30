import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import fuse_modules
from torch.nn.quantized import FloatFunctional

# =========================
#   Activation quantization module
# =========================
class ActQuantLayerwise(nn.Module):
    """
    Per-layer/per-channel activation quantization (unified interface).

    quant_mode:
      - "none"    : no quantization (FP32)
      - "calib"   : collect per-channel min/max without changing the forward values
      - "fake"    : QDQ q*s with trainable scale for PTQ refinement
      - "defense" : q*s with detached gradients for inference
    """

    def __init__(
        self,
        num_channels: int,
        init_bits: int = 8,
        init_scale: float = 1.0,
        scale_min: float = 1e-4,
        scale_max: float = 10.0,
        unsigned: bool = False,
        cl : bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.bits = int(init_bits)
        self.unsigned = bool(unsigned)
        self.cl = bool(cl)

        init = torch.full(
            (1, num_channels, 1, 1),
            float(init_scale),
            dtype=torch.float32,
        )
        self.log_scale = nn.Parameter(init.log())
        self.register_buffer("base_log_scale", self.log_scale.data.clone())

        self.register_buffer("scale_min", torch.tensor(float(scale_min)))
        self.register_buffer("scale_max", torch.tensor(float(scale_max)))

        # Running per-channel min/max for calibration
        self.register_buffer(
            "calib_min",
            torch.full((1, num_channels, 1, 1), float("inf")),
        )
        self.register_buffer(
            "calib_max",
            torch.full((1, num_channels, 1, 1), float("-inf")),
        )

        self.quant_mode = "none"  # "none" | "calib" | "fake" | "defense"

    # ---------- Configuration API ----------
    def set_quant_mode(self, mode: str):
        assert mode in ("none", "calib", "fake", "defense")
        self.quant_mode = mode

    def set_bits(self, bits: int):
        self.bits = int(bits)

    def reset_calib(self):
        self.calib_min.fill_(float("inf"))
        self.calib_max.fill_(float("-inf"))

    # ---------- Internal helpers ----------
    def _get_scale(self):
        # Softplus enforces positivity, then clamp to [scale_min, scale_max]
        s = F.softplus(self.log_scale)
        s = torch.clamp(
            s,
            min=self.scale_min.item(),
            max=self.scale_max.item(),
        )
        return s  # [1, C, 1, 1]

    def _get_qmin_qmax(self):
        b = self.bits
        if self.unsigned:
            qmin = 0.0
            qmax = float(2**b - 1)
        else:
            qmax = float(2 ** (b - 1) - 1)
            qmin = float(-2 ** (b - 1))   # Use -qmax for a strictly symmetric code range
        return qmin, qmax

    # ---- QDQ: q*s with a trainable scale ----
    def _quant_dequant(self, x: torch.Tensor, cl = False) -> torch.Tensor:
        if self.bits is None:
            return x

        qmin, qmax = self._get_qmin_qmax()
        s = self._get_scale()  # [1,C,1,1]

        x_ = x
        if x_.dim() == 2:
            x_ = x_.unsqueeze(-1).unsqueeze(-1)

        q = x_ / s
        q = torch.round(q)
        q = torch.clamp(q, qmin, qmax)

        if cl:
            x_q = q
        else:
            x_q = q * s  # Dequantize back to floating point


        if x.dim() == 2:
            x_q = x_q.squeeze(-1).squeeze(-1)
        return x_q

    # Optional integer-grid helper; not used in forward
    def _quant_int(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None:
            return x

        qmin, qmax = self._get_qmin_qmax()
        s = self._get_scale()

        x_ = x
        if x_.dim() == 2:
            x_ = x_.unsqueeze(-1).unsqueeze(-1)

        q = x_ / s
        q = torch.round(q)
        q = torch.clamp(q, qmin, qmax)

        if x.dim() == 2:
            q = q.squeeze(-1).squeeze(-1)
        return q

    # ---- Initialize scale from calibrated min/max ----
    def finalize_calib_scale(self):
        """
        Initialize log_scale from calib_min / calib_max in a PTQ-style procedure:
          - unsigned=True  : treat as a ReLU output, range [0, max]
          - unsigned=False : symmetric range [-a, a], a = max(|min|, |max|)
        """
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

            scale = torch.clamp(
                scale,
                min=self.scale_min.item(),
                max=self.scale_max.item(),
            )
            self.log_scale.copy_(scale.log())

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.quant_mode

        # FP32: no quantization
        if mode == "none":
            return x

        # Calibration: collect min/max without changing values
        if mode == "calib":
            with torch.no_grad():
                x_detach = x.detach()
                if x_detach.dim() == 2:
                    x_detach = x_detach.unsqueeze(-1).unsqueeze(-1)

                cur_min = x_detach.amin(dim=(0, 2, 3), keepdim=True)
                cur_max = x_detach.amax(dim=(0, 2, 3), keepdim=True)

                self.calib_min = torch.minimum(self.calib_min, cur_min)
                self.calib_max = torch.maximum(self.calib_max, cur_max)
            return x

        # Inference: QDQ values with detached gradients
        if mode == "defense":
            x_detach = x.detach()
            x_q = self._quant_dequant(x_detach, cl=self.cl)  # q*s
            return x_q.detach()

        # PTQ scale refinement: fake quantization (STE)
        if mode == "fake":
            x_detach = x.detach()
            x_q = self._quant_dequant(x_detach)  # q*s
            return x + (x_q - x_detach)

        return x


# =========================
#   ResNet50 Bottleneck
# =========================
class BottleneckPTQ(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
        act_bits: int = 8,
        act_scale_min: float = 1e-4,
        act_scale_max: float = 10.0,
    ):
        super().__init__()

        # conv1: 1x1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.act1 = ActQuantLayerwise(
            num_channels=out_channels,
            init_bits=act_bits,
            init_scale=1.0,
            scale_min=act_scale_min,
            scale_max=act_scale_max,
            unsigned=True,  # Nonnegative after ReLU
        )

        # conv2: 3x3
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.act2 = ActQuantLayerwise(
            num_channels=out_channels,
            init_bits=act_bits,
            init_scale=1.0,
            scale_min=act_scale_min,
            scale_max=act_scale_max,
            unsigned=True,
        )

        # conv3: 1x1
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu_final = nn.ReLU(inplace=True)
        self.act_out = ActQuantLayerwise(
            num_channels=out_channels * self.expansion,
            init_bits=act_bits,
            init_scale=1.0,
            scale_min=act_scale_min,
            scale_max=act_scale_max,
            unsigned=True,  # Nonnegative after ReLU
        )

        self.skip_add = FloatFunctional()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, feat_list=None):
        identity = x

        # conv1 -> bn1 -> relu1 -> act1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.act1(out)
        if feat_list is not None:
            feat_list.append(out)

        # conv2 -> bn2 -> relu2 -> act2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.act2(out)
        if feat_list is not None:
            feat_list.append(out)

        # conv3 -> bn3
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu_final(out)
        out = self.act_out(out)
        if feat_list is not None:
            feat_list.append(out)

        return out

    def fuse_model(self):
        # conv1 + bn1 + relu1
        fuse_modules(self, ["conv1", "bn1", "relu1"], inplace=True)
        # conv2 + bn2 + relu2
        fuse_modules(self, ["conv2", "bn2", "relu2"], inplace=True)
        # conv3 + bn3
        fuse_modules(self, ["conv3", "bn3"], inplace=True)
        # downsample: Conv + BN
        if self.downsample is not None:
            fuse_modules(self.downsample, ["0", "1"], inplace=True)


# =========================
#   ResNet50 PTQ model
# =========================
class ResNet50_PTQ(nn.Module):
    """
    Three primary modes:

      1) FP32 training/evaluation:
         - set_fp32_train_mode()
         - no activation quantization (quant_mode="none")
         - weights are FP32 or previously QDQ-quantized q*s

      2) PTQ activation-scale refinement:
         - First:
             a) apply_weight_ptq(...)   # QDQ weights to q*s
             b) run calib_loader once in calibration mode, then finalize_act_calib()
         - Then:
             c) set_ptq_scale_optim_mode()   # activation QDQ; train log_scale

      3) Inference:
         - set_inference_mode(...)
         - optionally harden weights to integer codes q
         - activation quantization with detached gradients
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        img_size: int = 224,
        default_act_bits: int = 8,
        act_scale_min: float = 1e-4,
        act_scale_max: float = 10.0,
    ):
        super().__init__()
        self.inplanes = 64
        self.input_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes

        # stem
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu_stem = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # Quantize the stem output after ReLU (unsigned)
        self.act_stem = ActQuantLayerwise(
            num_channels=64,
            init_bits=default_act_bits,
            init_scale=1.0,
            scale_min=act_scale_min,
            scale_max=act_scale_max,
            unsigned=True,
        )

        # Residual blocks
        self.layer1 = self._make_layer(
            BottleneckPTQ,
            64,
            blocks=3,
            stride=1,
            act_bits=default_act_bits,
            act_scale_min=act_scale_min,
            act_scale_max=act_scale_max,
        )
        self.layer2 = self._make_layer(
            BottleneckPTQ,
            128,
            blocks=4,
            stride=2,
            act_bits=default_act_bits,
            act_scale_min=act_scale_min,
            act_scale_max=act_scale_max,
        )
        self.layer3 = self._make_layer(
            BottleneckPTQ,
            256,
            blocks=6,
            stride=2,
            act_bits=default_act_bits,
            act_scale_min=act_scale_min,
            act_scale_max=act_scale_max,
        )
        self.layer4 = self._make_layer(
            BottleneckPTQ,
            512,
            blocks=3,
            stride=2,
            act_bits=default_act_bits,
            act_scale_min=act_scale_min,
            act_scale_max=act_scale_max,
        )

        # head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Quantize the pre-head activation without ReLU (signed)
        self.act_head = ActQuantLayerwise(
            num_channels=512 * BottleneckPTQ.expansion,
            init_bits=default_act_bits,
            init_scale=1.0,
            scale_min=act_scale_min,
            scale_max=act_scale_max,
            unsigned=False,
        )
        self.fc = nn.Linear(512 * BottleneckPTQ.expansion, num_classes)

        # Parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

        # Default: FP32 mode
        self.set_fp32_train_mode()

    # ---------- Layer construction ----------
    def _make_layer(
        self,
        block,
        out_channels: int,
        blocks: int,
        stride: int,
        act_bits: int,
        act_scale_min: float,
        act_scale_max: float,
    ):
        downsample = None
        if stride != 1 or self.inplanes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                out_channels,
                stride=stride,
                downsample=downsample,
                act_bits=act_bits,
                act_scale_min=act_scale_min,
                act_scale_max=act_scale_max,
            )
        )
        self.inplanes = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    out_channels,
                    stride=1,
                    downsample=None,
                    act_bits=act_bits,
                    act_scale_min=act_scale_min,
                    act_scale_max=act_scale_max,
                )
            )

        return nn.Sequential(*layers)

    # ---------- Fuse Conv+BN(+ReLU) ----------
    def fuse_model(self):
        fuse_modules(self, ["conv1", "bn1", "relu_stem"], inplace=True)
        for m in self.modules():
            if isinstance(m, BottleneckPTQ):
                m.fuse_model()

    # ---------- Weight quantization ----------
    def _iter_weight_modules(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                yield name, m

    def apply_weight_ptq(self, bit_cfg: dict = None, default_bits: int = 8):
        """
        Apply QDQ weight quantization to all Conv/Linear modules:
          - per-channel symmetric quantization: max_abs -> scale
          - round to integer q, then write dequantized q*scale floating weights
          - save scale in m.weight_scale for optional hardening
        """
        for name, m in self._iter_weight_modules():
            bits = default_bits
            if bit_cfg is not None and name in bit_cfg:
                bits = int(bit_cfg[name])
            if bits is None or bits <= 0:
                continue

            w = m.weight.data
            C_out = w.size(0)
            w_view = w.view(C_out, -1)

            qmax = float(2 ** (bits - 1) - 1)
            qmin = float(-2 ** (bits - 1))

            max_abs = w_view.abs().amax(dim=1, keepdim=True)  # [C_out, 1]
            scale = max_abs / max(qmax, 1.0)
            scale[scale < 1e-8] = 1.0

            w_norm = w_view / scale
            q = torch.round(w_norm)
            q = torch.clamp(q, qmin, qmax)

            w_deq = (q * scale).view_as(w)
            m.weight.data.copy_(w_deq)

            if not hasattr(m, "weight_scale"):
                m.register_buffer("weight_scale", torch.ones(C_out))
            m.weight_scale.copy_(scale.squeeze(1))

    def harden_weights_to_int(self, bit_cfg: dict = None, default_bits: int = 8):
        """
        Convert q*scale weights to integer codes q (stored as floats), for optional inference.
        """
        for name, m in self._iter_weight_modules():
            bits = default_bits
            if bit_cfg is not None and name in bit_cfg:
                bits = int(bit_cfg[name])
            if bits is None or bits <= 0:
                continue

            if not hasattr(m, "weight_scale"):
                continue

            w = m.weight.data  # Currently q*scale
            C_out = w.size(0)
            w_view = w.view(C_out, -1)

            qmax = float(2 ** (bits - 1) - 1)
            qmin = float(-2 ** (bits - 1))

            scale = m.weight_scale.view(C_out, 1)

            w_norm = w_view / scale
            q = torch.round(w_norm)
            q = torch.clamp(q, qmin, qmax)

            w_int = q.view_as(w)
            m.weight.data.copy_(w_int)

    # ---------- Activation quantization ----------
    def _iter_act_layers(self):
        for m in self.modules():
            if isinstance(m, ActQuantLayerwise):
                yield m

    def act_scale_parameters(self):
        for m in self._iter_act_layers():
            yield m.log_scale

    def _set_all_act_quant_mode(self, mode: str):
        for m in self._iter_act_layers():
            m.set_quant_mode(mode)

    def set_all_act_bits(self, bits: int):
        for m in self._iter_act_layers():
            m.set_bits(bits)

    def assign_act_bits_from_weight_cfg(self, bit_cfg: dict, default_bits: int = 8):
        """
        Map weight bit_cfg to activation bit widths:
          - act_stem corresponds to stem conv1
          - act_head corresponds to fc
          - each Bottleneck:
              act1 corresponds to conv1
              act2 corresponds to conv2
              act_out corresponds to conv3 (block output)
        """
        # stem
        stem_bits = bit_cfg.get("conv1", default_bits)
        self.act_stem.set_bits(stem_bits)

        # head
        head_bits = bit_cfg.get("fc", default_bits)
        self.act_head.set_bits(head_bits)

        # Each block
        for prefix, m in self.named_modules():
            if isinstance(m, BottleneckPTQ):
                # Prefix example: 'layer1.0'
                b1 = bit_cfg.get(f"{prefix}.conv1", default_bits)
                b2 = bit_cfg.get(f"{prefix}.conv2", default_bits)
                b3 = bit_cfg.get(f"{prefix}.conv3", default_bits)

                m.act1.set_bits(b1)
                m.act2.set_bits(b2)
                m.act_out.set_bits(b3)

    # ---------- Mode transitions ----------
    def set_fp32_train_mode(self):
        """
        FP32 mode:
          - activations are not quantized (quant_mode="none")
          - all parameters have requires_grad=True
        """
        self._set_all_act_quant_mode("none")
        for p in self.parameters():
            p.requires_grad = True

    def set_calib_mode(self):
        """
        Activation calibration mode:
          - forward does not quantize; only per-channel extrema are collected
        """
        self._set_all_act_quant_mode("calib")

    def reset_act_calib(self):
        for m in self._iter_act_layers():
            m.reset_calib()

    def finalize_act_calib(self):
        """
        Initialize all activation scales from calibrated min/max.
        """
        for m in self._iter_act_layers():
            m.finalize_calib_scale()

    def set_ptq_scale_optim_mode(self):
        """
        PTQ activation-scale refinement mode:
          - assumes apply_weight_ptq() and finalize_act_calib() have run
          - activations enter trainable QDQ mode
          - train only log_scale; freeze other parameters
        """
        self._set_all_act_quant_mode("fake")
        for p in self.parameters():
            p.requires_grad = False
        for log_s in self.act_scale_parameters():
            log_s.requires_grad = True

    def set_inference_mode(self, bit_cfg: dict = None, default_bits: int = 8):
        """
        Inference mode:
          - activations use detached-gradient QDQ values
          - weights retain QDQ q*s values after apply_weight_ptq()
          - all parameters have requires_grad=False
        """
        self._set_all_act_quant_mode("defense")

        # Keep QDQ weight values; do not harden to integer codes here
        # self.harden_weights_to_int(bit_cfg=bit_cfg, default_bits=default_bits)

        for p in self.parameters():
            p.requires_grad = False

    # ---------- Forward ----------

    def forward(self, x, release: bool = False, return_feats: bool = False):
        feats = [] if return_feats else None

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu_stem(x)
        x = self.maxpool(x)
        x = self.act_stem(x)
        if feats is not None:
            feats.append(x)

        # Residual blocks
        for block in self.layer1:
            x = block(x, feat_list=feats)
        for block in self.layer2:
            x = block(x, feat_list=feats)
        for block in self.layer3:
            x = block(x, feat_list=feats)
        for block in self.layer4:
            x = block(x, feat_list=feats)

        # head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if feats is not None:
            feats.append(x)

        x = self.act_head(x)
        if feats is not None:
            feats.append(x)

        x = self.fc(x)

        if release:
            out = F.softmax(x, dim=1)
        else:
            out = F.log_softmax(x, dim=1)

        if return_feats:
            return feats, out
        else:
            return out

def resnet50_ptq(
    in_channels: int = 1,
    num_classes: int = 50,
    img_size: int = 224,
    default_act_bits: int = 8,
    act_scale_min: float = 1e-4,
    act_scale_max: float = 10.0,
):
    return ResNet50_PTQ(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        default_act_bits=default_act_bits,
        act_scale_min=act_scale_min,
        act_scale_max=act_scale_max,
    )