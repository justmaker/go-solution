#!/usr/bin/env python3
"""
Convert KataGo text-format model (v11, b6c64) to ONNX.

Supports ordinary_block and gpool_block residual blocks,
MISH activation, and the full KataGo head structure:
  - Policy head (spatial + gpool + pass)
  - Value head (3-class win/loss/noresult)
  - Score value head (6 outputs)
  - Ownership head (1 channel per position)

Key insight: KataGo's BN stores (mean, variance, scale, bias) where
variance is raw variance (std^2 - epsilon). We compute
inv_std = 1/sqrt(variance + epsilon) and fuse BN into an affine
transform for clean ONNX export.

Conv weights are stored as [kh, kw, in_c, out_c] and transposed to
PyTorch's [out_c, in_c, kh, kw]. Matmul weights are [in_f, out_f].
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Token reader ───────────────────────────────────────────────────────────

class TextReader:
    """Read KataGo text format model, one token at a time."""
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.tokens = f.read().split()
        self.pos = 0

    def read_str(self):
        s = self.tokens[self.pos]
        self.pos += 1
        return s

    def read_int(self):
        return int(self.read_str())

    def read_float(self):
        return float(self.read_str())

    def read_floats(self, n):
        end = self.pos + n
        vals = [float(self.tokens[i]) for i in range(self.pos, end)]
        self.pos = end
        return vals

    def peek(self):
        return self.tokens[self.pos]


# ─── Weight reading helpers ─────────────────────────────────────────────────

def read_conv(reader):
    """Read conv: name kh kw in_c out_c sh sw weights...

    KataGo stores conv weights in [kh, kw, in_c, out_c] order.
    PyTorch expects [out_c, in_c, kh, kw].
    """
    name = reader.read_str()
    kh = reader.read_int()
    kw = reader.read_int()
    in_c = reader.read_int()
    out_c = reader.read_int()
    reader.read_int()  # sh
    reader.read_int()  # sw
    n = out_c * in_c * kh * kw
    w = np.array(reader.read_floats(n), dtype=np.float32).reshape(kh, kw, in_c, out_c)
    w = w.transpose(3, 2, 0, 1).copy()  # [out_c, in_c, kh, kw]
    print(f"  conv '{name}': [{in_c},{kh},{kw}] -> {out_c}")
    return w


def read_bn(reader):
    """Read BN: name nc epsilon has_scale has_bias mean[nc] variance[nc] scale[nc] bias[nc]

    KataGo stores raw variance (approximately std^2 - epsilon).
    C++ loader computes: mergedScale = scale / sqrt(variance + epsilon)
    This matches PyTorch's nn.BatchNorm2d behavior exactly.
    """
    name = reader.read_str()
    nc = reader.read_int()
    eps = reader.read_float()  # epsilon
    reader.read_int()    # has_scale
    reader.read_int()    # has_bias
    mean = np.array(reader.read_floats(nc), dtype=np.float32)
    variance = np.array(reader.read_floats(nc), dtype=np.float32)
    scale = np.array(reader.read_floats(nc), dtype=np.float32)
    bias = np.array(reader.read_floats(nc), dtype=np.float32)
    print(f"  bn '{name}': nc={nc}, eps={eps}")
    return nc, eps, mean, variance, scale, bias


def read_actv(reader):
    """Read activation: name type (e.g. ACTIVATION_MISH)"""
    name = reader.read_str()
    actv_type = reader.read_str()
    print(f"  actv '{name}': {actv_type}")
    return actv_type


def read_matmul(reader):
    """Read matmul: name in_f out_f weights[in_f*out_f]

    KataGo stores weights in [in_channels][out_channels] order.
    PyTorch nn.Linear expects weight shape [out_features, in_features].
    So we reshape to (in_f, out_f) then transpose.
    """
    name = reader.read_str()
    in_f = reader.read_int()
    out_f = reader.read_int()
    w = np.array(reader.read_floats(in_f * out_f), dtype=np.float32).reshape(in_f, out_f).T
    print(f"  matmul '{name}': {in_f} -> {out_f}")
    return w


def read_matbias(reader):
    """Read matbias: name n bias[n]"""
    name = reader.read_str()
    n = reader.read_int()
    b = np.array(reader.read_floats(n), dtype=np.float32)
    print(f"  matbias '{name}': {n}")
    return b


# ─── Fused BN as affine transform ─────────────────────────────────────────

class FusedBN2d(nn.Module):
    """BatchNorm fused into a channel-wise affine: output = x * fused_scale + fused_bias

    KataGo BN stores (mean, variance, gamma, beta).
    C++ computes: mergedScale = gamma / sqrt(variance + epsilon)
                  mergedBias = beta - mean * mergedScale
    Then: output = input * mergedScale + mergedBias
    """
    def __init__(self, nc, eps, mean, variance, gamma, beta):
        super().__init__()
        inv_std = 1.0 / np.sqrt(variance + eps)
        fused_scale = gamma * inv_std
        fused_bias = beta - mean * fused_scale
        self.register_buffer('fused_scale', torch.from_numpy(fused_scale).view(1, nc, 1, 1))
        self.register_buffer('fused_bias', torch.from_numpy(fused_bias).view(1, nc, 1, 1))

    def forward(self, x):
        return x * self.fused_scale + self.fused_bias


def make_bn(nc, eps, mean, variance, scale, bias):
    """Create a fused BN module from KataGo parameters."""
    return FusedBN2d(nc, eps, mean, variance, scale, bias)


# ─── MISH activation ────────────────────────────────────────────────────────

def mish(x):
    return x * torch.tanh(F.softplus(x))


# ─── Global pooling helpers ─────────────────────────────────────────────────

def kata_gpool(x):
    """KataGo global pooling for trunk gpool blocks and policy head.
    Output [B, 3*C]: mean, mean*(sqrt(N)-14)/10, max
    """
    B, C, H, W = x.shape
    N = float(H * W)
    sqrt_N = N ** 0.5
    mean = x.sum(dim=(2, 3)) / N                        # [B, C]
    max_val = x.amax(dim=(2, 3))                         # [B, C]
    scaled_mean = mean * (sqrt_N - 14.0) / 10.0          # [B, C]
    return torch.cat([mean, scaled_mean, max_val], dim=1) # [B, 3*C]


def kata_value_gpool(x):
    """KataGo global pooling for value head.
    Output [B, 3*C]: mean, mean*(sqrt(N)-14)/10, mean*((sqrt(N)-14)^2/100 - 0.1)
    """
    B, C, H, W = x.shape
    N = float(H * W)
    sqrt_N = N ** 0.5
    mean = x.sum(dim=(2, 3)) / N                          # [B, C]
    scale1 = (sqrt_N - 14.0) / 10.0
    scale2 = (sqrt_N - 14.0) ** 2 / 100.0 - 0.1
    scaled_mean1 = mean * scale1                           # [B, C]
    scaled_mean2 = mean * scale2                           # [B, C]
    return torch.cat([mean, scaled_mean1, scaled_mean2], dim=1)  # [B, 3*C]


# ─── Model components ───────────────────────────────────────────────────────

class OrdinaryBlock(nn.Module):
    def __init__(self, trunk_c):
        super().__init__()
        self.bn1 = None  # Will be set during weight loading
        self.conv1 = nn.Conv2d(trunk_c, trunk_c, 3, padding=1, bias=False)
        self.bn2 = None
        self.conv2 = nn.Conv2d(trunk_c, trunk_c, 3, padding=1, bias=False)

    def forward(self, x):
        out = mish(self.bn1(x))
        out = self.conv1(out)
        out = mish(self.bn2(out))
        out = self.conv2(out)
        return out + x


class GPoolBlock(nn.Module):
    def __init__(self, trunk_c, reg_c, gpool_c):
        super().__init__()
        self.bn1 = None
        self.conv1a = nn.Conv2d(trunk_c, reg_c, 3, padding=1, bias=False)
        self.conv1b = nn.Conv2d(trunk_c, gpool_c, 3, padding=1, bias=False)
        self.bn1b = None
        self.w1r = nn.Linear(3 * gpool_c, reg_c, bias=False)
        self.bn2 = None
        self.conv2 = nn.Conv2d(reg_c, trunk_c, 3, padding=1, bias=False)

    def forward(self, x):
        out = mish(self.bn1(x))
        reg = self.conv1a(out)
        gp = self.conv1b(out)
        gp = mish(self.bn1b(gp))
        gp_pooled = kata_gpool(gp)
        gp_out = self.w1r(gp_pooled)
        reg = reg + gp_out.unsqueeze(-1).unsqueeze(-1)
        out2 = mish(self.bn2(reg))
        out2 = self.conv2(out2)
        return out2 + x


class PolicyHead(nn.Module):
    def __init__(self, trunk_c, policy_c, gpool_c_policy):
        super().__init__()
        self.p1_conv = nn.Conv2d(trunk_c, policy_c, 1, bias=False)
        self.g1_conv = nn.Conv2d(trunk_c, gpool_c_policy, 1, bias=False)
        self.g1_bn = None
        self.g2_matmul = nn.Linear(3 * gpool_c_policy, policy_c, bias=False)
        self.p1_bn = None
        self.p2_conv = nn.Conv2d(policy_c, 1, 1, bias=False)
        self.pass_matmul = nn.Linear(3 * gpool_c_policy, 1, bias=False)

    def forward(self, trunk_out):
        p = self.p1_conv(trunk_out)
        g = self.g1_conv(trunk_out)
        g = mish(self.g1_bn(g))
        g_pooled = kata_gpool(g)
        g_out = self.g2_matmul(g_pooled)
        p = p + g_out.unsqueeze(-1).unsqueeze(-1)
        p = mish(self.p1_bn(p))
        spatial = self.p2_conv(p)
        pass_logit = self.pass_matmul(g_pooled)
        B, _, H, W = spatial.shape
        spatial_flat = spatial.reshape(B, H * W)
        policy = torch.cat([spatial_flat, pass_logit], dim=1)
        return policy


class ValueHead(nn.Module):
    def __init__(self, trunk_c, v1_c, v2_hidden):
        super().__init__()
        self.v1_conv = nn.Conv2d(trunk_c, v1_c, 1, bias=False)
        self.v1_bn = None
        self.v2_fc = nn.Linear(3 * v1_c, v2_hidden)
        self.v3_fc = nn.Linear(v2_hidden, 3)
        self.sv3_fc = nn.Linear(v2_hidden, 6)

    def forward(self, trunk_out):
        v = self.v1_conv(trunk_out)
        v = mish(self.v1_bn(v))
        v_pooled = kata_value_gpool(v)
        v_hidden = mish(self.v2_fc(v_pooled))
        value = self.v3_fc(v_hidden)
        score_value = self.sv3_fc(v_hidden)
        return value, score_value


class KataGoNet(nn.Module):
    def __init__(self, num_blocks, trunk_c, reg_c, gpool_c, block_types,
                 bin_features=22, global_features=19, board_size=19,
                 policy_c=24, gpool_c_policy=24, v1_c=24, v2_hidden=48):
        super().__init__()
        self.board_size = board_size
        self.init_conv = nn.Conv2d(bin_features, trunk_c, 3, padding=1, bias=False)
        self.g_linear = nn.Linear(global_features, trunk_c, bias=False)
        blocks = []
        for bt in block_types:
            if bt == 'ordinary_block':
                blocks.append(OrdinaryBlock(trunk_c))
            elif bt == 'gpool_block':
                blocks.append(GPoolBlock(trunk_c, reg_c, gpool_c))
        self.blocks = nn.ModuleList(blocks)
        self.trunk_bn = None
        self.policy_head = PolicyHead(trunk_c, policy_c, gpool_c_policy)
        self.value_head = ValueHead(trunk_c, v1_c, v2_hidden)
        self.own_conv = nn.Conv2d(v1_c, 1, 1, bias=False)

    def forward(self, input_binary, input_global):
        x = self.init_conv(input_binary)
        g = self.g_linear(input_global)
        x = x + g.unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            x = block(x)
        trunk_out = mish(self.trunk_bn(x))

        policy = self.policy_head(trunk_out)

        v1_out = mish(self.value_head.v1_bn(self.value_head.v1_conv(trunk_out)))
        v_pooled = kata_value_gpool(v1_out)
        v_hidden = mish(self.value_head.v2_fc(v_pooled))
        value = self.value_head.v3_fc(v_hidden)

        ownership = torch.tanh(self.own_conv(v1_out))

        return policy, value, ownership


# ─── Weight loading ─────────────────────────────────────────────────────────

def load_all_weights(reader, model, num_blocks, trunk_c, reg_c, gpool_c):
    """Load ALL weights from the text model file."""

    # 1. Initial conv
    print("Loading initial conv...")
    w = read_conv(reader)
    model.init_conv.weight.data = torch.from_numpy(w)

    # 2. Global input matmul
    print("Loading global input weights...")
    gw = read_matmul(reader)
    model.g_linear.weight.data = torch.from_numpy(gw)

    # 3. Residual blocks
    for i in range(num_blocks):
        block_type = reader.read_str()
        block_name = reader.read_str()
        print(f"\nBlock {i}: {block_type} ({block_name})")
        block = model.blocks[i]

        if block_type == 'ordinary_block':
            nc, eps, mean, variance, scale, bias = read_bn(reader)
            block.bn1 = make_bn(nc, eps, mean, variance, scale, bias)
            read_actv(reader)
            w1 = read_conv(reader)
            block.conv1.weight.data = torch.from_numpy(w1)
            nc, eps, mean, variance, scale, bias = read_bn(reader)
            block.bn2 = make_bn(nc, eps, mean, variance, scale, bias)
            read_actv(reader)
            w2 = read_conv(reader)
            block.conv2.weight.data = torch.from_numpy(w2)

        elif block_type == 'gpool_block':
            nc, eps, mean, variance, scale, bias = read_bn(reader)
            block.bn1 = make_bn(nc, eps, mean, variance, scale, bias)
            read_actv(reader)
            w1a = read_conv(reader)
            block.conv1a.weight.data = torch.from_numpy(w1a)
            w1b = read_conv(reader)
            block.conv1b.weight.data = torch.from_numpy(w1b)
            nc, eps, mean, variance, scale, bias = read_bn(reader)
            block.bn1b = make_bn(nc, eps, mean, variance, scale, bias)
            read_actv(reader)
            w1r = read_matmul(reader)
            block.w1r.weight.data = torch.from_numpy(w1r)
            nc, eps, mean, variance, scale, bias = read_bn(reader)
            block.bn2 = make_bn(nc, eps, mean, variance, scale, bias)
            read_actv(reader)
            w2 = read_conv(reader)
            block.conv2.weight.data = torch.from_numpy(w2)

    # 4. Trunk final BN
    print("\nLoading trunk output BN...")
    nc, eps, mean, variance, scale, bias = read_bn(reader)
    model.trunk_bn = make_bn(nc, eps, mean, variance, scale, bias)
    read_actv(reader)

    # 5. Policy head
    print("\nLoading policy head...")
    section = reader.read_str()
    print(f"  Section: {section}")

    p1w = read_conv(reader)
    model.policy_head.p1_conv.weight.data = torch.from_numpy(p1w)
    g1w = read_conv(reader)
    model.policy_head.g1_conv.weight.data = torch.from_numpy(g1w)
    nc, eps, mean, variance, scale, bias = read_bn(reader)
    model.policy_head.g1_bn = make_bn(nc, eps, mean, variance, scale, bias)
    read_actv(reader)
    g2w = read_matmul(reader)
    model.policy_head.g2_matmul.weight.data = torch.from_numpy(g2w)
    nc, eps, mean, variance, scale, bias = read_bn(reader)
    model.policy_head.p1_bn = make_bn(nc, eps, mean, variance, scale, bias)
    read_actv(reader)
    p2w = read_conv(reader)
    model.policy_head.p2_conv.weight.data = torch.from_numpy(p2w)
    passw = read_matmul(reader)
    model.policy_head.pass_matmul.weight.data = torch.from_numpy(passw)

    # 6. Value head
    print("\nLoading value head...")
    section = reader.read_str()
    print(f"  Section: {section}")

    v1w = read_conv(reader)
    model.value_head.v1_conv.weight.data = torch.from_numpy(v1w)
    nc, eps, mean, variance, scale, bias = read_bn(reader)
    model.value_head.v1_bn = make_bn(nc, eps, mean, variance, scale, bias)
    read_actv(reader)
    v2w = read_matmul(reader)
    model.value_head.v2_fc.weight.data = torch.from_numpy(v2w)
    v2b = read_matbias(reader)
    model.value_head.v2_fc.bias.data = torch.from_numpy(v2b)
    read_actv(reader)
    v3w = read_matmul(reader)
    model.value_head.v3_fc.weight.data = torch.from_numpy(v3w)
    v3b = read_matbias(reader)
    model.value_head.v3_fc.bias.data = torch.from_numpy(v3b)
    sv3w = read_matmul(reader)
    model.value_head.sv3_fc.weight.data = torch.from_numpy(sv3w)
    sv3b = read_matbias(reader)
    model.value_head.sv3_fc.bias.data = torch.from_numpy(sv3b)
    vow = read_conv(reader)
    model.own_conv.weight.data = torch.from_numpy(vow)

    print(f"\nAll weights loaded! (pos={reader.pos}/{len(reader.tokens)})")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_katago_to_onnx.py <input.txt> <output.onnx>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    board_size = 19

    print(f"Reading: {input_file}")
    reader = TextReader(input_file)

    # Header
    model_name = reader.read_str()
    version = reader.read_int()
    bin_features = reader.read_int()
    global_features = reader.read_int()

    print(f"Model: {model_name}")
    print(f"Version: {version}, BinFeatures: {bin_features}, GlobalFeatures: {global_features}")

    # Trunk config
    section = reader.read_str()  # "trunk"
    num_blocks = reader.read_int()
    trunk_c = reader.read_int()
    mid_c = reader.read_int()
    reg_c = reader.read_int()
    dil_c = reader.read_int()
    gpool_c = reader.read_int()

    print(f"Trunk: blocks={num_blocks}, channels={trunk_c}")
    print(f"  mid={mid_c}, regular={reg_c}, dilated={dil_c}, gpool={gpool_c}")

    # Scan block types (save position, scan ahead, restore)
    saved_pos = reader.pos

    # Skip initial conv
    reader.read_str()
    kh = reader.read_int(); kw = reader.read_int()
    in_c = reader.read_int(); out_c = reader.read_int()
    reader.read_int(); reader.read_int()
    reader.pos += out_c * in_c * kh * kw

    # Skip ginputw
    reader.read_str()
    in_f = reader.read_int(); out_f = reader.read_int()
    reader.pos += in_f * out_f

    block_types = []
    for _ in range(num_blocks):
        bt = reader.read_str()
        block_types.append(bt)
        block_name = reader.read_str()
        if bt == 'ordinary_block':
            # norm1: name + nc + 3(eps/has_s/has_b) + 4*nc
            reader.read_str(); nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            reader.pos += 2  # actv1
            reader.read_str()
            kkh = reader.read_int(); kkw = reader.read_int()
            ic = reader.read_int(); oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
            reader.read_str(); nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            reader.pos += 2  # actv2
            reader.read_str()
            kkh = reader.read_int(); kkw = reader.read_int()
            ic = reader.read_int(); oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
        elif bt == 'gpool_block':
            reader.read_str(); nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            reader.pos += 2
            reader.read_str()
            kkh = reader.read_int(); kkw = reader.read_int()
            ic = reader.read_int(); oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
            reader.read_str()
            kkh = reader.read_int(); kkw = reader.read_int()
            ic = reader.read_int(); oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
            reader.read_str(); nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            reader.pos += 2
            reader.read_str()
            inf = reader.read_int(); outf = reader.read_int()
            reader.pos += inf * outf
            reader.read_str(); nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            reader.pos += 2
            reader.read_str()
            kkh = reader.read_int(); kkw = reader.read_int()
            ic = reader.read_int(); oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw

    print(f"Block types: {block_types}")
    reader.pos = saved_pos

    # Head sizes
    policy_c = 24
    gpool_c_policy = 24
    v1_c = 24
    v2_hidden = 48

    # Build model
    model = KataGoNet(
        num_blocks, trunk_c, reg_c, gpool_c, block_types,
        bin_features, global_features, board_size,
        policy_c, gpool_c_policy, v1_c, v2_hidden,
    )

    # Load weights
    print("\n=== Loading weights ===")
    load_all_weights(reader, model, num_blocks, trunk_c, reg_c, gpool_c)

    model.eval()

    # Test forward pass
    print(f"\nForward pass test (board={board_size}x{board_size})...")
    dummy_bin = torch.zeros(1, bin_features, board_size, board_size)
    # KataGo binary features for empty board (black to play):
    # Ch 0: board mask (1 for all valid positions)
    # Ch 1: own stones (current player) - empty board = 0
    # Ch 2: opponent stones - empty board = 0
    # Ch 8: 1 if current player is black
    dummy_bin[0, 0, :, :] = 1.0  # board mask
    dummy_bin[0, 8, :, :] = 1.0  # black to play
    dummy_glob = torch.zeros(1, global_features)
    dummy_glob[0, 0] = 7.5 / 20.0  # komi
    dummy_glob[0, 1] = 1.0          # black to play
    with torch.no_grad():
        policy, value, ownership = model(dummy_bin, dummy_glob)
    print(f"  Policy: {policy.shape}, range=[{policy.min():.4f}, {policy.max():.4f}]")
    print(f"  Value: {value.shape}, values={value[0].numpy()}")
    print(f"  Ownership: {ownership.shape}, range=[{ownership.min():.4f}, {ownership.max():.4f}]")

    # Export ONNX
    print(f"\nExporting ONNX: {output_file}")
    torch.onnx.export(
        model,
        (dummy_bin, dummy_glob),
        output_file,
        input_names=['input_binary', 'input_global'],
        output_names=['output_policy', 'output_value', 'output_ownership'],
        dynamic_axes={
            'input_binary': {0: 'batch', 2: 'height', 3: 'width'},
            'input_global': {0: 'batch'},
            'output_policy': {0: 'batch'},
            'output_value': {0: 'batch'},
            'output_ownership': {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=13,
    )

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nDone! {output_file} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
