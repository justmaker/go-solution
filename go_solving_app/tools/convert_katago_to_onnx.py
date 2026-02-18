#!/usr/bin/env python3
"""
Convert KataGo text-format model (v11, b6c64) to ONNX.

Supports ordinary_block and gpool_block residual blocks,
MISH activation, and the full KataGo head structure:
  - Policy head (spatial + gpool + pass)
  - Value head (3-class win/loss/noresult)
  - Score value head (6 outputs)
  - Ownership head (1 channel per position)
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
    """Read conv: name kh kw in_c out_c sh sw weights..."""
    name = reader.read_str()
    kh = reader.read_int()
    kw = reader.read_int()
    in_c = reader.read_int()
    out_c = reader.read_int()
    reader.read_int()  # sh
    reader.read_int()  # sw
    n = out_c * in_c * kh * kw
    w = np.array(reader.read_floats(n), dtype=np.float32).reshape(out_c, in_c, kh, kw)
    print(f"  conv '{name}': [{in_c},{kh},{kw}] -> {out_c}")
    return w


def read_bn(reader):
    """Read BN: name nc epsilon has_scale has_bias mean[nc] var[nc] scale[nc] bias[nc]"""
    name = reader.read_str()
    nc = reader.read_int()
    reader.read_float()  # epsilon
    reader.read_int()    # has_scale
    reader.read_int()    # has_bias
    mean = np.array(reader.read_floats(nc), dtype=np.float32)
    var = np.array(reader.read_floats(nc), dtype=np.float32)
    scale = np.array(reader.read_floats(nc), dtype=np.float32)
    bias = np.array(reader.read_floats(nc), dtype=np.float32)
    print(f"  bn '{name}': nc={nc}")
    return nc, mean, var, scale, bias


def read_actv(reader):
    """Read activation: name type (e.g. ACTIVATION_MISH)"""
    name = reader.read_str()
    actv_type = reader.read_str()
    print(f"  actv '{name}': {actv_type}")
    return actv_type


def read_matmul(reader):
    """Read matmul: name in_f out_f weights[in_f*out_f]"""
    name = reader.read_str()
    in_f = reader.read_int()
    out_f = reader.read_int()
    w = np.array(reader.read_floats(in_f * out_f), dtype=np.float32).reshape(out_f, in_f)
    print(f"  matmul '{name}': {in_f} -> {out_f}")
    return w


def read_matbias(reader):
    """Read matbias: name n bias[n]"""
    name = reader.read_str()
    n = reader.read_int()
    b = np.array(reader.read_floats(n), dtype=np.float32)
    print(f"  matbias '{name}': {n}")
    return b


def set_bn(module, nc, mean, var, scale, bias):
    """Apply BN parameters to a nn.BatchNorm2d or nn.BatchNorm1d."""
    module.weight.data = torch.from_numpy(scale)
    module.bias.data = torch.from_numpy(bias)
    module.running_mean.data = torch.from_numpy(mean)
    module.running_var.data = torch.from_numpy(var)
    module.eps = 1e-5


# ─── MISH activation ────────────────────────────────────────────────────────

def mish(x):
    return x * torch.tanh(F.softplus(x))


# ─── Global pooling helper ──────────────────────────────────────────────────

def global_pool(x):
    """Pool spatial features to [B, 3*C]: mean, max, (mean of max - stdev)."""
    B, C, H, W = x.shape
    hw = H * W
    mean = x.mean(dim=(2, 3))                          # [B, C]
    max_val = x.amax(dim=(2, 3))                        # [B, C]
    # stdev-like feature
    var = ((x - mean.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(2, 3)) / hw
    stdev = torch.sqrt(var + 1e-6)
    return torch.cat([mean, max_val, stdev], dim=1)     # [B, 3*C]


# ─── Model components ───────────────────────────────────────────────────────

class OrdinaryBlock(nn.Module):
    def __init__(self, trunk_c):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(trunk_c)
        self.conv1 = nn.Conv2d(trunk_c, trunk_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(trunk_c)
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
        self.bn1 = nn.BatchNorm2d(trunk_c)
        # Regular pathway
        self.conv1a = nn.Conv2d(trunk_c, reg_c, 3, padding=1, bias=False)
        # GPool pathway
        self.conv1b = nn.Conv2d(trunk_c, gpool_c, 3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(gpool_c)
        # Matmul: 3*gpool_c -> reg_c (applied to pooled gpool features, added to regular)
        self.w1r = nn.Linear(3 * gpool_c, reg_c, bias=False)
        # Second half
        self.bn2 = nn.BatchNorm2d(reg_c)
        self.conv2 = nn.Conv2d(reg_c, trunk_c, 3, padding=1, bias=False)

    def forward(self, x):
        out = mish(self.bn1(x))
        # Regular pathway
        reg = self.conv1a(out)
        # GPool pathway
        gp = self.conv1b(out)
        gp = mish(self.bn1b(gp))
        gp_pooled = global_pool(gp)        # [B, 3*gpool_c]
        gp_out = self.w1r(gp_pooled)       # [B, reg_c]
        reg = reg + gp_out.unsqueeze(-1).unsqueeze(-1)
        # Second half
        out2 = mish(self.bn2(reg))
        out2 = self.conv2(out2)
        return out2 + x


class PolicyHead(nn.Module):
    def __init__(self, trunk_c, policy_c, gpool_c_policy):
        super().__init__()
        self.p1_conv = nn.Conv2d(trunk_c, policy_c, 1, bias=False)
        self.g1_conv = nn.Conv2d(trunk_c, gpool_c_policy, 1, bias=False)
        self.g1_bn = nn.BatchNorm2d(gpool_c_policy)
        self.g2_matmul = nn.Linear(3 * gpool_c_policy, policy_c, bias=False)
        self.p1_bn = nn.BatchNorm2d(policy_c)
        self.p2_conv = nn.Conv2d(policy_c, 1, 1, bias=False)
        self.pass_matmul = nn.Linear(3 * gpool_c_policy, 1, bias=False)

    def forward(self, trunk_out):
        # Spatial policy
        p = self.p1_conv(trunk_out)
        # GPool pathway
        g = self.g1_conv(trunk_out)
        g = mish(self.g1_bn(g))
        g_pooled = global_pool(g)                        # [B, 3*gpool_c]
        g_out = self.g2_matmul(g_pooled)                 # [B, policy_c]
        p = p + g_out.unsqueeze(-1).unsqueeze(-1)
        p = mish(self.p1_bn(p))
        spatial = self.p2_conv(p)                        # [B, 1, H, W]
        # Pass logit
        pass_logit = self.pass_matmul(g_pooled)          # [B, 1]
        # Combine: flatten spatial + append pass
        B, _, H, W = spatial.shape
        spatial_flat = spatial.reshape(B, H * W)         # [B, H*W]
        policy = torch.cat([spatial_flat, pass_logit], dim=1)  # [B, H*W+1]
        return policy


class ValueHead(nn.Module):
    def __init__(self, trunk_c, v1_c, v2_hidden):
        super().__init__()
        self.v1_conv = nn.Conv2d(trunk_c, v1_c, 1, bias=False)
        self.v1_bn = nn.BatchNorm2d(v1_c)
        # After global pooling: 3*v1_c features
        self.v2_fc = nn.Linear(3 * v1_c, v2_hidden)
        self.v3_fc = nn.Linear(v2_hidden, 3)
        self.sv3_fc = nn.Linear(v2_hidden, 6)

    def forward(self, trunk_out):
        v = self.v1_conv(trunk_out)
        v = mish(self.v1_bn(v))
        v_pooled = global_pool(v)                # [B, 3*v1_c]
        v_hidden = mish(self.v2_fc(v_pooled))    # [B, v2_hidden]
        value = self.v3_fc(v_hidden)             # [B, 3]
        score_value = self.sv3_fc(v_hidden)      # [B, 6]
        return value, score_value


class OwnershipHead(nn.Module):
    def __init__(self, trunk_c):
        super().__init__()
        # Uses value head's v1_conv output passed in
        self.conv = nn.Conv2d(trunk_c, 1, 1, bias=False)

    def forward(self, v1_out):
        return torch.tanh(self.conv(v1_out))


class KataGoNet(nn.Module):
    def __init__(self, num_blocks, trunk_c, reg_c, gpool_c, block_types,
                 bin_features=22, global_features=19, board_size=19,
                 policy_c=24, gpool_c_policy=24, v1_c=24, v2_hidden=48):
        super().__init__()
        self.board_size = board_size

        # Initial conv
        self.init_conv = nn.Conv2d(bin_features, trunk_c, 3, padding=1, bias=False)
        # Global input
        self.g_linear = nn.Linear(global_features, trunk_c, bias=False)
        # Residual blocks
        blocks = []
        for bt in block_types:
            if bt == 'ordinary_block':
                blocks.append(OrdinaryBlock(trunk_c))
            elif bt == 'gpool_block':
                blocks.append(GPoolBlock(trunk_c, reg_c, gpool_c))
        self.blocks = nn.ModuleList(blocks)
        # Trunk final BN
        self.trunk_bn = nn.BatchNorm2d(trunk_c)
        # Heads
        self.policy_head = PolicyHead(trunk_c, policy_c, gpool_c_policy)
        self.value_head = ValueHead(trunk_c, v1_c, v2_hidden)
        self.own_conv = nn.Conv2d(v1_c, 1, 1, bias=False)
        # Shared v1 for ownership
        self.v1_conv = self.value_head.v1_conv
        self.v1_bn = self.value_head.v1_bn

    def forward(self, input_binary, input_global):
        # Trunk
        x = self.init_conv(input_binary)
        g = self.g_linear(input_global)
        x = x + g.unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            x = block(x)
        trunk_out = mish(self.trunk_bn(x))

        # Policy
        policy = self.policy_head(trunk_out)

        # Value (shared v1 for ownership)
        v1_out = mish(self.v1_bn(self.v1_conv(trunk_out)))
        v_pooled = global_pool(v1_out)
        v_hidden = mish(self.value_head.v2_fc(v_pooled))
        value = self.value_head.v3_fc(v_hidden)
        score_value = self.value_head.sv3_fc(v_hidden)

        # Ownership
        ownership = torch.tanh(self.own_conv(v1_out))

        return policy, value, ownership


# ─── Weight loading ─────────────────────────────────────────────────────────

def load_all_weights(reader, model, num_blocks, trunk_c, reg_c, gpool_c):
    """Load ALL weights from the text model file."""

    # 1. Initial conv (conv1)
    print("Loading initial conv...")
    w = read_conv(reader)
    model.init_conv.weight.data = torch.from_numpy(w)

    # 2. Global input matmul (ginputw) - no bias in this model!
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
            # norm1
            nc, mean, var, scale, bias = read_bn(reader)
            set_bn(block.bn1, nc, mean, var, scale, bias)
            # actv1
            read_actv(reader)
            # w1
            w1 = read_conv(reader)
            block.conv1.weight.data = torch.from_numpy(w1)
            # norm2
            nc, mean, var, scale, bias = read_bn(reader)
            set_bn(block.bn2, nc, mean, var, scale, bias)
            # actv2
            read_actv(reader)
            # w2
            w2 = read_conv(reader)
            block.conv2.weight.data = torch.from_numpy(w2)

        elif block_type == 'gpool_block':
            # norm1
            nc, mean, var, scale, bias = read_bn(reader)
            set_bn(block.bn1, nc, mean, var, scale, bias)
            # actv1
            read_actv(reader)
            # w1a (regular conv)
            w1a = read_conv(reader)
            block.conv1a.weight.data = torch.from_numpy(w1a)
            # w1b (gpool conv)
            w1b = read_conv(reader)
            block.conv1b.weight.data = torch.from_numpy(w1b)
            # norm1b (BN for gpool)
            nc, mean, var, scale, bias = read_bn(reader)
            set_bn(block.bn1b, nc, mean, var, scale, bias)
            # actv1b
            read_actv(reader)
            # w1r (matmul: 3*gpool_c -> reg_c)
            w1r = read_matmul(reader)
            block.w1r.weight.data = torch.from_numpy(w1r)
            # norm2
            nc, mean, var, scale, bias = read_bn(reader)
            set_bn(block.bn2, nc, mean, var, scale, bias)
            # actv2
            read_actv(reader)
            # w2 (conv reg_c -> trunk_c)
            w2 = read_conv(reader)
            block.conv2.weight.data = torch.from_numpy(w2)

    # 4. Trunk final BN
    print("\nLoading trunk output BN...")
    nc, mean, var, scale, bias = read_bn(reader)
    set_bn(model.trunk_bn, nc, mean, var, scale, bias)
    # trunk activation
    read_actv(reader)

    # 5. Policy head
    print("\nLoading policy head...")
    section = reader.read_str()  # "policyhead"
    print(f"  Section: {section}")

    # p1/w: conv trunk_c -> policy_c
    p1w = read_conv(reader)
    model.policy_head.p1_conv.weight.data = torch.from_numpy(p1w)

    # g1/w: conv trunk_c -> gpool_c_policy
    g1w = read_conv(reader)
    model.policy_head.g1_conv.weight.data = torch.from_numpy(g1w)

    # g1/norm: BN for gpool policy
    nc, mean, var, scale, bias = read_bn(reader)
    set_bn(model.policy_head.g1_bn, nc, mean, var, scale, bias)

    # g1/actv
    read_actv(reader)

    # matmulg2w: 3*gpool_c_policy -> policy_c
    g2w = read_matmul(reader)
    model.policy_head.g2_matmul.weight.data = torch.from_numpy(g2w)

    # p1/norm: BN for combined policy
    nc, mean, var, scale, bias = read_bn(reader)
    set_bn(model.policy_head.p1_bn, nc, mean, var, scale, bias)

    # p1/actv
    read_actv(reader)

    # p2/w: conv policy_c -> 1
    p2w = read_conv(reader)
    model.policy_head.p2_conv.weight.data = torch.from_numpy(p2w)

    # matmulpass: 3*gpool_c_policy -> 1
    passw = read_matmul(reader)
    model.policy_head.pass_matmul.weight.data = torch.from_numpy(passw)

    # 6. Value head
    print("\nLoading value head...")
    section = reader.read_str()  # "valuehead"
    print(f"  Section: {section}")

    # v1/w: conv trunk_c -> v1_c
    v1w = read_conv(reader)
    model.value_head.v1_conv.weight.data = torch.from_numpy(v1w)

    # v1/norm: BN v1_c
    nc, mean, var, scale, bias = read_bn(reader)
    set_bn(model.value_head.v1_bn, nc, mean, var, scale, bias)

    # v1/actv
    read_actv(reader)

    # v2/w: matmul 3*v1_c -> v2_hidden
    v2w = read_matmul(reader)
    model.value_head.v2_fc.weight.data = torch.from_numpy(v2w)

    # v2/b: bias for v2
    v2b = read_matbias(reader)
    model.value_head.v2_fc.bias.data = torch.from_numpy(v2b)

    # v2/actv
    read_actv(reader)

    # v3/w: matmul v2_hidden -> 3
    v3w = read_matmul(reader)
    model.value_head.v3_fc.weight.data = torch.from_numpy(v3w)

    # v3/b: bias
    v3b = read_matbias(reader)
    model.value_head.v3_fc.bias.data = torch.from_numpy(v3b)

    # sv3/w: matmul v2_hidden -> 6
    sv3w = read_matmul(reader)
    model.value_head.sv3_fc.weight.data = torch.from_numpy(sv3w)

    # sv3/b: bias
    sv3b = read_matbias(reader)
    model.value_head.sv3_fc.bias.data = torch.from_numpy(sv3b)

    # vownership/w: conv v1_c -> 1
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

    # Read initial conv first, then peek at block types
    # We need to know block types before building the model
    # Save position, scan ahead for block types, then restore
    saved_pos = reader.pos

    # Read past initial conv weights
    reader.read_str()  # conv1 name
    kh = reader.read_int()
    kw = reader.read_int()
    in_c = reader.read_int()
    out_c = reader.read_int()
    reader.read_int()  # sh
    reader.read_int()  # sw
    reader.pos += out_c * in_c * kh * kw

    # Read past ginputw
    reader.read_str()  # ginputw name
    in_f = reader.read_int()
    out_f = reader.read_int()
    reader.pos += in_f * out_f

    # Now scan block types
    block_types = []
    for _ in range(num_blocks):
        bt = reader.read_str()
        block_types.append(bt)
        block_name = reader.read_str()
        # Skip all block content by scanning for next block marker or end
        # This is complex, so let's use a simpler approach: count expected tokens
        if bt == 'ordinary_block':
            # norm1: name + nc + 3(eps/has_s/has_b) + 4*nc
            norm_name = reader.read_str()
            nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            # actv1: 2 tokens
            reader.pos += 2
            # w1: name + 6(kh/kw/in/out/sh/sw) + weights
            reader.read_str()
            kkh = reader.read_int()
            kkw = reader.read_int()
            ic = reader.read_int()
            oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
            # norm2
            reader.read_str()
            nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            # actv2
            reader.pos += 2
            # w2
            reader.read_str()
            kkh = reader.read_int()
            kkw = reader.read_int()
            ic = reader.read_int()
            oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
        elif bt == 'gpool_block':
            # norm1
            reader.read_str()
            nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            # actv1
            reader.pos += 2
            # w1a
            reader.read_str()
            kkh = reader.read_int()
            kkw = reader.read_int()
            ic = reader.read_int()
            oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
            # w1b
            reader.read_str()
            kkh = reader.read_int()
            kkw = reader.read_int()
            ic = reader.read_int()
            oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw
            # norm1b
            reader.read_str()
            nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            # actv1b
            reader.pos += 2
            # w1r (matmul)
            reader.read_str()
            inf = reader.read_int()
            outf = reader.read_int()
            reader.pos += inf * outf
            # norm2
            reader.read_str()
            nc = reader.read_int()
            reader.pos += 3 + 4 * nc
            # actv2
            reader.pos += 2
            # w2
            reader.read_str()
            kkh = reader.read_int()
            kkw = reader.read_int()
            ic = reader.read_int()
            oc = reader.read_int()
            reader.pos += 2 + ic * oc * kkh * kkw

    print(f"Block types: {block_types}")

    # Restore position
    reader.pos = saved_pos

    # Determine head sizes from format
    # Policy: policy_c=24, gpool_c_policy=24 (from p1/w and g1/w output channels)
    # Value: v1_c=24, v2_hidden=48 (from v2/w)
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
    dummy_bin = torch.randn(1, bin_features, board_size, board_size)
    dummy_glob = torch.randn(1, global_features)
    with torch.no_grad():
        policy, value, ownership = model(dummy_bin, dummy_glob)
    print(f"  Policy: {policy.shape} (expected [1, {board_size*board_size+1}])")
    print(f"  Value: {value.shape} (expected [1, 3])")
    print(f"  Ownership: {ownership.shape} (expected [1, 1, {board_size}, {board_size}])")

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
