import torch
from neuron import (
    SpikeCountBinaryLIFNode,
    SpikeCountTernaryLIFNode,
    SpikeCountBitwiseNode
)
torch.manual_seed(42)
mean, std = 0, 1
size = (1024, 2048)
x = torch.normal(mean=mean, std=std, size=size)
xmin, xmax = x.min(), x.max()

# === Demo: Convert int4 tensor into 0/1 binary spike trains ===
qmin, qmax = 0, 15
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif1 = SpikeCountBinaryLIFNode()
spike1 = lif1(x_q1.to(torch.float32)) 
rate1 = lif1.firing_rate()
print(f"int4 to 0/1: Time steps = {spike1.shape[0]}, firing rate = {rate1}")
lif1.visualize_spike(filename="Binary_Lif_T15_N20.png", title="BinaryLif(TimeStep=15 Tokens=20)")

# === Demo: Convert int4 tensor into -1/0/1 ternary spike trains ===
qmin, qmax = -8, 7
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif2 = SpikeCountTernaryLIFNode()
spike2 = lif2(x_q2.to(torch.float32)) 
rate2 = lif2.firing_rate()
print(f"int4 to -1/0/1: Time steps = {spike2.shape[0]}, firing rate = {rate2}")
lif2.visualize_spike(filename="Ternary_Lif_T8_N20.png", title="Ternary_Lif(TimeStep=8 Tokens=20)")

# === Demo: Convert int4 tensor into bitwise spike trains ===
qmin, qmax = 0, 15
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif3 = SpikeCountBitwiseNode()
spike3 = lif3(x_q1.to(torch.float32)) 
rate3 = lif3.firing_rate()
print(f"int4 to Bitwise: Time steps = {spike3.shape[0]}, firing rate = {rate3}")
lif3.visualize_spike(filename="Bitwise_Lif_T4_N20.png", title="Bitwise_Lif(TimeStep=4 Tokens=20)")

# # === Demo: Convert int6 tensor into 0/1 binary spike trains ===
# qmin, qmax = 0, 63
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif1 = SpikeCountBinaryLIFNode()
# spike1 = lif1(x_q1.to(torch.float32)) 
# rate1 = lif1.firing_rate()
# print(f"int6 to 0/1: Time steps = {spike1.shape[0]}, firing rate = {rate1}")

# # === Demo: Convert int6 tensor into -1/0/1 ternary spike trains ===
# qmin, qmax = -32, 31
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif2 = SpikeCountTernaryLIFNode()
# spike2 = lif2(x_q2.to(torch.float32)) 
# rate2 = lif2.firing_rate()
# print(f"int6 to -1/0/1: Time steps = {spike2.shape[0]}, firing rate = {rate2}")

# # === Demo: Convert int6 tensor into bitwise spike trains ===
# qmin, qmax = 0, 63
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif3 = SpikeCountBitwiseNode()
# spike3 = lif3(x_q1.to(torch.float32)) 
# rate3 = lif3.firing_rate()
# print(f"int6 to Bitwise: Time steps = {spike3.shape[0]}, firing rate = {rate3}")

# # === Demo: Convert int8 tensor into 0/1 binary spike trains ===
# qmin, qmax = 0, 255
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif1 = SpikeCountBinaryLIFNode()
# spike1 = lif1(x_q1.to(torch.float32)) 
# rate1 = lif1.firing_rate()
# print(f"int8 to 0/1: Time steps = {spike1.shape[0]}, firing rate = {rate1}")

# # === Demo: Convert int8 tensor into -1/0/1 ternary spike trains ===
# qmin, qmax = -128, 127
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif2 = SpikeCountTernaryLIFNode()
# spike2 = lif2(x_q2.to(torch.float32)) 
# rate2 = lif2.firing_rate()
# print(f"int8 to -1/0/1: Time steps = {spike2.shape[0]}, firing rate = {rate2}")

# # === Demo: Convert int8 tensor into bitwise spike trains ===
# qmin, qmax = 0, 255
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif3 = SpikeCountBitwiseNode()
# spike3 = lif3(x_q1.to(torch.float32)) 
# rate3 = lif3.firing_rate()
# print(f"int8 to Bitwise: Time steps = {spike3.shape[0]}, firing rate = {rate3}")