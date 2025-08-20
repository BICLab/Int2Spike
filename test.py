from neuron import(
    test_spike_count_binary_lif_node,
    test_spike_matmul_equivalence,
    test_spike_count_ternary_lif_node,
    test_spike_count_bitwise_node,
    SpikeCountBinaryLIFNode,
    SpikeCountTernaryLIFNode,
    SpikeCountBitwiseNode
)

# --- Binary LIF Node: Numerical match test for spike count decoding ---
test_spike_count_binary_lif_node(high=15, size=(12, 2048, 2048))    # 4-bit unsigned input (0~15)
test_spike_count_binary_lif_node(high=31, size=(12, 2048, 2048))    # 5-bit unsigned input (0~31)
test_spike_count_binary_lif_node(high=63, size=(12, 2048, 2048))    # 6-bit unsigned input (0~63)
test_spike_count_binary_lif_node(high=255, size=(12, 2048, 2048))   # 8-bit unsigned input (0~255)

# --- Binary LIF Node: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_low=0, x_high=15, lif_quantizer=SpikeCountBinaryLIFNode())     # 4-bit
test_spike_matmul_equivalence(x_low=0, x_high=31, lif_quantizer=SpikeCountBinaryLIFNode())     # 5-bit
test_spike_matmul_equivalence(x_low=0, x_high=63, lif_quantizer=SpikeCountBinaryLIFNode())     # 6-bit
test_spike_matmul_equivalence(x_low=0, x_high=255, lif_quantizer=SpikeCountBinaryLIFNode())    # 8-bit

# --- Binary LIF Node: Numerical match test for matmul (zero-point adjustment) ---
test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBinaryLIFNode())     # 4-bit signed
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBinaryLIFNode())   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBinaryLIFNode())   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBinaryLIFNode()) # 8-bit signed

# --- Ternary LIF Node: Numerical match test for spike count decoding ---
test_spike_count_ternary_lif_node(low=-8, high=7, size=(12, 2048, 2048))      # 4-bit signed input
test_spike_count_ternary_lif_node(low=-16, high=15, size=(12, 2048, 2048))    # 5-bit signed input
test_spike_count_ternary_lif_node(low=-32, high=31, size=(12, 2048, 2048))    # 6-bit signed input
test_spike_count_ternary_lif_node(low=-128, high=127, size=(12, 2048, 2048))  # 8-bit signed input

# --- Ternary LIF Node: Numerical match test for matmul (zero-point adjustment) ---
test_spike_matmul_equivalence(x_low=0, x_high=15, lif_quantizer=SpikeCountTernaryLIFNode())     # 4-bit unsigned
test_spike_matmul_equivalence(x_low=0, x_high=31, lif_quantizer=SpikeCountTernaryLIFNode())     # 5-bit unsigned
test_spike_matmul_equivalence(x_low=0, x_high=63, lif_quantizer=SpikeCountTernaryLIFNode())     # 6-bit unsigned
test_spike_matmul_equivalence(x_low=0, x_high=255, lif_quantizer=SpikeCountTernaryLIFNode())    # 8-bit unsigned

# --- Ternary LIF Node: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountTernaryLIFNode())     # 4-bit signed
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountTernaryLIFNode())   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountTernaryLIFNode())   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountTernaryLIFNode()) # 8-bit signed

# --- Bitwise LIF Node: Numerical match test for spike count decoding ---
test_spike_count_bitwise_node(high=15, size=(12, 2048, 2048))    # 4-bit unsigned input
test_spike_count_bitwise_node(high=31, size=(12, 2048, 2048))    # 5-bit unsigned input
test_spike_count_bitwise_node(high=63, size=(12, 2048, 2048))    # 6-bit unsigned input
test_spike_count_bitwise_node(high=255, size=(12, 2048, 2048))   # 8-bit unsigned input

# --- Bitwise LIF Node: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_low=0, x_high=15, lif_quantizer=SpikeCountBitwiseNode())     # 4-bit
test_spike_matmul_equivalence(x_low=0, x_high=31, lif_quantizer=SpikeCountBitwiseNode())     # 5-bit
test_spike_matmul_equivalence(x_low=0, x_high=63, lif_quantizer=SpikeCountBitwiseNode())     # 6-bit
test_spike_matmul_equivalence(x_low=0, x_high=255, lif_quantizer=SpikeCountBitwiseNode())    # 8-bit

# --- Bitwise LIF Node: Numerical match test for matmul (zero-point adjustment) ---
test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBitwiseNode())     # 4-bit signed
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBitwiseNode())   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBitwiseNode())   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBitwiseNode()) # 8-bit signed

# --- Bitwise LIF Node and Bitwise Weight: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_low=0, x_high=15, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 4-bit
test_spike_matmul_equivalence(x_low=0, x_high=31, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 5-bit
test_spike_matmul_equivalence(x_low=0, x_high=63, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 6-bit
test_spike_matmul_equivalence(x_low=0, x_high=255, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)    # 8-bit

# --- Bitwise LIF Node and Bitwise Weight: Numerical match test for matmul (zero-point adjustment) ---
test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 4-bit signed
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True) # 8-bit signed
