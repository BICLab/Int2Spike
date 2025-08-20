import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.ticker import MaxNLocator
class SpikeCountBaseLIFNode(nn.Module):
    def __init__(self,):
        super().__init__()
        self.T = None # Number of timesteps (int)
        self.x_remain = None 
        self.spike_seq = None # Generated spike sequence, shape: (T, *input_shape)
        self.is_bidirectional = False # True if coding is bidirectional (-1/0/1)
        self.is_bitwise_coding = False # True if coding is bitwise representation
        
    def forward(self,):
        """
        Forward pass placeholder for subclasses.

        Args:
            x (torch.Tensor): Input spike count tensor.
            T (int | None): Optional number of timesteps. If None, determine from x.

        Returns:
            torch.Tensor: Spike sequence of shape [T, *x.shape].
        """
        raise NotImplementedError
    
    def neuronal_charge(self,):
        raise NotImplementedError

    def neuronal_fire(self,):
        raise NotImplementedError

    def neuronal_reset(self,):
        raise NotImplementedError

    def firing_rate(self,):
        """
        Calculate the average firing rate of the spike sequence.

        For bidirectional coding: use absolute spike values.
        For unidirectional coding: use raw spike values.

        Returns:
            torch.Tensor: scalar firing rate = total_spikes / (num_elements * T)

        Raises:
            ValueError: if spike_seq or T is not set before calling.
        """
        if self.spike_seq is None:
            raise ValueError("spike_seq is None. Run forward() before calling firing_rate().")
        if self.T is None:
            raise ValueError("T is None. Set self.T before calling firing_rate().")
     
        if self.is_bidirectional:
            return self.spike_seq.abs().sum()/ self.spike_seq.numel() 
        else:
            return self.spike_seq.sum()/ self.spike_seq.numel()     
    
    def visualize_spike(self, max_neurons=30, max_token=20, filename="sample.png", title="", seed=42):
        plt.rcParams.update({'font.size': 10}) 
        if seed is not None:
            random.seed(seed)

        if self.spike_seq is None:
            raise ValueError("spike_seq is None. Run forward() before visualization.")
        if self.spike_seq.dim() != 3:
            raise ValueError("Expected spike_seq of shape [T, N, D]")

        T, N, D = self.spike_seq.shape  # Time, Tokens, Neurons
        seq = self.spike_seq  # [T, N, D]
        seq = seq.permute(1, 0, 2)  # [N, T, D]
        seq = seq.reshape(N * T, D)  # Flatten to [time_steps, neurons]

        total_neurons = D
        sampled = random.sample(range(total_neurons), min(max_neurons, total_neurons))

        max_time = min(N * T, max_token * T)
        plt.figure(figsize=(6, 3))

        for idx, i in enumerate(sampled):
            spikes = (seq[:max_time, i].abs() > 0).nonzero(as_tuple=True)[0]
            plt.scatter(spikes.cpu().numpy(), [idx] * len(spikes), s=2, marker='|', color='black')

        plt.xlabel("Time")
        plt.ylabel("Neuron")
        plt.title(title, fontsize=10)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        plt.tight_layout()
        plt.savefig(f"png/{filename}")
    

class SpikeCountBinaryLIFNode(SpikeCountBaseLIFNode):
    """
    Spike count to binary spike sequence (0/1) over T timesteps.
    Emits 1 each step until count reaches zero.
    Input x must be a non-negative integer tensor representing spike counts (spike counts ≥ 0).
    """
    def __init__(self,):
        super().__init__()
        self.T = None
        self.x_remain = None
        self.spike_seq = None
        self.is_bidirectional = False   
        self.is_bitwise_coding = False 
    
    def forward(self, x: torch.Tensor, T: int | None = None):
        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must be integer-valued (all elements must be whole numbers).")
        
        if not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative (no values below 0).")

        if T is not None:
            if not isinstance(T, int) or T < 0:
                raise ValueError("T must be a non-negative integer.")
            self.T = T

        if self.T is None:
            self.T = int(x.max().item())

        self.neuronal_charge(x)

        self.spike_seq = torch.zeros((self.T,) + x.shape, dtype=torch.float32, device=x.device)

        for t in range(self.T):
            self.spike_seq[t] = self.neuronal_fire()
            self.neuronal_reset(self.spike_seq[t])

        return self.spike_seq # shape: [T, *x.shape]
    
    def neuronal_charge(self, x: torch.Tensor):
        self.x_remain = x

    def neuronal_fire(self,):
        # Emit 1 if there is remaining count, else 0
        return (self.x_remain > 0).to(torch.float32)
    
    def neuronal_reset(self, spike):
        self.x_remain = self.x_remain - spike

class SpikeCountTernaryLIFNode(SpikeCountBaseLIFNode):
    """
    Spike count to ternary spike sequence (-1/0/1) over T timesteps.
    Positive count emits +1, negative count emits -1.
    Input x must be an integer tensor representing spike counts, and may contain both positive and negative values.
    """
    def __init__(self,):
        super().__init__()
        self.T = None
        self.x_remain = None
        self.spike_seq = None
        self.is_bidirectional = True   
        self.is_bitwise_coding = False 

    def forward(self, x: torch.Tensor, T: int | None = None):
        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must be integer-valued (all elements must be whole numbers).")

        if T is not None:
            if not isinstance(T, int) or T < 0:
                raise ValueError("T must be a non-negative integer.")
            self.T = T

        if self.T is None:
            self.T = int(torch.abs(x).max().item())
         
        self.neuronal_charge(x)

        self.spike_seq = torch.zeros((self.T,) + x.shape, dtype=torch.float32, device=x.device)
        
        for t in range(self.T):
            self.spike_seq[t] = self.neuronal_fire()
            self.neuronal_reset(self.spike_seq[t])

        return self.spike_seq # shape: [T, *x.shape]
    
    def neuronal_charge(self, x: torch.Tensor):
        self.x_remain = x

    def neuronal_fire(self,):
        return torch.sign(self.x_remain) 
        
    def neuronal_reset(self, spike):
        self.x_remain = self.x_remain - spike

class SpikeCountBitwiseNode(SpikeCountBaseLIFNode):
    """
    Spike count to bitwise-coded spike sequence.
    Each timestep emits one bit from the binary representation of the count.
    Input x must be a non-negative integer tensor representing spike counts (spike counts ≥ 0).
    """
    def __init__(self,):
        super().__init__()
        self.T = None
        self.x_remain = None
        self.spike_seq = None
        self._bit_idx = 0 # Tracks current bit index
        self.is_bidirectional = False   
        self.is_bitwise_coding = True 

    def forward(self, x: torch.Tensor, T: int | None = None):
        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must be integer-valued (all elements must be whole numbers).")
        
        if not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative (no values below 0).")

        if T is not None:
            if not isinstance(T, int) or T < 0:
                raise ValueError("T must be a non-negative integer.")
            self.T = T

        if self.T is None:
            x_max = int(x.max().item())
            self.T = max(1, math.ceil(math.log2(x_max + 1)))

        self.neuronal_charge(x)

        self.spike_seq = torch.zeros((self.T,) + x.shape, dtype=torch.float32, device=x.device)

        for t in range(self.T):
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            self.spike_seq[t] = spike

        return self.spike_seq # shape: [T, *x.shape]

    def neuronal_charge(self, x: torch.Tensor):
        self.x_remain = x.to(torch.long)
        self._bit_idx = 0  

    def neuronal_fire(self):
        # # Mask selects the current bit (starting from MSB) 2^(T-1-bit_idx)
        mask = 1 << (self.T - 1 - self._bit_idx)
        return ((self.x_remain & mask) != 0).to(torch.float32)

    def neuronal_reset(self, spike):
        self._bit_idx += 1

def spike_matmul(x: torch.Tensor, w: torch.Tensor, x_zero: torch.Tensor = None,  lif_quantizer = None,
                 w_zero: torch.Tensor = None, weight_bitwise: bool = False) -> torch.Tensor:
    """
    Perform matrix multiplication for spike-based inputs with optional spike sequence
    conversion (LIF nodes) and optional bitwise processing.

    Processing steps:
        1. Ensure `x` is composed of integer-valued spike counts.
        2. Optionally process spike counts using `lif_quantizer` (must be a subclass of SpikeCountBaseLIFNode).
        - If bidirectional and x >= 0: split positive counts into ± values (requires `x_zero`).
        - If non-bidirectional and x contains negatives: shift counts into non-negative range (requires `x_zero`).
        3. Optionally convert weights into bitwise format if `weight_bitwise` is True.
        4. Perform matrix multiplication: `y = x @ w`.
        5. If `is_bitwise_coding` is True:
            Interpret time dimension (T) as bit positions (MSB to LSB) and compute weighted sum.
        Else:
            Sum outputs over time dimension.

    Args:
        x (torch.Tensor): Input spike count tensor (must be integer-valued).
        w (torch.Tensor): Weight matrix (can be float or integer, depending on bitwise mode).
        x_zero (torch.Tensor, optional): Zero-point tensor for x, used in spike decomposition.
        lif_quantizer (SpikeCountBaseLIFNode, optional): Converts spike counts to spike sequences.
        w_zero (torch.Tensor, optional): Zero-point tensor for weights, used in bitwise quantization.
        weight_bitwise (bool): Whether to treat weights as bitwise-encoded values.

    Returns:
        torch.Tensor: Output after spike processing and matrix multiplication.

    Raises:
        ValueError: If `x` is not integer-valued.
        TypeError: If `lif_quantizer` is not a SpikeCountBaseLIFNode instance.
        ValueError: If `x_zero` is required but not provided.
    """
    if not torch.allclose(x, x.round()):
        raise ValueError(
        "Input x must contain only integer-valued elements when act_quantizer is None."
    )

    if lif_quantizer is not None:
        if not isinstance(lif_quantizer, SpikeCountBaseLIFNode):
            raise TypeError(
                f"lif_quantizer must be an instance of SpikeCountBaseLIFNode or its subclasses, "
                f"but got {type(lif_quantizer).__name__}"
            )
        
        if lif_quantizer.is_bidirectional and torch.all(x >= 0):
            if x_zero is None:
                raise ValueError(
                    "x_zero is required when lif_quantizer is bidirectional and x is non-negative."
                )
            if not isinstance(x_zero, torch.Tensor):
                raise TypeError(f"x_zero must be a torch.Tensor, but got {type(x_zero).__name__}")
            
            half_up = torch.ceil(x.to(torch.float32) / 2.0).to(x.dtype)
            x = x - half_up
            x_zero.sub_(half_up.to(x_zero.dtype))

        elif not lif_quantizer.is_bidirectional and torch.any(x < 0):
            if x_zero is None:
                raise ValueError(
                    "x_zero is required when lif_quantizer is non-bidirectional and x contains negative values."
                )
            if not isinstance(x_zero, torch.Tensor):
                raise TypeError(f"x_zero must be a torch.Tensor, but got {type(x_zero).__name__}")
            
            x_min = x.min()
            x = x - x_min
            x_zero.sub_(x_min.to(x_zero.dtype))

        x = lif_quantizer(x)
        
    if weight_bitwise:
        if not torch.allclose(w, w.round()):
            raise ValueError("w must be integer-valued.")
        w = weight_to_bitwise(w, w_zero)

        y = torch.stack([x @ t for t in w], dim=0)
    else:
        y = x @ w

    x_is_bitwise = bool(getattr(lif_quantizer, "is_bitwise_coding", False))
    w_is_bitwise = bool(weight_bitwise) 

    if x_is_bitwise and w_is_bitwise:
        Tx, Tw = x.shape[0], w.shape[0]
        pow2x = (2 ** torch.arange(Tx - 1, -1, -1, device=y.device, dtype=y.dtype))  
        pow2w = (2 ** torch.arange(Tw - 1, -1, -1, device=y.device, dtype=y.dtype)) 
 
        wx = pow2x.view(1, Tx, *([1] * (y.dim() - 2)))
        ww = pow2w.view(Tw, 1, *([1] * (y.dim() - 2)))
        weights = wx * ww
        out = (y * weights).sum(dim=(0, 1))

    elif w_is_bitwise:
        Tw = w.shape[0]
        pow2w = (2 ** torch.arange(Tw - 1, -1, -1, device=y.device, dtype=y.dtype))  
        weights = pow2w.view(Tw, *([1] * (y.dim() - 1)))  
        out = (y * weights).sum(dim=(0, 1))
  
    elif x_is_bitwise:
        Tx = x.shape[0]
        pow2x = (2 ** torch.arange(Tx - 1, -1, -1, device=y.device, dtype=y.dtype))  
        weights = pow2x.view(Tx, *([1] * (y.dim() - 1)))  
        out = (y * weights).sum(dim=0)

    else:
        out = y.sum(dim=0) if (lif_quantizer is not None) else y  

    return out

def weight_to_bitwise(w: torch.Tensor, w_zero: torch.Tensor):
    """
    Convert integer weight tensor into its bitwise representation.

    - Ensures `w` is integer-valued.
    - If `w` contains negative values, shifts it into a non-negative range (adjusting `w_zero` accordingly).
    - Computes the minimum number of bits T required to represent the largest value in `w`.
    - Returns a binary sequence tensor of shape [T, *w.shape], where each slice along dim 0 is one bit plane.

    Args:
        w (torch.Tensor): Integer weight tensor.
        w_zero (torch.Tensor): Zero-point tensor (must match `w` shape).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            bit_seq: Tensor of shape [T, *w.shape], containing 0/1 bit planes.
            w_zero: Adjusted zero-point tensor.
    """
    if not torch.allclose(w, w.round()):
        raise ValueError("w must be integer-valued.")
    if not isinstance(w_zero, torch.Tensor):
        raise TypeError(f"w_zero must be a torch.Tensor, but got {type(w_zero).__name__}")

    if torch.any(w < 0):
        w_min = w.min()
        w = w - w_min
        w_zero.sub_(w_min.to(w_zero.dtype))

    w_max = int(w.max().item())
    T = max(1, math.ceil(math.log2(w_max + 1)))

    masks = 2 ** torch.arange(T - 1, -1, -1, device=w.device)
    bit_seq = ((w.unsqueeze(0).long() & masks.view(-1, *[1]*w.dim())) != 0).float()

    return bit_seq

def test_spike_count_binary_lif_node(x: torch.Tensor = None, high: int = 15, size: tuple = (12, 1024, 2048)) -> bool:
    """
    Test function for SpikeCountBinaryLIFNode.

    Args:
        x (torch.Tensor, optional): Input tensor with non-negative integer values.
                                    If None, a random tensor will be generated.
        high (int): Upper bound for random generation (exclusive).
        size (tuple): Shape of randomly generated tensor.

    Returns:
        bool: True if the spike sequence sums match the original input; False otherwise.
    """
    if x is None:
        x = torch.randint(low=0, high=high+1, size=size)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        if not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative")

        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must contain integer values only")

    lif = SpikeCountBinaryLIFNode()
    spike = lif(x.to(torch.float32)) 
    spike_sum = spike.sum(dim=0)

    match = torch.allclose(x.to(dtype=spike_sum.dtype), spike_sum, rtol=1e-3, atol=1e-3)
    
    if match:
        print(f"\u2714 Numerical match: {lif.__class__.__name__} spike sum matches input spike counts. x.shape = {x.shape}, spike.shape = {spike.shape}")
    else:
        print(f"\u2718 Mismatch: {lif.__class__.__name__} spike sum does not match input spike counts.")

    return match

def test_spike_count_ternary_lif_node(x: torch.Tensor = None, low: int = -8, high: int = 7, size: tuple = (12, 1024, 2048)
) -> bool:
    """
    Test function for SpikeCountTernaryLIFNode.

    Args:
        x (torch.Tensor, optional): Input tensor with integer values.
                                    If None, a random tensor will be generated.
        low (int): Lower bound for random input generation (inclusive).
        high (int): Upper bound for random input generation (exclusive).
        size (tuple): Shape of the input tensor.

    Returns:
        bool: True if spike sequence sums match the original input.
    """
    if x is None:
        x = torch.randint(low=low, high=high+1, size=size)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must contain integer values only")

    lif = SpikeCountTernaryLIFNode()
    spike = lif(x.to(torch.float32))
    spike_sum = spike.sum(dim=0)

    match = torch.allclose(x.to(dtype=spike_sum.dtype), spike_sum, rtol=1e-3, atol=1e-3)

    if match:
        print(f"\u2714 Numerical match: {lif.__class__.__name__} spike sum matches input spike counts. x.shape = {x.shape}, spike.shape = {spike.shape}")
    else:
        print(f"\u2718 Mismatch: {lif.__class__.__name__} spike sum does not match input spike counts.")

    return match

def test_spike_count_bitwise_node(x: torch.Tensor = None, high: int = 15, size: tuple = (12, 1024, 2048)
) -> bool:
    """
    Test function for SpikeCountBitwiseNode.

    Args:
        x (torch.Tensor, optional): Input tensor with non-negative integer values.
                                    If None, a random tensor will be generated.
        high (int): Upper bound (exclusive) for random input generation.
        size (tuple): Shape of the input tensor.

    Returns:
        bool: True if the reconstructed spike value equals the input spike count.
    """
    if x is None:
        x = torch.randint(low=0, high=high+1, size=size)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        if not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative")

        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must contain integer values only")

    lif = SpikeCountBitwiseNode()
    spike = lif(x.to(torch.float32))

    T = spike.shape[0]
    pow2 = 2 ** torch.arange(T - 1, -1, -1, device=spike.device, dtype=spike.dtype)
    weights = pow2.view(T, *([1] * (spike.dim() - 1)))
    spike_sum = (spike * weights).sum(dim=0)

    match = torch.allclose(x.to(dtype=spike_sum.dtype), spike_sum, rtol=1e-3, atol=1e-3)

    if match:
        print(f"\u2714 Numerical match: {lif.__class__.__name__} spike sum matches input spike counts. x.shape = {x.shape}, spike.shape = {spike.shape}")
    else:
        print(f"\u2718 Mismatch: {lif.__class__.__name__} spike sum does not match input spike counts.")

    return match


def test_spike_matmul_equivalence(x: torch.Tensor = None, x_low: int = 0, x_high: int = 15, x_size: tuple = (12, 1024, 2048), 
                                  w: torch.Tensor = None, w_size: tuple = (2048, 2048), 
                                  lif_quantizer: SpikeCountBaseLIFNode = None, weight_bitwise: bool = False
                                   ) -> bool:
    """
    General test for verifying spike_matmul matches dense matmul output
    when using a given SpikeCount LIF quantizer.

    Args:
        x (torch.Tensor, optional): Integer input tensor. Randomly generated if None.
        w (torch.Tensor, optional): Weight tensor. Randomly generated if None.
        lif_quantizer (SpikeCountBaseLIFNode): Quantizer to convert spike count to spike sequences.
        high (int): Max value (inclusive) for input generation.
        size (tuple): (B, K, N), where x.shape=(B,K), w.shape=(K,N)
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:    
        bool: True if spike_matmul output is close to x @ w.
    """
    if x is None:
        x = torch.randint(low=x_low, high=x_high + 1, size=x_size).float()
        x_zero = torch.zeros_like(x)
    else:
        if not torch.allclose(x, x.round()):
            raise ValueError("x must be integer-valued.")
        x = x.to(torch.float32)

    if w is None:
        w = torch.randint(low=-8, high=7, size=w_size).float()
        w_zero = torch.zeros_like(w)

    if lif_quantizer is None:
        raise ValueError("lif_quantizer must be provided (e.g., SpikeCountBinaryLIFNode()).")

    y_spike = spike_matmul(x, w, x_zero=x_zero, lif_quantizer=lif_quantizer, w_zero=w_zero, weight_bitwise=weight_bitwise)
    y_ref = (x + x_zero) @ (w + w_zero)

    match = torch.allclose(y_ref, y_spike, rtol=1e-3, atol=1e-3)
  
    if match:
        if isinstance(lif_quantizer, SpikeCountBinaryLIFNode):
            print(f"\u2714 Numerical match: Binary spike sequence (0/1) matmul with {lif_quantizer.__class__.__name__} matches spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountTernaryLIFNode):
            print(f"\u2714 Numerical match: Ternary spike sequence (-1/0/1) matmul with {lif_quantizer.__class__.__name__} matches spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountBitwiseNode):
            print(f"\u2714 Numerical match: Bitwise spike sequence matmul with {lif_quantizer.__class__.__name__} matches spike count matmul.")
    else:
        if isinstance(lif_quantizer, SpikeCountBinaryLIFNode):
            print(f"\u2718 Mismatch: Binary spike sequence (0/1) matmul with {lif_quantizer.__class__.__name__} does not match spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountTernaryLIFNode):
            print(f"\u2718 Mismatch: Ternary spike sequence (-1/0/1) matmul with {lif_quantizer.__class__.__name__} does not match spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountBitwiseNode):
            print(f"\u2718 Mismatch: Bitwise spike sequence matmul with {lif_quantizer.__class__.__name__} does not match spike count matmul.")

    return match

def test_weight_to_bitwise(w: torch.Tensor = None, low: int = -8, high: int = 7, size: tuple = (1024, 1024)) -> bool:
    w = torch.randint(low=low, high=high, size=size).float()
    w_zero = torch.zeros_like(w)
    bit_seq= weight_to_bitwise(w, w_zero)

    T = bit_seq.shape[0]
    powers = 2 ** torch.arange(T - 1, -1, -1).view(T, *[1] * (bit_seq.dim() - 1))
    recovered = (bit_seq * powers).sum(dim=0) 

    w = w + w_zero

    match = torch.allclose(recovered, w, rtol=1e-3, atol=1e-3)

    if match:
        print(f"\u2714 Passed: Recovered weights match original weights. shape={w.shape}")
    else:
        print(f"\u2718 Failed: Recovered weights do not match original. shape={w.shape}")

  