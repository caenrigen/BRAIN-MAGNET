"""
PyTorch Implementation of Multinomial Convolution from MuSeAM

This module provides a PyTorch implementation of the multinomial convolution layer
from the paper "Multinomial Convolutions for Joint Modeling of Sequence Motifs and Enhancer Activities"
(MuSeAM: https://github.com/minjunp/MuSeAM)

Key Features:
- MultinomialConv1d: Core layer that transforms standard convolution weights into
  multinomial distributions over nucleotides (biologically meaningful for DNA sequences)
- MuSeAMBlock: Complete processing block with dual-strand DNA processing
- Compatible with standard PyTorch training workflows
- Configurable alpha/beta parameters and background nucleotide frequencies

Example Usage:
    # Simple multinomial convolution
    conv = MultinomialConv1d(in_channels=4, out_channels=64, kernel_size=12)

    # Complete MuSeAM-style block
    model = MuSeAMBlock(out_channels=512, kernel_size=12)
    output = model(forward_sequences, reverse_sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultinomialConv1d(nn.Conv1d):
    """
    Multinomial Convolution Layer for DNA Sequence Analysis

    This layer implements the multinomial convolution from MuSeAM:
    "Multinomial Convolutions for Joint Modeling of Sequence Motifs and Enhancer Activities"

    The layer transforms standard convolution weights into multinomial distributions
    over nucleotides, which is more biologically meaningful for DNA sequence analysis.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha=120,
        beta=None,
        bkg_freq=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        """
        Args:
            in_channels (int): Number of input channels (typically 4 for DNA: A, T, G, C)
            out_channels (int): Number of output channels (filters)
            kernel_size (int): Size of the convolving kernel
            alpha (float): Temperature parameter for multinomial transformation (default: 120)
            beta (float): Scaling parameter, if None defaults to 1/(alpha+10) (default: None)
            bkg_freq (list): Background nucleotide frequencies [A, T, G, C],
                           if None defaults to [0.295, 0.205, 0.205, 0.295]
            stride (int): Stride of the convolution (default: 1)
            padding (int): Zero-padding added to both sides of the input (default: 0)
            dilation (int): Spacing between kernel elements (default: 1)
            groups (int): Number of blocked connections from input to output channels (default: 1)
            bias (bool): If True, adds a learnable bias to the output (default: True)
            padding_mode (str): Padding mode ('zeros', 'reflect', 'replicate', 'circular') (default: 'zeros')
            device: Device to place the layer on
            dtype: Data type for the layer
        """
        super(MultinomialConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        self.alpha = alpha
        self.beta = beta if beta is not None else 1.0 / (alpha + 10)

        # Default background frequencies for DNA nucleotides [A, T, G, C]
        if bkg_freq is None:
            bkg_freq = [0.295, 0.205, 0.205, 0.295]

        # Register background frequencies as buffer (not trainable parameter)
        self.register_buffer("bkg_freq", torch.tensor(bkg_freq, dtype=torch.float32))

        # Counter to track training iterations (similar to TF implementation)
        self.register_buffer("run_value", torch.tensor(1, dtype=torch.int32))

    def forward(self, x):
        """
        Forward pass with multinomial convolution transformation

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, sequence_length)
                       For DNA sequences: (batch_size, 4, sequence_length)

        Returns:
            Tensor: Output of multinomial convolution
        """
        # Apply multinomial transformation after a few initial iterations
        # This allows for initial training with standard convolution
        if self.run_value > 2:
            output = F.conv1d(
                x,
                self.calculate_multinomial_weight(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            # Use standard convolution for initial iterations
            output = F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        # Increment run counter
        self.run_value += 1
        return output

    def calculate_multinomial_weight(self):
        """
        Apply multinomial transformation to convolution weights

        This transforms standard convolution weights into multinomial distributions
        over nucleotides using the alpha/beta parameters and background frequencies.

        Returns:
            Tensor: Transformed weights with same shape as original
        """
        # Get weight tensor: shape (out_channels, in_channels, kernel_size)
        weight = self.weight

        # Reshape for processing: (out_channels, kernel_size, in_channels)
        weight_reshaped = weight.transpose(1, 2)

        # Apply alpha scaling
        scaled_weight = self.alpha * weight_reshaped

        # Compute stable softmax with temperature
        # (out_channels, kernel_size, 1)
        max_vals = torch.max(scaled_weight, dim=2, keepdim=True)[0]
        # Subtract max for numerical stability. This is mathematically valid because:
        # exp(x_i-max)/Σ exp(x_j-max) = exp(x_i)/exp(max)/Σ exp(x_j)/exp(max)
        #                             = exp(x_i)/Σ exp(x_j)
        stable_scaled_weight = scaled_weight - max_vals

        # Compute log-sum-exp for normalization
        exp_weight = torch.exp(stable_scaled_weight)
        # (out_channels, kernel_size, 1)
        sum_exp = torch.sum(exp_weight, dim=2, keepdim=True)
        log_sum_exp = torch.log(sum_exp)

        # Create background frequency tensor with proper shape
        # Shape: (out_channels, kernel_size, in_channels)
        bkg_expanded = (
            self.bkg_freq.unsqueeze(0)
            .unsqueeze(0)
            .expand(weight_reshaped.shape[0], weight_reshaped.shape[1], -1)
        )
        log_bkg = torch.log(bkg_expanded)

        # Apply multinomial transformation
        # transformed = beta * (alpha * weight - max - log_sum_exp - log_bkg)
        transformed = self.beta * (stable_scaled_weight - log_sum_exp - log_bkg)

        # Reshape back to original weight shape: (out_channels, in_channels, kernel_size)
        transformed_weight = transformed.transpose(1, 2)

        return transformed_weight

    def extra_repr(self):
        """String representation of the layer"""
        base_repr = super().extra_repr()
        return f"{base_repr}, alpha={self.alpha}, beta={self.beta}, bkg_freq={self.bkg_freq}"


class MuSeAMBlock(nn.Module):
    """
    Complete MuSeAM-style block with forward and reverse complement processing

    This implements the dual-strand DNA processing approach from MuSeAM,
    where both forward and reverse complement sequences are processed and combined.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=512,
        kernel_size=12,
        alpha=120,
        beta=None,
        seq_length=1000,
    ):
        """
        Args:
            in_channels (int): Number of input channels (default: 4 for DNA)
            out_channels (int): Number of filters/output channels (default: 512)
            kernel_size (int): Convolution kernel size (default: 12)
            alpha (float): Temperature parameter (default: 120)
            beta (float): Scaling parameter (default: None, auto-computed)
            seq_length (int): Expected sequence length for the model (default: 1000)
        """
        super(MuSeAMBlock, self).__init__()

        # Shared multinomial convolution layer for both strands
        self.conv = MultinomialConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            alpha=alpha,
            beta=beta,
            bias=True,
        )

        # Activation and pooling layers
        self.activation = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Final dense layer for regression/classification
        self.dense = nn.Linear(out_channels, 1)

    def forward(self, forward_seq, reverse_seq):
        """
        Forward pass with dual-strand processing

        Args:
            forward_seq (Tensor): Forward strand sequence (batch_size, 4, seq_length)
            reverse_seq (Tensor): Reverse complement sequence (batch_size, 4, seq_length)

        Returns:
            Tensor: Final output predictions
        """
        # Apply convolution to both strands
        fw_conv = self.conv(forward_seq)  # (batch_size, out_channels, conv_length)
        rc_conv = self.conv(reverse_seq)  # (batch_size, out_channels, conv_length)

        # Concatenate along sequence dimension (as in original MuSeAM)
        # (batch_size, out_channels, 2*conv_length)
        concat = torch.cat([fw_conv, rc_conv], dim=2)

        # Apply activation
        activated = self.activation(concat)

        # Global max pooling
        # (batch_size, out_channels)
        pooled = self.global_max_pool(activated).squeeze(-1)

        # Final dense layer
        output = self.dense(pooled)  # (batch_size, 1)

        return output


# Example usage and testing
if __name__ == "__main__":
    # Test the multinomial convolution layer
    print("Testing MultinomialConv1d layer...")

    # Create sample DNA sequences (batch_size=2, channels=4, sequence_length=100)
    batch_size, seq_length = 2, 100
    forward_seq = torch.randn(batch_size, 4, seq_length)
    reverse_seq = torch.randn(batch_size, 4, seq_length)

    # Test individual multinomial convolution
    conv_layer = MultinomialConv1d(in_channels=4, out_channels=64, kernel_size=12)
    conv_output = conv_layer(forward_seq)
    print(f"Input shape: {forward_seq.shape}")
    print(f"Conv output shape: {conv_output.shape}")

    # Test complete MuSeAM block
    print("Testing complete MuSeAM block...")
    museam_block = MuSeAMBlock(out_channels=64, kernel_size=12, seq_length=seq_length)
    final_output = museam_block(forward_seq, reverse_seq)
    print(f"Final output shape: {final_output.shape}")
    print(f"Final output: {final_output}")

    print("Multinomial convolution implementation complete!")
