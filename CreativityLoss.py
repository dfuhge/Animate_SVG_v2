import torch
from torch import nn, Tensor


class CreativityLoss(nn.Module):
    def __init__(self, weight_deep_svg=1, weight_type=0.1, weight_parameters=20, weight_eos=0.2):
        super(CreativityLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, target_padding_mask: Tensor, device="cpu") -> Tensor:
        """
        Args:
            input: tensor of shape [batch_size, sequence_length, num_features]
            target: unused
            target_padding_mask: tensor of shape [batch_size, sequence_length]
            device: cpu / gpu

        Returns: Combined loss
        """
        # Expand target_padding_mask to match the shape of 'input'
        expanded_mask = target_padding_mask.unsqueeze(-1).expand_as(input)

        # Set padding positions to NaN in 'input' for ignoring in calculations
        input[expanded_mask] = float('nan')
        target[expanded_mask] = float('nan')

        # Calculate variance across the batch (dim=0), ignoring padding
        variance_across_batch = calculate_variance(input).sum()
        variance_across_batch_tgt = calculate_variance(target).sum()

        # Calculate variance for each sequence individually, ignoring padding
        sequence_variances = torch.stack([calculate_variance(seq) for seq in input], dim=0)
        valid_seq_mask = ~torch.isnan(sequence_variances)
        average_variance_per_sequence = sequence_variances[valid_seq_mask].sum(dim=0)

        sequence_variances_tgt = torch.stack([calculate_variance(seq) for seq in target], dim=0)
        valid_seq_mask = ~torch.isnan(sequence_variances_tgt)
        average_variance_per_sequence_tgt = sequence_variances_tgt[valid_seq_mask].sum(dim=0)

        print(f"Batch: {variance_across_batch:.9f} / {variance_across_batch_tgt:.9f}"
              f"Sequence: {average_variance_per_sequence}/ {average_variance_per_sequence_tgt:.9f}")

        return torch.tensor([variance_across_batch / variance_across_batch_tgt,
                             average_variance_per_sequence / average_variance_per_sequence_tgt])


def calculate_variance(input: Tensor) -> Tensor:
    """Calculate variance of tensor, ignoring NaN values."""
    # Mean of non-NaN values
    mean = torch.where(torch.isnan(input), torch.zeros_like(input), input).sum(dim=0) / (
        ~torch.isnan(input)).float().sum(dim=0)
    # Squared deviations
    deviations = torch.where(torch.isnan(input), torch.zeros_like(input), input - mean) ** 2
    # Variance of non-NaN values
    variance = deviations.sum(dim=0) / (~torch.isnan(deviations)).float().sum(dim=0)
    return variance
