import torch
from torch import nn, Tensor

from prototype_dataset_helper import unpack_embedding


class CreativityLoss(nn.Module):
    def __init__(self, weight_deep_svg=1, weight_type=0.1, weight_parameters=20, weight_eos=0.2):
        super(CreativityLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, target_padding_mask: Tensor, device="cpu"):
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

        # Slice embedding into sections
        input_deep_svg, input_type, input_parameters, input_eos = unpack_embedding(input, dim=2, device=device)
        target_deep_svg, target_type, target_parameters, target_eos = unpack_embedding(target, dim=2, device=device)

        input_type = torch.softmax(input_type, dim=2)
        input_eos = torch.softmax(input_eos, dim=2)

        return {
            "batch_variance":
                calc_variance(input).mean() / calc_variance(target).mean(),
            "sequence_variance":
                avg_sequence_variance(input) / avg_sequence_variance(target),
            "batch_variance_deep_svg":
                calc_variance(input_deep_svg).mean() / calc_variance(target_deep_svg).mean(),
            "sequence_variance_deep_svg":
                avg_sequence_variance(input_deep_svg) / avg_sequence_variance(target_deep_svg),
            "batch_variance_type":
                calc_variance(input_type).mean() / calc_variance(target_type).mean(),
            "sequence_variance_type":
                avg_sequence_variance(input_type) / avg_sequence_variance(target_type),
            "batch_variance_parameters":
                calc_variance(input_parameters).mean() / calc_variance(target_parameters).mean(),
            "sequence_variance_parameters":
                avg_sequence_variance(input_parameters) / avg_sequence_variance(target_parameters),
            "batch_variance_eos":
                calc_variance(input_eos).mean() / calc_variance(target_eos).mean(),
            "sequence_variance_eos":
                avg_sequence_variance(input_eos) / avg_sequence_variance(target_eos)
        }


def calc_variance(tensor: Tensor) -> Tensor:
    """Calculate variance of tensor, ignoring NaN values."""
    # Mean of non-NaN values
    mean = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor).sum(dim=0) / (
        ~torch.isnan(tensor)).float().sum(dim=0)
    # Squared deviations
    deviations = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor - mean) ** 2
    # Variance of non-NaN values
    variance = deviations.sum(dim=0) / (~torch.isnan(deviations)).float().sum(dim=0)
    return variance


def avg_sequence_variance(tensor: Tensor) -> Tensor:
    """Calculate average sequence variance of tensor, ignoring NaN values."""
    sequence_variances = torch.stack([calc_variance(seq) for seq in tensor], dim=0)
    valid_seq_mask = ~torch.isnan(sequence_variances)
    return sequence_variances[valid_seq_mask].mean(dim=0)


def add_result_dicts(dictionary1: dict, dictionary2: dict):
    """Add two dictionaries"""
    for key, value in dictionary2.items():
        if key in dictionary1:
            dictionary1[key] += value
    return dictionary1


def print_dict(dictionary: dict):
    print(f"bth var: {dictionary['batch_variance']:.9f} / "
          f"seq var: {dictionary['sequence_variance']:.9f}")
    print(f"bth var: {dictionary['batch_variance_deep_svg']:.9f} / "
          f"seq var: {dictionary['sequence_variance_deep_svg']:.9f} (deep_svg)")
    print(f"bth var: {dictionary['batch_variance_type']:.9f} / "
          f"seq var: {dictionary['sequence_variance_type']:.9f} (type)")
    print(f"bth var: {dictionary['batch_variance_parameters']:.9f} / "
          f"seq var: {dictionary['sequence_variance_parameters']:.9f} (parameters)")
    print(f"bth var: {dictionary['batch_variance_eos']:.9f} / "
          f"seq var: {dictionary['sequence_variance_eos']:.9f} (eos)")
    return


def dict_list_to_list_dict(list_of_dicts: list) -> dict:
    """Converts a list of dictionaries to a dictionary of lists"""
    result_dict = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
    return result_dict
