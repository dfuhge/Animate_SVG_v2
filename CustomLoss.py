import torch
from torch import nn, Tensor

import dataset_helper
from dataset_helper import unpack_embedding

def _ignore_values_when_target_is_eos(input: Tensor, target: Tensor) -> Tensor:
    """
    Ignores parameters that are not related to EOS when EOS is in target sequence, to avoid a backpropagation to 0.
    Assumes that the last two features are EOS_NO and EOS_YES, one-hot encoded with all other elements in the target
    sequence set to 0. Then the target sequence equals the input sequence after the execution of this function.

    Args:
        input: tensor of shape [batch_size, sequence_length, num_features]
        target: tensor of shape [batch_size, sequence_length, num_features]
    """
    # Check where the last feature of the target is 1
    condition_mask = (target[..., -26] == 1).unsqueeze(-1).to(input.device)

    # Multiply all but the last two feature of input by 0/1 Mask
    input[..., :-26] *= condition_mask
    input[..., -16:] *= condition_mask
    return input

class CustomEmbeddingSliceLoss(nn.Module):


    def __init__(self, weight_deep_svg=100, weight_type=8, weight_parameters=1):
        super(CustomEmbeddingSliceLoss, self).__init__()
        # Loss functions
        self.loss_function_type = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_function_parameter = nn.MSELoss()
        self.loss_function_deep_svg = nn.MSELoss()
        # Weights
        self.weight_deep_svg = weight_deep_svg
        self.weight_type = weight_type
        self.weight_parameters = weight_parameters
        # Constants
        self.PADDING_VALUE = dataset_helper.PADDING_VALUE
        self.ANIMATION_PARAMETER_INDICES = dataset_helper.ANIMATION_PARAMETER_INDICES

    def forward(self, input: Tensor, target: Tensor, target_padding_mask: Tensor, device="cpu") -> Tensor:
        """

                Args:
                    input: tensor of shape [batch_size, sequence_length, num_features]
                    target: tensor of shape [batch_size, sequence_length, num_features]
                    target_padding_mask: tensor of shape [batch_size, sequence_length]
                    device: cpu / gpu

                Returns: Combined loss

                """
        
         # ignore part of input sequence, when target sequence is EOS
        input = _ignore_values_when_target_is_eos(input, target)

        # Expand target_padding_mask to match the shape of 'input' and set padding positions to -100 in 'input'
        expanded_mask = target_padding_mask.unsqueeze(-1).expand_as(input).to(device)
        input[expanded_mask] = -100

        # Slice embedding into sections
        input_deep_svg, input_type, input_parameters = unpack_embedding(input, dim=2, device=device)
        target_deep_svg, target_type, target_parameters = unpack_embedding(target, dim=2, device=device)

        # Flatten to 2D input: [batch_size * seq_length, num_classes]
        input_type_flat = input_type.view(-1, input_type.shape[-1]).to(device)

        # Create Padding mask to ignore padding
        type_padding_mask = (target_type == self.PADDING_VALUE).all(dim=-1).to(device)
        target_type = target_type.argmax(dim=-1).to(device)
        target_type[type_padding_mask] = -1
        # Flatten to 1D Targets: [batch_size * seq_length]
        target_type_flat = target_type.view(-1).to(device)
        loss_type = self.loss_function_type(input_type_flat, target_type_flat).to(device)

        # Ignore features by equalizing vectors on certain indices
        input_parameters = self._ignore_parameters(target_type, input_parameters, target_parameters, device=device)

        # Calculate loss
        loss_deep_svg = self.loss_function_deep_svg(input_deep_svg, target_deep_svg).to(device)
        loss_parameter = self.loss_function_parameter(input_parameters, target_parameters).to(device)

        # print(f"Loss: {self.weight_deep_svg * loss_deep_svg:.5f} === "
        #        f"{self.weight_type * loss_type:.5f} === "
        #        f"{self.weight_parameters * loss_parameter:.5f}")

        # Should roughly be balance 33% - 33% - 33%
        loss_overall = (self.weight_deep_svg * loss_deep_svg +
                        self.weight_type * loss_type +
                        self.weight_parameters * loss_parameter)

        return loss_overall

    def _ignore_parameters(self, target_type: Tensor, input_parameters: Tensor, target_parameters: Tensor, device):
        """

        Args:
            target_type: tensor of shape [batch_size, sequence_length] with types
            input_parameters: tensor [batch_size, sequence_length, num_parameters]
            target_parameters: tensor [batch_size, sequence_length, num_parameters] is ground truth
            device: cpu / gpu

        Returns: input_parameters, but features are overwritten depending on target_type

        """
        # Move tensors to correct device
        target_type = target_type.to(device)
        input_parameters = input_parameters.to(device)
        target_parameters = target_parameters.to(device)

        # Initialize a full overwrite mask to True (that means initially all values can be overwritten)
        overwrite_mask = torch.ones_like(input_parameters, dtype=torch.bool)

        # Iterate through each animation type
        for type_id, non_overwrite_indices in self.ANIMATION_PARAMETER_INDICES.items():
            # Find positions the animation type matches
            type_mask = (target_type == type_id).unsqueeze(-1)

            # Iterate through the parameter indices that should not be overwritten for this type
            for idx in non_overwrite_indices:
                # Use advanced indexing to update the overwrite mask for these indices to False
                overwrite_mask[:, :, idx] &= ~type_mask.squeeze(-1)

        # Apply the combined mask to selectively overwrite input_parameters with target_parameters
        input_parameters[overwrite_mask] = target_parameters[overwrite_mask]

        return input_parameters
