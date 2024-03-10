import torch
from torch import nn, Tensor

import prototype_dataset_helper as dataset_helper
from prototype_dataset_helper import unpack_embedding


class CustomEmbeddingSliceLoss(nn.Module):
    def __init__(self, weight_deep_svg=1, weight_type=0.1, weight_parameters=20, weight_eos=0.2):
        super(CustomEmbeddingSliceLoss, self).__init__()
        # Loss functions
        self.loss_function_type = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_function_eos = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_function_parameter = nn.MSELoss()
        self.loss_function_deep_svg = nn.MSELoss()
        # Weights
        self.weight_deep_svg = weight_deep_svg
        self.weight_type = weight_type
        self.weight_parameters = weight_parameters
        self.weight_eos = weight_eos
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
        # Expand target_padding_mask to match the shape of 'input' and set padding positions to -100 in 'input'
        expanded_mask = target_padding_mask.unsqueeze(-1).expand_as(input).to(device)
        input[expanded_mask] = -100

        # Slice embedding into sections
        input_deep_svg, input_type, input_parameters, input_eos = unpack_embedding(input, dim=2, device=device)
        target_deep_svg, target_type, target_parameters, target_eos = unpack_embedding(target, dim=2, device=device)

        loss_type, target_type = self._categorical_loss(self.loss_function_type, input_type, target_type, device=device)
        loss_eos, _ = self._categorical_loss(self.loss_function_eos, input_eos, target_eos, device=device)

        # Ignore features by equalizing vectors on certain indices
        input_parameters = self._ignore_parameters(target_type, input_parameters, target_parameters, device=device)

        # Calculate loss
        loss_deep_svg = self.loss_function_deep_svg(input_deep_svg, target_deep_svg).to(device)
        loss_parameter = self.loss_function_parameter(input_parameters, target_parameters).to(device)

        # print(f"Loss: {self.weight_deep_svg * loss_deep_svg:.5f} === "
        #       f"{self.weight_type * loss_type:.5f} === "
        #       f"{self.weight_parameters * loss_parameter:.5f} === "
        #       f"{self.weight_eos * loss_eos:.5f}"
        #       )

        # Should roughly be balance 33% - 33% - 33%
        loss_overall = (self.weight_deep_svg * loss_deep_svg +
                        self.weight_eos * loss_eos +
                        self.weight_type * loss_type +
                        self.weight_parameters * loss_parameter
                        )

        return loss_overall

    def _categorical_loss(self, loss_function, input, target, device):
        # Flatten to 2D input: [batch_size * seq_length, num_classes]
        input_flat = input.view(-1, input.shape[-1]).to(device)

        # Create Padding mask to ignore padding
        padding_mask = (target == self.PADDING_VALUE).all(dim=-1).to(device)

        # Convert target tensor to class indices if it's one-hot encoded and apply padding mask
        target = target.argmax(dim=-1).to(device)
        target[padding_mask] = -1

        # Flatten to 1D Targets: [batch_size * seq_length]
        target_flat = target.view(-1).to(device)
        loss = loss_function(input_flat, target_flat).to(device)

        return loss, target

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
