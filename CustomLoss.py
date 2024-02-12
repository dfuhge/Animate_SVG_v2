from torch import nn, Tensor

import dataset_helper
from dataset_helper import unpack_embedding


class CustomEmbeddingSliceLoss(nn.Module):
    def __init__(self, weight_deep_svg=10, weight_type=0.1, weight_parameters=1):
        super(CustomEmbeddingSliceLoss, self).__init__()
        self.loss_function_type = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_function_parameter = nn.MSELoss()
        self.loss_function_deep_svg = nn.MSELoss()
        self.dataset_helper = dataset_helper
        self.weight_deep_svg = weight_deep_svg
        self.weight_type = weight_type
        self.weight_parameters = weight_parameters

    def forward(self, input: Tensor, target: Tensor, target_padding_mask: Tensor, device="cpu") -> Tensor:
        """

        Args:
            input: tensor of shape [batch_size, sequence_length, num_features]
            target: tensor of shape [batch_size, sequence_length, num_features]
            target_padding_mask: tensor of shape [batch_size, sequence_length]
            device: cpu / gpu

        Returns: Combined loss

        """
        # Expand target_padding_mask to match the shape of 'input'
        expanded_mask = target_padding_mask.unsqueeze(-1).expand_as(input).to(device)

        # Set padding positions to -100 in 'input'
        input[expanded_mask] = -100

        # Slice embedding into sections
        input_deep_svg, input_type, input_parameters = unpack_embedding(input, dim=2, device=device)
        target_deep_svg, target_type, target_parameters = unpack_embedding(target, dim=2, device=device)

        # Ignore features by equalizing vectors on certain indices
        for i in range(input.size(0)):
            for j in range(input.size(1)):
                if target_type[i][j][0] == dataset_helper.PADDING_VALUE:
                    # if input_type[i][j][0] != dataset_helper.PADDING_VALUE:
                    #     print(f"PADDING WARNING: {target_type[i][j][0]} --  {input_type[i][j][0]}")
                    break  # no continue as sequence ends here

                target_value = (target_type[i][j] == 1).nonzero(as_tuple=True)[0].item()

                for parameter_index in dataset_helper.ANIMATION_PARAMETER_INDICES[target_value]:
                    input_parameters[i][j][parameter_index] = target_parameters[i][j][parameter_index].to(device)

        # Flatten to 2D input: [batch_size * seq_length, num_classes]
        input_type_flat = input_type.view(-1, input_type.shape[-1]).to(device)

        # Create Padding mask to ignore padding
        type_padding_mask = (target_type == dataset_helper.PADDING_VALUE).all(dim=-1).to(device)
        target_type_flat = target_type.argmax(dim=-1).to(device)
        target_type_flat[type_padding_mask] = -1
        # Flatten to 1D Targets: [batch_size * seq_length]
        target_type_flat = target_type_flat.view(-1).to(device)

        # Calculate loss
        loss_deep_svg = self.loss_function_deep_svg(input_deep_svg, target_deep_svg).to(device)
        loss_type = self.loss_function_type(input_type_flat, target_type_flat).to(device)
        loss_parameter = self.loss_function_parameter(input_parameters, target_parameters).to(device)

        #print(f"Loss: {round(float(self.weight_deep_svg * loss_deep_svg), 5)} === "
        #      f"{round(float(self.weight_type * loss_type), 5)} === "
        #      f"{round(float(self.weight_parameters * loss_parameter), 5)}")

        # Should roughly be balance 33% - 33% - 33%
        loss_overall = (self.weight_deep_svg * loss_deep_svg +
                        self.weight_type * loss_type +
                        self.weight_parameters * loss_parameter)

        return loss_overall
