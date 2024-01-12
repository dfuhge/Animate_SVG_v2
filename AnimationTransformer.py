import torch
import torch.nn as nn


class AnimationTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout_prob):
        super(AnimationTransformer, self).__init__()

        # Define the layers and components of the transformer
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,  # Adjust as needed
            dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_size, output_size)  # Adjust output_size as needed
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt):
        # Forward pass through the transformer
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)
        transformer_output = self.transformer(src_embedded, tgt_embedded)

        # Apply linear layer and softmax
        output = self.fc(transformer_output)
        output = self.softmax(output)

        return output
