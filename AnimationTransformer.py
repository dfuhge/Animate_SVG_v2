import math

import torch
import torch.nn as nn

import dataset_helper


class AnimationTransformer(nn.Module):
    def __init__(
            self,
            dim_model,  # hidden_size; corresponds to embedding length
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p,
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # TODO: Currently left out, as input sequence shuffled. Later check if use is beneficial.
        # self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.softmax = nn.Softmax(dim=2)  # Todo: Softmax over Categorical values
        # self.out = nn.Linear(dim_model, dim_model)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask)
        # TODO: Add Softmax:
        # out = self.out(out)

        return out


def get_tgt_mask(size) -> torch.tensor:
    # Generates a square matrix where each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask


def create_pad_mask(matrix: torch.tensor) -> torch.tensor:
    pad_masks = []

    # Iterate over each sequence in the batch.
    for i in range(0, matrix.size(0)):
        sequence = []

        # Iterate over each element in the sequence and append True if padding value
        for j in range(0, matrix.size(1)):
            sequence.append(matrix[i, j, 0] == dataset_helper.PADDING_VALUE)

        pad_masks.append(sequence)

    return torch.tensor(pad_masks)


def train_loop(model, opt, loss_function, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        print("TRAIN BATCH")
        source, target = batch[0], batch[1]

        if torch.isnan(source).any() or torch.isnan(target).any():
            raise ValueError("Input data contains NaN values.")

        source, target = source.to(device), target.to(device)

        # TODO doesn't work with padded sequences, does it? No batches then?
        # First index is all batch entries, second is
        target_input = target[:, :-1]  # trg input is offset by one (SOS token and excluding EOS)
        target_expected = target[:, 1:]  # trg is offset by one (excluding SOS token)

        # Get mask to mask out the next words
        tgt_mask = get_tgt_mask(target_input.size(1)).to(device)

        # Standard training except we pass in y_input and tgt_mask
        prediction = model(source, target_input,
                           tgt_mask=tgt_mask,
                           src_pad_mask=create_pad_mask(source).to(device),
                           tgt_pad_mask=create_pad_mask(target_input).to(device))

        if torch.isnan(prediction).any():
            print("Nan prediction detected")

        loss = loss_function(prediction, target_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_function, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            print("VALIDATION BATCH")
            source, target = batch[0], batch[1]
            source, target = source.clone().detach().to(device), target.clone().detach().to(device)

            target_input = target[:, :-1]  # trg input is offset by one (SOS token and excluding EOS)
            target_expected = target[:, 1:]  # trg is offset by one (excluding SOS token)

            # Get mask to mask out the next words
            tgt_mask = get_tgt_mask(target_input.size(1)).to(device)

            # Standard training except we pass in y_input and src_mask
            prediction = model(source, target_input,
                               tgt_mask=tgt_mask,
                               src_pad_mask=create_pad_mask(source).to(device),
                               tgt_pad_mask=create_pad_mask(target_input).to(device))

            loss = loss_function(prediction, target_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, optimizer, loss_function, train_dataloader, val_dataloader, epochs, device):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, optimizer, loss_function, train_dataloader, device)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_function, val_dataloader, device)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
