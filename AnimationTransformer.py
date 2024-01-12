import math

import torch
import torch.nn as nn


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

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None, batch_first=True):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask)
        # TODO: Add Softmax:
        out = transformer_out

        return out


def get_tgt_mask(size) -> torch.tensor:
    # Todo: Check if this is suitable for 3D as example is 2D
    # Generates a square matrix where the each row allows one word more to be seen
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


def create_pad_mask(self, matrix: torch.tensor, pad_value: int) -> torch.tensor:
    # Todo: Check if this is needed, as padding is already set to float(-inf)
    # If matrix = [1,2,3,0,0,0] where pad_value=0, the result mask is
    # [False, False, False, True, True, True]
    return matrix == pad_value


def train_loop(model, opt, loss_function, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        source, target = batch[0], batch[1]
        # Move the tensor to a GPU (if available)
        source, target = torch.tensor(source).to(device), torch.tensor(target).to(device)

        # TODO doesn't work with padded sequences, does it? No batches then?
        # First index is all batch entries, second is
        target_input = target[:, :-1]  # trg input is offset by one (SOS token and excluding EOS)
        target_expected = target[:, 1:]  # trg is offset by one (excluding SOS token)

        # Get mask to mask out the next words
        sequence_length = target_input.size(1)
        tgt_mask = get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        prediction = model(source, target_input, tgt_mask)  # TODO adapt forward method: tgt_mask

        # Permute prediction to have batch size first again
        prediction = prediction.permute(1, 2, 0)
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
            source, target = batch[0], batch[1]
            source, target = torch.tensor(source, device=device), torch.tensor(target, device=device)

            target_input = target[:, :-1]  # trg input is offset by one (SOS token and excluding EOS)
            target_expected = target[:, 1:]  # trg is offset by one (excluding SOS token)

            # Get mask to mask out the next words
            sequence_length = target_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            prediction = model(source, target_input, tgt_mask)

            # Permute pred to have batch size first again
            prediction = prediction.permute(1, 2, 0)
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
