import math
import time

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
            use_positional_encoder=True
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # TODO: Currently left out, as input sequence shuffled. Later check if use is beneficial.
        self.use_positional_encoder = use_positional_encoder
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model,
            dropout_p=dropout_p
        )

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        if self.use_positional_encoder:
            src = self.positional_encoder(src)
            tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
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


def _transformer_call_in_loops(model, batch, device, loss_function):
    source, target = batch[0], batch[1]
    source, target = source.to(device), target.to(device)

    # First index is all batch entries, second is
    target_input = target[:, :-1]  # trg input is offset by one (SOS token and excluding EOS)
    target_expected = target[:, 1:]  # trg is offset by one (excluding SOS token)

    # SOS -  1  -  2  -  3  -  4  - EOS - PAD - PAD // target_input
    #  1  -  2  -  3  -  4  - EOS - PAD - PAD - PAD // target_expected

    # Get mask to mask out the next words
    tgt_mask = get_tgt_mask(target_input.size(1)).to(device)

    # Standard training except we pass in y_input and tgt_mask
    prediction = model(source, target_input,
                       tgt_mask=tgt_mask,
                       src_key_padding_mask=create_pad_mask(source).to(device),
                       # Mask with expected as EOS is no input (see above)
                       tgt_key_padding_mask=create_pad_mask(target_expected).to(device))

    return loss_function(prediction, target_expected, create_pad_mask(target_expected).to(device))


def train_loop(model, opt, loss_function, dataloader, device):
    model.train()
    total_loss = 0

    t0 = time.time()
    i = 1
    for batch in dataloader:
        loss = _transformer_call_in_loops(model, batch, device, loss_function)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

        if i == 1 or i % 10 == 0:
            elapsed_time = time.time() - t0
            total_expected = elapsed_time / i * len(dataloader)
            print(f">> {i}: Time per Batch {elapsed_time / i : .2f}s | "
                  f"Total expected {total_expected / 60 : .2f} min | "
                  f"Remaining {(total_expected - elapsed_time) / 60 : .2f} min ")
        i += 1

    print(f">> Epoch time: {(time.time() - t0)/60:.2f} min")
    return total_loss / len(dataloader)


def validation_loop(model, loss_function, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            loss = _transformer_call_in_loops(model, batch, device, loss_function)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, optimizer, loss_function, train_dataloader, val_dataloader, epochs, device):
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


def predict(model, source_sequence, sos_token: torch.Tensor, device, max_length=32, eos_scaling=1, backpropagate=False):
    if backpropagate:
        model.train()
    else:
        model.eval()

    source_sequence = source_sequence.float().to(device)
    y_input = torch.unsqueeze(sos_token, dim=0).float().to(device)

    i = 0
    while i < max_length:
        # Get source mask
        prediction = model(source_sequence.unsqueeze(0), y_input.unsqueeze(0),  # un-squeeze for batch
                           # tgt_mask=get_tgt_mask(y_input.size(0)).to(device),
                           src_key_padding_mask=create_pad_mask(source_sequence.unsqueeze(0)).to(device))

        next_embedding = prediction[0, -1, :]  # prediction on last token
        pred_deep_svg, pred_type, pred_parameters = dataset_helper.unpack_embedding(next_embedding, dim=0)
        pred_deep_svg, pred_type, pred_parameters = pred_deep_svg.to(device), pred_type.to(device), pred_parameters.to(
            device)

        # === TYPE ===
        # Apply Softmax
        type_softmax = torch.softmax(pred_type, dim=0)
        type_softmax[0] = type_softmax[0] * eos_scaling  # Reduce EOS
        animation_type = torch.argmax(type_softmax, dim=0)

        # Break if EOS is most likely
        if animation_type == 0:
            print("END OF ANIMATION")
            y_input = torch.cat((y_input, sos_token.unsqueeze(0).to(device)), dim=0)
            return y_input

        pred_type = torch.zeros(11)
        pred_type[animation_type] = 1

        # === DEEP SVG ===
        # Find the closest path
        distances = [torch.norm(pred_deep_svg - embedding[:-26]) for embedding in source_sequence]
        closest_index = distances.index(min(distances))
        closest_token = source_sequence[closest_index]

        # === PARAMETERS ===
        # overwrite unused parameters
        for j in range(len(pred_parameters)):
            if j in dataset_helper.ANIMATION_PARAMETER_INDICES[int(animation_type)]:
                continue
            pred_parameters[j] = -1

        # === SEQUENCE ===
        y_new = torch.concat([closest_token[:-26], pred_type.to(device), pred_parameters], dim=0)
        y_input = torch.cat((y_input, y_new.unsqueeze(0)), dim=0)

        # === INFO PRINT ===
        print(f"{int(y_input.size(0))}: Path {closest_index} ({round(float(distances[closest_index]), 3)}) "
              f"got animation {animation_type} ({round(float(type_softmax[animation_type]), 3)}%) "
              f"with parameters {[round(num, 2) for num in pred_parameters.tolist()]}")

        i += 1

    return y_input


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=5000):
        """
        Initializes the PositionalEncoding module which injects information about the relative or absolute position
        of the tokens in the sequence. The positional encodings have the same dimension as the embeddings so that the
        two can be summed. Uses a sinusoidal pattern for positional encoding.

        Args:
            dim_model (int): The dimension of the embeddings and the expected dimension of the positional encoding.
            dropout_p (float): Dropout probability to be applied to the summed embeddings and positional encodings.
            max_len (int): The max length of the sequences for which positional encodings are precomputed and stored.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pos_encoding = torch.zeros(max_len, 1, dim_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input embeddings and applies dropout.

        Args:
            embedding (torch.Tensor): The input embeddings with shape [batch_size, seq_len, dim_model]

        Returns:
            torch.Tensor: The embeddings with positional encoding applied, and dropout, having the same shape as the
            input token embeddings [seq_len, batch_size, dim_model].
        """
        return self.dropout(embedding + self.pos_encoding[:embedding.size(0), :])
