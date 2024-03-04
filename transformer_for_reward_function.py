import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
class RewardTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None, batch_first=True):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        #src = self.embedding(src) * math.sqrt(self.dim_model)
            #tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        #print(src.shape, tgt.shape)
        #transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        transformer_out = self.transformer(src, tgt, src_key_padding_mask=src_pad_mask)
        
        out = self.out(transformer_out)
        
        return out


    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        mask = []
        
        for i in range(0, matrix.size(0)):
            seq = []
            for j in range(0, matrix.size(1)):
                if matrix[i,j,0] == pad_token:
                    seq.append(True)
                else:
                    seq.append(False)
            mask.append(seq)
        result = torch.tensor(mask)
        #print(matrix, result, result.shape)
        return result
    def train_loop(model, opt, loss_fn, dataloader):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            X, y = batch[0], batch[1]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

            
            
            #tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            pad_mask_src = model.create_pad_mask(X, 500).to(device)
            #pad_mask_tgt = model.create_pad_mask(y, 10).to(device)

            # Standard training except we pass in y_input and tgt_mask
            #print(X.shape, y.shape)
            pred = model(X, y, src_pad_mask=pad_mask_src)

            print('Predictions: ',pred[:,:,:1])

            # Permute pred to have batch size first again
            #pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
        
            total_loss += loss.detach().item()
            
        return total_loss / len(dataloader)
    
    def validation_loop(model, loss_fn, dataloader):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch[0], batch[1]
                X, y = torch.tensor(X, device=device), torch.tensor(y, device=device)

                #tgt_mask = model.get_tgt_mask(sequence_length).to(device)
                pad_mask_src = model.create_pad_mask(X, 500).to(device)
                #pad_mask_tgt = model.create_pad_mask(y, 10).to(device)

                # Standard training except we pass in y_input and src_mask
                #print("val ", X.shape, y.shape, X.dtype, y.dtype)
                #pred = model(X, y, tgt_mask, src_pad_mask=pad_mask_src, tgt_pad_mask=pad_mask_tgt)
                pred = model(X, y, src_pad_mask=pad_mask_src)


                # Permute pred to have batch size first again
                #pred = pred.permute(1, 2, 0)      
                loss = loss_fn(pred, y)
                total_loss += loss.detach().item()
            
        return total_loss / len(dataloader)
    
    def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        
        # Used for plotting later on
        train_loss_list, validation_loss_list = [], []
        
        print("Training and validating model")
        for epoch in range(epochs):
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            
            train_loss = train_loop(model, opt, loss_fn, train_dataloader)
            train_loss_list += [train_loss]
            
            validation_loss = validation_loop(model, loss_fn, val_dataloader)
            validation_loss_list += [validation_loss]
            
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()
            
        return train_loss_list, validation_loss_list
    
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        #typeLoss = nn.CrossEntropyLoss()
        ParameterLoss = nn.MSELoss()

        #loss = 0.5 * typeLoss(inputs[:,:,:6], targets[:,:,:6]) + 0.5 * ParameterLoss(inputs[:,:,6:12], targets[:,:,6:12])
        #print(inputs[:,:,:6], inputs[:,:,:6].shape, targets[:,:,6:12], targets[:,:,6:12].shape)

        loss = ParameterLoss(inputs[:,:,:1], targets[:,:,:1])

        return loss
    
