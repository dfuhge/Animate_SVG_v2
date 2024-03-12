from src.preprocessing.preprocessing import compute_embedding
import AnimationTransformer
import torch.nn as torch
import torch

def animateLogo(path : str):
    # HYPERPARAMETERS
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01

    #transformer
    NUM_HEADS = 6 # Dividers of 282: {1, 2, 3, 6, 47, 94, 141, 282}
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 8
    DROPOUT=0.1
    # CONSTANTS
    FEATURE_DIM = 282

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AnimationTransformer(
    dim_model=FEATURE_DIM,
    num_heads=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dropout_p=DROPOUT,
    use_positional_encoder=True
    ).to(device)

    model.transformer.load_state_dict(torch.load("data/models/animation_transformer.pth"))


logo = "data/examples/logo_168.svg"
animateLogo(logo)