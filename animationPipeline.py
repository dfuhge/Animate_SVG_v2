from src.preprocessing.preprocessing import compute_embedding
from src.postprocessing.postprocessing import animate_logo
from AnimationTransformer import AnimationTransformer
from AnimationTransformer import predict
import torch.nn as torch
import torch
import pandas as pd

def animateLogo(path : str, targetPath : str):
    #transformer
    NUM_HEADS = 47 # Dividers of 282: {1, 2, 3, 6, 47, 94, 141, 282}
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 4
    DROPOUT=0.2
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

    model.load_state_dict(torch.load("data/models/animation_transformer2.pth"))


    df = compute_embedding(path, "src/preprocessing/deepsvg/deepsvg_models/deepSVG_hierarchical_ordered.pth.tar")
    #print(df.shape)
    df = df.drop("animation_id", axis=1)

    df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=range(df.shape[1], df.shape[1] + 26))], axis=1, ignore_index=True).astype(float)
    inp = torch.tensor(df.values)
    #print(inp, inp.shape)


    sos_token = torch.zeros(282)
    sos_token[256] = 1
    result = predict(model, inp, sos_token=sos_token, device=device, max_length=inp.shape[0], eos_scaling=0.1)
    result = pd.DataFrame(result[1:, -26:].cpu().detach().numpy())
    result = pd.DataFrame({"model_output" : [row.tolist() for index, row in result.iterrows()]})
    result["animation_id"] = range(len(result))
    #print(result, path)
    animate_logo(result, target)

logo = "data/examples/c.svg"
target = "data/examples/c_animated.svg"
animateLogo(logo, target)