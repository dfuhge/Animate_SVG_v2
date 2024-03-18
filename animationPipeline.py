from src.preprocessing.preprocessing import compute_embedding
from src.postprocessing.postprocessing import animate_logo
from AnimationTransformer import AnimationTransformer
from AnimationTransformer import predict
import torch.nn as torch
import torch
import pandas as pd
import shutil
import os

def animateLogo(path : str):

    if not os.path.exists(path):
        print("The original file does not exist.")
        return None
    
    # Extract the filename and extension
    filename, extension = os.path.splitext(os.path.basename(path))
    
    # Construct the new filename with "_animated" attached
    new_filename = filename + "_animated" + extension
    
    # Construct the path for the new file
    targetPath = os.path.join(os.path.dirname(path), new_filename)
    
    try:
        # Copy the original file to the new location with the new filename
        shutil.copyfile(path, targetPath)
        print(f"File copied and renamed to {new_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


    #transformer
    NUM_HEADS = 47 # Dividers of 282: {1, 2, 3, 6, 47, 94, 141, 282}
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 4
    DROPOUT=0.21
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
    result = predict(model, inp, sos_token=sos_token, device=device, max_length=inp.shape[0], eos_scaling=0.5, temperature=1000000)
    result = pd.DataFrame(result[1:, -26:].cpu().detach().numpy())
    result = pd.DataFrame({"model_output" : [row.tolist() for index, row in result.iterrows()]})
    result["animation_id"] = range(len(result))
    #print(result, path)
    animate_logo(result, targetPath)

logo = "data/examples/sariyer.svg"
animateLogo(logo)