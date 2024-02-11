import pandas as pd
import os
import sys
import torch
from xml.dom import minidom

sys.path.append(os.getcwd()) # os.getcwd() needs to be root folder
from src.postprocessing_old.insert_animation import create_animated_svg



def postprocess_logo(output: pd.DataFrame, path: str):
    """
    Takes the model output for one logo and inserts animations into svg.
    Args:
        - model_output: DataFrame with column "animation_id" and column "model_output" animation parameters
        - path: file path of svg to be animated

    Example:
    animation_id                          model_output
               1  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
               2  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
               3  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    # TODO what shape has the output? 


    for i, row in output.iterrows():
        create_animated_svg(path, output['animation_id'], output['model_output'])

if __name__ == "__main__":
    data_nd = [
        [1, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        [2, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [3, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    ]

    data = pd.DataFrame(data_nd, columns=["animation_id", "model_output"])
    postprocess_logo(data, "/Users/dfuhge/Documents/Studium/Uni Mannheim/Master/1. Semester/TP 500 Team Project - AnimateSVG/Animate_SVG_v2/Animate_SVG_v2/data/1_inserted_animation_id/logo_0.svg")
