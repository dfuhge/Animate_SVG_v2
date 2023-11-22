import pandas as pd
import os
import sys
import torch
from xml.dom import minidom

sys.path.append(os.getcwd()) # os.getcwd() needs to be root folder
from src.preprocessing.deepsvg.deepsvg_svglib.svg import SVG
from src.postprocessing.transform_animation_predictor_output import transform_animation_predictor_output
from src.postprocessing.insert_animation import create_animated_svg



def postprocess_logo(output: pd.DataFrame, path: str):
    """
    Takes the model output for one logo and inserts animations into svg.
    Args:
        - model_output: DataFrame with column "animation_id" and column "model_output" animation parameters
        - path: file path of svg to be animated
    """
    # TODO what shape has the output? 


    for i, row in output.iterrows():
        create_animated_svg(path, output['animation_id'], output['model_output'])