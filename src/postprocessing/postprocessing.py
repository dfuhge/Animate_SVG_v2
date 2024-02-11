import pandas as pd
import os
import sys
from xml.dom import minidom

sys.path.append(os.getcwd())


def animate_logo(model_output: pd.DataFrame, logo_path: str):

    # Load logo
    document = minidom.parse(logo_path)

    # Insert every animation
    paths = document.getElementsByTagName('path') # TODO needs to be changed if other elements can have animation id too!
    for i, row in model_output.iterrows():
        # Find element to animate:
        i = 0
        current_path = paths[i]
        while current_path.getAttribute('animation_id') is not row['animation_id']:
            i += 1
            if i >= len(paths):
                print('animation_id ' + row['animation_id'] + ' could not be found')
                break
            current_path = paths[i]
        else:
            # Find out animation type
            try:
                index = row['model_output'].index(1) # TODO check if int or str
            except:
                print('no valid model output')
                continue
            if index == 0:
                print('animation: translate')
            elif index == 1:
                print('animation: scale')
            elif index == 2:
                print('animation: rotate')
            elif index == 3:
                print('animation: skew')
            elif index == 4:
                print('animation: fill')
            elif index == 5:
                print('animation: opacity')
            elif index > 5 or index < 0: # This means that no animation type is set, and the first value is already a parameter
                print('no valid model output')
                continue


