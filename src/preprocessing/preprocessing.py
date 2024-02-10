# Imports
import os
import copy
import torch
import glob
import pandas as pd
import pickle
from xml.dom import minidom
from svgpathtools import svg2paths2
from svgpathtools import wsvg
import sys
sys.path.append(os.getcwd())
from deepsvg.deepsvg_svglib.svg import SVG
from deepsvg.deepsvg_config import config_hierarchical_ordered
from deepsvg.deepsvg_utils import train_utils
from deepsvg.deepsvg_utils import utils
from deepsvg.deepsvg_dataloader import svg_dataset

# ---- Methods for embedding logos ----

def compute_embedding_folder(folder_path: str, model_path: str, save: str = None) -> pd.DataFrame:
    data_list = []
    for file in os.listdir(folder_path):
        embedding = compute_embedding(os.path.join(folder_path, file), model_path)
        embedding['filename'] = file
        data_list.append(embedding)
    data = pd.concat(data_list)
    if not save == None:
        output = open(os.path.join(save, 'svg_embedding.pkl'), 'wb')
        pickle.dump(data, output)
        output.close()
    return data
    

def compute_embedding(path: str, model_path: str, save: str = None) -> pd.DataFrame:
    # Convert all primitives to SVG paths - TODO text
    paths, attributes, svg_attributes = svg2paths2(path) # In previous project, this is performed at the end
    wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=path)
    
    svg = SVG.load_svg(path)
    svg.normalize() # Using DeepSVG normalize instead of expanding viewbox - TODO check is this equal?
    svg_str = svg.to_str()
    
    # Assign animation id to every path
    document = minidom.parseString(svg_str)
    paths = document.getElementsByTagName('path')
    for i in range(len(paths)):
        paths[i].setAttribute('animation_id', str(i))

    # Decompose SVGs

    decomposed_svgs = {}
    
    for i in range(len(paths)):
        doc_temp = copy.deepcopy(document)
        paths_temp = doc_temp.getElementsByTagName('path')
        current_path = paths_temp[i]
        # Iteratively choose path i and remove all others
        remove_temp = paths_temp[:i] + paths_temp[i+1:]
        for path in remove_temp:
            if not path.parentNode.nodeName == 'clipPath':
                path.parentNode.removeChild(path)
        # Check for style attributes; add in case there are none
        if len(current_path.getAttribute('style')) <= 0:
            current_path.setAttribute('stroke', 'black')
            current_path.setAttribute('stroke-width', '2')
        id = current_path.getAttribute('animation_id')
        decomposed_svgs[id] = doc_temp.toprettyxml(encoding='iso-8859-1')
        doc_temp.unlink()
    #print(decomposed_svgs)
    meta = {}
    for id in decomposed_svgs:
        svg_d_str = decomposed_svgs[id]
        # Load into SVG and canonicalize
        current_svg = SVG.from_str(svg_d_str)
        # Canonicalize
        current_svg.canonicalize() # Applies DeepSVG canonicalize; previously custom methods were used
        decomposed_svgs[id] = current_svg.to_str()
        if not os.path.exists('svgdatadir'):
            os.mkdir('svgdatadir')
        with open(('svgdatadir/path_' + str(id)) + '.svg', 'w') as svg_file:
            svg_file.write(decomposed_svgs[id])

        # Collect metadata
        len_groups = [path_group.total_len() for path_group in current_svg.svg_path_groups]
        start_pos = [path_group.svg_paths[0].start_pos for path_group in current_svg.svg_path_groups]
        try:
            total_len = sum(len_groups)
            nb_groups = len(len_groups)
            max_len_group = max(len_groups)
        except:
            total_len = 0
            nb_groups = 0
            max_len_group = 0
        
        meta[id] = {
            'id': id,
            'total_len': total_len,
            'nb_groups': nb_groups,
            'len_groups': len_groups,
            'max_len_group': max_len_group,
            'start_pos': start_pos
        }
    metadata = pd.DataFrame(meta.values())
    metadata.to_csv('metadata.csv', index=False)
    # Load pretrained DeepSVG model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = config_hierarchical_ordered.Config()
    model = cfg.make_model().to(device)
    train_utils.load_model(model_path, model)
    model.eval()
    # Load dataset
    cfg.data_dir = 'svgdatadir/'
    cfg.meta_filepath = 'metadata.csv'
    dataset = svg_dataset.load_dataset(cfg)
    svg_files = glob.glob('svgdatadir/*.svg')
    svg_list = []
    for svg_file in svg_files:
        id = svg_file.split('_')[1].split('.')[0]
        # Preprocessing
        svg = SVG.load_svg(svg_file)
        svg = dataset.simplify(svg)
        svg = dataset.preprocess(svg, augment=False)
        data = dataset.get(svg=svg)
        # Get embedding
        model_args = utils.batchify((data[key] for key in cfg.model_args), device)
        with torch.no_grad():
            z = model(*model_args, encode_mode=True).numpy()[0][0][0]
        dict_data = {
            'animation_id': id,
            'embedding': z
        }
        svg_list.append(dict_data)
    data = pd.DataFrame.from_records(svg_list, index='animation_id')['embedding'].apply(pd.Series)
    data.reset_index(level=0, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    if not save == None:
        output = open(os.path.join(save, 'svg_embedding.pkl'), 'wb')
        pickle.dump(data, output)
        output.close()
    print('Embedding computed')
    return data
