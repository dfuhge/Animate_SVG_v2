import os
import torch
import pandas as pd
import pickle
import glob
import logging
import shutil
import sys

from svgpathtools import svg2paths
from xml.dom import minidom
from pathlib import Path
from concurrent import futures
from tqdm import tqdm

sys.path.append(os.getcwd()) # os.getcwd() needs to be root folder
from src.preprocessing.deepsvg.deepsvg_config import config_hierarchical_ordered as deepsvg_config
#sys.path.append(os.path.relpath('src/preprocessing/deepsvg/deepsvg_utils'))
from src.preprocessing.deepsvg.deepsvg_utils import utils as deepsvg_utils
from src.preprocessing.deepsvg.deepsvg_utils import train_utils as deepsvg_train_utils
#sys.path.append(os.path.relpath('src/preprocessing/deepsvg/deepsvg_dataloader'))
from src.preprocessing.deepsvg.deepsvg_dataloader import svg_dataset as deepsvg_dataset
#sys.path.append(os.path.relpath('src/preprocessing/deepsvg/deepsvg_svglib'))
from src.preprocessing.deepsvg.deepsvg_svglib.svg import SVG

def insert_animation_id_folder(path: str, target_path: str) -> bool:
    """
    (Adapted from AnimateSVG)
    Inserts an animation id to every svg file in the path folder.

    Args:
        - path(str): Path of the dataset folder
    Return:
        - True if insertion successful, False otherwise
    """
    for file in os.listdir(path):
        if file.endswith('.svg'):
            file_path = os.path.join(path, file)
            successful = _insert_animation_id(file_path, target_path)
            if not successful:
                return False
    return True

def _insert_animation_id(path: str, target_path) -> bool:
    """
    (Adapted from AnimateSVG)
    Inserts an animation id to the svg file at the path.

    Args:
        - path(str): Path of the file
    Return:
        - True if insertion successful, False otherwise
    """
    #filename = path.replace('.svg', '').split("/")[-1]
    try:
        filename = path.split("/")[-1]
        doc = minidom.parse(path)
        # Store all elements in list
        elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
            'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
            'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
            'rect') + doc.getElementsByTagName('text')
        for i in range(len(elements)):
            elements[i].setAttribute('animation_id', str(i))
        # write svg
        textfile = open(os.path.join(target_path, filename), 'wb')
        textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
        textfile.close()
        doc.unlink()
        return True
    except:
        return False
    
def expand_viewbox_folder(path: str, target_path: str, percent: int = 50) -> bool:
    """
    (Adapted from AnimateSVG)
    Expands the viewbox of every svg file in the path folder.

    Args:
        - path(str): Path of the dataset folder
    Return:
        - True if expanding successful, False otherwise
    """
    for file in os.listdir(path):
        if file.endswith('.svg'):
            file_path = os.path.join(path, file)
            successful = _expand_viewbox(file_path, target_path, percent)
            if not successful:
                return False
    return True

def _expand_viewbox(path: str, target_path: str, percent: int = 50) -> bool:
    """ Expand the viewbox of a given SVG.

    Args:
        path (svg): Path of SVG file.
        percent (int): Percentage in %: How much do we want to expand the viewbox? Default is 50%.
        target_path (str): Path of folder containing the expanded SVGs.

    """
    #Path(new_folder).mkdir(parents=True, exist_ok=True)
    pathelements = path.split('/')
    filename = pathelements[len(pathelements) - 1]#.replace('.svg', '')

    doc = minidom.parse(path)
    x, y = '', ''
    # get width and height of logo
    try:
        width = doc.getElementsByTagName('svg')[0].getAttribute('width')
        height = doc.getElementsByTagName('svg')[0].getAttribute('height')
        if not width[-1].isdigit():
            width = width.replace('px', '').replace('pt', '')
        if not height[-1].isdigit():
            height = height.replace('px', '').replace('pt', '')
        x = float(width)
        y = float(height)
        check = True
    except:
        check = False
    if not check:
        try:
            # get bounding box of svg
            xmin_svg, xmax_svg, ymin_svg, ymax_svg = 0, 0, 0, 0
            paths, attributes = svg2paths(path)
            for path in paths:
                xmin, xmax, ymin, ymax = path.bbox()
                if xmin < xmin_svg:
                    xmin_svg = xmin
                if xmax > xmax_svg:
                    xmax_svg = xmax
                if ymin < ymin_svg:
                    ymin_svg = ymin
                if ymax > ymax_svg:
                    ymax_svg = ymax
                x = xmax_svg - xmin_svg
                y = ymax_svg - ymin_svg
        except:
            print('Error: ' + filename)
            return False
    # Check if viewBox exists
    if doc.getElementsByTagName('svg')[0].getAttribute('viewBox') == '':
        v1, v2, v3, v4 = 0, 0, 0, 0
        # Calculate new viewBox values
        x_new = x * (100 + percent) / 100
        y_new = y * (100 + percent) / 100
    else:
        v1 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[0].replace('px', '').replace('pt', '').replace(',', ''))
        v2 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[1].replace('px', '').replace('pt', '').replace(',', ''))
        v3 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[2].replace('px', '').replace('pt', '').replace(',', ''))
        v4 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[3].replace('px', '').replace('pt', '').replace(',', ''))
        x = v3
        y = v4
        # Calculate new viewBox values
        x_new = x * percent / 100
        y_new = y * percent / 100
    x_translate = - x * percent / 200
    y_translate = - y * percent / 200
    coordinates = str(v1 + x_translate) + ' ' + str(v2 + y_translate) + ' ' + str(v3 + x_new) + ' ' + str(v4 + y_new)
    doc.getElementsByTagName('svg')[0].setAttribute('viewBox', coordinates)
    # write to svg
    textfile = open(os.path.join(target_path, filename), 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()
    doc.unlink()
    return True

def decompose_svg_folder(path: str, target_path: str) -> bool:
    """
    (Adapted from AnimateSVG)
    Expands the viewbox of every svg file in the path folder.

    Args:
        - path(str): Path of the dataset folder
    Return:
        - True if expanding successful, False otherwise
    """
    for file in os.listdir(path):
        if file.endswith('.svg'):
            file_path = os.path.join(path, file)
            successful = _decompose_svg(file_path, target_path)
            if not successful:
                return False
    return True

def _decompose_svg(path: str, target_path: str) -> bool:
    """ Decompose one SVG.

    Args:
        path (str): Path of SVG that needs to be decomposed.
    Return:
        - True if decomposing successful, False otherwise

    """
    doc = minidom.parse(path)
    # store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName('polygon') + doc.getElementsByTagName(
        'polyline') + doc.getElementsByTagName('rect') + doc.getElementsByTagName('text')
    num_elements = len(elements)

    # Change name and path for writing element svgs
    filename = path.split('/')[-1].replace('.svg', '')
    #Path("data/decomposed_svgs").mkdir(parents=True, exist_ok=True)

    # Write each element to a svg file
    for i in range(num_elements):
        # load svg again: necessary because we delete elements in each loop
        doc_temp = minidom.parse(path)
        elements_temp = doc_temp.getElementsByTagName('path') + doc_temp.getElementsByTagName(
            'circle') + doc_temp.getElementsByTagName('ellipse') + doc_temp.getElementsByTagName(
            'line') + doc_temp.getElementsByTagName('polygon') + doc_temp.getElementsByTagName(
            'polyline') + doc_temp.getElementsByTagName('rect') + doc_temp.getElementsByTagName('text')
        # select all elements besides one
        elements_temp_remove = elements_temp[:i] + elements_temp[i + 1:]
        for element in elements_temp_remove:
            # Check if current element is referenced clip path
            if not element.parentNode.nodeName == "clipPath":
                parent = element.parentNode
                parent.removeChild(element)
        # Add outline to element (to handle white elements on white background)
        elements_temp[i].setAttribute('stroke', 'black')
        elements_temp[i].setAttribute('stroke-width', '2')
        # If there is a style attribute, remove stroke:none
        if len(elements_temp[i].getAttribute('style')) > 0:
            elements_temp[i].attributes['style'].value = elements_temp[i].attributes['style'].value.replace('stroke:none', '')
        # save element svgs
        animation_id = elements_temp[i].getAttribute('animation_id')
        textfile = open(os.path.join(target_path, filename) + '_' + animation_id + '.svg', 'wb')
        textfile.write(doc_temp.toxml(encoding="iso-8859-1")) # needed to handle "Umlaute"
        textfile.close()
        doc_temp.unlink()

    doc.unlink()
    return True

def get_metadata_folder(path: str, target_path: str) -> pd.DataFrame:
    """ Get meta data of all SVGs in a given folder.

    Note: There are some elements (like text tags or matrices or clip paths) that can't be processed here. The meta
    file only considers "normal" elements.

    Args:
        data_folder (str): Path of the folder containing all SVGs.

    Returns:
        pd.DataFrame: Dataframe containing metadata of SVGs.

    """
    """
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        svg_files = glob.glob(os.path.join(path, "*.svg"))
        meta_data = {}

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [
                executor.submit(_get_svg_meta_data, svg_file, meta_data)
                for svg_file in svg_files]
            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)
    """
    meta_data = {}
    for svg_file in glob.glob(os.path.join(path, "*.svg")):
        _get_svg_meta_data(svg_file, meta_data)
    df = pd.DataFrame(meta_data.values())
    print(df)
    df.to_csv(os.path.join(target_path, "meta_data.csv"), index=False)
    return df

def _get_svg_meta_data(svg_file, meta_data):
    print("Test: " + str(svg_file))
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    #svg = SVG.load_svg(svg_file)  # THIS ONE
    # svg.fill_(False)
    # svg.normalize()
    # svg.zoom(0.9)
    # svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])

    #svg.canonicalize(normalize=True)  # THIS ONE

    svg = _canonicalize(svg_file, normalize=True)

    # svg = svg.simplify_heuristic()

    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]
    start_pos = [path_group.svg_paths[0].start_pos for path_group in svg.svg_path_groups]
    try:
        total_len = sum(len_groups)
        nb_groups = len(len_groups)
        max_len_group = max(len_groups)
    except:
        total_len = 0
        nb_groups = 0
        max_len_group = 0
    meta_data[filename] = {
        "id": filename,
        "total_len": total_len,
        "nb_groups": nb_groups,
        "len_groups": len_groups,
        "max_len_group": max_len_group,
        "start_pos": start_pos
    }

def _canonicalize(svg_file, normalize=False):
    svg = SVG.load_svg(svg_file)
    svg.to_path().simplify_arcs()

    if normalize:
        svg.normalize()

    #svg.split_paths()
    svg.filter_consecutives()
    svg.filter_empty()
    svg._apply_to_paths("reorder")
    svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
    print(svg.svg_path_groups)
    svg._apply_to_paths("canonicalize")
    svg.recompute_origins()

    svg.drop_z()

    return svg

def encode_svg_folder(path, split_paths=True, save=True, metadata_path="", model_path="", target_path=""):
    """ Get embeddings of all SVGs in a given folder.

    Args:
        path (str): path of folder to be embedded
        split_paths (bool): If true, additional preprocessing step is carried out, where paths consisting of multiple
                            paths are split into multiple paths.
        save (bool): If true, SVG embedding is saved as pd.DataFrame in folder data/embeddings.
        model_path (str): path to model to be used

    Returns:
        pd.DataFrame: Dataframe containing filename, animation_id and embedding.

    """
    dataset, model, device, cfg = _load_model_and_dataset(path=path, metadata_path=metadata_path, model_path=model_path)
    cfg.data_dir = path
    """
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        svg_files = glob.glob(os.path.join(cfg.data_dir, "*.svg"))
        svg_list = []
        with tqdm(total=len(svg_files)) as pbar:
            embedding_requests = [
                executor.submit(_apply_embedding_model_to_svg, dataset, svg_file, svg_list, model, device, cfg,
                                split_paths)
                for svg_file in svg_files]

            for _ in futures.as_completed(embedding_requests):
                pbar.update(1)
    """
    svg_files = glob.glob(os.path.join(cfg.data_dir, "*.svg"))
    svg_list = []
    for file in svg_files:
        _apply_embedding_model_to_svg(dataset, file, svg_list, model, device, cfg, split_paths)

    df = pd.DataFrame.from_records(svg_list, index='filename')['embedding'].apply(pd.Series)
    print(df)
    df.reset_index(level=0, inplace=True)

    df['animation_id'] = df['filename'].apply(lambda row: row.split('_')[-1])
    cols = list(df.columns)
    cols = [cols[0], cols[-1]] + cols[1:-1]
    df = df.reindex(columns=cols)
    df['filename'] = df['filename'].apply(lambda row: "_".join(row.split('_')[0:-1]))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    if save:
        output = open(os.path.join(target_path, 'svg_embedding.pkl'), 'wb')
        pickle.dump(df, output)
        output.close()

    logging.info("Embedding complete.")
    return df

def _load_model_and_dataset(path, metadata_path="", model_path=""):
    # Load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = deepsvg_config.Config()
    model = cfg.make_model().to(device)
    deepsvg_train_utils.load_model(model_path, model)
    model.eval()

    # Load dataset
    cfg.data_dir = f"{path}/"
    cfg.meta_filepath = metadata_path
    print(cfg.meta_filepath)
    dataset = deepsvg_dataset.load_dataset(cfg)
    return dataset, model, device, cfg

def _apply_embedding_model_to_svg(dataset, svg_file, svg_list, model, device, cfg, split_paths):
    z = _encode_svg(dataset, svg_file, model, device, cfg, split_paths).numpy()[0][0][0]
    filename = os.path.splitext(os.path.basename(svg_file))[0]
    print(filename)
    dict_data = {"filename": filename,
                 "embedding": z}
    svg_list.append(dict_data)

def _encode_svg(dataset, filename, model, device, cfg, split_paths):
    # Note: Only 30 segments per path are allowed. Paths are cut after the first 30 segments.
    svg = SVG.load_svg(filename)
    if split_paths:
        svg = dataset.simplify(svg)
        svg = dataset.preprocess(svg, augment=False)
        data = dataset.get(svg=svg)
    else:  # Here: paths are not split
        svg = _canonicalize_without_path_splitting(svg, normalize=True)
        svg = dataset.preprocess(svg, augment=False)
        data = dataset.get(svg=svg)

    model_args = deepsvg_utils.batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z

def _canonicalize_without_path_splitting(svg, normalize=False):
    svg.to_path().simplify_arcs()
    if normalize:
        svg.normalize()
    svg.filter_consecutives()
    svg.filter_empty()
    svg._apply_to_paths("reorder")
    svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
    svg._apply_to_paths("canonicalize")
    svg.recompute_origins()
    svg.drop_z()
    return svg


def preprocessing_pipeline(path: str = os.getcwd()) -> pd.DataFrame:
    current_path = os.path.join(path, "data", "0_raw_dataset")

    # --- 1 Insert Animation ID ---
    target_folder_name = "1_inserted_animation_id"
    target_folder_path = os.path.join(os.getcwd(), "data", target_folder_name)

    # Create target folder if not existent
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)

    # Insert animation IDs
    insert_animation_id_folder(current_path, target_folder_path)

    # Set new current path
    current_path = target_folder_path

    # --- 2 Expand Viewbox ---
    target_folder_name = "2_expanded_viewbox"
    target_folder_path = os.path.join(os.getcwd(), "data", target_folder_name)

    # Create target folder if not existent
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)

    # Expand viewbox
    expand_viewbox_folder(current_path, target_folder_path)
    # Set new current path
    current_path = target_folder_path

    # --- 3 Decompose SVG ---
    target_folder_name = "3_decomposed_svg"
    target_folder_path = os.path.join(os.getcwd(), "data", target_folder_name)

    # Create target folder if not existent
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)

    # Decompose SVGs in folder
    decompose_svg_folder(current_path, target_folder_path)
    # Set new current path
    current_path = target_folder_path

    # --- 4 Extract Metadata ---
    target_folder_name = "4_svg_metadata"
    target_folder_path = os.path.join(os.getcwd(), "data", target_folder_name)

    # Create target folder if not existent
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)

    # Extract SVG metadata
    get_metadata_folder(current_path, target_folder_path)

    # --- 5 Encode SVG Embedding ---
    target_folder_name = "5_svg_embedding"
    target_folder_path = os.path.join(os.getcwd(), "data", target_folder_name)

    metadata_path = "data/4_svg_metadata/meta_data.csv"
    model_path = "src/preprocessing/deepsvg/deepsvg_models/deepSVG_hierarchical_ordered.pth.tar"

    # Create target folder if not existent
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)

    # Encode SVGs
    embedding = encode_svg_folder(current_path, metadata_path=metadata_path, model_path=model_path, target_path=target_folder_path)
    
    return embedding


print(preprocessing_pipeline())