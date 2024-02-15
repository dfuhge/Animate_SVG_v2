import pandas as pd
import os
import sys
from xml.dom import minidom
from collections import defaultdict

sys.path.append(os.getcwd())
from get_svg_size_pos import get_svg_size
from get_style_attributes import get_style_attributes_path

filter_id = 0

def animate_logo(model_output: pd.DataFrame, logo_path: str):
    width, height = get_svg_size(logo_path)
    # ---- Normalize model output ----
    animations_by_id = defaultdict(list)
    for row in model_output.iterrows():
        # Structure animations by animation id
        animation_id = row[1]['animation_id']
        output = row[1]['model_output']
        animations_by_id[animation_id].append(output)
    total_animations = []
    for animation_id in animations_by_id.keys():
        # Structure animations by type (check first 7 parameters)
        animations_by_type = defaultdict(list)
        for animation in animations_by_id[animation_id]:
            animation_type = animation[:8].index(1)
            try:
                animation_type = animation[:8].index(1)
                animations_by_type[animation_type].append(animation)
            except:
                # No value found
                print('Model output invalid: no animation type found')
                return
        # Normalize begin and duration per type
        for animation_type in animations_by_type.keys():
            # Set up list of animations for later distribution
            current_animations = []
            # Sort animations by begin
            animations_by_type[animation_type].sort(key=lambda l : l[8]) # TODO Begin is index 7???
            # For every animation, check consistency of begin and duration, then set parameters
            for i in range(len(animations_by_type[animation_type])):
                # Check if begin is equal to next animation's begin - in this case, set second begin to average of first and third animation
                # Get next animation with different begin time
                if len(animations_by_type[animation_type]) > 1:
                    j = 1
                    next_animation = animations_by_type[animation_type][j]
                    while (i + j) < len(animations_by_type[animation_type]) and animations_by_type[animation_type][i][8] == next_animation[8]:
                        j += 1
                        next_animation = animations_by_type[animation_type][j]
                    if j != 1:
                        # Get difference
                        difference = animations_by_type[animation_type][j][8] - animations_by_type[animation_type][i][8]
                        interval = difference / (j - i)
                        factor = 0
                        for a in range(i, j):
                            animations_by_type[animation_type][a][8] = interval * factor
                            factor += 1
                    # Check if duration and begin of next animation are consistent - if not, shorten duration
                    if i < len(animations_by_type[animation_type]) - 1:
                        max_duration = animations_by_type[animation_type][i+1][8] - animations_by_type[animation_type][i][8]
                        if animations_by_type[animation_type][i][9] > max_duration:
                            animations_by_type[animation_type][i][9] = max_duration
                # General parameters
                begin = animations_by_type[animation_type][i][8]
                dur = animations_by_type[animation_type][i][9]
                # Check type and call method
                if animation_type == 0:
                    # animation: translate
                    # For every value, check if it is in svg width/height TODO!
                    from_x = animations_by_type[animation_type][i][10]
                    from_y = animations_by_type[animation_type][i][11]
                    # Check if there is a next translate animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next translate animation's starting point
                        to_x = animations_by_type[animation_type][i+1][10]
                        to_y = animations_by_type[animation_type][i+1][11]
                    else:
                        # animation endpoint is final position of object
                        to_x = 0
                        to_y = 0
                    current_animations.append(_animation_translate(animation_id, begin, dur, from_x, from_y, to_x, to_y))
                elif animation_type == 1:
                    # animation: scale
                    from_f = animations_by_type[animation_type][i][12]
                    # Check if there is a next translate animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next translate animation's starting point
                        to_f = animations_by_type[animation_type][i+1][12]
                    else:
                        # animation endpoint is final position of object
                        to_f = 1
                    current_animations.append(_animation_scale(animation_id, begin, dur, from_f, to_f))
                elif animation_type == 2:
                    # animation: rotate
                    from_degree = animations_by_type[animation_type][i][13]
                    # Check if there is a next scale animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next scale animation's starting point
                        to_degree = animations_by_type[animation_type][i+1][13]
                    else:
                        # animation endpoint is final position of object
                        to_degree = 360
                    current_animations.append(_animation_rotate(animation_id, begin, dur, from_degree, to_degree))
                elif animation_type == 3:
                    # animation: skewX
                    from_x = animations_by_type[animation_type][i][14]
                    # Check if there is a next skewX animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next skewX animation's starting point
                        to_x = animations_by_type[animation_type][i+1][14]
                    else:
                        # animation endpoint is final position of object
                        to_x = 1
                    current_animations.append(_animation_skewX(animation_id, begin, dur, from_x, to_x))
                elif animation_type == 4:
                    # animation: skewY
                    from_y = animations_by_type[animation_type][i][15]
                    # Check if there is a next skewY animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next skewY animation's starting point
                        to_y = animations_by_type[animation_type][i+1][15]
                    else:
                        # animation endpoint is final position of object
                        to_y = 1
                    current_animations.append(_animation_skewY(animation_id, begin, dur, from_y, to_y))
                elif animation_type == 5:
                    # animation: fill
                    from_rgb = '#' + _convert_to_hex_str(animations_by_type[animation_type][i][16]) + _convert_to_hex_str(animations_by_type[animation_type][i][17]) + _convert_to_hex_str(animations_by_type[animation_type][i][18])
                    # Check if there is a next fill animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next fill animation's starting point
                        to_rgb = '#' + _convert_to_hex_str(animations_by_type[animation_type][i+1][16]) + _convert_to_hex_str(animations_by_type[animation_type][i+1][17]) + _convert_to_hex_str(animations_by_type[animation_type][i+1][18])
                    else:
                        fill_style = get_style_attributes_path(logo_path, animation_id, "fill")
                        stroke_style = get_style_attributes_path(logo_path, animation_id, "stroke")
                        if fill_style == "none" and stroke_style != "none":
                            color_hex = stroke_style
                        else:
                            color_hex = fill_style
                        to_rgb = color_hex
                    current_animations.append(_animation_fill(animation_id, begin, dur, from_rgb, to_rgb))
                elif animation_type == 6:
                    # animation: opacity
                    from_f = animations_by_type[animation_type][i][19] / 100 # percent
                    # Check if there is a next opacity animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next opacity animation's starting point
                        to_f = animations_by_type[animation_type][i+1][19] / 100 # percent
                    else:
                        # animation endpoint is final position of object
                        to_f = 1
                    current_animations.append(_animation_opacity(animation_id, begin, dur, from_f, to_f))
                elif animation_type == 7:
                    # animation: blur
                    from_f = animations_by_type[animation_type][i][20]
                    # Check if there is a next blur animation
                    if i < len(animations_by_type[animation_type]) - 1:
                        # animation endpoint is next blur animation's starting point
                        to_f = animations_by_type[animation_type][i+1][20]
                    else:
                        # animation endpoint is final position of object
                        to_f = 1
                    current_animations.append(_animation_blur(animation_id, begin, dur, from_f, to_f))
            total_animations += current_animations
    _insert_animations(total_animations, logo_path, logo_path)

def _convert_to_hex_str(i: int):
    h = str(hex(i))[2:]
    if i < 16:
        h = '0' + h
    return h

        
def _animation_translate(animation_id: int, begin: float, dur: float, from_x: int, from_y: int, to_x: int, to_y: int):
    print('animation: translate')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'translate'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = f'{from_x} {from_y}'
    animation_dict['to'] = f'{to_x} {to_y}'
    return animation_dict

def _animation_scale(animation_id: int, begin: float, dur: float, from_f: float, to_f: float):
    print('animation: scale')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'scale'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = str(from_f)
    animation_dict['to'] = str(to_f)
    return animation_dict

def _animation_rotate(animation_id: int, begin: float, dur: float, from_degree: int, to_degree: int):
    print('animation: rotate')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'rotate'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = f'{from_degree}' #TODO maybe necessary to define midpoint!
    animation_dict['to'] = f'{to_degree}'
    return animation_dict

def _animation_skewX(animation_id: int, begin: float, dur: float, from_i: int, to_i: int):
    print('animation: skew')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'skewX'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = f'{from_i}'
    animation_dict['to'] = f'{to_i}'
    return animation_dict

def _animation_skewY(animation_id: int, begin: float, dur: float, from_i: int, to_i: int):
    print('animation: skew')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'skewY'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = f'{from_i}'
    animation_dict['to'] = f'{to_i}'
    return animation_dict

def _animation_fill(animation_id: int, begin: float, dur: float, from_rgb: str, to_rgb: str):
    print('animation: fill')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'fill'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = from_rgb
    animation_dict['to'] = to_rgb
    return animation_dict

def _animation_opacity(animation_id: int, begin: float, dur: float, from_f: float, to_f: float):
    print('animation: opacity')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_transform'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'opacity'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = str(from_f)
    animation_dict['to'] = str(to_f)
    return animation_dict

def _animation_blur(animation_id: int, begin: float, dur: float, from_f: float, to_f: float):
    print('animation: blur')
    animation_dict = {}
    animation_dict['animation_id'] = animation_id
    animation_dict['animation_type'] = 'animate_filter'
    animation_dict['attributeName'] = 'transform'
    animation_dict['attributeType'] = 'XML'
    animation_dict['type'] = 'blur'
    animation_dict['begin'] = str(begin)
    animation_dict['dur'] = str(dur)
    animation_dict['fill'] = 'freeze'
    animation_dict['from'] = str(from_f)
    animation_dict['to'] = str(to_f)
    return animation_dict

def _insert_animations(animations: list, path: str, target_path: str):
    print('Insert animations')
    # Load XML
    document = minidom.parse(path)
    # Collect all elements
    elements = document.getElementsByTagName('path') + document.getElementsByTagName('circle') + document.getElementsByTagName(
        'ellipse') + document.getElementsByTagName('line') + document.getElementsByTagName(
        'polygon') + document.getElementsByTagName('polyline') + document.getElementsByTagName(
        'rect') + document.getElementsByTagName('text')
    # Create statement
    for animation in animations:
        
        # Search for element
        current_element = None
        for element in elements:
            if element.getAttribute('animation_id') == str(animation['animation_id']):
                current_element = element
        if current_element == None:
            # Animation id not found - take next animation
            continue
        if animation['animation_type'] == 'animate_transform':
            animate_statement = _create_animate_transform_statement(animation)
            current_element.appendChild(document.createElement(animate_statement))
        elif animation['animation_type'] == 'animate':
            animate_statement = _create_animate_statement(animation)
            current_element.appendChild(document.createElement(animate_statement))
        elif animation['animation_type'] == 'animate_filter':
            filter_element, fe_element, animate_statement = _create_animate_filter_statement(animation, document)
            defs = document.getElementsByTagName('defs')
            current_defs = None
            # Check if defs tag exists; create otherwise
            if len(defs) == 0:
                svg = document.getElementsByTagName('svg')[0]
                current_defs = document.createElement('defs')
                svg.appendChild(current_defs)
            else:
                current_defs = defs[0]
            # Check if filter to be appended
            if filter_element != None:
                # Create filter
                print('append filter')
                current_defs.appendChild(filter_element)
            # Check if FE to be created
            if fe_element != None:
                print('create fe statement')
                # Check if filter set; else search
                if filter_element == None:
                    # Search for filter
                    id = 'filter_' + str(animation['animation_id'])
                    for f in document.getElementsByTagName('filter'):
                        if f.getAttribute('id') == id:
                            filter_element = f
                # Create FE
                filter_element.appendChild(fe_element)
            current_defs.appendChild(document.createElement(animate_statement))
            current_element.setAttribute('filter', f'url(#filter_{animation["animation_id"]})')

    # Save XML to target path
    with open(target_path, 'wb') as f:
        f.write(document.toprettyxml(encoding="iso-8859-1"))
        


def _create_animate_transform_statement(animation_dict: dict):
    """ Set up animation statement from model output for ANIMATETRANSFORM animations 
        (Adapted from AnimateSVG)
    """
    animation = f'animateTransform attributeName="transform" attributeType="XML" ' \
                f'type="{animation_dict["type"]}" ' \
                f'begin="{str(animation_dict["begin"])}" ' \
                f'dur="{str(animation_dict["dur"])}" ' \
                f'from="{str(animation_dict["from"])}" ' \
                f'to="{str(animation_dict["to"])}" ' \
                f'fill="{str(animation_dict["fill"])}"'

    return animation

def _create_animate_statement(animation_dict: dict):
    """ Set up animation statement from model output for ANIMATE animations 
        (adapted from AnimateSVG)
    """
    animation = f'animate attributeName="{animation_dict["type"]}" ' \
                f'begin="{str(animation_dict["begin"])}" ' \
                f'dur="{str(animation_dict["dur"])}" ' \
                f'from="{str(animation_dict["from"])}" ' \
                f'to="{str(animation_dict["to"])}" ' \
                f'fill="{str(animation_dict["fill"])}"'

    return animation

def _create_animate_filter_statement(animation_dict: dict, document: minidom.Document):
    global filter_id
    filter_id += 1
    filter_element = None
    fe_element = None
    animate_statement = None
    if animation_dict['type'] == 'blur':
        # Check if filter already exists
        filters = document.getElementsByTagName('filter')
        current_filter = None
        current_fe = None
        for f in filters:
            #print(f.getAttribute('id') == f'filter_{str(animation_dict["animation_id"])}')
            if f.getAttribute('id') == f'filter_{str(animation_dict["animation_id"])}':
                current_filter = f
        fe_elements = document.getElementsByTagName('feGaussianBlur')
        for fe in fe_elements:
            if fe.getAttribute('id') == f'filter_blur_{str(animation_dict["animation_id"])}':
                current_fe = fe
        if current_filter == None:
            filter_element = document.createElement('filter')
            filter_element.setAttribute('id', f'filter_{str(animation_dict["animation_id"])}')
        if current_fe == None:
            fe_element = document.createElement('feGaussianBlur')
            fe_element.setAttribute('id', f'filter_blur_{str(animation_dict["animation_id"])}')
            fe_element.setAttribute('stdDeviation', '0')
        animate_statement = f'animate href="#filter_blur_{str(animation_dict["animation_id"])}" ' \
                f'attributeName="stdDeviation" ' \
                f'begin="{str(animation_dict["begin"])}" ' \
                f'dur="{str(animation_dict["dur"])}" ' \
                f'from="{str(animation_dict["from"])}" ' \
                f'to="{str(animation_dict["to"])}" ' \
                f'fill="{str(animation_dict["fill"])}"'
    return filter_element, fe_element, animate_statement








def randomly_animate_logo(number_of_animations: int, previously_generated: pd.DataFrame = None):
    # Creates model output equal to defined number of animations. They are then randomly distributed over the paths.
    print('Random animations')

model_output = [
    {
        'animation_id': 1,
        'model_output': [0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    },
    {
        'animation_id': 1,
        'model_output': [0, 0, 0, 0, 0, 0, 0, 1, 5, 3, 4, 5, 2, 1, 2, 3, 4, 5, 6, 7, 1000, 20]
    }
]
model_output = pd.DataFrame(model_output)
#print(model_output)
path = 'src/postprocessing/logo_0.svg'
# Assign animation id to every path - TODO this changes the original logo!
document = minidom.parse(path)
paths = document.getElementsByTagName('path')
for i in range(len(paths)):
    paths[i].setAttribute('animation_id', str(i))
with open(path, 'wb') as svg_file:
    svg_file.write(document.toxml(encoding='iso-8859-1'))
#print('Inserted animation id')
animate_logo(model_output, path)
