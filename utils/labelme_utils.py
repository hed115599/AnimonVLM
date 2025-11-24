"""
Labelme Utility Class (Simplified Version)
"""
import json
import uuid
import numpy as np
import cv2
from PIL import Image
import PIL.ImageDraw
import math


def shape_to_mask(img_shape, points, shape_type='polygon'):
    """Convert shape to mask"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    
    if shape_type == "circle":
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        draw.rectangle(xy, outline=1, fill=1)
    else:  # polygon
        draw.polygon(xy=xy, outline=1, fill=1)
    
    return np.array(mask, dtype=bool)


def mask_to_bbox(mask):
    """Convert mask to bbox"""
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return [0, 0, 0, 0]
    xmin, xmax = cols.min(), cols.max()
    ymin, ymax = rows.min(), rows.max()
    return [xmin, ymin, xmax, ymax]


class LabelmeJSON:
    """Labelme JSON File Processing Class (Simplified Version)"""
    
    def __init__(self, json_path, img_path=None):
        self.json_path = json_path
        self.img_path = json_path.replace('.json', '.jpg') if img_path is None else img_path
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_dict = json.load(f)
        
        self.json_dict['imageData'] = None
        self.img = cv2.imread(self.img_path)
        self.imgsize = self.img.shape[:2]  # (h, w)
    
    def remove(self, with_name, key='label'):
        """Remove shapes containing specified characters"""
        new_shapes = []
        for shape in self.json_dict['shapes']:
            if with_name not in shape[key]:
                new_shapes.append(shape)
        self.json_dict['shapes'] = new_shapes
    
    def remove_index(self, with_name='piglet', keep_P=False):
        """Remove index numbers and unify label names"""
        for shape in self.json_dict['shapes']:
            if with_name in shape['label']:
                if keep_P and ('_P' in shape['label'] or '_p' in shape['label']):
                    shape['label'] = with_name + '_P'
                else:
                    shape['label'] = with_name
    
    def get_mask_dict(self):
        """
        Get mask dictionary list
        Returns: [{'label': 'piglet', 'mask': np.array, 'bbox': [x1,y1,x2,y2], 'polygon': [...], 'occluded': bool}]
        """
        # Only select target shapes
        normal_shapes = [s for s in self.json_dict['shapes'] 
                        if any(name in s['label'] for name in ['sow', 'piglet', 'pig', 'mouse'])]
        
        # Group by group_id
        from collections import defaultdict
        groups = defaultdict(list)
        
        for shape in normal_shapes:
            group_id = shape.get('group_id') or uuid.uuid1()
            key = (shape['label'], group_id)
            
            mask = shape_to_mask(self.imgsize, np.array(shape['points']), shape['shape_type'])
            groups[key].append({
                'mask': mask,
                'polygon': np.array(shape['points'])
            })
        
        # Merge masks in the same group
        result_list = []
        for (label, group_id), items in groups.items():
            # Merge masks
            combined_mask = np.zeros(self.imgsize, dtype=bool)
            polygons = []
            for item in items:
                combined_mask = np.logical_or(combined_mask, item['mask'])
                polygons.append(item['polygon'])
            
            # Determine if occluded
            is_occluded = len(items) > 1 or '_P' in label
            
            # Extract base label name
            base_label = label.split('_')[0] if '_' in label else label
            for name in ['sow', 'piglet', 'pig', 'mouse']:
                if name in base_label:
                    base_label = name
                    break
            
            result_list.append({
                'label': base_label,
                'mask': combined_mask,
                'bbox': mask_to_bbox(combined_mask),
                'polygon': polygons,
                'occluded': is_occluded
            })
        
        return result_list, self.img
