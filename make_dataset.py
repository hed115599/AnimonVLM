"""
Dataset Creation Script
Supports conversion from Labelme and COCO formats to Florence2 format
"""
import os
import json
import cv2
import numpy as np
from tqdm.auto import tqdm

from utils.labelme_utils import LabelmeJSON, shape_to_mask, mask_to_bbox
from utils.data_utils import (
    convert_polygons_2_FLlabel,
    convert_bbox_2_FLbbox,
    convert_point_2_FLpoint
)

# ⚠️ COCO functionality requires pycocotools installation
try:
    from pycocotools.coco import COCO
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("⚠️  pycocotools not installed, COCO conversion unavailable")
    print("   Installation: pip install pycocotools")


# ==================== Utility Functions ====================

class DictList:
    """Dictionary list data structure"""
    def __init__(self):
        self.dict = {}
    
    def add(self, key, item):
        if key in self.dict:
            self.dict[key].append(item)
        else:
            self.dict[key] = [item]


def generate_full_mask(polygons, img_shape):
    """Generate complete mask"""
    full_mask = np.zeros(img_shape[:2], dtype=bool)
    for poly in polygons:
        mask = shape_to_mask(img_shape, poly, shape_type="polygon")
        full_mask = np.logical_or(full_mask, mask)
    return full_mask


def compute_center_from_mask(mask, bbox, ratio=0.9):
    """Calculate weighted center from mask and bbox"""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return None
    
    cx_mask = np.mean(x_coords)
    cy_mask = np.mean(y_coords)
    
    x_min, y_min, x_max, y_max = bbox
    cx_bbox = (x_min + x_max) / 2
    cy_bbox = (y_min + y_max) / 2
    
    cx_new = cx_mask * ratio + cx_bbox * (1 - ratio)
    cy_new = cy_mask * ratio + cy_bbox * (1 - ratio)
    
    return (cx_new, cy_new)


def list_2_jsonl(data_list, filename):
    """Save as JSONL format"""
    filename = filename if filename.endswith('.jsonl') else filename + '.jsonl'
    with open(filename, "w") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")
    print(f"✓ Data saved to {filename}")


# ==================== Labelme Conversion ====================

def make_labelme_dataset(json_files, out_filename='data', epsilon=2):
    """
    Generate Florence2 dataset from Labelme annotation files
    
    Args:
        json_files: JSON file list or folder path
        out_filename: Output filename
        epsilon: Polygon simplification parameter
    """
    # Get file list
    if isinstance(json_files, str) and os.path.isdir(json_files):
        # ✅ Fix: Recursively find all .json files
        json_paths = []
        for root, dirs, files in os.walk(json_files):
            for file in files:
                if file.endswith('.json'):
                    json_paths.append(os.path.join(root, file))
        print(f"📂 Found {len(json_paths)} JSON files in {json_files}")
    else:
        json_paths = json_files if isinstance(json_files, list) else [json_files]
    
    if len(json_paths) == 0:
        print("❌ No JSON files found!")
        return
    
    data_list = []
    skipped = 0
    
    for json_path in tqdm(json_paths, desc="Processing Labelme files"):
        # Read image path
        jpg_path = json_path.replace('.json', '.jpg')
        png_path = json_path.replace('.json', '.png')
        img_path = jpg_path if os.path.exists(jpg_path) else png_path
        
        if not os.path.exists(img_path):
            print(f"⚠ Image not found: {img_path}")
            skipped += 1
            continue
        
        try:
            # Process JSON
            Ljson = LabelmeJSON(json_path)
            Ljson.remove('_fill')
            Ljson.remove_index('sow')
            Ljson.remove('center')
            Ljson.remove_index('piglet', keep_P=True)
            
            labels, img = Ljson.get_mask_dict()
            
            if len(labels) == 0:
                print(f"⚠ {json_path} has no valid annotations")
                skipped += 1
                continue
            
            OD_dict = DictList()
            PD_dict = DictList()
            
            for label in labels:
                FLbbox = convert_bbox_2_FLbbox(label['label'], label['bbox'], imgsize=img.shape[:2])
                FLbbox_wo_name = convert_bbox_2_FLbbox('', label['bbox'], imgsize=img.shape[:2])
                OD_dict.add(label['label'], FLbbox_wo_name)
                
                # Calculate center point
                center_point = compute_center_from_mask(label['mask'], label['bbox'])
                if center_point:
                    FLpoint_wo_name = convert_point_2_FLpoint(center_point, name='', imgsize=img.shape[:2])
                    PD_dict.add(label['label'], FLpoint_wo_name)
                
                # Segmentation task
                FLlabel = convert_polygons_2_FLlabel(label['polygon'], epsilon=epsilon, imgsize=img.shape[:2])
                data_list.append({
                    'image': img_path,
                    'task': '<REGION_TO_SEGMENTATION>',
                    'bbox': FLbbox,
                    'answer': FLlabel
                })
            
            # OD task
            OD = ''.join([k + ''.join(v) for k, v in OD_dict.dict.items()])
            data_list.append({'image': img_path, 'task': '<OD>', 'answer': OD})
            
            # POINT_COUNT task
            PD = ''.join([k + ''.join(v) for k, v in PD_dict.dict.items()])
            data_list.append({'image': img_path, 'task': '<POINT_COUNT>', 'answer': PD})
        
        except Exception as e:
            print(f"❌ Error processing {json_path}: {e}")
            skipped += 1
            continue
    
    # Save
    if len(data_list) > 0:
        list_2_jsonl(data_list, out_filename)
        print(f"✅ Successfully processed {len(json_paths) - skipped} files")
        print(f"⚠️  Skipped {skipped} files")
        print(f"📊 Generated {len(data_list)} training samples")
    else:
        print("❌ No data generated!")


# ==================== COCO Conversion ====================

def make_coco_dataset(json_file, img_folder, out_filename='data', epsilon=2):
    """
    Generate Florence2 dataset from COCO format
    
    Args:
        json_file: COCO JSON file path
        img_folder: Image folder path
        out_filename: Output filename
        epsilon: Polygon simplification parameter
    """
    if not COCO_AVAILABLE:
        raise ImportError("❌ COCO functionality unavailable, please install: pip install pycocotools")
    
    coco = COCO(json_file)
    
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    keypoints_labels = categories[0].get('keypoints', [])
    
    data_list = []
    images = coco.loadImgs(coco.getImgIds())
    
    for img_dict in tqdm(images, desc="Processing COCO images"):
        image_id = img_dict['id']
        img_path = os.path.join(img_folder, img_dict['file_name'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠ Cannot read image: {img_path}")
            continue
        
        annotation_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(annotation_ids)
        
        OD_dict = DictList()
        PD_dict = DictList()
        
        for ann in annotations:
            category_id = ann.get("category_id")
            category_name = category_id_to_name.get(category_id, "Unknown")
            
            # Bbox
            bbox = ann.get("bbox", [])
            x_min, y_min, width, height = bbox
            bbox_converted = [x_min, y_min, x_min + width, y_min + height]
            
            FLbbox = convert_bbox_2_FLbbox(category_name, bbox_converted, imgsize=img.shape[:2])
            FLbbox_wo_name = convert_bbox_2_FLbbox('', bbox_converted, imgsize=img.shape[:2])
            OD_dict.add(category_name, FLbbox_wo_name)
            
            # Segmentation
            segmentations = ann.get("segmentation", [])
            polygons = [np.array(seg).reshape(-1, 2) for seg in segmentations]
            mask = generate_full_mask(polygons, img_shape=img.shape[:2])
            
            center_point = compute_center_from_mask(mask, bbox_converted)
            if center_point:
                FLpoint_wo_name = convert_point_2_FLpoint(center_point, name='', imgsize=img.shape[:2])
                PD_dict.add(category_name, FLpoint_wo_name)
            
            FLlabel = convert_polygons_2_FLlabel(polygons, epsilon=epsilon, imgsize=img.shape[:2])
            data_list.append({
                'image': img_path,
                'task': '<REGION_TO_SEGMENTATION>',
                'bbox': FLbbox,
                'answer': FLlabel
            })
            
            # Keypoints
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                FLpoint = ''
                for idx, label in enumerate(keypoints_labels):
                    point = keypoints[idx * 3: idx * 3 + 2]
                    visibility = keypoints[idx * 3 + 2]
                    if visibility > 0:
                        FLpoint += convert_point_2_FLpoint(point, name=label, imgsize=img.shape[:2])
                if FLpoint:
                    data_list.append({
                        'image': img_path,
                        'task': '<KEYPOINT>',
                        'bbox': FLbbox,
                        'answer': FLpoint
                    })
        
        # OD and POINT_COUNT
        OD = ''.join([k + ''.join(v) for k, v in OD_dict.dict.items()])
        data_list.append({'image': img_path, 'task': '<OD>', 'answer': OD})
        
        PD = ''.join([k + ''.join(v) for k, v in PD_dict.dict.items()])
        data_list.append({'image': img_path, 'task': '<POINT_COUNT>', 'answer': PD})
    
    list_2_jsonl(data_list, out_filename)


# ==================== Main Function ====================

if __name__ == "__main__":
    # Example: Labelme conversion
    make_labelme_dataset(
        json_files='/sdb1_hdisk/pub_data/DATAS/old_processed',
        out_filename='/sdb1_hdisk/pub_data/chenhong/Florence2/code/train_labelme',
        epsilon=2
    )
    
    # Example: COCO conversion
    # make_coco_dataset(
    #     json_file='/path/to/coco/annotations.json',
    #     img_folder='/path/to/coco/images',
    #     out_filename='data/train_coco',
    #     epsilon=2
    # )
    
    print("Please configure data paths in the code before running")
