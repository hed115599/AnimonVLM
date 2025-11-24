"""
Data augmentation and coordinate conversion utilities
"""
import re
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance


# ==================== Coordinate Conversion ====================

def FL2pixel(x, y, imgsize):
    """Convert Florence coordinates to pixel coordinates"""
    w, h = imgsize
    return int(round(x / 1000 * w)), int(round(y / 1000 * h))


def pixel2FL(x, y, imgsize):
    """Convert pixel coordinates to Florence coordinates"""
    w, h = imgsize
    x_nor = int(round(np.clip(x / w * 1000, 0, 999)))
    y_nor = int(round(np.clip(y / h * 1000, 0, 999)))
    return x_nor, y_nor


# ==================== Polygon Processing ====================

def ensure_clockwise(polygons):
    """Ensure polygon vertices are arranged clockwise"""
    def is_clockwise(points):
        n = len(points)
        area = 0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += (x2 - x1) * (y2 + y1)
        return area < 0
    
    result = []
    for poly in polygons:
        if not is_clockwise(poly):
            poly = poly[::-1]
        result.append(poly)
    return result


def ramer_douglas_peucker(points, epsilon=5):
    """Simplify polygon using RDP algorithm"""
    if len(points) <= 4:
        return points
    
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_len if line_len != 0 else line_vec
    distances = np.abs(np.cross(line_unit_vec, points - start))
    
    max_distance = np.max(distances)
    farthest_idx = np.argmax(distances)
    
    if max_distance > epsilon:
        left = ramer_douglas_peucker(points[:farthest_idx + 1], epsilon)
        right = ramer_douglas_peucker(points[farthest_idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])


# ==================== Florence Format Conversion ====================

def convert_polygons_2_FLlabel(polygons, imgsize=(2160, 3840), epsilon=None):
    """Convert polygons to Florence2 format"""
    h, w = imgsize
    result = ''
    polygons = ensure_clockwise(polygons)
    
    for i, array in enumerate(polygons):
        if epsilon:
            array = ramer_douglas_peucker(array, epsilon)
        for x, y in array:
            x_nor = np.clip(int(x / w * 1000), 0, 999)
            y_nor = np.clip(int(y / h * 1000), 0, 999)
            result += f'<loc_{x_nor}><loc_{y_nor}>'
        if i < len(polygons) - 1:
            result += '<sep>'
    return result


def convert_bbox_2_FLbbox(name, bbox, imgsize=(2160, 3840)):
    """Convert BBox to Florence2 format"""
    h, w = imgsize
    result = name
    for i, coord in enumerate(bbox):
        if i % 2 == 0:  # x coordinate
            normalized = np.clip(int(coord / w * 1000), 0, 999)
        else:  # y coordinate
            normalized = np.clip(int(coord / h * 1000), 0, 999)
        result += f'<loc_{normalized}>'
    return result


def convert_point_2_FLpoint(point, name='', imgsize=(2160, 3840)):
    """Convert point to Florence2 format"""
    h, w = imgsize
    result = name
    for i, coord in enumerate(point):
        if i % 2 == 0:  # x coordinate
            normalized = np.clip(int(coord / w * 1000), 0, 999)
        else:  # y coordinate
            normalized = np.clip(int(coord / h * 1000), 0, 999)
        result += f'<loc_{normalized}>'
    return result


# ==================== Data Augmentation ====================

def augment_image(image, p=0.5, p_vflip=0.3, p_rotate=0.3, p_color=0.3,
                  allow_hflip=True, allow_rotate_90_270=True, only_color=False):
    """Image augmentation"""
    ops = []
    
    if only_color:
        if random.random() < p_color:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            ops.append('color')
        return image, ops
    
    # Horizontal flip
    if allow_hflip and random.random() < p:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        ops.append('hflip')
    
    # Vertical flip
    if random.random() < p_vflip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        ops.append('vflip')
    
    # Rotation
    if allow_rotate_90_270:
        rotate_angle = random.choice([0, 90, 180, 270]) if random.random() < p_rotate else 0
    else:
        rotate_angle = 180 if random.random() < p_rotate else 0
    
    if rotate_angle != 0:
        image = image.rotate(rotate_angle, expand=True)
        ops.append(f'rotate_{rotate_angle}')
    
    # Color enhancement
    if random.random() < p_color:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        ops.append('color')
    
    return image, ops


def aug(image, data, p=0.5, p_vflip=0.3, p_rotate=0.3, p_color=0.3):
    """
    Main data augmentation function
    Simplified version: only performs image augmentation, does not sync labels
    """
    task = data['task']
    
    # KEYPOINT task only performs color enhancement
    if task == "<KEYPOINT>":
        only_color = True
        allow_hflip = False
        allow_rotate_90_270 = False
    else:
        only_color = False
        allow_hflip = True
        allow_rotate_90_270 = True
    
    # Image augmentation
    image, ops = augment_image(
        image, p, p_vflip, p_rotate, p_color,
        allow_hflip=allow_hflip,
        allow_rotate_90_270=allow_rotate_90_270,
        only_color=only_color
    )
    
    return image, data


# ==================== Visualization ====================

import cv2
import numpy as np
import re

def vis_FLlabels(
    img,
    FLlabel: str = None,
    FLbbox: str = None,
    FLpoint: str = None,
    FLkeypoint: str = None,
    FLbehavior: str = None,
    FLocr=None, 
    resize=1.0,  # Default no scaling, maintain original image clarity
    show=True,
    use_matplotlib=True
):
    '''
    Visualize data in <loc_xx> annotation format.
    Supports multiple types: bbox, polygon, point, keypoint, behavior, OCR, etc.
    Returns: Image with annotations drawn (numpy array, BGR format)
    '''
    # 1. Image preprocessing
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Unable to read image path: {img}")
    
    # Make a copy to avoid modifying original image
    img_vis = img.copy()
    h, w = img.shape[:2]

    # ==========================
    # 0. Visualize OCR (if draw_chinese_text function exists)
    # ==========================
    if FLocr:
        if isinstance(FLocr, str):
            ocr_pattern = r"(.*?)((?:<loc_\d+>)+)"
            matches = re.findall(ocr_pattern, FLocr.replace('</s>', '').strip())
            
            for match in matches:
                ocr_text = match[0].strip()
                locs = match[1]
                coords = [int(x) for x in re.findall(r"<loc_(\d+)>", locs)]
                
                if len(coords) >= 4 and len(coords) % 2 == 0: # As long as it's even and >= 4 points
                    polygon = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] / 1000 * w)
                        y = int(coords[i + 1] / 1000 * h)
                        polygon.append([x, y])
                    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                    color = tuple(np.random.randint(128, 255, 3).tolist())
                    cv2.polylines(img_vis, [pts], isClosed=True, color=color, thickness=2)
                    
                    # Try to call Chinese text drawing function, fallback to English if not available
                    try:
                        x0, y0 = polygon[0]
                        # Assume draw_chinese_text is defined externally
                        img_vis = draw_chinese_text(
                            img_vis, ocr_text, (x0, max(y0 - 30, 0)),
                            font_size=20, color=(255,255,255), bg_color=color
                        )
                    except NameError:
                        # If draw_chinese_text doesn't exist, use cv2.putText
                        cv2.putText(img_vis, ocr_text, (polygon[0][0], max(polygon[0][1]-5, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            print("[Warning] OCR format not supported")

    # ==========================
    # 1. Visualize bbox (OD task)
    # ==========================
    if FLbbox:
        pattern = r"([a-zA-Z0-9_\s]+)((?:<loc_\d+>){4,})"
        matches = re.findall(pattern, FLbbox)
        for match in matches:
            name = match[0].strip()
            locs = match[1]
            color = tuple(np.random.randint(0, 255, 3).tolist())
            coords = [int(loc) for loc in re.findall(r"<loc_(\d+)>", locs)]
            
            if len(coords) % 4 != 0:
                continue
                
            for i in range(0, len(coords), 4):
                x1 = int(coords[i] / 1000 * w)
                y1 = int(coords[i + 1] / 1000 * h)
                x2 = int(coords[i + 2] / 1000 * w)
                y2 = int(coords[i + 3] / 1000 * h)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img_vis, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                cv2.putText(img_vis, name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ==========================
    # 2. Visualize polygon (Segmentation)
    # ==========================
    if FLlabel:
        parts = FLlabel.split('<sep>')
        for part in parts:
            color = tuple(np.random.randint(100, 255, 3).tolist())
            locs = re.findall(r'<loc_(\d+)>', part)
            
            if len(locs) < 6 or len(locs) % 2 != 0:
                continue
                
            polygon = []
            for i in range(0, len(locs), 2):
                x = int(int(locs[i]) / 1000 * w)
                y = int(int(locs[i + 1]) / 1000 * h)
                polygon.append([x, y])
            
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_vis, [pts], isClosed=True, color=color, thickness=2)
            # Optional: fill with semi-transparent color (requires additional layer blending)

    # ==========================
    # 3. Visualize Point (point detection)
    # ==========================
    if FLpoint:
        pattern = r"([a-zA-Z0-9_\s]+)((?:<loc_\d+>){2,})"
        matches = re.findall(pattern, FLpoint)
        for match in matches:
            name = match[0].strip()
            locs = match[1]
            color = tuple(np.random.randint(0, 200, 3).tolist()) # Darker color for better visibility
            coords = [int(loc) for loc in re.findall(r"<loc_(\d+)>", locs)]
            
            if len(coords) % 2 != 0:
                continue
                
            for i in range(0, len(coords), 2):
                x = int(coords[i] / 1000 * w)
                y = int(coords[i + 1] / 1000 * h)
                cv2.circle(img_vis, (x, y), radius=5, color=color, thickness=-1)
                cv2.circle(img_vis, (x, y), radius=5, color=(255,255,255), thickness=1) # White border
                cv2.putText(img_vis, name, (x + 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ==========================
    # 4. Visualize Keypoint (keypoint) - Fixed
    # ==========================
    if FLkeypoint:
        # Fixed regex: {2} -> {2,}, allows one label followed by multiple points
        pattern = r"([a-zA-Z0-9_\s]+)((?:<loc_\d+>){2,})"
        matches = re.findall(pattern, FLkeypoint)
        
        keypoint_colors = {}
        color_palette = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (128, 128, 128), (0, 64, 255), (255, 128, 0), (0, 128, 255)
        ]
        color_count = 0
        
        for match in matches:
            name = match[0].strip()
            locs = match[1]
            
            # Assign color
            if name not in keypoint_colors:
                keypoint_colors[name] = color_palette[color_count % len(color_palette)]
                color_count += 1
            color = keypoint_colors[name]
            
            coords = [int(loc) for loc in re.findall(r"<loc_(\d+)>", locs)]
            
            if len(coords) % 2 != 0:
                print(f"[Warning] keypoint coordinates not even: {len(coords)}")
                continue
            
            # Fixed: iterate through all points
            for i in range(0, len(coords), 2):
                x = int(coords[i] / 1000 * w)
                y = int(coords[i + 1] / 1000 * h)
                
                # Draw point
                cv2.circle(img_vis, (x, y), radius=6, color=color, thickness=-1)
                cv2.circle(img_vis, (x, y), radius=2, color=(255,255,255), thickness=-1) # Center white dot
                
                # Draw label (only on first point, or on every point, here we choose every point with smaller font)
                cv2.putText(img_vis, name, (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ==========================
    # 5. Visualize Behavior
    # ==========================
    if FLbehavior:
        pattern = r"([^<]+)((?:<loc_\d+>)+)"
        matches = re.findall(pattern, FLbehavior)
        for match in matches:
            behavior = match[0].strip()
            locs = match[1]
            color = tuple(np.random.randint(0, 255, 3).tolist())
            coords = [int(loc) for loc in re.findall(r"<loc_(\d+)>", locs)]
            
            for i in range(0, len(coords), 4):
                if i + 3 >= len(coords): break
                x1 = int(coords[i] / 1000 * w)
                y1 = int(coords[i + 1] / 1000 * h)
                x2 = int(coords[i + 2] / 1000 * w)
                y2 = int(coords[i + 3] / 1000 * h)
                
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_vis, behavior, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ==========================
    # 6. Resize and display
    # ==========================
    if resize != 1.0:
        img_vis = cv2.resize(img_vis, None, fx=resize, fy=resize) 
    
    if show:
        # Only effective in local environment
        cv2.imshow("Visualization", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return img_vis
