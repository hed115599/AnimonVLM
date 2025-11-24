"""
Florence-2 Dataset Class
"""
import copy
import json
from PIL import Image
from torch.utils.data import Dataset
from .data_utils import aug


class FLDataset(Dataset):
    """Florence-2 Dataset"""
    
    def __init__(self, jsonl_paths, task=["<OD>", "<POINT_COUNT>"], augment=False):
        """
        Args:
            jsonl_paths: JSONL file path (string or list)
            task: List of task types
            augment: Whether to perform data augmentation
        """
        self.task = task
        self.augment = augment
        
        # Ensure jsonl_paths is a list
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        
        # Read all JSONL files
        self.jsonl_list = []
        for path in jsonl_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data['task'] in task:
                        self.jsonl_list.append(data)
        
        print(f"Loaded {len(self.jsonl_list)} samples for tasks: {task}")
    
    def __len__(self):
        return len(self.jsonl_list)
    
    def __getitem__(self, idx):
        data = copy.deepcopy(self.jsonl_list[idx])
        image = Image.open(data['image']).convert('RGB')
        
        # Data augmentation
        if self.augment:
            image, data = aug(image, data, p=0.5, p_vflip=0.3, p_rotate=0.3, p_color=0.3)
        
        task = data['task']
        bbox = data.get('bbox', None)
        
        # Build question and answer
        if task == "<REGION_TO_SEGMENTATION>":
            question = task + bbox[bbox.find("<"):]
            answer = data['answer']
        elif task in ["<OD>", "<POINT_COUNT>"]:
            question = task
            answer = data['answer']
            bbox = None
        elif task == "<KEYPOINT>":
            question = task + bbox[bbox.find("<"):]
            answer = data['answer']
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return question, answer, bbox, image
