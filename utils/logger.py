"""
Simplified Training Logger
"""
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class SimpleLogger:
    """Simple training logger"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.metrics = defaultdict(list)
        self.counts = defaultdict(int)
    
    def add(self, name, value, count=1):
        """Add metric"""
        self.metrics[name].append(value * count)
        self.counts[name] += count
    
    def dump(self, epoch):
        """Log and clear metrics"""
        for name, values in self.metrics.items():
            avg_value = sum(values) / self.counts[name]
            self.writer.add_scalar(name, avg_value, epoch)
            print(f"  {name}: {avg_value:.4f}")
        
        # Clear
        self.metrics.clear()
        self.counts.clear()
    
    def close(self):
        """Close writer"""
        self.writer.close()
