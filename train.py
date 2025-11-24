"""
Florence-2 Training Script
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, get_scheduler, AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from tqdm import tqdm

from utils.dataset import FLDataset
from utils.logger import SimpleLogger


# ==================== Configuration ====================

# Model paths
BASE_MODEL_PATH = '/sdb1_hdisk/pub_data/MODELS/Florence-2-large-ft/'
PEFT_MODEL_PATH = None  # None if training from scratch
MODEL_REVISION = 'refs/pr/6'

# Data paths
JSONL_PATHS = [
    '/sdb1_hdisk/pub_data/chenhong/Florence2/new_data/Data1_train.jsonl',
    '/sdb1_hdisk/pub_data/chenhong/Florence2/new_data/Data2_train.jsonl',
    '/sdb1_hdisk/pub_data/chenhong/Florence2/new_data/Data3.1_train.jsonl',
    '/sdb1_hdisk/pub_data/chenhong/Florence2/new_data/Data4_train.jsonl',
    '/sdb1_hdisk/pub_data/chenhong/Florence2/new_data/Data5__train.jsonl',
    '/sdb1_hdisk/pub_data/chenhong/Florence2/new_data/Data6_OP_train.jsonl',
]

# Task types
TASKS = ["<OD>", "<POINT_COUNT>","<KEYPOINT>","<REGION_TO_SEGMENTATION>"]

# Training parameters
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-6
NUM_WORKERS = 0

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"]

# Save parameters
EXP_NAME = "OD_POINT_ONLY"
OUTPUT_DIR = f"results/{EXP_NAME}"
LOG_DIR = f"logs/{EXP_NAME}"
SAVE_INTERVAL = 10  # Save every 10 epochs

# Resume training
RESUME_EPOCH = 0  # Resume from which epoch, 0 means start from scratch

# GPU settings
GPU_IDS = "0,1,2,3"


# ==================== Utility Functions ====================

def setup_environment():
    """Setup environment variables"""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def create_dataloader(dataset, processor, batch_size=2, num_workers=0, shuffle=False):
    """Create DataLoader"""
    def collate_fn(batch):
        questions, answers, bboxes, images = zip(*batch)
        return list(questions), list(answers), list(bboxes), list(images)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        shuffle=shuffle
    )


def load_model():
    """Load model and processor"""
    print("Loading model...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True, 
        revision=MODEL_REVISION,
        device_map="cpu"
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True, 
        revision=MODEL_REVISION
    )
    
    # Load or create LoRA model
    peft_path = PEFT_MODEL_PATH
    if RESUME_EPOCH > 0:
        peft_path = os.path.join(OUTPUT_DIR, f"epoch_{RESUME_EPOCH}")
    
    if peft_path:
        print(f"Loading LoRA weights from {peft_path}")
        model = PeftModel.from_pretrained(base_model, peft_path, is_trainable=True)
    else:
        print("Creating new LoRA model...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=LORA_DROPOUT,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
            revision=MODEL_REVISION
        )
        model = get_peft_model(base_model, lora_config)
    
    model.print_trainable_parameters()
    return model, processor


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, processor, accelerator, logger, epoch):
    """Train one epoch"""
    model.train()
    train_loss = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}",
        disable=not accelerator.is_main_process
    )
    
    for questions, answers, bboxes, images in progress_bar:
        batch_size = len(answers)
        
        # Process inputs
        inputs = processor(
            text=questions,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(accelerator.device)
        
        # Process labels
        labels = processor.tokenizer(
            text=answers,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        ).input_ids.to(accelerator.device)
        
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            labels=labels
        )
        loss = outputs.loss
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Log loss
        train_loss += loss.item()
        logger.add('loss', loss.item(), batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = train_loss / len(train_loader)
    return avg_loss


def save_checkpoint(model, processor, optimizer, lr_scheduler, output_dir):
    """Save checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    print(f"✓ Checkpoint saved to {output_dir}")


def load_checkpoint(optimizer, lr_scheduler, checkpoint_dir):
    """Load checkpoint"""
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
    
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"✓ Loaded optimizer state from {optimizer_path}")
    else:
        print(f"⚠ Optimizer state not found in {checkpoint_dir}")
    
    if os.path.exists(scheduler_path):
        lr_scheduler.load_state_dict(torch.load(scheduler_path))
        print(f"✓ Loaded scheduler state from {scheduler_path}")
    else:
        print(f"⚠ Scheduler state not found in {checkpoint_dir}")


# ==================== Main Training Function ====================

def main():
    # Setup environment
    setup_environment()
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Load model
    model, processor = load_model()
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = FLDataset(
        jsonl_paths=JSONL_PATHS,
        task=TASKS
    )
    train_loader = create_dataloader(
        train_dataset,
        processor,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    # Resume optimizer state
    if RESUME_EPOCH > 0:
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"epoch_{RESUME_EPOCH}")
        load_checkpoint(optimizer, lr_scheduler, checkpoint_dir)
    
    # Accelerate preparation
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    
    # Initialize Logger
    logger = SimpleLogger(log_dir=LOG_DIR)
    
    # Print training info
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print(f"Experiment: {EXP_NAME}")
        print(f"Tasks: {TASKS}")
        print(f"Total epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Output dir: {OUTPUT_DIR}")
        print("="*60 + "\n")
    
    # Training loop
    print("Start training...")
    for epoch in range(RESUME_EPOCH, EPOCHS):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler,
            processor, accelerator, logger, epoch
        )
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
            logger.dump(epoch)
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % SAVE_INTERVAL == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
            save_checkpoint(unwrapped_model, processor, optimizer, lr_scheduler, output_dir)
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_dir = os.path.join(OUTPUT_DIR, "final")
        save_checkpoint(unwrapped_model, processor, optimizer, lr_scheduler, final_dir)
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Final model saved to {final_dir}")
        print("="*60)
    
    logger.close()


if __name__ == "__main__":
    main()
