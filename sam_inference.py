import numpy as np
import torch
from transformers import SamModel, SamProcessor

# Global Variables for Preloaded SAM Models
SAM_MODEL = None
SAM_PROCESSOR = None

def preload_models():
    """
    Preload SAM model and processor for faster inference.
    """
    global SAM_MODEL, SAM_PROCESSOR
    device = "cuda"

    # Load SAM model and processor
    SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-base")

    print("SAM models preloaded successfully.")

def mask_to_rgb(mask):
    """
    Transforms a binary mask into an RGBA image for visualization.
    """
    bg_transparent = np.zeros(mask.shape + (4,), dtype=np.uint8)
    bg_transparent[mask == 1] = [0, 255, 0, 127]  # Green color
    
    return bg_transparent

def get_processed_inputs(image, input_points):
    """
    Generate inputs for SAM and process the outputs to get the best mask.
    """
    global SAM_MODEL, SAM_PROCESSOR

    # Prepare inputs for SAM
    inputs = SAM_PROCESSOR(image, input_points=input_points, return_tensors="pt").to("cuda")
    
    # Inference with SAM
    with torch.inference_mode():
        outputs = SAM_MODEL(**inputs)

    # Post-process SAM outputs
    masks = SAM_PROCESSOR.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    # Select the best mask (highest IoU score)
    best_mask = masks[0][0][outputs.iou_scores.argmax()]

    # Invert the mask (subject = 0, background = 1)
    return ~best_mask.cpu().numpy()

# Preload SAM models at startup
preload_models()