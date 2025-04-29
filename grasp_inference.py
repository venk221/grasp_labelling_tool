# inference.py
import numpy as np
import sys
from scipy.ndimage import gaussian_filter
import torch
import cv2
from Datasets.dataset_processing import grasp, image
from Datasets.dataset_processing.grasp import Grasp
from Experts.FullyGenerative.grconvnet.grconvnet import GenerativeResnet
from Experts.GPNN.model import GatedPixelCNN
from Experts.HiFormer.HiFormer import HiFormer
torch.serialization.add_safe_globals([GenerativeResnet, GatedPixelCNN, HiFormer])
import matplotlib.pyplot as plt
sys.path.append('Experts/Generative')
sys.path.insert(0, 'Experts/Residual')
sys.path.insert(0, 'Experts/GPNN')
sys.path.append('Experts/HiFormer')
MODEL_CACHE = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    # Convert to numpy and squeeze
    q_img = q_img.cpu().numpy().squeeze()
    cos_img = cos_img.cpu().numpy().squeeze()
    sin_img = sin_img.cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze()

    # Apply sigmoid to q_img if it looks like logits 
    if q_img.min() < 0 or q_img.max() > 1: 
        q_img = 1 / (1 + np.exp(-q_img))  

    # Normalize q_img to [0, 1] 
    if q_img.max() - q_img.min() > 1e-6:  
        q_img = (q_img - q_img.min()) / (q_img.max() - q_img.min())
    else:
        q_img = np.zeros_like(q_img) 

    # Compute angle
    ang_img = np.arctan2(sin_img, cos_img) / 2.0

    # Scale width_img
    width_img = width_img * 150.0
    # Apply Gaussian filtering 
    q_img = gaussian_filter(q_img, sigma=1.0)  
    ang_img = gaussian_filter(ang_img, sigma=1.0)  
    width_img = gaussian_filter(width_img, sigma=0.5) 

    return q_img, ang_img, width_img

def load_model(model_path, model_name):
    """Loads a specific grasp detection model."""
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]

    model = torch.load(model_path, map_location=device, weights_only=False)

    if model:
        MODEL_CACHE[model_path] = model
        return model
    else:
        print(f"Error: Failed to load model {model_path}")
        return None

def preprocess_image(image_data):
    """Prepares the image data for your specific model."""
    processed_image = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0).float()
    return processed_image

def run_inference(model, processed_image):
    """Runs inference and returns a list of Grasp objects."""
    # print("Running inference...")
    predicted_grasps = []

    processed_image = processed_image.to(device)

    with torch.no_grad():
        outputs = model(processed_image) 
        q_img, ang_img, width_img = post_process_output(
                    outputs[0], outputs[1],
                    outputs[2], outputs[3]
                )
        q_img = cv2.resize(q_img, (1024, 1024))
        ang_img = cv2.resize(ang_img, (1024, 1024))
        width_img = cv2.resize(width_img, (1024, 1024))

        gs = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
        for g in gs:
            predicted_grasps.append(g.as_gr) 

    return predicted_grasps

def predict_grasps_for_image(image_data, model_path, model_type='cornell'):
    grasp_list = []
    """High-level function to load, preprocess, and predict."""
    paths = {
    "GGCNND_Cornell": "/home/venk/Downloads/LabelGrasp/Experts/Generative/pretrained_weights/cornell/depth/ggcnn_epoch_23_cornell",
    "GGCNND_Jacquard": "/home/venk/Downloads/LabelGrasp/Experts/Generative/pretrained_weights/jacquard/depth/epoch_34_iou_0.89",
    "GRCNN_Cornell": "/home/venk/Downloads/LabelGrasp/Experts/Residual/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97",
    "GRCNN_Jacquard": "/home/venk/Downloads/LabelGrasp/Experts/Residual/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93",
    "GPNN_Cornell": "/home/venk/Downloads/LabelGrasp/Experts/GPNN/pretrained_models/cornell/epoch_40_iou_0.76",
    "GPNN_Jacquard": "/home/venk/Downloads/LabelGrasp/Experts/GPNN/pretrained_models/jacquard/epoch_48_iou_0.75"
    }
    for model_name, model_path in paths.items():
        model = load_model(model_path, model_name)
        if not model:
            return []
        
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            print("Error during preprocessing.")
            return []

        # Ensure the model input matches what it was trained on
        if model_name.startswith("GGCNND"):  # depth-only
            if processed_image.shape[1] == 4:
                processed_image = processed_image[:, 0, :, :]  # Take depth only
            elif processed_image.shape[1] == 1:
                pass  # already depth-only
            else:
                print(f"[ERROR] Model {model_name} requires depth, but got shape: {processed_image.shape}")
                return []
        elif model_name.startswith("GRCNN"):  # RGBD required
            if processed_image.shape[1] != 4:
                print(f"[ERROR] Model {model_name} requires RGB+Depth (4 channels), but got shape: {processed_image.shape}")
                return []
        
        grasps = run_inference(model, processed_image)

        grasp_list.extend(grasps)
    # print(f"Total predicted grasps: {(grasp_list)}")   
    return grasp_list



img = cv2.imread('/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/1_rgb.png')
depth_img = cv2.imread('/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/1_depth.tiff', cv2.IMREAD_UNCHANGED)
depth_norm = depth_img.astype('float32')
depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
depth_norm = depth_norm[..., None]  
stacked = np.concatenate([img.astype('float32') / 255.0, depth_norm], axis=-1) 


def visualize_grasps(image, grasps):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(image[..., :3])  
    for g in grasps:
        g.plot(ax, color='g')  
    plt.title("Predicted Grasps")
    plt.axis('off')
    plt.show()

# predicted_grasps = predict_grasps_for_image(stacked, 0, model_type='cornell')
# for g in predicted_grasps:
#     print(g)
# visualize_grasps(img, predicted_grasps)
