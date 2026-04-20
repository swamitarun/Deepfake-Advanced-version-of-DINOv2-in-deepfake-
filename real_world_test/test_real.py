import os
import sys
import shutil
import logging
import cv2
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import load_config, get_device
from src.data.transforms import get_val_transforms
from src.models.classifier import DeepfakeClassifier
from src.utils.face_detect import FaceDetector

def extract_frames(video_path, temp_dir, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, (total if total > 0 else 300) // num_frames)
    indices = set(range(0, total if total > 0 else 300, step))
    
    saved = []
    for i in range(total if total > 0 else 300):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            path = os.path.join(temp_dir, f"frame_{len(saved):03d}.jpg")
            Image.fromarray(rgb).save(path)
            saved.append(path)
            if len(saved) >= num_frames: break
    cap.release()
    return saved

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (best_model.pth or best_mlp.pth)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    # Parent directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Specific directories for Real World Testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "tests")
    temp_dir = os.path.join(script_dir, "temp_frames_and_faces")

    logger.info("=" * 60)
    logger.info("  REAL-WORLD VIDEO DEEPFAKE TESTING")
    logger.info("=" * 60)

    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(tests_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        logger.warning(f"No videos found in {tests_dir}")
        return

    config = load_config(os.path.join(base_dir, 'configs/config.yaml'))
    device = get_device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(base_dir, 'models/checkpoints/best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(base_dir, 'models/checkpoints/best_mlp.pth')
            
    if not os.path.exists(checkpoint_path):
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    
    # Dual input handling
    dual_input = config['model']['dual_input']
    is_fast_mlp = checkpoint_path.endswith('best_mlp.pth')
    if is_fast_mlp and 'feat_dim' in checkpoint:
        dual_input = (checkpoint['feat_dim'] == 1536)

    face_detector = FaceDetector(
        margin=config['face_detection']['margin'],
        confidence_threshold=config['face_detection']['confidence_threshold'],
        image_size=config['data']['image_size'],
        device=str(device)
    ) if dual_input else None

    # Load model
    model = DeepfakeClassifier(
        dino_variant=config['model']['dino_variant'],
        freeze_backbone=not is_fast_mlp,  # For full model eval, it doesn't matter since we do eval()
        unfreeze_last_n_blocks=config['model']['unfreeze_last_n_blocks'] if not is_fast_mlp else 0, 
        dual_input=dual_input
    )
    
    if is_fast_mlp:
        model.classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
    model = model.to(device).eval()
    transform = get_val_transforms(config['data']['image_size'])

    for v_name in videos:
        logger.info(f"Analyzing: {v_name}")
        v_temp = os.path.join(temp_dir, v_name.split('.')[0])
        frames_dir, faces_dir = os.path.join(v_temp, "frames"), os.path.join(v_temp, "faces")
        os.makedirs(frames_dir, exist_ok=True)
        if dual_input: os.makedirs(faces_dir, exist_ok=True)

        paths = extract_frames(os.path.join(tests_dir, v_name), frames_dir)
        preds = []

        with torch.no_grad():
            for i, p in enumerate(paths):
                img = Image.open(p).convert('RGB')
                t_img = transform(img).unsqueeze(0).to(device)
                t_face = t_img
                
                if dual_input:
                    face = face_detector.detect_and_crop(p)
                    if face:
                        face.save(os.path.join(faces_dir, f"face_{i:03d}.jpg"))
                        t_face = transform(face).unsqueeze(0).to(device)
                        
                probs = torch.softmax(model(t_img, t_face if dual_input else None), dim=1)
                preds.append(probs[0, 1].item() * 100)

        if preds:
            avg_fake = np.mean(preds)
            logger.info(f"  Result: {'🚨 FAKE' if avg_fake > 50 else '✅ REAL'}")
            logger.info(f"  Score: {avg_fake:.1f}% Fake | {100-avg_fake:.1f}% Real\n")

    shutil.rmtree(temp_dir)
    logger.info("Temporary files deleted. Testing complete!")

if __name__ == "__main__":
    main()
