"""
api.py — Flask REST API for deepfake detection.

Endpoints:
    POST /predict/image  — Upload image → JSON (label + confidence)
    POST /predict/video  — Upload video → JSON (label + confidence + per-frame)
    GET  /health         — Health check

Usage:
    pip install flask
    python scripts/api.py --config configs/config.yaml

Example curl:
    curl -X POST -F "file=@test.jpg" http://localhost:5000/predict/image
    curl -X POST -F "file=@test.mp4" http://localhost:5000/predict/video
"""

import os
import sys
import argparse
import logging
import tempfile

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, get_device
from src.data.transforms import get_inference_transforms
from src.models.classifier import DualInputDeepfakeClassifier
from src.utils.face_detect import FaceDetector
from src.data.video_loader import VideoFrameExtractor

logger = logging.getLogger(__name__)

# ---- Global model objects (loaded once at startup) ----
model = None
transform = None
face_detector = None
device = None
config = None


def load_model(config_path: str, checkpoint_path: str = None):
    """Load model and transforms into global state."""
    global model, transform, face_detector, device, config

    config = load_config(config_path)
    setup_logging()
    device = get_device(config['device'])

    model_config = config['model']
    dual_input = model_config.get('dual_input', True)
    pooling_mode = model_config.get('pooling_mode', 'multi')

    # ---- Load model ----
    model = DualInputDeepfakeClassifier(
        dino_variant=model_config['dino_variant'],
        hidden_dims=model_config['classifier']['hidden_dims'],
        num_classes=model_config['classifier']['num_classes'],
        dropout=model_config['classifier']['dropout'],
        freeze_backbone=model_config['freeze_backbone'],
        unfreeze_last_n_blocks=model_config['unfreeze_last_n_blocks'],
        pooling_mode=pooling_mode,
        dual_input=dual_input,
    )

    ckpt_path = checkpoint_path or os.path.join(
        config['paths']['checkpoint_dir'], 'best_model.pth'
    )
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=str(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}, using untrained model")

    model.to(device)
    model.eval()

    # ---- Transform ----
    transform = get_inference_transforms(config['data']['image_size'])

    # ---- Face detector ----
    if config['face_detection']['enabled']:
        face_config = config['face_detection']
        face_detector = FaceDetector(
            margin=face_config['margin'],
            confidence_threshold=face_config['confidence_threshold'],
            image_size=config['data']['image_size'],
        )

    logger.info("API model loaded and ready")


def predict_image_api(image_path: str) -> dict:
    """Run prediction on a single image."""
    full_image = Image.open(image_path).convert('RGB')

    dual_input = config['model'].get('dual_input', True)

    # Face crop
    face_crop = None
    if dual_input:
        if face_detector:
            face_crop = face_detector.detect_and_crop(full_image)
        if face_crop is None:
            face_crop = full_image.copy()

    # Preprocess
    full_tensor = transform(full_image).unsqueeze(0).to(device)
    face_tensor = None
    if face_crop:
        face_tensor = transform(face_crop).unsqueeze(0).to(device)

    # Predict
    result = model.predict(full_tensor, face_tensor)

    return {
        'label': result['labels'][0],
        'confidence': round(result['confidence'][0].item(), 4),
        'probability_real': round(result['prob_real'][0].item(), 4),
        'probability_fake': round(result['prob_fake'][0].item(), 4),
    }


def predict_video_api(video_path: str, num_frames: int = 16) -> dict:
    """Run prediction on a video."""
    import numpy as np

    dual_input = config['model'].get('dual_input', True)

    extractor = VideoFrameExtractor(num_frames=num_frames)
    pil_frames = extractor.extract_to_pil(video_path)

    if not pil_frames:
        return {'error': 'No frames extracted'}

    all_probs = []
    all_preds = []
    frame_results = []

    model.eval()
    with torch.no_grad():
        for i, frame in enumerate(pil_frames):
            face_crop = None
            if dual_input:
                if face_detector:
                    face_crop = face_detector.detect_and_crop(frame)
                if face_crop is None:
                    face_crop = frame.copy()

            full_tensor = transform(frame).unsqueeze(0).to(device)
            face_tensor = None
            if face_crop:
                face_tensor = transform(face_crop).unsqueeze(0).to(device)

            logits = model(full_tensor, face_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            prob_r = probs[0][0].item()
            prob_f = probs[0][1].item()

            all_probs.append([prob_r, prob_f])
            all_preds.append(pred.item())
            frame_results.append({
                'frame': i,
                'label': 'REAL' if pred.item() == 0 else 'FAKE',
                'probability_real': round(prob_r, 4),
                'probability_fake': round(prob_f, 4),
            })

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    avg_probs = all_probs.mean(axis=0)
    final_label = 'REAL' if avg_probs[0] > avg_probs[1] else 'FAKE'

    return {
        'label': final_label,
        'confidence': round(float(max(avg_probs)), 4),
        'probability_real': round(float(avg_probs[0]), 4),
        'probability_fake': round(float(avg_probs[1]), 4),
        'total_frames': len(pil_frames),
        'frames_real': int((all_preds == 0).sum()),
        'frames_fake': int((all_preds == 1).sum()),
        'frame_results': frame_results,
    }


def create_app():
    """Create and configure the Flask application."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model': config['model']['dino_variant'],
            'dual_input': config['model'].get('dual_input', True),
            'pooling': config['model'].get('pooling_mode', 'multi'),
        })

    @app.route('/predict/image', methods=['POST'])
    def predict_image_endpoint():
        """
        Predict deepfake on an uploaded image.

        Request: POST with file upload (key: 'file')
        Response: JSON with label, confidence, probabilities
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Save to temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = predict_image_api(tmp_path)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.unlink(tmp_path)

    @app.route('/predict/video', methods=['POST'])
    def predict_video_endpoint():
        """
        Predict deepfake on an uploaded video.

        Request: POST with file upload (key: 'file')
        Response: JSON with label, confidence, per-frame results
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            num_frames = request.form.get('num_frames', 16, type=int)
            result = predict_video_api(tmp_path, num_frames=num_frames)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.unlink(tmp_path)

    return app


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Deepfake Detection API")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--host', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()

    # Load model
    load_model(args.config, args.checkpoint)

    # Create Flask app
    app = create_app()

    # Run server
    api_config = config.get('api', {})
    host = args.host or api_config.get('host', '0.0.0.0')
    port = args.port or api_config.get('port', 5000)

    print(f"\n{'='*50}")
    print(f"  DINOv2 Deepfake Detection API")
    print(f"  Running on http://{host}:{port}")
    print(f"{'='*50}")
    print(f"  Endpoints:")
    print(f"    POST /predict/image  — Image upload")
    print(f"    POST /predict/video  — Video upload")
    print(f"    GET  /health         — Health check")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
