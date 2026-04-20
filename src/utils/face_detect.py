"""
Face Detection utility using MTCNN (Multi-task Cascaded Convolutional Networks).

Used to crop face regions from images before feeding to DINOv2.
This improves detection accuracy by focusing on facial features.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    MTCNN-based face detector for preprocessing deepfake images.

    Usage:
        detector = FaceDetector(margin=40, confidence=0.9)
        face_img = detector.detect_and_crop("image.jpg")
    """

    def __init__(
        self,
        margin: int = 40,
        confidence_threshold: float = 0.9,
        image_size: int = 224,
        device: str = 'cuda',
    ):
        """
        Args:
            margin: Pixel margin to add around detected face bounding box.
            confidence_threshold: Minimum confidence for a valid face detection.
            image_size: Output size for cropped face image.
            device: Device for MTCNN model.
        """
        self.margin = margin
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        self._mtcnn_unavailable_warned = False
        # Always use CPU — CUDA MTCNN breaks in DataLoader worker subprocesses (fork + CUDA)
        self._init_device = 'cpu'
        self._init_params = dict(image_size=image_size, margin=margin)
        self.mtcnn = self._try_init_mtcnn()
        if self.mtcnn is not None:
            logger.info("MTCNN face detector initialized successfully (CPU)")

    def _try_init_mtcnn(self):
        """Initialize MTCNN on CPU (safe in DataLoader worker subprocesses)."""
        try:
            from facenet_pytorch import MTCNN
            return MTCNN(
                image_size=self._init_params['image_size'],
                margin=self._init_params['margin'],
                keep_all=False,
                post_process=False,
                device=self._init_device,
            )
        except ImportError:
            logger.error("facenet-pytorch not installed. Run: pip install facenet-pytorch")
            return None
        except Exception as e:
            logger.error(f"MTCNN init failed: {type(e).__name__}: {e}")
            return None

    def detect_and_crop(
        self,
        image,
        return_bbox: bool = False,
    ) -> Optional[Image.Image]:
        """
        Detect and crop the largest face from an image.

        Args:
            image: PIL Image or path to image file.
            return_bbox: If True, also return the bounding box.

        Returns:
            Cropped face as PIL Image, or None if no face detected.
            If return_bbox=True, returns (face_image, bbox) tuple.
        """
        if self.mtcnn is None:
            # Lazy re-init for DataLoader worker processes (CUDA broken after fork → CPU fallback)
            self.mtcnn = self._try_init_mtcnn()
        if self.mtcnn is None:
            if not self._mtcnn_unavailable_warned:
                logger.warning("MTCNN unavailable — face crops disabled. Run: pip install facenet-pytorch")
                self._mtcnn_unavailable_warned = True
            return (None, None) if return_bbox else None

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Detect faces
        boxes, probs = self.mtcnn.detect(image)

        if boxes is None or len(boxes) == 0:
            logger.debug("No face detected, returning None")
            if return_bbox:
                return None, None
            return None

        # Get the highest confidence detection
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]

        if best_prob < self.confidence_threshold:
            logger.debug(
                f"Face confidence ({best_prob:.2f}) below threshold "
                f"({self.confidence_threshold}), returning None"
            )
            if return_bbox:
                return None, None
            return None

        # Crop face with margin
        box = boxes[best_idx]
        face_crop = self._crop_face(image, box)

        logger.debug(f"Face detected with confidence {best_prob:.2f}")

        if return_bbox:
            return face_crop, box
        return face_crop

    def _crop_face(self, image: Image.Image, box: np.ndarray) -> Image.Image:
        """
        Crop face region from image with margin.

        Args:
            image: Source PIL Image.
            box: Bounding box [x1, y1, x2, y2].

        Returns:
            Cropped face as PIL Image.
        """
        w, h = image.size
        x1, y1, x2, y2 = [int(b) for b in box]

        # Add margin
        x1 = max(0, x1 - self.margin)
        y1 = max(0, y1 - self.margin)
        x2 = min(w, x2 + self.margin)
        y2 = min(h, y2 + self.margin)

        face = image.crop((x1, y1, x2, y2))
        face = face.resize((self.image_size, self.image_size), Image.LANCZOS)

        return face

    def detect_batch(self, images: List[Image.Image]) -> List[Optional[Image.Image]]:
        """
        Detect and crop faces from a batch of images.

        Args:
            images: List of PIL Images.

        Returns:
            List of cropped face images (or originals if no face detected).
        """
        results = []
        for img in images:
            face = self.detect_and_crop(img)
            results.append(face)
        return results
