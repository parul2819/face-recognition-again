import cv2
import numpy as np
from insightface.app import FaceAnalysis

_app = FaceAnalysis(name="buffalo_l")
_app.prepare(ctx_id=-1)


def get_face_embeddings_from_array(img: np.ndarray) -> list[dict]:
    """
    Core detection logic. Takes an already-loaded image (as a numpy array,
    OpenCV BGR format) and returns a list of detected faces with their
    embedding, bbox, and confidence. Used both when reading from disk
    (get_face_embeddings) and when reading an in-memory upload (the API).
    """
    faces = _app.get(img)

    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        results.append(
            {
                "embedding": face.embedding,
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "confidence": float(face.det_score),
            }
        )
    return results


def get_face_embeddings(image_path: str) -> list[dict]:
    """Convenience wrapper: reads an image from disk, then detects faces."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return get_face_embeddings_from_array(img)
