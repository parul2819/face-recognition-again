import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

_app = FaceAnalysis(name="buffalo_l")
_app.prepare(ctx_id=-1)


def _build_single_threaded_face_app() -> FaceAnalysis:
    """
    A second copy of the model, with each onnxruntime session capped to one
    internal thread. Used for bulk uploads, which run several images'
    detection concurrently (see api/controllers/admin_controller.py):
    without the cap, each image's inference tries to use every CPU core on
    its own, so running them "concurrently" would just have them fight over
    the same cores instead of actually overlapping.

    The onnxruntime patch is only active while this app is being built, so
    it doesn't affect the default `_app` above (used by single-image
    endpoints, where using every core for that one image is the right
    choice) or anything else that creates its own sessions later.
    """
    original_init = ort.InferenceSession.__init__

    def capped_init(self, path_or_bytes, sess_options=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        original_init(self, path_or_bytes, sess_options=sess_options, **kwargs)

    ort.InferenceSession.__init__ = capped_init
    try:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1)
    finally:
        ort.InferenceSession.__init__ = original_init
    return app


_bulk_app = _build_single_threaded_face_app()


def _extract_faces(app: FaceAnalysis, img: np.ndarray) -> list[dict]:
    faces = app.get(img)

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


def get_face_embeddings_from_array(img: np.ndarray) -> list[dict]:
    """
    Core detection logic. Takes an already-loaded image (as a numpy array,
    OpenCV BGR format) and returns a list of detected faces with their
    embedding, bbox, and confidence. Used both when reading from disk
    (get_face_embeddings) and when reading an in-memory upload (the API).
    """
    return _extract_faces(_app, img)


def get_face_embeddings_from_array_bulk(img: np.ndarray) -> list[dict]:
    """
    Same as get_face_embeddings_from_array, but uses the single-threaded
    model instance -- safe to call from several concurrent threads at once
    (e.g. asyncio.to_thread from multiple in-flight photos) without each
    call trying to hog every CPU core for itself.
    """
    return _extract_faces(_bulk_app, img)


def get_face_embeddings(image_path: str) -> list[dict]:
    """Convenience wrapper: reads an image from disk, then detects faces."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return get_face_embeddings_from_array(img)
