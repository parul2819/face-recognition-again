import cv2
import numpy as np


def embedding_to_pgvector(embedding: np.ndarray) -> str:
    """Convert a numpy embedding to pgvector's text format: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def blob_path_to_url(blob_path: str) -> str:
    """'pics/9.jpg' (as stored in the DB) -> '/pics/9.jpg' (browser-facing URL)"""
    return "/" + blob_path.lstrip("/")


def draw_labeled_box(img, x, y, w, h, label=None, matched=True):
    """
    Draws a bounding box (and an optional name label) on img, scaled to the
    face's own size so photos with several small/close faces don't get
    giant overlapping labels. Shared by /identify, the admin annotated
    endpoints, and the bulk-download-as-zip endpoint.
    """
    x1, y1, x2, y2 = x, y, x + w, y + h
    color = (0, 200, 0) if matched else (0, 0, 220)

    face_size = max(w, h)
    scale = max(0.35, min(1.0, face_size / 180))
    box_thickness = max(1, round(scale * 3))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

    if not label:
        return

    font_scale = round(0.45 * scale + 0.15, 2)
    font_thickness = max(1, round(scale * 2))
    pad = max(2, round(4 * scale))

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    label_y = max(y1, text_h + pad * 2)
    cv2.rectangle(img, (x1, label_y - text_h - pad * 2), (x1 + text_w + pad * 2, label_y), color, -1)
    cv2.putText(
        img, label, (x1 + pad, label_y - pad),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness,
    )
