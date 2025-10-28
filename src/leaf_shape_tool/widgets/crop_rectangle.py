"""
crop_rectangle.py
-----------------
A napari widget for interactively defining rectangular (or polygonal) ROIs,
cropping them from an image, and creating associated landmark layers.

Main features:
- Interactive ROI selection via the "ROIs" Shapes layer.
- Automatic ROI numbering and metadata tracking.
- Optional saving of cropped image regions to disk.
- Creation of a paired landmark Points layer for each ROI.
- Automatic maintenance of ROI number labels.

Author: Maple
License: BSD-3-Clause
"""

import re
import numpy as np
from magicgui import magicgui
from napari.utils.notifications import show_info
from napari.layers import Shapes
import pandas as pd
from magicgui.widgets import Label
import copy
import cv2
from pathlib import Path
import napari
from qtpy.QtCore import QTimer

COLOR_CYCLE = ["#FF4B00", "#005AFF"]  # base(orange), tip(blue)
LABELS = ("base", "tip")


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------


def _next_index_from_layers(viewer) -> int:
    """Return the next available ROI index by scanning existing layer names 'ROI_XX'."""
    pat = re.compile(r"^ROI_(\d+)$")
    mx = 0
    for lyr in viewer.layers:
        m = pat.match(lyr.name)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except ValueError:
                pass
    return mx + 1 if mx >= 1 else 1


def _unique_roi_name(viewer, base_idx: int) -> tuple[str, int]:
    """Return a unique layer name 'ROI_{:02d}' and its numeric index."""
    idx = int(base_idx)
    while True:
        name = f"ROI_{idx:02d}"
        if name not in [lyr.name for lyr in viewer.layers]:
            return name, idx
        idx += 1


def bounding_box_from_polygon(pts):
    """Compute the bounding box (ymin, ymax, xmin, xmax) from polygon vertices (N,2)."""
    ys = pts[:, 0]
    xs = pts[:, 1]
    ymin, ymax = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    xmin, xmax = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    return ymin, ymax, xmin, xmax


def current_plane_2d(img_layer, viewer):
    """Return the currently displayed 2D plane (…, y, x) from an image layer."""
    data = img_layer.data
    if data.ndim <= 2:
        return data
    steps = list(viewer.dims.current_step)
    steps[-2:] = [slice(None), slice(None)]
    return data[tuple(steps)]


def crop_from_shape(img_layer, shape_vertices, viewer):
    """Crop a rectangular region defined by a Shapes layer polygon."""
    plane2d = current_plane_2d(img_layer, viewer)
    ymin, ymax, xmin, xmax = bounding_box_from_polygon(shape_vertices)
    h, w = plane2d.shape[:2]
    ymin = max(0, ymin)
    xmin = max(0, xmin)
    ymax = min(h, ymax)
    xmax = min(w, xmax)
    if ymin >= ymax or xmin >= xmax:
        raise ValueError("The selected rectangle is outside the image area.")
    return plane2d[ymin:ymax, xmin:xmax, ...]


def _get_shapes_layer(viewer):
    """Return an active or non-empty Shapes layer (prefer 'ROIs' if present)."""
    active = viewer.layers.selection.active
    if isinstance(active, Shapes) and len(active.data) > 0:
        return active
    rois = viewer.layers["ROIs"] if "ROIs" in viewer.layers else None
    if isinstance(rois, Shapes) and len(rois.data) > 0:
        return rois
    for lyr in viewer.layers:
        if isinstance(lyr, Shapes) and len(lyr.data) > 0:
            return lyr
    return None


# -------------------------------------------------------------------------
# ROI label layer utilities
# -------------------------------------------------------------------------


# ---- 追加: ROI番号テキスト用レイヤの作成/取得 ----
def _get_or_create_rois_label_layer(
    viewer, yx: tuple[float, float], roi_number: int
) -> "napari.layers.Points":
    """
    Retrieve or create a 'ROIs_label' Points layer that displays ROI numbers as text.
    Each ROI number is added as a single point at the given (y, x) coordinate.
    """

    layer_name = "ROIs_label"
    lbl = viewer.layers[layer_name] if layer_name in viewer.layers else None

    roi_layer = viewer.layers["ROIs"] if "ROIs" in viewer.layers else None
    scale = roi_layer.scale if roi_layer else (1.0, 1.0)

    # --- text visuals for labels ---
    text = {
        "string": "{roi}",
        "anchor": "lower_right",  # tune as you like
        "translation": [0, 0],
        "size": 45,
        "color": "#FF4B00",
        "visible": True,
    }

    if lbl is None:
        # create new layer
        lbl = viewer.add_points(
            data=np.zeros((0, 2), dtype=float),
            name=layer_name,
            face_color="transparent",
            size=1,  # minimal size
            ndim=2,
            scale=scale,
        )

    # --- Append the new point ---
    cur_data = np.asarray(lbl.data) if len(lbl.data) else np.zeros((0, 2), float)
    new_data = np.vstack([cur_data, np.asarray(yx, float)])
    lbl.data = new_data
    # --- Rebuild features to match N rows ---
    if "roi" not in lbl.features:
        lbl.features["roi"] = pd.Series(dtype="Int64")
    lbl.features.loc[:, "roi"] = range(1, len(lbl.data) + 1)
    lbl.text = text

    # Keep the layer just below "ROIs"
    try:
        idx_rois = viewer.layers.index("ROIs")
        idx_lbl = viewer.layers.index(layer_name)
        if idx_lbl != idx_rois - 1:
            viewer.layers.move(idx_lbl, idx_rois)
    except Exception:
        pass
    return lbl


# -------------------------------------------------------------------------
# Main widget
# -------------------------------------------------------------------------


def make_add_roi_widget(viewer, labels=LABELS, point_size=30, show_text=True):
    """
    Create a napari widget for cropping rectangular ROIs from the current image.
    Also creates a landmark Points layer and optional numbered label overlay.
    """

    def _ensure_rois_label_text_visible(event=None):
        """
        Ensure that the 'ROIs_label' Points layer always displays text properly.
        Some napari versions hide text temporarily when switching layer modes.
        """
        layer_name = "ROIs_label"
        active = getattr(event, "value", None)
        if not (active and active.name == layer_name):
            return

        def _apply():
            lbl = viewer.layers[layer_name] if layer_name in viewer.layers else None
            if lbl is None:
                return
            try:
                lbl.mode = "pan_zoom"  # Avoid add/select mode
            except Exception:
                pass

            # Keep features length in sync
            n = len(lbl.data)
            if "roi" not in lbl.features or len(lbl.features) != n:
                lbl.features = pd.DataFrame(
                    {"roi": pd.Series(range(1, n + 1), dtype="Int64")}
                )

            # Re-apply text
            lbl.text = {
                "string": "{roi}",
                "anchor": "lower_right",
                "translation": [0, 0],
                "size": 45,
                "color": "#FF4B00",
                "visible": True,
            }

        # Run after other callbacks
        QTimer.singleShot(0, _apply)

    # Connect
    viewer.layers.selection.events.active.connect(_ensure_rois_label_text_visible)

    @magicgui(
        call_button="Add ROI layer",
        roi_index={
            "widget_type": "SpinBox",
            "label": "Next ROI index",
            "min": 1,
            "step": 1,
        },
        save_cropped={
            "widget_type": "CheckBox",
            "label": "Save cropped image",
        },
        save_dir={
            "widget_type": "FileEdit",
            "mode": "d",
            "label": "Output Folder",
        },
    )
    def add_roi(
        roi_index: int = 1,
        save_cropped: bool = True,
        save_dir: str = "output/cropped_images",
    ):
        """Add a cropped ROI image and its landmark layer."""
        ROIs_layer = viewer.layers["ROIs"]
        if ROIs_layer is None:
            show_info("Shapes layer 'ROIs' not found.")
            return
        image_name = ROIs_layer.metadata.get("source_image", "input")
        img_layer = viewer.layers[image_name]
        if img_layer is None:
            show_info(f"Image layer {image_name!r} not found.")
            return
        metadata_img_layer = copy.deepcopy(img_layer.metadata or {})

        shapes = _get_shapes_layer(viewer)
        if shapes is None or len(shapes.data) == 0:
            show_info("Add a rectangle or polygon shape in the 'ROIs' layer first.")
            return

        # Determine which shape to crop (selected one if any)
        sel = shapes.selected_data
        idx_shape = next(iter(sel)) if len(sel) > 0 else (len(shapes.data) - 1)
        verts = np.asarray(shapes.data[idx_shape])  # (N,2) = (y,x)

        try:
            cropped = crop_from_shape(img_layer, verts, viewer)
        except Exception as e:
            show_info(f"Cropping failed: {e}")
            return

        # Generate unique ROI name
        wanted_idx = int(roi_index)
        roi_name, used_idx = _unique_roi_name(viewer, wanted_idx)

        # --- Metadata ---
        ymin, ymax, xmin, xmax = bounding_box_from_polygon(verts)
        roi_corners_yx = [
            (ymin, xmin),
            (ymin, xmax),
            (ymax, xmax),
            (ymax, xmin),
        ]
        src_path = img_layer.metadata.get("source_path")

        meta = {
            **metadata_img_layer,  # copy from image layer
            "source_path": src_path,
            "roi_index": int(used_idx),  # start from 1 (not zero-based)
            "roi_polygon_yx": verts.tolist(),
            "roi_bbox_ymin_ymax_xmin_xmax": [
                int(ymin),
                int(ymax),
                int(xmin),
                int(xmax),
            ],
            "roi_corners_yx": roi_corners_yx,
            "slice_indices": list(viewer.dims.current_step),
            "from_layer": img_layer.name,
        }

        # --- Add cropped image layer ---
        scale2d = tuple(np.atleast_1d(img_layer.scale)[-2:])  # (sy, sx)
        layer_cropped = viewer.add_image(
            cropped, name=roi_name, metadata=meta, scale=scale2d
        )
        show_info(f"Added new layer: {roi_name}")

        # Save cropped image if requested
        if save_cropped:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{img_layer.name}_{roi_name}.png"
            # Convert to uint8 BGR for saving
            arr = np.asarray(cropped)
            if arr.dtype.kind == "f":
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] == 3:  # RGB → BGR
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr_bgr = arr
            cv2.imwrite(str(save_path), arr_bgr)  # Save the image

        # Add landmark Points layer
        features_landmarks = pd.DataFrame(
            {"label": pd.Categorical([], categories=list(labels))}
        )

        pts = viewer.add_points(
            data=np.zeros((0, 2), dtype=float),  # initially empty
            name=f"{roi_name}_landmarks",
            ndim=2,
            features=features_landmarks,
            face_color="transparent",
            border_color="label",
            border_color_cycle=COLOR_CYCLE[: len(labels)],
            size=point_size,
            border_width=0.5,  # fraction of point size
            symbol="o",
            metadata={**meta, "attached_to": roi_name},
            scale=img_layer.scale,
        )

        pts.feature_defaults = {"label": [labels[0]]}
        if show_text:
            pts.text = {
                "string": "{label}",
                "anchor": "upper_left",
                "translation": [-20, 0],  # offset from ancher [dx, dy]
                "color": (255 / 255, 75 / 255, 0 / 255, 1),  # orange
                "size": point_size / 1.5,
            }
        pts.mode = "add"

        # Focus camera on the cropped region
        ext = np.asarray(layer_cropped.extent.world)
        y_min, y_max = ext[0, 0], ext[1, 0]
        x_min, x_max = ext[0, 1], ext[1, 1]
        dy = ext[1, 0] - ext[0, 0]
        dx = ext[1, 1] - ext[0, 1]

        canvas = viewer.screenshot(canvas_only=True, flash=False)
        canvas_h, canvas_w = canvas.shape[:2]
        viewer.camera.center = (
            (y_min + y_max) / 2,
            (x_min + x_max) / 2,
        )
        zoom = min(canvas_w / dx, canvas_h / dy)
        viewer.camera.zoom = float(zoom)

        # Add ROI label point
        _get_or_create_rois_label_layer(viewer, (ymin + 1, xmax - 1), used_idx)

        # Select the new Points layer
        viewer.layers.selection.active = pts

        # Increment next ROI index
        try:
            add_roi.roi_index.value = _next_index_from_layers(viewer)
        except Exception:
            pass

    add_roi.insert(
        0,
        Label(value="1. Draw rectangle in 'ROIs' layer\n2. Click the button below"),
    )

    # Initialize ROI index SpinBox
    try:
        add_roi.roi_index.value = _next_index_from_layers(viewer)
    except Exception:
        pass

    return add_roi
