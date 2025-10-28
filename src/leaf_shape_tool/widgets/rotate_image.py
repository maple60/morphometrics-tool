"""
rotate_image.py
---------------
A napari widget for rotating cropped images based on landmark points.

The widget reads "base" and "tip" landmarks from a Points layer, rotates the
corresponding image so that the base→tip axis becomes horizontally rightward,
and then adds the rotated image as a new layer with full metadata.

Features:
- Detects associated image layer via Points.metadata['attached_to']
- Computes precise affine rotation with white background fill
- Updates and stores rotation information in metadata
- Optionally saves rotated images to disk

Author: Maple
License: BSD-3-Clause
"""

from typing import Any, Callable, Sequence
import pandas as pd
from magicgui import magicgui
from napari.utils.notifications import show_info
import napari
import cv2
import numpy as np
from pathlib import Path
from numbers import Integral
from magicgui.widgets import Label
import copy


# -------------------------------------------------------------------------
# Layer and metadata helpers
# -------------------------------------------------------------------------


def get_active_points_layer(viewer: "napari.Viewer") -> "napari.layers.Points":
    """Return the active layer if it is a Points layer, otherwise raise an error."""
    layer = viewer.layers.selection.active
    if layer is None:
        raise RuntimeError("No active layer found.")
    if not isinstance(layer, napari.layers.Points):
        raise TypeError(f"Active layer '{layer.name}' is not a Points layer.")
    return layer


def summarize_points_layer(pl: "napari.layers.Points") -> dict[str, Any]:
    """
    Collect basic metadata and summary information from a Points layer.

    Returns
    -------
    dict
        Dictionary summarizing label types, metadata, and feature structure.
    """
    feats = pl.features if hasattr(pl, "features") else pd.DataFrame()
    cols = list(feats.columns)

    label_cats = None
    if "label" in feats.columns:
        try:
            if pd.api.types.is_categorical_dtype(feats["label"]):
                label_cats = list(feats["label"].cat.categories)
        except Exception:
            pass

    info: dict[str, Any] = {
        "name": pl.name,
        "n_points": len(pl.data),
        "ndim": pl.ndim,
        "data_example": pl.data[:3].tolist(),
        "features_columns": cols,
        "label_categories": label_cats,
        "properties_keys": list(pl.properties.keys()),
        "metadata": dict(pl.metadata),
        "size": pl.size,
        "symbol": pl.symbol,
        "face_color": pl.face_color.name
        if hasattr(pl.face_color, "name")
        else pl.face_color,
        "border_color": pl.border_color.name
        if hasattr(pl.border_color, "name")
        else pl.border_color,
    }
    return info


def find_layer_by_name(
    viewer: "napari.Viewer",
    name: str,
    allow_prefix_fallback: bool = True,
) -> "napari.layers.Layer":
    """
    Find a layer by exact name or, optionally, by prefix fallback.

    Parameters
    ----------
    viewer : napari.Viewer
    name : str
        Layer name to search for.
    allow_prefix_fallback : bool
        If True, also match layers whose names start with `name`.

    Returns
    -------
    napari.layers.Layer
    """
    for ly in viewer.layers:
        if ly.name == name:
            return ly

    if allow_prefix_fallback:
        cands = [ly for ly in viewer.layers if ly.name.startswith(name)]
        if len(cands) == 1:
            return cands[0]
        elif len(cands) > 1:
            return cands[-1]

    raise LookupError(f"Layer '{name}' not found.")


def get_attached_image_layer_from_points(
    viewer: "napari.Viewer",
    get_active_points_layer: Callable[["napari.Viewer"], "napari.layers.Points"],
    find_layer_by_name: Callable[["napari.Viewer", str], "napari.layers.Layer"],
) -> "napari.layers.Image":
    """
    Retrieve the image layer attached to the active Points layer.

    The function looks for the key 'attached_to' in the Points metadata.

    Returns
    -------
    napari.layers.Image
        The corresponding image layer.
    """
    pl = get_active_points_layer(viewer)  # Landmark layer
    md = dict(pl.metadata) if hasattr(pl, "metadata") else {}  # Metadata dict
    attached_name = md.get("attached_to", None)  # attached_to = Image layer name
    if not attached_name or not isinstance(attached_name, str):
        raise KeyError("Points.metadata['attached_to'] is missing or invalid.")
    ly = find_layer_by_name(viewer, attached_name)  # Find image layer
    return ly


# -------------------------------------------------------------------------
# Image rotation core
# -------------------------------------------------------------------------


def rotate_image_bese_left_tip_right(
    img: np.ndarray,
    points_xy: Sequence[Sequence[float]],
    labels: Sequence[str],
    yx_input: bool = True,
):
    """
    Rotate the image so that the 'base'→'tip' line becomes horizontally rightward.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    points_xy : Sequence of [y, x] or [x, y]
        Landmark coordinates.
    labels : Sequence[str]
        Landmark labels containing at least 'base' and 'tip'.
    yx_input : bool, optional
        Whether coordinates are provided as [y, x]. Default is True.

    Returns
    -------
    tuple
        (rotated_image, rotated_points, rotation_metadata)
    """
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    if "base" not in lbl_to_idx or "tip" not in lbl_to_idx:
        raise ValueError("Labels must include 'base' and 'tip'.")
    base_idx = lbl_to_idx["base"]
    tip_idx = lbl_to_idx["tip"]
    P = np.asarray(points_xy, dtype=float).copy()
    # [y,x] -> [x,y]
    if yx_input:
        P = P[:, ::-1]
    base_xy = P[base_idx]  # Base coordinates
    tip_xy = P[tip_idx]  # Tip Coordinates

    # --- Calculate rotation angle ---
    dx = tip_xy[0] - base_xy[0]
    dy = tip_xy[1] - base_xy[1]
    angle_rad = np.arctan2(dy, dx)  # Angle in radians
    angle_deg = np.degrees(angle_rad)  # Convert to degrees
    h, w = img.shape[:2]  # Original image size
    cx, cy = w / 2.0, h / 2.0  # Image center

    # Compute rotation matrix with translation correction
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)  # Width after rotation
    new_h = int(h * cos + w * sin)  # Height after rotation
    # Adjust translation to keep image centered
    M[0, 2] += (new_w / 2) - cx  # Translation in x
    M[1, 2] += (new_h / 2) - cy  # Translation in y

    # Apply rotation to image and points
    border_val = (255,) * img.shape[2] if img.ndim == 3 else (255,)  # White background
    img_rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=border_val,
    )
    # P: (N,2)
    ones = np.ones((P.shape[0], 1))  # (N,1)
    P_h = np.hstack([P, ones])  # (N,3) Coords in homogeneous form
    P_rot = (M @ P_h.T).T  # (N,2)
    base_rot = P_rot[base_idx]
    tip_rot = P_rot[tip_idx]

    # Ensure orientation (base.x < tip.x)
    if base_rot[0] > tip_rot[0]:
        img_rotated = cv2.flip(img_rotated, 1)  # Horizontal flip
        P_rot[:, 0] = img_rotated.shape[1] - P_rot[:, 0]
        base_rot = P_rot[base_idx]
        tip_rot = P_rot[tip_idx]

    if yx_input:
        P_rot = P_rot[:, ::-1]  # [x,y] -> [y,x]
    else:
        P_rot = P_rot

    # --- Summary ---
    rotation_info = {
        "angle_deg": angle_deg,
        "original_size": (h, w),
        "rotated_size": img_rotated.shape[:2],
        "base_original": base_xy.tolist(),
        "tip_original": tip_xy.tolist(),
        "base_rotated": base_rot.tolist(),
        "tip_rotated": tip_rot.tolist(),
    }

    return img_rotated, P_rot, rotation_info


# -------------------------------------------------------------------------
# Widget factory
# -------------------------------------------------------------------------


def make_points_metadata_widget(
    viewer: "napari.Viewer",
    get_active_points_layer: Callable[["napari.Viewer"], "napari.layers.Points"],
    summarize_points_layer: Callable[["napari.layers.Points"], dict[str, Any]],
):
    """
    Create a widget that rotates the associated image based on the
    'base' and 'tip' points in the active Points layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    get_active_points_layer : callable
        Function returning the current active Points layer.
    summarize_points_layer : callable
        Function returning summary info from a Points layer.

    Returns
    -------
    magicgui.widgets.Widget
        The rotation widget.
    """

    @magicgui(
        call_button="Rotate Image Based on Points",
        save_rotated={
            "widget_type": "CheckBox",
            "label": "Save rotated image",
        },
        save_dir={
            "widget_type": "FileEdit",
            "mode": "d",
            "label": "Output Folder",
        },
    )
    def points_metadata_widget(
        viewer: "napari.Viewer",
        save_rotated: bool = True,
        save_dir: str = "output/rotated_images",
    ) -> dict[str, Any]:
        """Rotate the image corresponding to the current Points layer."""
        try:
            pl = get_active_points_layer(viewer)
            info = summarize_points_layer(pl)
            msg = (
                f"[{info['name']}] {info['n_points']}点 / ndim={info['ndim']} / "
                f"features={info['features_columns']} / labels={info['label_categories']} / "
                f"source={info['metadata'].get('source_path')} / index={info['metadata'].get('roi_index')}"
            )
            metadata_inner = info.get("metadata", {})

            # Get cropped image layer
            ly = get_attached_image_layer_from_points(
                viewer, get_active_points_layer, find_layer_by_name
            )
            medatada_layer = copy.deepcopy(ly.metadata or {})
            labels = list(pl.features["label"])

            # Prepare image for OpenCV
            img = ly.data
            img_for_cv2 = img.copy()
            if img_for_cv2.dtype != np.uint8:
                img_for_cv2 = cv2.normalize(
                    img_for_cv2, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
            if img_for_cv2.shape[-1] == 4:  # RGBA → BGR
                img_for_cv2 = cv2.cvtColor(img_for_cv2, cv2.COLOR_RGBA2BGR)
            elif img_for_cv2.shape[-1] == 3:  # RGB → BGR
                img_for_cv2 = cv2.cvtColor(img_for_cv2, cv2.COLOR_RGB2BGR)

            # Compute world → image coordinates
            P_world = pl.data * pl.scale + pl.translate  # (y, x)
            P_img = (P_world - ly.translate) / ly.scale  # (y, x) in image pixels

            img_rotated, points_rotated, rot_info = rotate_image_bese_left_tip_right(
                img_for_cv2, P_img, labels, yx_input=True
            )

            # --- Save image ---
            if save_rotated:
                image_id = info["metadata"].get("from_layer")
                leaf_id = info["metadata"].get("roi_index")
                leaf_id = f"{leaf_id:02}" if isinstance(leaf_id, int) else leaf_id
                save_path = Path(save_dir) / f"{image_id}_{leaf_id}.png"
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), img_rotated)

            # --- Prepare metadata for rotated image layer ---
            safe_rot_info = {
                "angle_deg": float(rot_info["angle_deg"]),
                "original_size": tuple(map(int, rot_info["original_size"])),
                "rotated_size": tuple(map(int, rot_info["rotated_size"])),
                "base_original": list(map(float, rot_info["base_original"])),
                "tip_original": list(map(float, rot_info["tip_original"])),
                "base_rotated": list(map(float, rot_info["base_rotated"])),
                "tip_rotated": list(map(float, rot_info["tip_rotated"])),
            }

            source_path = metadata_inner.get("source_path")
            roi_index = metadata_inner.get("roi_index")
            image_id = Path(source_path).stem if source_path else "unknown"
            if isinstance(roi_index, Integral):
                roi_index = int(roi_index)

            # Metadata summary for rotated image
            points_summary = {
                "points_layer_name": info.get("name"),
                "points_n": int(info.get("n_points", 0)),
                "points_labels": info.get("label_categories"),
                "source_path": source_path,
                "image_id": image_id,
                "roi_index": roi_index,
            }

            color_summary = {
                "face_color_type": type(info.get("face_color")).__name__,
                "border_color_type": type(info.get("border_color")).__name__,
            }

            final_meta = {
                **medatada_layer,
                **safe_rot_info,
                **points_summary,
                **color_summary,
                "binarization_method": info.get("binarization_method") or "Otsu",
                "threshold": float(info.get("threshold", 0))
                if "threshold" in info
                else None,
            }

            # Convert BGR to RGB for display
            img_view = img_rotated.copy()
            img_view = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
            layer_rotated = viewer.add_image(
                img_view,
                name=f"{ly.name}_rotated",
                metadata=final_meta,
                scale=ly.scale.copy(),
            )

            # Auto-zoom to fit the rotated image
            ext = np.asarray(layer_rotated.extent.world)
            y_min, y_max = ext[0, 0], ext[1, 0]
            x_min, x_max = ext[0, 1], ext[1, 1]
            dy = ext[1, 0] - ext[0, 0]  # y方向（高さ）
            dx = ext[1, 1] - ext[0, 1]  # x方向（幅）
            margin = 0  # 任意で余白をつける
            # キャンバスサイズを取得
            canvas = viewer.screenshot(canvas_only=True, flash=False)
            canvas_h, canvas_w = canvas.shape[:2]
            # 中心に移動
            viewer.camera.center = (
                (y_min + y_max) / 2,
                (x_min + x_max) / 2,
            )
            # ズームサイズの計算
            zoom = min(canvas_w / dx, canvas_h / dy)
            viewer.camera.zoom = float(zoom)

            return info

        except Exception as e:
            show_info(f"Error retrieving metadata: {e}")
            return {}

    points_metadata_widget.insert(
        0,
        Label(value="3. Add landmarks at the base and tip\n4. Click the button below"),
    )
    return points_metadata_widget
