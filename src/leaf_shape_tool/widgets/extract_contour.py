"""
extract_contour.py
------------------
A napari widget for extracting contours from binary label images.

This module detects contours (e.g., leaf outlines) from a binary `Labels` layer,
adds the contour as a new layer in napari, and exports corresponding
CSV and JSON metadata files for reproducible downstream analyses.

Features:
- Extracts the largest external contour from a binary mask.
- Saves contour coordinates, mask images, and structured metadata.
- Adds a visual contour overlay as a new labels layer.
- Supports saving both the edited binary mask and contour mask.

Author: Maple
License: BSD-3-Clause
"""

import napari
from magicgui import magicgui
from napari.utils.notifications import show_info
import numpy as np
import cv2
import pandas as pd
import copy
from pathlib import Path
from magicgui.widgets import Label
import json
from datetime import datetime
from napari.utils.colormaps import DirectLabelColormap


def make_extract_contour_widget(viewer: "napari.Viewer"):
    """
    Create a napari widget that extracts contours from the active Labels layer
    and exports contour data (CSV + metadata JSON).

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.

    Returns
    -------
    magicgui.widgets.Widget
        Configured contour extraction widget.
    """

    # Define colormap for the contour overlay
    color_dict = {
        None: "transparent",
        0: "transparent",  # background
        1: (
            75 / 255,
            196 / 255,
            255 / 255,
        ),  # sky blue (contour)
    }
    colormap = DirectLabelColormap(color_dict=color_dict)

    @magicgui(
        folder_csv={"mode": "d", "label": "Output Folder"},
        folder_final_mask={"mode": "d", "label": " Final Mask Folder"},
        folder_blob_mask={"mode": "d", "label": "Contour Image Folder"},
        save_final_mask={
            "widget_type": "CheckBox",
            "label": "Save final (edited) mask",
            "value": True,
        },
        save_blob_mask={
            "widget_type": "CheckBox",
            "label": "Save chosen blob mask",
            "value": True,
        },
        call_button="Extract Contour",
    )
    def extract_contour(
        folder_csv=Path("./output/contour"),
        folder_final_mask=Path("./output/binarized_image_final"),
        folder_blob_mask=Path("./output/contour_image"),
        save_final_mask=True,
        save_blob_mask=True,
    ):
        """
        Extract the largest external contour from the active Labels layer
        and export contour data and metadata to disk.
        """
        folder_csv.mkdir(parents=True, exist_ok=True)

        # --- Validate input ---
        layer = viewer.layers.selection.active
        if layer is None or layer._type_string != "labels":
            show_info("Please activate a Labels layer.")
            return
        labels = layer.data.astype(np.uint8)
        meta = copy.deepcopy(layer.metadata or {})

        # --- Contour detection ---
        contours, hierarchy = cv2.findContours(
            labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            show_info("No contour detected.")
            return
        # Select the largest contour by area
        max_contour = max(contours, key=cv2.contourArea)

        # --- Save final binary mask ---
        if save_final_mask:
            folder_final_mask.mkdir(parents=True, exist_ok=True)
            image_id = Path(meta.get("source_path", "unknown")).stem
            leaf_id = meta.get("roi_index", "unknown")
            if isinstance(leaf_id, int):
                leaf_id = f"{leaf_id:02}"
            final_mask_path = folder_final_mask / f"{image_id}_{leaf_id}.png"
            img_save = (labels > 0).astype("uint8") * 255
            cv2.imwrite(str(final_mask_path), img_save)

        # --- Save contour visualization mask ---
        if save_blob_mask:
            folder_blob_mask.mkdir(parents=True, exist_ok=True)
            image_id = Path(meta.get("source_path", "unknown")).stem
            leaf_id = meta.get("roi_index", "unknown")
            if isinstance(leaf_id, int):
                leaf_id = f"{leaf_id:02}"
            blob_mask_path = folder_blob_mask / f"{image_id}_{leaf_id}.png"
            mask_blob = np.zeros_like(labels)
            cv2.drawContours(mask_blob, [max_contour], -1, color=255, thickness=3)
            cv2.imwrite(str(blob_mask_path), mask_blob)

        # --- Add contour overlay to napari ---
        mask = np.zeros_like(labels)
        cv2.drawContours(mask, [max_contour], -1, color=1, thickness=3)
        viewer.add_labels(
            mask,
            name=f"{layer.name}_contour",
            opacity=1,
            metadata=meta,
            scale=layer.scale,
            colormap=colormap,
        )

        # --- Save contour coordinates (CSV) ---
        contour = max_contour.squeeze()  # shape as (N, 2)
        df_contour = pd.DataFrame(contour, columns=["x", "y"])

        # Reorder points starting from the rightmost x
        idx_x_max = np.argmax(df_contour["x"].values)  # index of max x
        reordered = np.concatenate(
            (
                df_contour.iloc[idx_x_max:].values,
                df_contour.iloc[:idx_x_max].values,
            ),
            axis=0,
        )
        df_contour = pd.DataFrame(reordered, columns=["x", "y"])

        image_id = meta.get("source_path", "unknown")
        image_id = Path(image_id).stem  # File name without extension
        leaf_id = meta.get("roi_index", "unknown")
        if isinstance(leaf_id, int):
            leaf_id = f"{leaf_id:02d}"
        csv_filename = f"{image_id}_{leaf_id}.csv"
        csv_filename = folder_csv / csv_filename
        df_contour.to_csv(csv_filename, index=False)

        # --- Construct metadata JSON ---
        meta_out = copy.deepcopy(meta)
        source_path = Path(meta.get("source_path"))
        project_root = Path.cwd()

        if project_root in source_path.parents:
            file_relative_path = str(source_path.relative_to(project_root))
        else:
            file_relative_path = str(source_path.resolve())

        meta_out = {
            "metadata_version": "1.0.0",
            "source": {
                "absolute_path": str(source_path.resolve()),
                "relative_path": file_relative_path,
                "image_id": meta.get("image_id"),
                "roi_index": meta.get("roi_index"),
            },
            "scale": {
                "px_per_cm": meta.get("px_per_cm"),
                "unit": meta.get("scale_unit", "cm"),
                "scale_factors": list(map(float, np.asarray(layer.scale))),
                "dpi": meta.get("last_dpi"),
            },
            "roi": {
                "polygon_yx": meta.get("roi_polygon_yx"),
                "bbox_ymin_ymax_xmin_xmax": meta.get("roi_bbox_ymin_ymax_xmin_xmax"),
                "corners_yx": meta.get("roi_corners_yx"),
                "slice_indices": meta.get("slice_indices"),
            },
            "rotation": {
                "angle_deg": meta.get("angle_deg"),
                "original_size": meta.get("original_size"),
                "rotated_size": meta.get("rotated_size"),
            },
            "landmarks": {
                "points_layer_name": meta.get("points_layer_name"),
                "points_n": meta.get("points_n"),
                "points_labels": meta.get("points_labels"),
                "base_original": meta.get("base_original"),
                "tip_original": meta.get("tip_original"),
                "base_rotated": meta.get("base_rotated"),
                "tip_rotated": meta.get("tip_rotated"),
            },
            "binarization": {
                "method": meta.get("binarization_method"),
                "threshold": meta.get("threshold"),
                "manually_edited": meta.get("manually_edited", False),
            },
            "contour": {
                "points": int(len(contour)),
                "area": float(cv2.contourArea(max_contour)),
            },
            "meta": {
                "created_time": pd.Timestamp(datetime.now().astimezone()).isoformat(),
                "cropped_from": meta.get("cropped_from"),
                "face_color_type": meta.get("face_color_type"),
                "border_color_type": meta.get("border_color_type"),
            },
            # Processing history
            "processing_history": [
                {
                    "step": "binarization",
                    "method": meta.get("binarization_method"),
                    "threshold": meta.get("threshold"),
                },
                {
                    "step": "contour_extraction",
                    "points": int(len(contour)),
                    "area": float(cv2.contourArea(max_contour)),
                },
            ],
        }

        # Meta output directory
        meta_out_dir = folder_csv.parent / "metadata"
        meta_out_dir.mkdir(parents=True, exist_ok=True)
        meta_filename = meta_out_dir / f"{image_id}_{leaf_id}.json"
        with open(meta_filename, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=4)

        # --- Save compact metadata (CSV) ---
        src_path_str = meta.get("source_path")
        source_path = (
            Path(src_path_str)
            if src_path_str
            else Path(meta_out["source"]["absolute_path"])
        )
        meta_csv_filename = meta_out_dir / f"{image_id}_{leaf_id}.csv"
        source_path = Path(meta["source_path"])
        scale = meta_out.get("scale", {})
        landmarks = meta_out.get("landmarks", {})

        base_rot = landmarks.get("base_rotated") or [None, None]
        tip_rot = landmarks.get("tip_rotated") or [None, None]

        meta_csv = {
            "file_absolute_path": str(Path(source_path).resolve()),
            "file_relative_path": meta_out["source"]["relative_path"],
            "id": image_id,
            "leaf_id": leaf_id,
            "px_per_cm": scale.get("px_per_cm"),
            "base_x": base_rot[0],
            "base_y": base_rot[1],
            "tip_x": tip_rot[0],
            "tip_y": tip_rot[1],
        }
        df = pd.DataFrame([meta_csv])
        df.to_csv(meta_csv_filename, index=False)

        # --- Return payload ---
        payload = {
            "df_contour": df_contour,
            "metadata": meta_out,
        }
        return payload

    extract_contour.insert(
        0,
        Label(value="7. Choose the output folder\n8. Click the button below"),
    )
    return extract_contour
