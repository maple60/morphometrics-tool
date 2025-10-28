"""
binarize_image.py
-----------------
A napari widget for image binarization using Otsu thresholding or SAM2 segmentation.
This module is part of the Leaf Shape Analysis Tool.
"""

import napari
from magicgui import magicgui
from napari.utils.notifications import show_info
import numpy as np
import cv2
import copy
from magicgui.widgets import Label
from napari.layers import Image
from napari.utils.colormaps import DirectLabelColormap
from pathlib import Path


def make_binarize_image_widget(viewer: "napari.Viewer"):
    """
    Create a widget to binarize the active image layer.

    Supports two methods: Otsu thresholding and Segment Anything (SAM2).
    The method is selected from a dropdown in the UI.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to which the widget is attached.

    Returns
    -------
    magicgui.widgets.Widget
        The configured binarization widget.

    Notes
    -----
    This function maintains the original source image layer name in
    ``state["source_layer_name"]`` so subsequent interactions (e.g.,
    responding to slider changes) can re-use the same image layer safely.
    """

    # --- Keep the original image-layer name globally for later reuse ---
    state = {"source_layer_name": None}

    def _attach_manual_edit_detector(labels_layer):
        """
        Attach a mouse-drag event detector to mark manual edits.
        """

        def _on_drag(layer, event):
            # Triggered only during edit modes (paint, erase, fill)
            if layer.mode in ("paint", "erase", "fill"):
                layer.metadata["manually_edited"] = True
            yield

        labels_layer.mouse_drag_callbacks.append(_on_drag)

    # --- Otsu thresholding and labels-layer update helper ---
    def _run_otsu_and_update_labels(img_layer, *, set_slider=True):
        """
        Apply Otsu thresholding and create/update a Labels layer.

        Parameters
        ----------
        img_layer : napari.layers.Image
            The source image layer (RGB/RGBA) to be binarized.
        set_slider : bool, optional
            Whether to update the UI slider with the Otsu-derived threshold
            (True) or keep the user-set slider value (False). Default is True.
        """

        # --- Define label colors ---
        color_dict = {
            None: "transparent",
            0: "transparent",  # Background
            1: (
                255 / 255,
                75 / 255,
                00 / 255,
            ),  # Foreground (leaf)
        }
        colormap = DirectLabelColormap(color_dict=color_dict)

        img = img_layer.data  # RGB image
        meta = copy.deepcopy(img_layer.metadata or {})

        # --- Prepare image for OpenCV---
        img_for_cv2 = img.copy()
        if img_for_cv2.dtype != np.uint8:
            # Normalize to 8-bit
            img_for_cv2 = cv2.normalize(
                img_for_cv2, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        # Convert RGB(A) → BGR
        if img_for_cv2.shape[-1] == 4:  # RGBA → BGR
            img_for_cv2 = cv2.cvtColor(img_for_cv2, cv2.COLOR_RGBA2BGR)
        elif img_for_cv2.shape[-1] == 3:  # RGB → BGR
            img_for_cv2 = cv2.cvtColor(img_for_cv2, cv2.COLOR_RGB2BGR)

        # --- Convert to grayscale---
        img_gray = cv2.cvtColor(img_for_cv2, cv2.COLOR_BGR2GRAY)
        # --- Apply Otsu thresholding ---
        if set_slider:
            thresh_value, img_binary = cv2.threshold(
                img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            binarize_image.threshold.value = int(round(thresh_value))
        else:
            thresh_value = binarize_image.threshold.value
            _, img_binary = cv2.threshold(
                img_gray, thresh_value, 255, cv2.THRESH_BINARY_INV
            )

        labels = (img_binary > 0).astype(np.uint8)  # 0=background, 1=object
        layer_name = f"{img_layer.name}_Otsu_labels"

        # --- Update or create label layer ---
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = labels
            viewer.layers[layer_name].metadata.update(
                {
                    "binarization_method": "Otsu",
                    "threshold": float(thresh_value),
                }
            )
        else:
            viewer.add_labels(
                labels,
                name=layer_name,
                blending="translucent",
                opacity=0.5,
                metadata={
                    **meta,
                    "binarization_method": "Otsu",
                    "threshold": float(thresh_value),
                    "cropped_from": img_layer.name,
                    "manually_edited": False,
                },
                scale=img_layer.scale,
                colormap=colormap,
            )
            labels_layer = viewer.layers[layer_name]
            labels_layer.metadata["manually_edited"] = False  # Initialize
            _attach_manual_edit_detector(labels_layer)

    # -------------------------------------------------------------------------
    # --- Event handlers ---
    # -------------------------------------------------------------------------
    def _on_threshold_changed(event=None):
        """Handle slider changes (Otsu only)"""

        if binarize_image.method.value != "Otsu":
            return
        src_name = state.get("source_layer_name")
        if src_name is None or src_name not in viewer.layers:
            print("Source image not found")
            return
        _run_otsu_and_update_labels(viewer.layers[src_name], set_slider=False)

    def _toggle_threshold_visibility(event=None):
        """Show or hide threshold slider"""
        show = binarize_image.method.value == "Otsu"
        binarize_image.threshold.visible = show
        binarize_image.threshold.enabled = show

    # -------------------------------------------------------------------------
    # --- Save binarized image ---
    # -------------------------------------------------------------------------
    def _save_binarized_image(
        labels: np.ndarray,
        meta: dict,
        out_dir: Path,
        *,
        method: str,
    ):
        """
        Save the binarized label image as a PNG file.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        image_id = Path(meta.get("source_path", "unknown")).stem
        roi = meta.get("roi_index", "unknown")
        roi = f"{roi:02}" if isinstance(roi, int) else roi
        if isinstance(roi, int):
            roi = f"{roi:02d}"
        method = method.lower()
        file_name = f"{image_id}_{roi}_{method}.png"
        save_path = out_dir / file_name
        img_save = (labels.astype(np.uint8) > 0).astype(np.uint8) * 255
        cv2.imwrite(save_path, img_save)
        print(f"[binarize] Saved: {save_path}")

    # -------------------------------------------------------------------------
    # --- Main magicgui widget definition ---
    # -------------------------------------------------------------------------
    @magicgui(
        method={"choices": ["Otsu", "SAM2"]},
        threshold={"widget_type": "Slider", "min": 0, "max": 255, "step": 1},
        call_button="Binarize Image",
        save_check={
            "widget_type": "CheckBox",
            "label": "Save binarized image",
        },
        save_dir={
            "widget_type": "FileEdit",
            "mode": "d",
            "label": "Output Folder",
        },
    )
    def binarize_image(
        method: str = "Otsu",
        threshold: int = 128,
        save_check: bool = True,
        save_dir: Path = Path("./output/binarized_images"),
    ):
        """
        Run the selected binarization method
        """
        _toggle_threshold_visibility()  # Update visibility
        img_layer = viewer.layers.selection.active  # Active image layer

        # --- Input validation ---
        if not isinstance(img_layer, Image) or img_layer.data.ndim not in (
            2,
            3,
        ):
            show_info("Please activate a 2D or 2D+RGB(A) image layer.")
            return

        # --- Update source layer name ---
        if img_layer.name != state["source_layer_name"]:
            state["source_layer_name"] = img_layer.name

        # --- Otsu method ---
        if method == "Otsu":
            _run_otsu_and_update_labels(img_layer, set_slider=True)
            # Save option
            if save_check:
                labels = viewer.layers[f"{img_layer.name}_Otsu_labels"].data
                meta = copy.deepcopy(
                    viewer.layers[f"{img_layer.name}_Otsu_labels"].metadata or {}
                )
                _save_binarized_image(labels, meta, save_dir, method="Otsu")

        # --- SAM2 method ---
        elif method == "SAM2":
            show_info("Running SAM2 segmentation...")
            import os

            if not os.path.exists("sam2"):
                print("Please place the 'sam2' folder in the root directory.")
            else:
                import torch

                old_dir = os.getcwd()  # save current directory
                os.chdir(os.path.join(old_dir, "sam2"))  # change to sam2 directory
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import (
                    SAM2AutomaticMaskGenerator,
                )

                color_dict = {
                    None: "transparent",
                    0: "transparent",  # background
                    1: (
                        0 / 255,
                        90 / 255,
                        255 / 255,
                    ),  # leaf
                }
                colormap = DirectLabelColormap(color_dict=color_dict)

                checkpoint = "checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                sam2_model = build_sam2(model_cfg, checkpoint)
                os.chdir(old_dir)  # restore original directory

                mask_generator = SAM2AutomaticMaskGenerator(
                    sam2_model, multimask_output=False
                )
                img = img_layer.data
                # --- Auto mask generation ---
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    masks = mask_generator.generate(img)

                mask_areas = [np.sum(m["segmentation"]) for m in masks]
                largest_mask = masks[np.argmax(mask_areas)]["segmentation"]  # bool

                # largest_mask : bool (True=leaf, False=background)
                labels = np.zeros(largest_mask.shape, dtype=np.uint8)
                labels[largest_mask] = 1
                meta = copy.deepcopy(img_layer.metadata or {})
                viewer.add_labels(
                    labels,  # 0/1 label image
                    name=f"{img_layer.name}_SAM2_labels",
                    blending="translucent",
                    opacity=0.5,
                    metadata={
                        **meta,  # copy from source image
                        "binarization_method": "SAM2",
                        "threshold": None,
                        "manually_edited": False,
                    },
                    scale=img_layer.scale,
                    colormap=colormap,
                )
                print("Labels layer added to napari viewer.")

                # --- Attach manual edit detector ---
                labels_layer = viewer.layers[f"{img_layer.name}_SAM2_labels"]
                labels_layer.metadata["manually_edited"] = False  # Initialize
                _attach_manual_edit_detector(labels_layer)

                # --- Save option ---
                if save_check:
                    labels = viewer.layers[f"{img_layer.name}_SAM2_labels"].data
                    meta = copy.deepcopy(
                        viewer.layers[f"{img_layer.name}_SAM2_labels"].metadata or {}
                    )
                    _save_binarized_image(labels, meta, save_dir, method="SAM2")
                print("SAM2 segmentation completed.")
        # --- Invalid method ---
        else:
            show_info(f"Invalid method: {method}")
            return

    # --- Setup initial states ---
    binarize_image.method.changed.connect(_toggle_threshold_visibility)
    binarize_image.threshold.changed.connect(_on_threshold_changed)
    _toggle_threshold_visibility()

    # --- Instruction label ---
    binarize_image.insert(
        0,
        Label(value="5. Select binarization method\n6. Click the button below"),
    )
    return binarize_image
