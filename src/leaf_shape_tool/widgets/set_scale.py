"""
set_scale.py
-------------
A napari widget for setting physical scale information (cm, mm, µm)
for images based on DPI or a drawn measurement line.

Features:
- Add a “Scale Measurement” Shapes layer for drawing a reference line.
- Measure pixel distance directly from the line.
- Convert to real-world units (via DPI or custom px/cm conversion).
- Apply scale to all visible layers and update metadata.
- Automatically configure the napari scale bar.

Author: Maple
License: BSD-3-Clause
"""

import napari
import numpy as np
from magicgui import magicgui
from napari.utils.notifications import show_info
from magicgui.widgets import PushButton, Container


def make_set_scale_widget(viewer: "napari.Viewer"):
    """
    Create a widget to set or calibrate physical scale for an image layer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.

    Returns
    -------
    magicgui.widgets.Container
        Configured scale-setting widget.
    """

    # ------------------------------------------------------------------
    # Button: Add Shapes layer for scale measurement
    # ------------------------------------------------------------------
    add_btn = PushButton(text="Add Layer")

    def _add_layer():
        """Add a Shapes layer to draw a line for measuring pixel distance."""
        shapes = viewer.add_shapes(
            name="Scale Measurement",
            shape_type="line",
            edge_width=5,
            edge_color="#FF4B00",  # Orange
            face_color="transparent",
        )
        try:
            shapes.mode = "ADD_LINE"
        except Exception:
            pass  # Old napari version may not support mode setting
        viewer.layers.selection.active = shapes  # Activate the new layer

    add_btn.native.clicked.connect(_add_layer)

    # ------------------------------------------------------------------
    # Helper: get measurement line length (px)
    # ------------------------------------------------------------------
    def _get_measurement_px() -> float | None:
        """Return the Euclidean length (in pixels) of the selected or last drawn line."""
        shapes = None
        if "Scale Measurement" in viewer.layers:
            ly = viewer.layers["Scale Measurement"]
            if ly._type_string == "shapes":
                shapes = ly
        else:
            ly = viewer.layers.selection.active
            if ly is not None and getattr(ly, "_type_string", "") == "shapes":
                shapes = ly

        if shapes is None:
            return None

        # If multiple lines exist, use the selected one or the last drawn one
        # line data: [[y0, x0], [y1, x1]]
        try:
            if len(shapes.data) == 0:
                return None
            indices = list(getattr(shapes, "selected_data", []) or [])
            if indices:
                idx = sorted(indices)[-1]
            else:
                idx = len(shapes.data) - 1
            coords = np.asarray(shapes.data[idx])
            if coords.shape[0] < 2:
                return None
            # napari 座標は (y, x)。ユークリッド距離（px）を返す
            (y0, x0), (y1, x1) = coords[:2]
            length_px = float(np.hypot(y1 - y0, x1 - x0))
            return length_px
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main scale setting function
    # ------------------------------------------------------------------
    @magicgui(
        mode={"choices": ["dpi", "px/cm"]},
        dpi_value={"label": "DPI", "value": 300},
        px_length={"label": "Length (px)", "value": 0.0, "visible": False},
        real_length={"label": "Real length", "value": 1.0, "visible": False},
        real_unit={
            "choices": ["mm", "cm", "µm"],
            "label": "Unit",
            "visible": False,
        },
        call_button="Set Scale",
    )
    def set_scale(
        mode: str = "dpi",
        dpi_value: float = 300,
        px_length: float = 0.0,
        real_length: float = 1.0,
        real_unit: str = "cm",
    ):
        """
        Set image scale either by specifying DPI or by measuring a drawn line.

        The resulting scale (cm/px) is applied to all visible image layers.
        """
        # Use the first visible image layer or the active layer
        imgs = [ly for ly in viewer.layers if ly.visible and ly._type_string == "image"]
        target_img = imgs[0] if imgs else viewer.layers.selection.active
        if target_img is None or getattr(target_img, "_type_string", "") != "image":
            show_info("Select an image layer to set the scale.")
            return

        # Save camera state for zoom consistency
        cam = viewer.camera
        old_zoom = cam.zoom
        old_center = np.array(cam.center) if cam.center is not None else None

        # Determine px/cm ratio
        try:
            old_cm_per_px = float(target_img.scale[0])
            if not np.isfinite(old_cm_per_px) or old_cm_per_px <= 0:
                old_cm_per_px = None
        except Exception:
            old_cm_per_px = None

        # --- px/cm を決める ---
        if mode == "dpi":
            px_per_cm = dpi_value / 2.54
        elif mode == "px/cm":
            measured_px = _get_measurement_px()
            if measured_px is not None and measured_px > 0:
                set_scale.px_length.value = measured_px
                px = measured_px
            else:
                px = float(px_length)
            if px <= 0:
                show_info("Draw a measurement line in the Scale Measurement layer.")
                return

            # Unit conversion to cm
            if real_unit == "cm":
                real_cm = float(real_length)
            elif real_unit == "mm":
                real_cm = float(real_length) / 10.0
            elif real_unit == "µm":
                real_cm = float(real_length) / 10000.0
            else:
                show_info("Unknown unit.")
                return

            if real_cm <= 0:
                show_info("Real length must be positive.")
                return

            px_per_cm = px / real_cm

        # Apply scale to main image layer
        cm_per_px = 1.0 / px_per_cm
        target_img.scale = (cm_per_px, cm_per_px)

        meta = target_img.metadata
        meta["px_per_cm"] = px_per_cm
        if mode == "dpi":
            meta["scale_unit"] = "cm"
            meta["last_dpi"] = float(set_scale.dpi_value.value)
        else:
            meta["scale_unit"] = real_unit
            meta["last_px_per_cm"] = float(set_scale.px_length.value)
            meta["last_real_length"] = float(set_scale.real_length.value)
            meta["last_real_unit"] = set_scale.real_unit.value

        # Apply same scale to all visible layers
        for ly in viewer.layers:
            if ly is target_img or not ly.visible:
                continue
            try:
                if getattr(ly, "rgb", False) or ly.ndim == 2:
                    ly.scale = (cm_per_px, cm_per_px)
                elif ly.ndim == 3:
                    ly.scale = (1.0, cm_per_px, cm_per_px)
                else:
                    ly.scale = tuple(cm_per_px for _ in range(ly.ndim))

                ly.metadata["px_per_cm"] = px_per_cm
                ly.metadata["scale_unit"] = "cm" if mode == "dpi" else real_unit
            except Exception:
                pass

        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "cm" if mode == "dpi" else real_unit
        viewer.scale_bar.length = (
            1 if real_unit == "cm" else 10 if real_unit == "mm" else 1000
        )

        # Configure scale bar
        viewer.scale_bar.colored = True
        viewer.scale_bar.color = (255 / 255, 75 / 255, 0 / 255)  # Orange
        viewer.scale_bar.box = True
        viewer.scale_bar.box_color = (0, 0, 0, 0.2)
        viewer.scale_bar.position = "bottom_left"

        # Adjust zoom to maintain appearance
        try:
            if (
                old_cm_per_px is not None
                and np.isfinite(old_cm_per_px)
                and old_cm_per_px > 0
            ):
                # Adjust zoom based on old scale
                cam.zoom = float(old_zoom) * (old_cm_per_px / cm_per_px)
            else:
                cam.zoom = float(old_zoom) * (1.0 / cm_per_px)
            if old_center is not None:
                cam.center = old_center
            # Set center in px coordinates
            cam.center = old_center / px_per_cm
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI field visibility toggle
    # ------------------------------------------------------------------
    def _toggle_fields(event=None):
        """Toggle between DPI mode and px/cm mode input fields."""
        m = set_scale.mode.value
        dpi_fields = (set_scale.dpi_value,)
        pxcm_fields = (
            set_scale.px_length,
            set_scale.real_length,
            set_scale.real_unit,
        )
        if m == "dpi":
            for w in dpi_fields:
                w.visible = True
            for w in pxcm_fields:
                w.visible = False
        else:
            for w in dpi_fields:
                w.visible = False
            for w in pxcm_fields:
                w.visible = True

    set_scale.mode.changed.connect(_toggle_fields)
    _toggle_fields()  # Initial call

    ui = Container(widgets=[add_btn, set_scale], layout="vertical")
    return ui
