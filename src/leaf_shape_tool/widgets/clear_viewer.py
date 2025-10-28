"""
clear_viewer.py
---------------
A napari widget for safely resetting or clearing layers in the viewer.

Features:
- Soft reset: Keep base image and ROI layers.
- Hard reset: Optionally save a composite screenshot, then clear all layers.
- Asynchronous layer clearing to avoid race conditions with Qt draw cycles.
- Automatic UI updates for layer visibility and saving options.

Author: Maple
License: BSD-3-Clause
"""

import napari
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
from napari.utils.notifications import show_info, show_warning

from pathlib import Path
from napari.layers import Image as NapariImage
from PIL import Image as PILImage
from typing import Sequence

from qtpy.QtCore import QTimer


def _clear_except_async(viewer, keep_names: set[str], on_done=None):
    """
    Safely remove all layers except those specified, outside the current draw cycle.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    keep_names : set[str]
        Set of layer names to retain.
    on_done : callable, optional
        Callback to execute after clearing is complete.
    """
    draw_emitter = getattr(viewer.window._qt_viewer.canvas.events, "draw", None)
    draw_blocker = draw_emitter.blocker() if draw_emitter else None

    qcanvas = viewer.window._qt_viewer.canvas.native
    if hasattr(qcanvas, "setUpdatesEnabled"):
        qcanvas.setUpdatesEnabled(False)

    # Attempt to pause layer list updates if available
    qview = None
    try:
        qv = getattr(viewer.window, "_qt_viewer", None)
        for cand in (
            getattr(qv, "layerList", None),
            getattr(getattr(qv, "layers", None), "view", None),
            getattr(qv, "layerlist", None),
        ):
            if cand is not None:
                qview = cand
                break
    except Exception:
        qview = None
    if qview is not None and hasattr(qview, "setUpdatesEnabled"):
        qview.setUpdatesEnabled(False)

    def _do():
        entered = False  # flag to ensure proper unblocking
        try:
            if draw_blocker is not None:
                draw_blocker.__enter__()
                entered = True  # guarantees __exit__

            # Remove unwanted layers
            for lyr in list(viewer.layers):
                if lyr.name not in keep_names:
                    viewer.layers.remove(lyr)

            # Reset view and selection
            if len(viewer.layers) > 0:
                viewer.reset_view()
            if "ROIs" in viewer.layers:
                viewer.layers.selection.active = viewer.layers["ROIs"]
            else:
                viewer.layers.selection = []

        finally:
            # Restore drawing and updates
            if entered:
                draw_blocker.__exit__(None, None, None)

            # Re-enable UI updates
            if hasattr(qcanvas, "setUpdatesEnabled"):
                qcanvas.setUpdatesEnabled(True)
            if qview is not None and hasattr(qview, "setUpdatesEnabled"):
                qview.setUpdatesEnabled(True)

            # Force repaint
            try:
                qv = getattr(viewer.window, "_qt_viewer", None)
                if qv is not None and hasattr(qv, "update"):
                    qv.update()
                elif (
                    qv is not None
                    and hasattr(qv, "canvas")
                    and hasattr(qv.canvas, "update")
                ):
                    qv.canvas.update()
                elif hasattr(qcanvas, "update"):
                    qcanvas.update()
                else:
                    qwin = getattr(viewer.window, "_qt_window", None)
                    if qwin is not None and hasattr(qwin, "update"):
                        qwin.update()
            except Exception:
                pass

            if callable(on_done):
                on_done()

    # Execute asynchronously to avoid draw conflicts
    from qtpy.QtCore import QTimer

    QTimer.singleShot(80, _do)


def make_clear_viewer_widget(viewer: "napari.Viewer", on_hard_reset=None):
    """
    Create a widget that resets the napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    on_hard_reset : callable, optional
        Callback function executed after a full reset.

    Returns
    -------
    magicgui.widgets.Widget
        Configured clear/reset widget.
    """

    def _get_base_image_id() -> str:
        """Return the image_id of the base image layer if available."""
        for layer in viewer.layers:
            if isinstance(layer, NapariImage):
                meta = getattr(layer, "metadata", {})
                image_id = meta.get("image_id", layer.name)
                return image_id
        return "unknown"

    # ---- create FileEdit with dynamic default path ----
    default_image_id = _get_base_image_id()
    default_dir = Path("output/rois")
    # default_dir.mkdir(parents=True, exist_ok=True)
    default_path = default_dir / f"{default_image_id}.png"

    def _save_composite_cropped_by_base(
        viewer: "napari.Viewer",
        path: Path,
        include: Sequence[str],
        scale: float = 1.0,
        # clear_after: bool = False,
    ) -> bool:
        """
        Save a composite image tightly cropped to the data extent
        of specified visible layers.

        Parameters
        ----------
        viewer : napari.Viewer
            Napari viewer instance.
        path : Path
            Output file path. Defaults to `.png` if no extension is provided.
        include : Sequence[str]
            Names of layers to include in the composite.
        scale : float, optional
            Scale factor for `viewer.export_figure()`. Default is 1.0.

        Returns
        -------
        bool
            True if saved successfully, otherwise False.
        """
        # --- Pre-checks ---
        if scale <= 0:
            show_warning("Scale must be > 0. Using 1.0 instead.")
            scale = 1.0

        # Backup visibility states
        vis_backup = {ly.name: ly.visible for ly in viewer.layers}

        # Handle text label visibility (for ROI labels)
        rois_label_layer = (
            viewer.layers["ROIs_label"] if "ROIs_label" in viewer.layers else None
        )
        text_vis_backup = None
        if rois_label_layer is not None and hasattr(rois_label_layer, "text"):
            try:
                text_vis_backup = rois_label_layer.text.visible
            except Exception:
                text_vis_backup = None

        # Show only specified layers
        try:
            for ly in viewer.layers:
                ly.visible = ly.name in include

            # Force text visibility if present
            if rois_label_layer is not None and hasattr(rois_label_layer, "text"):
                try:
                    rois_label_layer.text.visible = True
                except Exception:
                    pass

            # --- Get a tight-to-data image ---
            rgba = viewer.export_figure(
                scale_factor=scale
            )  # ndarray (H, W, 4), dtype=uint8

            # --- Save to disk ---
            if path.suffix.lower() not in (
                ".png",
                ".tif",
                ".tiff",
                ".jpg",
                ".jpeg",
                ".bmp",
            ):
                path = path.with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)

            PILImage.fromarray(rgba).save(str(path))
            show_info(f"Saved (exact-crop): {path}")

            return True

        except Exception as e:
            # Restore state then notify on failure
            show_warning(f"Failed to save composite: {e}")
            return False

        finally:
            # Restore visibility states
            for ly in viewer.layers:
                if ly.name in vis_backup:
                    ly.visible = vis_backup[ly.name]

            # --- Restore text visibility if backed up ---
            if (
                rois_label_layer is not None
                and hasattr(rois_label_layer, "text")
                and text_vis_backup is not None
            ):
                try:
                    rois_label_layer.text.visible = text_vis_backup
                except Exception:
                    pass

    @magicgui(
        call_button="Reset Viewer",
        check_keep_base={
            "widget_type": "CheckBox",
            "label": "Keep base image && 'ROIs'",
            "value": True,
        },
        save_rois={
            "widget_type": "CheckBox",
            "label": "Save ROIs (Image + ROIs + ROIs_label)",
            "value": True,
        },
        save_path={
            "widget_type": "FileEdit",
            "label": "Save As",
            "mode": "w",  # choose a file to write
            "value": str(default_path),
            "tooltip": "Specify file path to save composite screenshot",
        },
    )
    def clear_viewer_with_confirmation(
        check_keep_base: bool = True,
        save_rois: bool = True,
        save_path: Path = default_path,
    ):
        """
        Reset the viewer.
        When `check_keep_base` is False, prompt the user for confirmation and
        optionally save a composite image before clearing all layers.
        """

        # --- UI visibility gating ---
        off = not check_keep_base
        clear_viewer_with_confirmation.save_rois.visible = off
        clear_viewer_with_confirmation.save_path.visible = off and save_rois

        # --- Hard reset mode ---
        if not check_keep_base:
            # --- Comfirmation dialog ---
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Confirm Reset")
            msg.setText("Do you want to delete all layers and reset the viewer?")
            msg.setInformativeText("This operation cannot be undone.")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            result = msg.exec_()

            if result == QMessageBox.Yes:
                # If checked, save composite AFTER confirmation
                if save_rois:
                    # refresh default filename with current image_id + timestamp
                    image_id = _get_base_image_id()
                    # If user didn't change save_path, auto-fill a sensible name
                    if not save_path:
                        sp = default_dir / f"{image_id}.png"
                        final_out = sp
                    else:
                        sp = Path(save_path)
                        # If user selected only a directory, append a filename
                        if sp.exists() and sp.is_dir():
                            final_out = sp / f"{image_id}.png"
                        else:
                            final_out = (
                                sp.with_suffix(".png") if sp.suffix == "" else sp
                            )

                    try:
                        wanted = ["ROIs", "ROIs_label", image_id]
                        include_list = [n for n in wanted if n in viewer.layers]
                        viewer.layers.selection.active = (
                            viewer.layers[image_id]
                            if image_id in viewer.layers
                            else None
                        )
                        _save_composite_cropped_by_base(
                            viewer,
                            final_out,
                            include=include_list,
                            scale=0.5,  # 0.5x is usually enough
                        )
                    except Exception as e:
                        show_info(f"Failed to save composite: {e}")

                # Clear all layers asynchronously
                _clear_except_async(
                    viewer,
                    keep_names=set(),
                    on_done=on_hard_reset,
                )
                # After clearing, set the checkbox back to True
                QTimer.singleShot(
                    0,
                    lambda: setattr(
                        clear_viewer_with_confirmation.check_keep_base,
                        "value",
                        True,
                    ),
                )
            else:
                print("Operation cancelled.")
            return

        # --- Soft reset mode ---
        active = viewer.layers.selection.active
        base_layer_name = None
        if active is not None:
            base_layer_name = active.metadata.get("from_layer", None)

        existing_names = {ly.name for ly in viewer.layers}
        keep_names = {
            n for n in (base_layer_name, "ROIs", "ROIs_label") if n is not None
        }
        keep_names = keep_names & existing_names

        if not keep_names:
            show_info("Base image layer not found from metadata; nothing to keep.")

        # --- Confirmation dialog ---
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Confirm Reset")
        msg.setText("Do you want to delete all layers except specified ones?")
        keep_preview = ", ".join(sorted(keep_names)) if keep_names else "(none)"
        msg.setInformativeText(f"The following layers will be kept:\n{keep_preview}")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        if msg.exec_() == QMessageBox.Yes:
            for lyr in list(viewer.layers):
                if lyr.name not in keep_names:
                    lyr.visible = False
            _clear_except_async(viewer, keep_names)
            print("Layers cleared except:", keep_names)
        else:
            print("Operation cancelled.")

    # --- Reactive UI updates ---
    def _update_visibility_on_toggle(event=None):
        """Update visibility of save options based on checkbox state."""
        off = not clear_viewer_with_confirmation.check_keep_base.value
        clear_viewer_with_confirmation.save_rois.visible = off
        clear_viewer_with_confirmation.save_path.visible = (
            off and clear_viewer_with_confirmation.save_rois.value
        )

        # Also refresh default filename with current image_id when fields become visible.
        if off:
            image_id = _get_base_image_id()
            default_dir.mkdir(parents=True, exist_ok=True)
            clear_viewer_with_confirmation.save_path.value = str(
                (default_dir / f"{image_id}.png")
            )

    # Connect toggles
    clear_viewer_with_confirmation.check_keep_base.changed.connect(
        _update_visibility_on_toggle
    )
    clear_viewer_with_confirmation.save_rois.changed.connect(
        _update_visibility_on_toggle
    )

    # Initialize visibility once
    _update_visibility_on_toggle()

    return clear_viewer_with_confirmation
