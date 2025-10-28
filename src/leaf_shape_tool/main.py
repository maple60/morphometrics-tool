"""
Main entry point for the Napari-based leaf shape analysis workflow.

This script initializes the Napari viewer and integrates multiple custom widgets
for region-of-interest (ROI) extraction, landmark annotation, image rotation,
binarization, contour extraction, scale calibration, and Elliptic Fourier Descriptor (EFD) analysis.

The GUI layout and event-driven behavior are designed for interactive leaf
image processing and morphometric analysis.

Author: Maple
License: BSD-3-Clause
"""

import napari
from functools import partial
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QDockWidget

# --- Custom widget imports ---
from leaf_shape_tool.widgets import crop_rectangle
from leaf_shape_tool.widgets.make_points_tool_widget import make_points_tools_widget
from leaf_shape_tool.widgets.rotate_image import (
    make_points_metadata_widget,
    get_active_points_layer,
    summarize_points_layer,
)
from leaf_shape_tool.widgets.binarize_image import make_binarize_image_widget
from leaf_shape_tool.widgets.extract_contour import make_extract_contour_widget
from leaf_shape_tool.widgets.set_scale import make_set_scale_widget
from leaf_shape_tool.widgets.calculate_efd import calculate_efd_and_save
from leaf_shape_tool.widgets.clear_viewer import make_clear_viewer_widget
from leaf_shape_tool.utils.add_ROIs_layer import add_ROIs


# ---------------------------------------------------------------------
# Initialize Napari viewer
# ---------------------------------------------------------------------

viewer = napari.Viewer()
# Automatically add ROI layer when a new image layer is added
viewer.layers.events.inserted.connect(partial(add_ROIs, viewer))

# ---------------------------------------------------------------------
# Create and register widgets
# ---------------------------------------------------------------------

roi_widget = crop_rectangle.make_add_roi_widget(viewer)
dock_00 = viewer.window.add_dock_widget(
    roi_widget, name="ROI Extractoin Widget", area="right", tabify=False
)


# Callback to reset SpinBox to 1 on hard reset (check_keep_base=False)
def _reset_roi_spinbox_to_1():
    # Set the SpinBox value to 1 safely
    try:
        roi_widget.roi_index.value = 1  # Reset to 1
    except Exception as e:
        print("Error resetting ROI SpinBox:", e)


# Landmark / point tools widget
dock = make_points_tools_widget(viewer)
viewer.window.add_dock_widget(dock, area="left", name="Point Tools")

# Image rotation widget (based on landmark metadata)
points_meta_widget = make_points_metadata_widget(
    viewer,
    get_active_points_layer,
    summarize_points_layer,
)
dock_01 = viewer.window.add_dock_widget(
    points_meta_widget, name="Rotation Widget", area="right", tabify=False
)

# Binarization widget (Otsu / SAM2)
binarize_widget = make_binarize_image_widget(viewer)
dock_02 = viewer.window.add_dock_widget(
    binarize_widget, name="Binarization Widget", area="right", tabify=False
)

# Contour extraction widget
contour_widget = make_extract_contour_widget(viewer)
dock_03 = viewer.window.add_dock_widget(
    contour_widget,
    name="Contour Extraction Widget",
    area="right",
    tabify=False,
)

# Clear viewer widget
clear_viewer_widget = make_clear_viewer_widget(
    viewer, on_hard_reset=_reset_roi_spinbox_to_1
)
dock_04 = viewer.window.add_dock_widget(
    clear_viewer_widget, name="Clear Viewer", area="right", tabify=False
)

# Scale calibration widget
set_scale_widget = make_set_scale_widget(viewer)
viewer.window.add_dock_widget(set_scale_widget, area="left", tabify=False)


# ---------------------------------------------------------------------
# Arrange dock widgets vertically (right side)
# ---------------------------------------------------------------------
def pack_right_docks_top():
    """
    Dynamically resize and align right-side dock widgets vertically.
    Ensures consistent spacing and visibility of all widgets.
    """
    main_window = viewer.window._qt_window  # QMainWindow instance
    qdocks = [
        d
        for d in main_window.findChildren(QDockWidget)
        if main_window.dockWidgetArea(d) == Qt.RightDockWidgetArea
        and not d.isFloating()
    ]
    if not qdocks:
        return

    # Sort by y-position (top to bottom)
    qdocks.sort(key=lambda d: d.geometry().top())

    sizes = []
    for d in qdocks[:-1]:
        h = d.widget().sizeHint().height()
        d.widget().setMinimumHeight(min(h, 300))
        sizes.append(max(h, 150))  # Ensure visibility

    sizes.append(10_000)  # Let the last dock absorb remaining space

    main_window.resizeDocks(qdocks, sizes, Qt.Vertical)


# Trigger dock layout adjustment after GUI initialization
QTimer.singleShot(0, pack_right_docks_top)

# ---------------------------------------------------------------------
# Signal connections
# ---------------------------------------------------------------------
# Trigger EFD calculation automatically after contour extraction
contour_widget.called.connect(calculate_efd_and_save)  # EFD calculation


# Add ROI shortcut (Shift + S)
@viewer.bind_key("Shift-s")
def _add_roi_shortcut(v):
    roi_widget()


# ---------------------------------------------------------------------
# Run Napari viewer
# ---------------------------------------------------------------------
viewer.title = "Leaf Shape Analysis Tool — powered by napari"  # Set window title
napari.run()  # Run the viewer
