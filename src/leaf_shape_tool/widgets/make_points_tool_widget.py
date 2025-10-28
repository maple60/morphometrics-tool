"""
make_points_tool_widget.py
--------------------------
A napari widget providing interactive tools for managing landmark Points layers.

This widget allows users to:
- Switch between predefined point labels (e.g., base, tip)
- Automatically advance label selection when adding new points
- Undo or clear points
- Display live counts of labeled points per category

Intended for use in shape analysis pipelines (e.g., leaf morphometrics)
where consistent labeling of anatomical landmarks is required.

Author: Maple
License: BSD-3-Clause
"""

from magicgui.widgets import ComboBox, Container, PushButton, CheckBox, Label
from napari.layers import Points
from napari.utils.notifications import show_info
import numpy as np
import pandas as pd
from typing import Sequence
import napari

DEFAULT_LABELS: Sequence[str] = ("base", "tip")


def make_points_tools_widget(
    viewer: "napari.Viewer",
    labels: Sequence[str] = DEFAULT_LABELS,
    *,
    area: str = "left",
    name: str = "Point Tools",
) -> Container:
    """
    Create an interactive napari dock widget for manipulating Points layers.

    Features:
    ----------
    - Switch between predefined labels (e.g., base/tip)
    - Auto-advance to the next label after point creation
    - Undo or clear points
    - Display live label counts

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    labels : Sequence[str], optional
        List of label names for point annotation. Default is ("base", "tip").
    area : str, optional
        Docking area of the widget in the napari window. Default is "left".
    name : str, optional
        Display name of the widget. Default is "Point Tools".

    Returns
    -------
    magicgui.widgets.Container
        Configured point tool widget.
    """

    # --- UI elements ---
    menu = ComboBox(
        label="label", choices=list(labels), value=labels[0], nullable=False
    )
    auto_adv = CheckBox(text="auto advance", value=True)
    btn_undo = PushButton(text="Undo last point")
    btn_clear = PushButton(text="Clear points")
    counts = Label(value=f"{labels[0]}: 0 | {labels[1]}: 0")

    dock = Container(
        widgets=[
            menu,
            auto_adv,
            btn_undo,
            btn_clear,
            counts,
        ]
    )

    # Store last known number of points per layer (for auto-advance logic)
    last_len = {}

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------

    def active_points() -> Points | None:
        """Return the active Points layer, if any."""
        lyr = viewer.layers.selection.active
        return lyr if isinstance(lyr, Points) else None

    # Points レイヤの 'label' 列と色設定を保証
    def ensure_label_feature(points: Points):
        """Ensure that the Points layer has a categorical 'label' feature with correct colors."""
        if "label" not in points.features.columns:
            points.features = pd.DataFrame(
                {"label": pd.Categorical([], categories=list(labels))}
            )
        else:
            col = points.features["label"]
            if not pd.api.types.is_categorical_dtype(col):
                points.features["label"] = pd.Categorical(
                    col.astype(str).values, categories=list(labels)
                )
            else:
                pass
        points.border_color = "label"
        points.border_color_mode = "cycle"
        COLOR_CYCLE = ["#FF4B00", "#005AFF"]
        points.border_color_cycle = COLOR_CYCLE[: len(labels)]

    def update_menu_from_points(points: Points):
        """Sync dropdown menu with the current default label of the Points layer."""
        if points is None:
            return
        ensure_label_feature(points)
        curr = str(points.feature_defaults.get("label", [labels[0]])[0])
        if curr in list(labels) and menu.value != curr:
            menu.value = curr

    def update_counts(points: Points):
        """Update the count display for each label."""
        if points is None or len(points.data) == 0 or "label" not in points.features:
            counts.value = f"{labels[0]}: 0 | {labels[1]}: 0"
            return
        ser = points.features["label"].astype(str)
        n0 = int((ser == labels[0]).sum())
        n1 = int((ser == labels[1]).sum())
        counts.value = f"{labels[0]}: {n0} | {labels[1]}: {n1}"

    def set_default_label(points: Points, value: str):
        """Set the default label for newly added points."""
        if points is None:
            return
        ensure_label_feature(points)
        defaults = points.feature_defaults.copy()
        defaults["label"] = value
        points.feature_defaults = defaults
        points.refresh_colors()

    def advance_label(points: Points):
        """Advance to the next label in the dropdown (cyclically)."""
        ch = list(menu.choices)
        i = ch.index(menu.value)
        menu.value = ch[(i + 1) % len(ch)]
        points.feature_defaults = {"label": [menu.value]}
        show_info(f"Label advanced to: {menu.value}")
        points.refresh_colors()

    # ---------------------------------------------------------------------
    # Event connections
    # ---------------------------------------------------------------------

    def on_active_change(event=None):
        """Update menu and counts when the active layer changes."""
        pts = active_points()
        update_menu_from_points(pts)
        update_counts(pts)
        if pts is not None and id(pts) not in last_len:
            last_len[id(pts)] = len(pts.data)

        if pts is not None:

            def _on_data(_event=None, _layer=pts):
                """Triggered on point data changes (for auto-advance and count updates)."""
                update_counts(_layer)
                # auto advance
                if auto_adv.value:
                    prev = last_len.get(id(_layer), 0)
                    now = len(_layer.data)
                    if now > prev and _layer.mode == "add":
                        ensure_label_feature(_layer)
                        feat = _layer.features.copy()
                        if len(feat) > 0:
                            last_idx = feat.index[-1]
                            feat.loc[last_idx, "label"] = str(menu.value)
                            _layer.features = feat
                        _layer.selected_data = set()
                        _layer.refresh_colors()
                        advance_label(_layer)
                    last_len[id(_layer)] = now

            # Avoid double connections
            try:
                pts.events.data.disconnect(_on_data)
            except Exception:
                pass
            pts.events.data.connect(_on_data)

    viewer.layers.selection.events.active.connect(on_active_change)

    def on_menu_changed(v):
        """When dropdown selection changes, update Points feature_defaults."""
        set_default_label(active_points(), v)

    menu.changed.connect(on_menu_changed)

    def on_undo_clicked():
        """Remove the last point."""
        pts = active_points()
        if pts is None or len(pts.data) == 0:
            return
        last_idx = len(pts.data) - 1
        pts.selected_data = {last_idx}
        pts.remove_selected()
        update_counts(pts)
        last_len[id(pts)] = len(pts.data)

    btn_undo.changed.connect(on_undo_clicked)

    def on_clear_clicked():
        """Clear all points and reset the layer."""
        pts = active_points()
        if pts is None:
            return
        pts.data = np.zeros((0, 2), dtype=float)
        if "label" in pts.features:
            pts.features = pd.DataFrame(
                {"label": pd.Categorical([], categories=list(labels))}
            )
        update_counts(pts)
        last_len[id(pts)] = 0

    btn_clear.changed.connect(on_clear_clicked)

    # Initial synchronization
    on_active_change()

    dock = Container(widgets=[menu, auto_adv, btn_undo, btn_clear, counts])
    return dock
