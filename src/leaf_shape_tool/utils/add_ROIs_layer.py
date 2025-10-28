from napari.layers import Image
from qtpy.QtCore import QTimer


def add_ROIs(viewer, event):
    """
    画像レイヤが追加されたときに、ROI描画用のShapesレイヤを追加する。
    viewer.layers.events.inserted.connect に接続して使う。

    Add a Shapes layer for drawing ROIs when a new image layer is added.

    Parameters
    ----------
    event : napari.utils.events.Event
        The event object containing information about the inserted layer.
    Returns
    -------
    None
    """
    layer = event.value

    # すでに 'ROIs' レイヤがあれば何もしない
    if "ROIs" in viewer.layers:
        return

    # 画像レイヤでなければ処理しない
    if not isinstance(layer, Image):
        return

    # 画像レイヤにメタデータを追加
    try:
        if getattr(layer, "source", None) and getattr(
            layer.source, "path", None
        ):
            #print(f"Layer source path: {layer.source.path}")
            layer.metadata["source_path"] = str(layer.source.path)
    except Exception as e:
        print(f"Error accessing layer source path: {e}")

    # ROI描画用の Shapes レイヤを追加
    ROIs_layer = viewer.add_shapes(
        name="ROIs",
        shape_type="rectangle",  # ツールで変更可。'polygon' でもOK
        edge_width=10,
        edge_color="#FF4B00",  # オレンジ
        face_color="transparent",
        metadata={**layer.metadata, "source_image": layer.name},
        scale=layer.scale,
    )
    ROIs_layer.mode = "add_rectangle"  # 最初から矩形を描けるように
    # viewer.layers.selection.active = ROIs_layer  # 追加したレイヤをアクティブに
    QTimer.singleShot(
        0, lambda: viewer.layers.selection.select_only(ROIs_layer)
    )
