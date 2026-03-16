"""Annotation toolbar - tool selection UI."""

from PyQt6.QtWidgets import (
    QToolBar, QToolButton, QButtonGroup, QColorDialog,
    QInputDialog, QWidget, QVBoxLayout, QLabel, QSpinBox,
    QHBoxLayout
)
from PyQt6.QtGui import QAction, QIcon, QColor
from PyQt6.QtCore import pyqtSignal, Qt

from app.annotations.tools import (
    LineTool, ArrowTool, CircleTool, AngleTool,
    FreehandTool, CurveTool, TextTool
)


class AnnotationToolbar(QToolBar):
    """Toolbar for selecting annotation drawing tools."""

    tool_changed = pyqtSignal(object)  # Emits the active tool instance

    TOOL_DEFS = [
        ("Line", "Draw a straight line", LineTool),
        ("Arrow", "Draw an arrow", ArrowTool),
        ("Circle", "Draw a circle", CircleTool),
        ("Angle", "Measure an angle (3 clicks)", AngleTool),
        ("Freehand", "Freehand drawing", FreehandTool),
        ("Curve", "Bezier curve (4 clicks)", CurveTool),
        ("Text", "Add text annotation", TextTool),
    ]

    def __init__(self, parent=None):
        super().__init__("Annotations", parent)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(False)

        self._color = (0, 255, 0)  # BGR green
        self._thickness = 2
        self._tools = {}
        self._active_tool = None
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        # Add tool buttons
        for name, tooltip, tool_cls in self.TOOL_DEFS:
            btn = QToolButton()
            btn.setText(name)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setMinimumWidth(80)
            self._button_group.addButton(btn)

            tool = tool_cls(color=self._color, thickness=self._thickness)
            self._tools[name] = tool
            btn.clicked.connect(lambda checked, n=name: self._on_tool_selected(n))
            self.addWidget(btn)

        self.addSeparator()

        # Color button
        color_btn = QToolButton()
        color_btn.setText("Color")
        color_btn.setToolTip("Change annotation color")
        color_btn.setMinimumWidth(80)
        color_btn.clicked.connect(self._pick_color)
        self.addWidget(color_btn)

        # Thickness control
        thickness_widget = QWidget()
        thickness_layout = QVBoxLayout(thickness_widget)
        thickness_layout.setContentsMargins(4, 2, 4, 2)
        thickness_label = QLabel("Width:")
        thickness_layout.addWidget(thickness_label)
        self._thickness_spin = QSpinBox()
        self._thickness_spin.setRange(1, 10)
        self._thickness_spin.setValue(2)
        self._thickness_spin.valueChanged.connect(self._on_thickness_changed)
        thickness_layout.addWidget(self._thickness_spin)
        self.addWidget(thickness_widget)

        self.addSeparator()

        # Undo / Redo / Clear
        undo_btn = QToolButton()
        undo_btn.setText("Undo")
        undo_btn.setToolTip("Undo last annotation (Ctrl+Z)")
        undo_btn.setMinimumWidth(80)
        undo_btn.clicked.connect(self._on_undo)
        self.addWidget(undo_btn)

        redo_btn = QToolButton()
        redo_btn.setText("Redo")
        redo_btn.setToolTip("Redo (Ctrl+Y)")
        redo_btn.setMinimumWidth(80)
        redo_btn.clicked.connect(self._on_redo)
        self.addWidget(redo_btn)

        clear_btn = QToolButton()
        clear_btn.setText("Clear All")
        clear_btn.setToolTip("Remove all annotations")
        clear_btn.setMinimumWidth(80)
        clear_btn.clicked.connect(self._on_clear)
        self.addWidget(clear_btn)

        # Callbacks for undo/redo/clear (set by main window)
        self.undo_callback = None
        self.redo_callback = None
        self.clear_callback = None

    def _on_tool_selected(self, name):
        tool = self._tools.get(name)
        if tool:
            if isinstance(tool, TextTool):
                text, ok = QInputDialog.getText(
                    self, "Text Annotation", "Enter text:"
                )
                if ok and text:
                    tool.set_text(text)
                else:
                    return
            self._active_tool = tool
            self.tool_changed.emit(tool)

    def _pick_color(self):
        # Convert BGR to QColor (RGB)
        qcolor = QColor(self._color[2], self._color[1], self._color[0])
        color = QColorDialog.getColor(qcolor, self, "Annotation Color")
        if color.isValid():
            self._color = (color.blue(), color.green(), color.red())  # BGR
            for tool in self._tools.values():
                tool.color = self._color

    def _on_thickness_changed(self, value):
        self._thickness = value
        for tool in self._tools.values():
            tool.thickness = value

    def _on_undo(self):
        if self.undo_callback:
            self.undo_callback()

    def _on_redo(self):
        if self.redo_callback:
            self.redo_callback()

    def _on_clear(self):
        if self.clear_callback:
            self.clear_callback()
