"""Annotation manager - stores and manages all annotations per frame."""

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Annotation:
    """A single annotation on the video."""
    tool_type: str          # "line", "angle", "circle", "curve", "freehand", "arrow", "text"
    points: list            # List of (x, y) tuples
    color: tuple = (0, 255, 0)  # BGR
    thickness: int = 2
    text: str = ""          # For text annotations
    angle_degrees: float = 0.0  # For angle tool
    start_frame: int = 0
    end_frame: int = -1     # -1 means persist until changed

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class AnnotationManager:
    """Stores all annotations with per-frame visibility and undo/redo."""

    def __init__(self):
        self._annotations: list[Annotation] = []
        self._undo_stack: list[list[Annotation]] = []
        self._redo_stack: list[list[Annotation]] = []
        self._temp_annotation: Annotation | None = None  # In-progress drawing

    @property
    def annotations(self):
        return self._annotations

    @property
    def temp_annotation(self):
        return self._temp_annotation

    @temp_annotation.setter
    def temp_annotation(self, ann):
        self._temp_annotation = ann

    def add(self, annotation: Annotation, current_frame: int):
        """Add a completed annotation."""
        self._save_undo_state()
        annotation.start_frame = current_frame
        if annotation.end_frame == -1:
            annotation.end_frame = current_frame + 300  # ~10 seconds at 30fps default
        self._annotations.append(annotation)
        self._redo_stack.clear()

    def get_visible(self, frame_index: int) -> list[Annotation]:
        """Get all annotations visible at given frame."""
        visible = []
        for ann in self._annotations:
            if ann.start_frame <= frame_index <= ann.end_frame:
                visible.append(ann)
        return visible

    def remove_last(self):
        """Remove the most recently added annotation."""
        if self._annotations:
            self._save_undo_state()
            self._annotations.pop()
            self._redo_stack.clear()

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(list(self._annotations))
            self._annotations = self._undo_stack.pop()

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(list(self._annotations))
            self._annotations = self._redo_stack.pop()

    def clear(self):
        self._save_undo_state()
        self._annotations.clear()
        self._redo_stack.clear()

    def _save_undo_state(self):
        self._undo_stack.append(list(self._annotations))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def to_dict(self) -> list[dict]:
        return [a.to_dict() for a in self._annotations]

    def from_dict_list(self, data: list[dict]):
        self._annotations = [Annotation.from_dict(d) for d in data]
        self._undo_stack.clear()
        self._redo_stack.clear()
