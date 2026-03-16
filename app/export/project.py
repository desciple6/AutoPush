"""Project manager - save and load review sessions."""

import json
import os
from typing import Optional

from app.annotations.manager import AnnotationManager
from app.audio.recorder import AudioRecorder


class ProjectManager:
    """Manages saving and loading .liftreview project files."""

    PROJECT_EXTENSION = ".liftreview"

    def save(self, path: str, video_path: str,
             annotation_manager: AnnotationManager = None,
             audio_recorder: AudioRecorder = None,
             velocity_data: dict = None,
             pose_enabled: bool = False):
        """Save the current review session to a project file.

        Args:
            path: Output .liftreview file path.
            video_path: Path to the source video.
            annotation_manager: Annotation data.
            audio_recorder: Audio recording data.
            velocity_data: Velocity tracker serialized data.
            pose_enabled: Whether pose overlay was enabled.
        """
        project = {
            'version': 1,
            'video_path': video_path,
            'pose_enabled': pose_enabled,
            'annotations': [],
            'audio_clips': [],
            'velocity': None,
        }

        if annotation_manager:
            project['annotations'] = annotation_manager.to_dict()

        if audio_recorder:
            project['audio_clips'] = audio_recorder.to_dict()

        if velocity_data:
            project['velocity'] = velocity_data

        with open(path, 'w') as f:
            json.dump(project, f, indent=2)

    def load(self, path: str) -> dict:
        """Load a project file and return the project data.

        Returns dict with keys: video_path, annotations, audio_clips,
        velocity, pose_enabled.
        """
        with open(path, 'r') as f:
            project = json.load(f)

        return project

    @classmethod
    def get_filter(cls) -> str:
        """Get file dialog filter string."""
        return f"LiftReview Projects (*{cls.PROJECT_EXTENSION});;All Files (*)"
