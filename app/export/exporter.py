"""Video exporter - renders final video with all annotations and audio baked in."""

import os
import subprocess
import tempfile
import cv2
import numpy as np

from app.video.player import VideoPlayer
from app.annotations.manager import AnnotationManager
from app.annotations.renderer import AnnotationRenderer
from app.pose.estimator import PoseEstimator
from app.pose.renderer import PoseRenderer
from app.pose.analysis import FormAnalysis
from app.audio.recorder import AudioRecorder


class VideoExporter:
    """Exports the reviewed video with all overlays composited and audio muxed."""

    def export(self, output_path: str, video_path: str,
               annotation_manager: AnnotationManager = None,
               annotation_renderer: AnnotationRenderer = None,
               pose_estimator: PoseEstimator = None,
               pose_renderer: PoseRenderer = None,
               form_analysis: FormAnalysis = None,
               pose_cache: dict = None,
               pose_enabled: bool = False,
               velocity_tracker=None,
               audio_recorder: AudioRecorder = None,
               progress_callback=None):
        """Export the complete annotated video.

        Args:
            output_path: Path for the output MP4 file.
            video_path: Path to the source video.
            annotation_manager: Annotation data.
            annotation_renderer: Renderer for annotations.
            pose_estimator: For pose detection (if not cached).
            pose_renderer: For drawing skeleton overlay.
            form_analysis: For form issue detection.
            pose_cache: Dict of frame_index -> landmarks.
            pose_enabled: Whether to include pose overlay.
            velocity_tracker: For drawing velocity markers.
            audio_recorder: For muxing audio clips.
            progress_callback: Called with percentage (0-100).
        """
        player = VideoPlayer()
        if not player.load(video_path):
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = player.total_frames
        fps = player.fps
        width = player.width
        height = player.height

        # Write video without audio first
        temp_video = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError("Failed to create video writer")

        for i in range(total_frames):
            frame = player.get_frame(i)
            if frame is None:
                break

            # Composite annotations
            if annotation_renderer and annotation_manager:
                frame = annotation_renderer.render(frame, annotation_manager, i)

            # Composite pose skeleton
            if pose_enabled and pose_renderer:
                landmarks = None
                if pose_cache and i in pose_cache:
                    landmarks = pose_cache[i]
                elif pose_estimator:
                    landmarks = pose_estimator.process(frame)

                if landmarks is not None:
                    analysis_result = None
                    if form_analysis:
                        analysis_result = form_analysis.analyze(landmarks, frame.shape)
                    frame = pose_renderer.render(frame, landmarks, analysis_result)

            # Draw velocity marker
            if velocity_tracker and velocity_tracker.is_tracking:
                frame = velocity_tracker.render_marker(frame, i)

            writer.write(frame)

            if progress_callback and total_frames > 0:
                pct = int((i + 1) / total_frames * 90)  # 0-90% for video
                progress_callback(pct)

        writer.release()
        player.release()

        # Mux audio if available
        has_audio = audio_recorder and len(audio_recorder.clips) > 0
        if has_audio:
            self._mux_audio(temp_video, output_path, audio_recorder, fps, total_frames)
            os.unlink(temp_video)
        else:
            # Just rename the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.rename(temp_video, output_path)

        if progress_callback:
            progress_callback(100)

    def _mux_audio(self, video_path: str, output_path: str,
                   audio_recorder: AudioRecorder, fps: float, total_frames: int):
        """Mux audio clips into the video using ffmpeg."""
        duration = total_frames / fps

        # Build a combined audio track
        # For simplicity, use the first audio clip with offset
        clips = audio_recorder.clips
        if not clips:
            os.rename(video_path, output_path)
            return

        # Build ffmpeg command
        # Use the first clip with its frame offset as a delay
        clip = clips[0]
        delay_seconds = clip.start_frame / fps

        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', clip.file_path,
                '-filter_complex',
                f'[1:a]adelay={int(delay_seconds * 1000)}|{int(delay_seconds * 1000)}[a]',
                '-map', '0:v',
                '-map', '[a]',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]

            # If multiple clips, we need a more complex filter
            if len(clips) > 1:
                inputs = ['-i', video_path]
                filter_parts = []
                for i, c in enumerate(clips):
                    inputs.extend(['-i', c.file_path])
                    d = c.start_frame / fps
                    filter_parts.append(
                        f'[{i + 1}:a]adelay={int(d * 1000)}|{int(d * 1000)}[a{i}]'
                    )

                mix_inputs = ''.join(f'[a{i}]' for i in range(len(clips)))
                filter_parts.append(
                    f'{mix_inputs}amix=inputs={len(clips)}:duration=longest[aout]'
                )

                cmd = [
                    'ffmpeg', '-y',
                    *inputs,
                    '-filter_complex', ';'.join(filter_parts),
                    '-map', '0:v',
                    '-map', '[aout]',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    output_path
                ]

            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ffmpeg not available or failed - just copy video without audio
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.rename(video_path, output_path)

    @staticmethod
    def open_containing_folder(file_path: str):
        """Open the folder containing the exported file."""
        folder = os.path.dirname(os.path.abspath(file_path))
        if os.name == 'nt':  # Windows
            os.startfile(folder)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', folder])
