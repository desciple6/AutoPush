"""Main application window."""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QMenuBar, QStatusBar,
    QDockWidget, QToolBar, QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence

from app.config import (
    APP_NAME, APP_VERSION, VIDEO_EXTENSIONS,
    WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
)
from app.video.player import VideoPlayer
from app.video.canvas import VideoCanvas
from app.video.controls import PlaybackControls
from app.annotations.manager import AnnotationManager
from app.annotations.renderer import AnnotationRenderer
from app.annotations.toolbar import AnnotationToolbar
from app.audio.recorder import AudioRecorder
from app.tracking.velocity_tracker import VelocityTracker
from app.tracking.velocity_graph import VelocityGraph
from app.pose.estimator import PoseEstimator
from app.pose.renderer import PoseRenderer
from app.pose.analysis import FormAnalysis
from app.export.exporter import VideoExporter
from app.export.project import ProjectManager
from app.workers.threads import PoseWorker, ExportWorker


class MainWindow(QMainWindow):
    """Top-level application window coordinating all modules."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Core components
        self._player = VideoPlayer()
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_speed = 1.0
        self._is_playing = False
        self._is_reverse = False
        self._current_frame = None

        # Annotations
        self._annotation_manager = AnnotationManager()
        self._annotation_renderer = AnnotationRenderer()
        self._active_tool = None

        # Audio
        self._audio_recorder = AudioRecorder()

        # Velocity tracker
        self._velocity_tracker = VelocityTracker()
        self._velocity_graph = VelocityGraph()
        self._velocity_tracking_mode = False

        # Pose estimation
        self._pose_estimator = PoseEstimator()
        self._pose_renderer = PoseRenderer()
        self._form_analysis = FormAnalysis()
        self._pose_worker = PoseWorker()
        self._pose_worker.set_estimator(self._pose_estimator)
        self._pose_worker.result_ready.connect(self._on_pose_result)
        self._pose_enabled = False
        self._pose_cache = {}

        # Export
        self._exporter = VideoExporter()
        self._project_manager = ProjectManager()
        self._export_worker = ExportWorker()

        # Build UI
        self._setup_menu()
        self._setup_canvas()
        self._setup_annotation_toolbar()
        self._setup_velocity_dock()
        self._setup_controls()
        self._setup_statusbar()

        # Apply dark theme
        self._apply_dark_theme()

    # ── UI Setup ──────────────────────────────────────────────

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_project_action = QAction("&Save Project...", self)
        save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        load_project_action = QAction("&Load Project...", self)
        load_project_action.setShortcut(QKeySequence("Ctrl+L"))
        load_project_action.triggered.connect(self._load_project)
        file_menu.addAction(load_project_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Video...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_video)
        file_menu.addAction(export_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self._pose_toggle_action = QAction("Show &Skeleton Overlay", self)
        self._pose_toggle_action.setCheckable(True)
        self._pose_toggle_action.setChecked(False)
        self._pose_toggle_action.triggered.connect(self._toggle_pose)
        view_menu.addAction(self._pose_toggle_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        velocity_action = QAction("&Track Barbell Velocity", self)
        velocity_action.setShortcut(QKeySequence("Ctrl+T"))
        velocity_action.setCheckable(True)
        velocity_action.triggered.connect(self._toggle_velocity_mode)
        tools_menu.addAction(velocity_action)

        clear_tracking_action = QAction("&Clear Velocity Data", self)
        clear_tracking_action.triggered.connect(self._clear_velocity)
        tools_menu.addAction(clear_tracking_action)

    def _setup_canvas(self):
        self._canvas = VideoCanvas()
        self.setCentralWidget(self._canvas)

        # Connect mouse events for annotation tools
        self._canvas.mouse_pressed.connect(self._on_canvas_mouse_press)
        self._canvas.mouse_moved.connect(self._on_canvas_mouse_move)
        self._canvas.mouse_released.connect(self._on_canvas_mouse_release)

    def _setup_annotation_toolbar(self):
        self._annotation_toolbar = AnnotationToolbar()
        self._annotation_toolbar.tool_changed.connect(self._on_annotation_tool_changed)
        self._annotation_toolbar.undo_callback = self._annotation_undo
        self._annotation_toolbar.redo_callback = self._annotation_redo
        self._annotation_toolbar.clear_callback = self._annotation_clear
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self._annotation_toolbar)

    def _setup_velocity_dock(self):
        dock = QDockWidget("Velocity Graph", self)
        dock.setWidget(self._velocity_graph)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_controls(self):
        self._controls = PlaybackControls()
        dock = QDockWidget("Playback", self)
        dock.setWidget(self._controls)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self._controls.play_pause_clicked.connect(self._toggle_play)
        self._controls.step_forward_clicked.connect(self._step_forward)
        self._controls.step_backward_clicked.connect(self._step_backward)
        self._controls.fast_forward_clicked.connect(self._fast_forward)
        self._controls.reverse_clicked.connect(self._reverse)
        self._controls.slider_moved.connect(self._seek)
        self._controls.speed_changed.connect(self._set_speed)
        self._controls.record_toggled.connect(self._toggle_record)

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready - Open a video to begin")

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QMenuBar { background-color: #333; color: #ddd; }
            QMenuBar::item:selected { background-color: #555; }
            QMenu { background-color: #333; color: #ddd; }
            QMenu::item:selected { background-color: #555; }
            QDockWidget { color: #ddd; }
            QDockWidget::title { background-color: #333; padding: 4px; }
            QStatusBar { background-color: #333; color: #aaa; }
            QPushButton {
                background-color: #444; color: #ddd; border: 1px solid #555;
                padding: 4px 8px; border-radius: 3px;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #666; }
            QPushButton:checked { background-color: #336; border-color: #55a; }
            QSlider::groove:horizontal {
                height: 6px; background: #555; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #aaa; width: 14px; margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #5588cc; border-radius: 3px; }
            QComboBox {
                background-color: #444; color: #ddd; border: 1px solid #555;
                padding: 2px 6px; border-radius: 3px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #444; color: #ddd; }
            QLabel { color: #ddd; }
            QToolBar { background-color: #333; border: none; spacing: 4px; }
            QToolButton {
                background-color: #444; color: #ddd; border: 1px solid #555;
                padding: 4px; border-radius: 3px;
            }
            QToolButton:hover { background-color: #555; }
            QToolButton:checked { background-color: #336; border-color: #55a; }
        """)

    # ── Video Playback ────────────────────────────────────────

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", VIDEO_EXTENSIONS
        )
        if path:
            if self._player.load(path):
                self._controls.set_range(self._player.total_frames)
                self._pose_cache.clear()
                frame = self._player.get_frame(0)
                self._current_frame = frame
                self._display_frame(frame)
                self._statusbar.showMessage(
                    f"Loaded: {os.path.basename(path)} | "
                    f"{self._player.width}x{self._player.height} | "
                    f"{self._player.fps:.1f} fps | "
                    f"{self._player.total_frames} frames"
                )
            else:
                QMessageBox.warning(self, "Error", f"Failed to open video:\n{path}")

    def _display_frame(self, frame: np.ndarray):
        """Composite all overlays onto frame and display."""
        if frame is None:
            return

        display = frame.copy()
        frame_idx = self._player.frame_index

        # Phase 2: Render annotations
        if self._annotation_renderer and self._annotation_manager:
            display = self._annotation_renderer.render(
                display, self._annotation_manager, frame_idx
            )

        # Phase 5: Render pose skeleton
        if self._pose_enabled and self._pose_renderer:
            landmarks = self._pose_cache.get(frame_idx)
            if landmarks is not None:
                analysis = None
                if self._form_analysis:
                    analysis = self._form_analysis.analyze(landmarks, frame.shape)
                display = self._pose_renderer.render(display, landmarks, analysis)

        # Phase 4: Render velocity tracker marker
        if self._velocity_tracker and self._velocity_tracker.is_tracking:
            display = self._velocity_tracker.render_marker(display, frame_idx)

        self._canvas.display_composited(display)
        self._controls.set_position(
            frame_idx, self._player.fps, self._player.total_frames
        )

    def _toggle_play(self):
        if not self._player.is_loaded:
            return
        self._is_reverse = False
        if self._is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self._is_playing = True
        self._controls.set_playing(True)
        interval = int(1000 / (self._player.fps * self._playback_speed))
        self._playback_timer.start(max(1, interval))

    def _stop_playback(self):
        self._is_playing = False
        self._controls.set_playing(False)
        self._playback_timer.stop()

    def _on_playback_tick(self):
        if not self._player.is_loaded:
            self._stop_playback()
            return

        if self._is_reverse:
            frame = self._player.prev_frame()
            if self._player.frame_index <= 1:
                self._stop_playback()
        else:
            frame = self._player.next_frame()
            if self._player.frame_index >= self._player.total_frames - 1:
                self._stop_playback()

        if frame is not None:
            self._current_frame = frame
            self._run_pose_if_enabled(frame)
            self._update_velocity_tracking(frame)
            self._display_frame(frame)

    def _step_forward(self):
        if not self._player.is_loaded:
            return
        self._stop_playback()
        frame = self._player.next_frame()
        if frame is not None:
            self._current_frame = frame
            self._run_pose_if_enabled(frame)
            self._update_velocity_tracking(frame)
            self._display_frame(frame)

    def _step_backward(self):
        if not self._player.is_loaded:
            return
        self._stop_playback()
        frame = self._player.prev_frame()
        if frame is not None:
            self._current_frame = frame
            self._run_pose_if_enabled(frame)
            self._display_frame(frame)

    def _fast_forward(self):
        if not self._player.is_loaded:
            return
        self._is_reverse = False
        speeds = [1.0, 2.0, 4.0]
        current_idx = 0
        for i, s in enumerate(speeds):
            if abs(self._playback_speed - s) < 0.01:
                current_idx = i
                break
        next_idx = min(current_idx + 1, len(speeds) - 1)
        self._playback_speed = speeds[next_idx]
        if self._is_playing:
            interval = int(1000 / (self._player.fps * self._playback_speed))
            self._playback_timer.setInterval(max(1, interval))
        else:
            self._start_playback()

    def _reverse(self):
        if not self._player.is_loaded:
            return
        self._is_reverse = True
        self._playback_speed = 1.0
        if not self._is_playing:
            self._start_playback()

    def _seek(self, frame_index: int):
        if not self._player.is_loaded:
            return
        frame = self._player.seek(frame_index)
        if frame is not None:
            self._current_frame = frame
            self._run_pose_if_enabled(frame)
            self._display_frame(frame)

    def _set_speed(self, speed: float):
        self._playback_speed = speed
        if self._is_playing:
            interval = int(1000 / (self._player.fps * speed))
            self._playback_timer.setInterval(max(1, interval))

    # ── Annotations ─────────────────────────────────────────

    def _on_annotation_tool_changed(self, tool):
        self._active_tool = tool
        self._velocity_tracking_mode = False

    def _annotation_undo(self):
        self._annotation_manager.undo()
        if self._current_frame is not None:
            self._display_frame(self._current_frame)

    def _annotation_redo(self):
        self._annotation_manager.redo()
        if self._current_frame is not None:
            self._display_frame(self._current_frame)

    def _annotation_clear(self):
        self._annotation_manager.clear()
        if self._current_frame is not None:
            self._display_frame(self._current_frame)

    def _on_canvas_mouse_press(self, pos):
        if not self._player.is_loaded:
            return
        # Velocity tracking mode: click to set tracking point
        if self._velocity_tracking_mode and self._current_frame is not None:
            self._velocity_tracker.start_tracking(
                self._current_frame, (pos.x(), pos.y()), self._player.frame_index
            )
            self._velocity_tracking_mode = False
            self._statusbar.showMessage("Tracking barbell - play video to collect velocity data")
            self._display_frame(self._current_frame)
            return
        # Annotation mode
        if self._active_tool:
            self._active_tool.on_press(pos, self._player.frame_index)
            self._display_frame(self._current_frame)

    def _on_canvas_mouse_move(self, pos):
        if not self._player.is_loaded:
            return
        if self._active_tool:
            self._active_tool.on_move(pos, self._player.frame_index)
            # Update preview
            preview = self._active_tool.get_preview()
            self._annotation_manager.temp_annotation = preview
            if not self._is_playing:
                self._display_frame(self._current_frame)

    def _on_canvas_mouse_release(self, pos):
        if not self._player.is_loaded:
            return
        if self._active_tool:
            annotation = self._active_tool.on_release(pos, self._player.frame_index)
            self._annotation_manager.temp_annotation = None
            if annotation:
                self._annotation_manager.add(annotation, self._player.frame_index)
            self._display_frame(self._current_frame)

    # ── Audio Recording ──────────────────────────────────────

    def _toggle_record(self, recording: bool):
        if not self._player.is_loaded:
            return
        if recording:
            self._audio_recorder.start(self._player.frame_index)
            self._statusbar.showMessage("Recording voice commentary...")
        else:
            self._audio_recorder.stop(self._player.frame_index)
            self._statusbar.showMessage("Recording saved.")

    # ── Velocity Tracking ────────────────────────────────────

    def _toggle_velocity_mode(self, enabled):
        self._velocity_tracking_mode = enabled
        if enabled:
            self._active_tool = None
            self._statusbar.showMessage("Click on the barbell center to start tracking")
        else:
            self._statusbar.showMessage("")

    def _clear_velocity(self):
        self._velocity_tracker.reset()
        self._velocity_graph.clear()
        self._statusbar.showMessage("Velocity data cleared.")
        if self._current_frame is not None:
            self._display_frame(self._current_frame)

    def _update_velocity_tracking(self, frame):
        if self._velocity_tracker.is_tracking:
            self._velocity_tracker.update(frame, self._player.frame_index)
            self._velocity_graph.update_data(
                self._velocity_tracker.velocities,
                self._velocity_tracker.rep_boundaries,
                self._player.frame_index
            )

    # ── Pose Estimation ──────────────────────────────────────

    def _toggle_pose(self, enabled):
        self._pose_enabled = enabled
        if enabled and self._current_frame is not None:
            self._run_pose_if_enabled(self._current_frame)
        if self._current_frame is not None:
            self._display_frame(self._current_frame)

    def _run_pose_if_enabled(self, frame):
        if not self._pose_enabled:
            return
        frame_idx = self._player.frame_index
        if frame_idx in self._pose_cache:
            return
        self._pose_worker.process_frame(frame, frame_idx)

    def _on_pose_result(self, frame_index, landmarks):
        self._pose_cache[frame_index] = landmarks
        if frame_index == self._player.frame_index and self._current_frame is not None:
            self._display_frame(self._current_frame)

    # ── Export / Project ─────────────────────────────────────

    def _save_project(self):
        if not self._player.is_loaded:
            QMessageBox.information(self, "Info", "No video loaded to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", ProjectManager.get_filter()
        )
        if path:
            if not path.endswith(ProjectManager.PROJECT_EXTENSION):
                path += ProjectManager.PROJECT_EXTENSION
            self._project_manager.save(
                path,
                video_path=self._player.file_path,
                annotation_manager=self._annotation_manager,
                audio_recorder=self._audio_recorder,
                velocity_data=self._velocity_tracker.to_dict() if self._velocity_tracker.is_tracking else None,
                pose_enabled=self._pose_enabled
            )
            self._statusbar.showMessage(f"Project saved: {os.path.basename(path)}")

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", ProjectManager.get_filter()
        )
        if not path:
            return
        try:
            project = self._project_manager.load(path)
            video_path = project.get('video_path', '')
            if video_path and os.path.exists(video_path):
                self._player.load(video_path)
                self._controls.set_range(self._player.total_frames)
                frame = self._player.get_frame(0)
                self._current_frame = frame

                # Restore annotations
                if project.get('annotations'):
                    self._annotation_manager.from_dict_list(project['annotations'])

                # Restore audio clips
                if project.get('audio_clips'):
                    self._audio_recorder.from_dict_list(project['audio_clips'])

                # Restore pose state
                self._pose_enabled = project.get('pose_enabled', False)
                self._pose_toggle_action.setChecked(self._pose_enabled)
                self._pose_cache.clear()

                self._display_frame(frame)
                self._statusbar.showMessage(f"Project loaded: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Error",
                                    f"Video file not found:\n{video_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load project:\n{e}")

    def _export_video(self):
        if not self._player.is_loaded:
            QMessageBox.information(self, "Info", "No video loaded to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "MP4 Video (*.mp4);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith('.mp4'):
            path += '.mp4'

        progress = QProgressDialog("Exporting video...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        def on_progress(pct):
            progress.setValue(pct)

        self._export_worker.setup(
            self._exporter.export,
            output_path=path,
            video_path=self._player.file_path,
            annotation_manager=self._annotation_manager,
            annotation_renderer=self._annotation_renderer,
            pose_estimator=self._pose_estimator,
            pose_renderer=self._pose_renderer,
            form_analysis=self._form_analysis,
            pose_cache=self._pose_cache,
            pose_enabled=self._pose_enabled,
            velocity_tracker=self._velocity_tracker,
            audio_recorder=self._audio_recorder,
        )
        self._export_worker.progress.connect(on_progress)
        self._export_worker.finished.connect(
            lambda p: self._on_export_finished(p, progress)
        )
        self._export_worker.error.connect(
            lambda e: self._on_export_error(e, progress)
        )
        self._export_worker.start()

    def _on_export_finished(self, output_path, progress):
        progress.close()
        self._statusbar.showMessage(f"Exported: {os.path.basename(output_path)}")
        VideoExporter.open_containing_folder(output_path)

    def _on_export_error(self, error_msg, progress):
        progress.close()
        QMessageBox.warning(self, "Export Error", f"Export failed:\n{error_msg}")

    # ── Keyboard Shortcuts ───────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Right:
            self._step_forward()
        elif key == Qt.Key.Key_Left:
            self._step_backward()
        elif key == Qt.Key.Key_R and not ctrl:
            is_rec = not self._controls._is_recording
            self._controls._btn_record.setChecked(is_rec)
        elif key == Qt.Key.Key_Z and ctrl:
            self._annotation_undo()
        elif key == Qt.Key.Key_Y and ctrl:
            self._annotation_redo()
        elif key == Qt.Key.Key_Delete:
            self._annotation_manager.remove_last()
            if self._current_frame is not None:
                self._display_frame(self._current_frame)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up resources on close."""
        self._pose_worker.stop()
        self._pose_estimator.close()
        self._player.release()
        super().closeEvent(event)
