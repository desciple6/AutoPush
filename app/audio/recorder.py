"""Audio recorder - captures microphone audio synced to video timeline."""

import os
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from dataclasses import dataclass, field
from typing import Optional

from app.config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS


@dataclass
class AudioClip:
    """A recorded audio segment with frame sync info."""
    file_path: str
    start_frame: int
    end_frame: int
    sample_rate: int = AUDIO_SAMPLE_RATE


class AudioRecorder:
    """Records audio from the microphone, synced to video frame positions."""

    def __init__(self, output_dir: str = None):
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="liftreview_audio_")
        self._clips: list[AudioClip] = []
        self._recording = False
        self._stream: Optional[sd.InputStream] = None
        self._buffer: list[np.ndarray] = []
        self._start_frame = 0
        self._clip_counter = 0

    @property
    def is_recording(self):
        return self._recording

    @property
    def clips(self):
        return self._clips

    @property
    def output_dir(self):
        return self._output_dir

    def start(self, frame_index: int):
        """Start recording from microphone at the given video frame."""
        if self._recording:
            return

        self._buffer = []
        self._start_frame = frame_index
        self._recording = True

        self._stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype='float32',
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self, frame_index: int) -> Optional[AudioClip]:
        """Stop recording and save the audio clip."""
        if not self._recording:
            return None

        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._buffer:
            return None

        # Combine buffer chunks
        audio_data = np.concatenate(self._buffer, axis=0)

        # Save to WAV file
        self._clip_counter += 1
        filename = f"clip_{self._clip_counter:03d}.wav"
        filepath = os.path.join(self._output_dir, filename)

        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(filepath, AUDIO_SAMPLE_RATE, audio_int16)

        clip = AudioClip(
            file_path=filepath,
            start_frame=self._start_frame,
            end_frame=frame_index,
            sample_rate=AUDIO_SAMPLE_RATE
        )
        self._clips.append(clip)
        self._buffer = []
        return clip

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream."""
        if self._recording:
            self._buffer.append(indata.copy())

    def get_clips_for_range(self, start_frame: int, end_frame: int) -> list[AudioClip]:
        """Get all audio clips that overlap the given frame range."""
        result = []
        for clip in self._clips:
            if clip.start_frame < end_frame and clip.end_frame > start_frame:
                result.append(clip)
        return result

    def clear(self):
        """Remove all recorded clips."""
        self._clips.clear()
        self._clip_counter = 0

    def to_dict(self) -> list[dict]:
        return [
            {
                "file_path": c.file_path,
                "start_frame": c.start_frame,
                "end_frame": c.end_frame,
                "sample_rate": c.sample_rate
            }
            for c in self._clips
        ]

    def from_dict_list(self, data: list[dict]):
        self._clips = [AudioClip(**d) for d in data]
        self._clip_counter = len(self._clips)
