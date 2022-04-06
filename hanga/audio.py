import enum
import functools
import os
import tempfile

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pydub
import soundfile as sf

from hanga import util as h_util


def bpm_to_fps(bpm: float) -> float:
    return bpm / 60


class AudioFormat(enum.Enum):
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    MP3 = "mp3"


class Track:
    def __init__(self, y: np.ndarray, sr: int, use_onset: bool = False):
        self.y = y
        self.sr = sr
        self.use_onset = use_onset

    @classmethod
    def from_file(cls, file_path: str, use_onset: bool = False) -> "Track":
        return cls(*librosa.load(file_path), use_onset=use_onset)

    @classmethod
    def from_file_snippet(
        cls,
        file_path: str,
        start: float,
        end: float,
        audio_format: AudioFormat = None,
        use_onset: bool = False,
        save_path: str = None,
    ) -> "Track":
        """Loads an audio track between specified times (in seconds).
        NB: if no audio_format is specified, this func attempts to work out the
        format from the file_path.
        """
        audio_format = None if audio_format is None else audio_format.name.lower()
        audio_segment = pydub.AudioSegment.from_file(file_path, format=audio_format)
        snippet = audio_segment[int(h_util.s_to_ms(start)) : int(h_util.s_to_ms(end))]
        # use tmp_dir only if not passed a save_path
        with tempfile.TemporaryDirectory() as tmp_dir:
            if save_path is None:
                save_path = os.path.join(tmp_dir, "tmp.wav")
            snippet.export(save_path, format="wav")
            return cls.from_file(save_path, use_onset=use_onset)

    @functools.cached_property
    def duration(self):
        # TODO: add different formatting options here
        return librosa.get_duration(y=self.y)

    @functools.cached_property
    def onset_env(self):
        return librosa.onset.onset_strength(self.y, sr=self.sr, aggregate=np.median)

    @functools.cached_property
    def tempo(self):
        if self.use_onset:
            tempo, _ = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=self.sr)
        else:
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return tempo

    @functools.cached_property
    def beats(self):
        if self.use_onset:
            _, beats = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=self.sr)
        else:
            _, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return beats

    @functools.cached_property
    def beat_times(self):
        return librosa.frames_to_time(self.beats, sr=self.sr)

    def display(self):
        return ipd.Audio(self.y, rate=self.sr)

    def plot_beats(self, hop_length=512, t_min=-1, t_max=float("inf")):
        # TODO: create a proper figure here rather than directly using plt
        # TODO: take fig size as argument

        times = librosa.times_like(self.onset_env, sr=self.sr, hop_length=hop_length)
        time_indices = np.where((t_min <= times) & (times <= t_max))
        plt.plot(times[time_indices], librosa.util.normalize(self.onset_env[time_indices]), label="Onset strength")
        print(len(self.beats))
        print(len(librosa.time_to_frames(times[time_indices])))  # NOT THIS, BUT SOMETHING LIKE IT
        # TODO: work out how to get the same time range for beats here ?? DAFUQ
        # plt.vlines(librosa.frames_to_time(self.beats, sr=self.sr), 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')

    def save(self, save_dir: str, file_name: str, audio_format: AudioFormat = AudioFormat.WAV):
        """Save audio track (librosa ndarray) to disk.
        Args:
            save_dir: dir in which to save audio file
            file_name: file name (excluding extension)
            audio_format: valid file format
        """
        file_path = os.path.join(save_dir, f"{file_name}.{audio_format.name.lower()}")
        subtype = "vorbis" if audio_format == audio_format.OGG else "PCM_24"
        sf.write(file_path, data=self.y, samplerate=self.sr, format=audio_format.name, subtype=subtype)
