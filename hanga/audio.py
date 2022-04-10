import enum
import functools
import os
import tempfile
from typing import Optional

import IPython.display as ipd
import librosa
import madmom as mm
import matplotlib.pyplot as plt
import numpy as np
import pydub
import scipy.signal
import soundfile as sf
import torch

from hanga import util as h_util


def bpm_to_fps(bpm: float) -> float:
    return bpm / 60


class AudioFormat(enum.Enum):
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    MP3 = "mp3"


# define a base class API here
class BaseTrack:
    pass


class PyDubTrack(BaseTrack):
    def __init__(self, y: pydub.AudioSegment):
        self.y = self.y

    @classmethod
    def from_file(
        cls,
        file_path: str,
        audio_format: Optional[AudioFormat] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        audio_format = None if audio_format is None else audio_format.name.lower()
        audio_segment = pydub.AudioSegment.from_file(file_path, format=audio_format)
        # TODO:
        # audio_segment = audio_segment[int(h_util.s_to_ms(start_time)) : int(h_util.s_to_ms(end_time))]
        # TODO: work out if I can also get sample rate
        return cls(y=audio_segment)

    # TODO: add a function to convert to Librosa and VV
    # TODO: add the remaining util functions


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
        plt.plot(
            times[time_indices],
            librosa.util.normalize(self.onset_env[time_indices]),
            label="Onset strength",
        )
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
        sf.write(
            file_path,
            data=self.y,
            samplerate=self.sr,
            format=audio_format.name,
            subtype=subtype,
        )


def onsets(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    margin: int = 8,
    fmin: int = 20,
    fmax: int = 8000,
    smooth: int = 1,
    clip: int = 100,
    power: int = 1,
    type: str = "mm",
) -> torch.Tensor:
    """Creates onset envelope from audio, taken from https://github.com/JCBrouwer/maua-stylegan2/

    Args:
        audio Audio signal
        sr: Sampling rate of the audio
        n_frames: Total number of frames to resample envelope to
        margin: For percussive source separation, higher values create more extreme separations. Defaults to 8.
        fmin: Minimum frequency for onset analysis. Defaults to 20.
        fmax: Maximum frequency for onset analysis. Defaults to 8000.
        smooth: Standard deviation of gaussian kernel to smooth with. Defaults to 1.
        clip: Percentile to clip onset signal to. Defaults to 100.
        power: Exponent to raise onset signal to. Defaults to 1.
        type: ["rosa", "mm"]. Whether to use librosa or madmom for onset analysis. Madmom is slower but often more accurate. Defaults to "mm".

    Returns:
        th.tensor, shape=(n_frames,): Onset envelope
    """
    y_perc = librosa.effects.percussive(y=audio, margin=margin)
    if type == "rosa":
        onset = librosa.onset.onset_strength(y=y_perc, sr=sr, fmin=fmin, fmax=fmax)
    elif type == "mm":
        sig = mm.audio.signal.Signal(y_perc, num_channels=1, sample_rate=sr)
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
        stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, circular_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, circular_shift=True)
        filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24, fmin=fmin, fmax=fmax)
        onset = np.sum(
            [
                mm.features.onsets.spectral_diff(filt_spec),
                mm.features.onsets.spectral_flux(filt_spec),
                mm.features.onsets.superflux(filt_spec),
                mm.features.onsets.complex_flux(filt_spec),
                mm.features.onsets.modified_kullback_leibler(filt_spec),
            ],
            axis=0,
        )
    onset = np.clip(scipy.signal.resample(onset, n_frames), onset.min(), onset.max())
    onset = torch.from_numpy(onset).float()
    onset = h_util.gaussian_filter(onset, smooth, causal=0)
    onset = h_util.percentile_clip(onset, clip)
    onset = onset**power
    return onset


def raw_chroma(y: np.ndarray, sr: int, type: str = "cens", nearest_neighbor: bool = True) -> np.ndarray:
    """Creates chromagram

    Args:
        y: Audio signal
        sr: Sampling rate of the audio
        type: ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        nearest_neighbor: Whether to post process using nearest neighbor smoothing. Defaults to True.

    Returns: ndarray shape (12, n_frames): Chromagram
    """
    if type == "cens":
        ch = librosa.feature.chroma_cens(y=y, sr=sr)
    elif type == "cqt":
        ch = librosa.feature.chroma_cqt(y=y, sr=sr)
    elif type == "stft":
        ch = librosa.feature.chroma_stft(y=y, sr=sr)
    elif type == "deep":
        sig = mm.audio.signal.Signal(y, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.DeepChromaProcessor().process(sig).T
    elif type == "clp":
        sig = mm.audio.signal.Signal(y, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.CLPChromaProcessor().process(sig).T
    else:
        print("chroma type not recognized, options are: [cens, cqt, deep, clp, or stft]. defaulting to cens...")
        ch = librosa.feature.chroma_cens(y=y, sr=sr)

    if nearest_neighbor:
        ch = np.minimum(ch, librosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    return ch


def chroma(
    y: np.ndarray,
    sr: int,
    n_frames: int,
    margin: int = 16,
    type: str = "cens",
    notes: int = 12,
) -> torch.tensor:
    """Creates chromagram for the harmonic component of the audio

    Args:
        y: Audio signal
        sr: Sampling rate of the audio
        n_frames: Total number of frames to resample envelope to
        margin: For harmonic source separation, higher values create more extreme separations. Defaults to 16.
        type: ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        notes (int, optional): Number of notes to use in output chromagram (e.g. 5 for pentatonic scale, 7 for standard western scales). Defaults to 12.

    Returns: Tensor shape=(n_frames, 12): Chromagram
    """
    y_harm = librosa.effects.harmonic(y=y, margin=margin)
    chroma = raw_chroma(y_harm, sr, type=type).T
    chroma = scipy.signal.resample(chroma, n_frames)
    notes_indices = np.argsort(np.median(chroma, axis=0))[:notes]
    chroma = chroma[:, notes_indices]
    chroma = torch.from_numpy(chroma / chroma.sum(1)[:, None]).float()
    return chroma
