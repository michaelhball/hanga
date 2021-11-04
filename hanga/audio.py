import functools

import librosa
import matplotlib.pyplot as plt
import numpy as np


def bpm_to_fps(bpm: float):
    return bpm / 60


class Track:
    def __init__(self, file_path, use_onset_env=False):
        self.y, self.sr = librosa.load(file_path)        
        self.use_onset = use_onset_env

    @functools.cached_property
    def duration(self):
        return librosa.get_duration(self.y)

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

    def plot_beats(self, hop_length=512, t_min=-1, t_max=float('inf')):
        # TODO: create a proper figure here rather than directly using plt
        # TODO: take fig size as argument
        
        times = librosa.times_like(self.onset_env, sr=self.sr, hop_length=hop_length)
        time_indices = np.where((t_min <= times) & (times <= t_max))
        plt.plot(times[time_indices], librosa.util.normalize(self.onset_env[time_indices]), label='Onset strength')
        print(len(self.beats))
        print(len(librosa.time_to_frames(times[time_indices]))) # NOT THIS, BUT SOMETHING LIKE IT
        # TODO: work out how to get the same time range for beats here ?? DAFUQ 
        # plt.vlines(librosa.frames_to_time(self.beats, sr=self.sr), 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
