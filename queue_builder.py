import json
import math
import random
import librosa
import librosa.feature
import numpy as np
import yt_dlp
from scipy.spatial.distance import cdist


class AutoDJ:
    current_song = None
    next_song = None
    def __init__(self, track_data_path: str = "track_data.json"):
        self.current_song = "QZJ842404156"
        self.played_songs = []
        with open(track_data_path, "r") as f:
            raw_data = json.load(f)
        self.tracks_data = raw_data["content"]
        self.tracks_data_by_isrc = {s["isrc"]: s for s in self.tracks_data if "isrc" in s}
        with open("tracks.json", "r") as f:
            tracks = json.load(f)
        self.tracks = tracks

    def _solve_preliminary_next_songs(self):
        def score_transition(current, candidate, mu: float = 0.08):
            target_energy = max(1.0, current["energy"]+mu)
            candidate_vec = np.array([candidate["energy"], candidate["danceability"], candidate["valence"]])
            target_vec = np.array([target_energy, current["danceability"], current["valence"]])
            delt_E = candidate["energy"] - current["energy"]
            sigma = 0.2 if delt_E >= mu else 0.08
            energy_score = math.exp(-((delt_E - mu)**2)/(2*sigma**2))
            cos_sim = np.dot(target_vec, candidate_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(candidate_vec))
            score = cos_sim * energy_score**2
            return score


        current = next((s for s in self.tracks_data if s.get("isrc") == self.current_song), None)

        current_bpm = current.get("tempo")
        current_key = current.get("key")
        current_mode = current.get("mode")
        if current_bpm is None or current_key is None or current_mode is None:
            return []

        candidate_songs = [song for song in self.tracks_data if (abs(song["tempo"] - current_bpm) <= 0.05 * current_bpm or
                                                              abs((0.5*song["tempo"]) - current_bpm) <= 0.05 * current_bpm or
                                                              abs((2*song["tempo"]) - current_bpm) <= 0.05 * current_bpm) and
                                                              (
                                                                  song["key"] != -1 and
                                                                  current_key != -1 and
                                                                  (
                                                                    (song["mode"] == current_mode and (song["key"] - current_key) % 12 in (0, 1, 11)) or
                                                                    (current_mode == 1 and song["mode"] == 0 and song["key"] == (current_key - 3) % 12) or
                                                                    (current_mode == 0 and song["mode"] == 1 and song["key"] == (current_key + 3) % 12)

                                                                  )
                                                              ) and song.get("isrc") != self.current_song and song.get("isrc") not in self.played_songs]

        best_songs = sorted(candidate_songs, key=lambda candidate: score_transition(current, candidate), reverse=True)[:8]
        for _ in range(3):
            best_songs.pop(random.randrange(len(best_songs)))
        return best_songs

    @staticmethod
    def _first_round_beat_analysis(filename):
        y, sr = librosa.load(filename)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

        chroma = librosa.util.sync(librosa.feature.chroma_cqt(y=y_harmonic, sr=sr), beats, aggregate=np.median)[:, 1:]
        mfcc = librosa.util.sync(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), beats, aggregate=np.mean)[:, 1:]

        chroma = librosa.util.normalize(chroma, axis=0)
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        beat_features = np.vstack([chroma, mfcc]).T

        ssm = 1.0 - cdist(beat_features, beat_features, metric="cosine")
        ssm = np.nan_to_num(ssm, nan=0.0)

        L = 32
        axis = np.arange(-L, L + 1)
        gx, gy = np.meshgrid(axis, axis)
        checkerboard = np.sign(gx) * np.sign(gy)
        gaussian = np.exp(-(gx ** 2 + gy ** 2) / (2 * (L / 2.0) ** 2))
        kernel = checkerboard * gaussian
        kernel /= np.sum(np.abs(kernel))

        N = ssm.shape[0]
        padded = np.pad(ssm, L, mode="reflect")
        novelty = np.zeros(N)
        for i in range(N):
            patch = padded[i : i + 2 * L + 1, i : i + 2 * L + 1]
            novelty[i] = np.sum(patch*kernel)

        wait_beats = 8 * 4
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max = 16,
            post_max = 16,
            pre_avg = 32,
            post_avg = 32,
            delta = 0.05,
            wait = wait_beats
        )

        num_phrases = len(beat_features.T) - 3
        phrase_vecs = np.hstack([
            beat_features[0:num_phrases],
            beat_features[1:num_phrases+1],
            beat_features[2:num_phrases+2],
            beat_features[3:num_phrases+3]
        ])
        phrase_vecs /= (np.linalg.norm(phrase_vecs, axis=1, keepdims=True) + 1e-8)

        snapped_beat_indices = np.unique(np.round(peaks / 4.0).astype(int) * 4)
        snapped_beat_indices = snapped_beat_indices[snapped_beat_indices < len(beats)]

        beat_times = librosa.frames_to_time(beats, sr=sr)
        phrase_times = beat_times[snapped_beat_indices]
        return {
            "phrase_beats" : snapped_beat_indices,
            "beat_times" : beat_times,
            "phrase_times" : phrase_times,
            "phrase_vecs" : phrase_vecs,
            "bpm": float(np.atleast_1d(tempo)[0])
        }






    def solve_next(self):
        songs = self._solve_preliminary_next_songs()
        for song_data in songs:
            song = next((s for s in self.tracks if s.get("isrc") == song_data.get("isrc")), None)
            query = f"ytsearch1:{song['artist']} - {song['title']} official audio"
            ydl_opts = {
                "format": "bestaudio/best",  # Implied automatically when using -x
                "outtmpl": "%(title)s.%(ext)s",  # Replace with your desired "-o" path/template
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",  # --audio-format wav
                        "preferredquality": "0",  # --audio-quality 0
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([query])

# dj = AutoDJ()
# dj.solve_next()
print(AutoDJ()._first_round_beat_analysis("euphoria.wav"))
