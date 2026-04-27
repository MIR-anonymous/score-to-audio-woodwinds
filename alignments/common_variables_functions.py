"""
This module provides utilities to align symbolic musical annotations
(e.g., unfolded scores in CSV format) with audio recordings using
feature-based synchronization techniques. It is designed for research
in Music Information Retrieval (MIR), with a focus on baroque solo
repertoire (e.g., Telemann Fantasias) and wind instruments.

Main functionalities
--------------------
- Conversion of symbolic note sequences into MIDI-based representations
- Construction of SyncToolbox-compatible annotation DataFrames
- Extraction of pitch, chroma, and onset-based features from both
  symbolic annotations and audio signals
- Post-processing of alignments, including silence handling and timing refinement


Dependencies
------------
- librosa
- numpy
- pandas
- matplotlib
- scipy
- music21
- synctoolbox
- libfmp

Data organization
-----------------
The module assumes the following directory structure:
- Audio files stored in a dedicated folder (e.g., `../audios/`)
- Precomputed symbolic annotations in CSV format (unfolded scores)
- Optional precomputed features stored as `.npz`
- Output alignment results written to dedicated folders
"""

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from music21 import converter, note, chord
from synctoolbox.feature.csv_tools import music_xml_to_csv_musical_time
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic
from synctoolbox.feature.csv_tools import read_csv_to_df, df_to_pitch_features, df_to_pitch_onset_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning
import libfmp.c2
from libfmp.b import list_to_pitch_activations, plot_chromagram, plot_signal, plot_matrix

######################################################                 
                     
NPZ = '../npz/'
AUDIOS='../audios/'
ALIGNMENTS_NEURAL = 'output/neural/'
ALIGNMENTS_ST = 'output/st/'
ALIGNMENTS_CHROMA = 'output/chroma/'
UNFOLDED = '../scores/unfolded-scores/'

FLUTE = ['pahud', 'rampal']
TRAVERSO = ['kuijken', 'pitelina']
OBOE=['leleux', 'indermuhle']
RECORDER=['verbruggen', 'nahajowski']
ALL = FLUTE + OBOE + TRAVERSO + RECORDER

SR = 22050
HOP_LENGTH = 512
NFFT= 1024

#####################################

def notes_to_midi_pitches(pitches: list[str])->list[int]:
    """Converts a sequence of pitch names and rests into a sequence of MIDI note numbers,
    optionally stretched to match a specified number of frames.

    Args:
        - pitches: sequence of pitches names and rests(None) according to the score and the performance

    Returns:
        Sequence of MIDI note numbers (-1 for rest)
    """
    res=[]
    for p in pitches:
        if p == '0':
            res.append(-1)
        else:
            res.append(librosa.note_to_midi(p))    
    return res


def produce_df_from_unfolded_scores(performer: str, fantasia: int) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Generate a SyncToolbox-compatible annotation DataFrame from an unfolded scores.

    The function reads an unfolded score (CSV), rescales symbolic durations
    to match the feature rate of the audio representation, and converts pitch
    names into MIDI numbers.

    Args:
        performer : Name of the performer (used to locate the unfolded score file).
        fantasia : Fantasia number (1-12).

    Returns:
        df : 
            Annotation DataFrame formatted for SyncToolbox, containing:
            - 'start' (frame index)
            - 'duration' (in frames)
            - 'pitch' (MIDI note numbers)
            - 'velocity'
            - 'instrument'
        df_rests :
            Subset of the original DataFrame containing rest events.
        rest_indexes_list :
            Original indices of rest events in the unfolded score.

    Notes
    -----
    - Grace notes (duration = 0) are assigned a minimal duration.
    - Rests are temporarily removed from the main DataFrame but tracked separately.
    - Durations are scaled according to the feature rate (SR / HOP_LENGTH).
    """

    sequencepath= f'{UNFOLDED}{performer}/unfolded_{fantasia}.csv'
    df = pd.read_csv(sequencepath)
    df = df[~((df['Score_duration'] == 0) & (df['Note'] == '0'))]
    
    df_rests = df[(df["Note"] == '0')  & (df['Score_duration'] != 0)]
    rest_indexes_list = df_rests.index.tolist()
    
    df = df.rename(columns={
                    'Onset (ms)': 'start',
                    'Time_duration (ms)': 'duration'
                    })
    
    df['duration'] = df['Score_duration']
    # handle appogiatures
    min_dur = 0.05
    df.loc[df['duration'] == 0, 'duration'] = min_dur
    
    # Score at scale of audio
    feature_rate = SR / HOP_LENGTH
    scale= feature_rate/5
    df['duration'] *= scale

    df['start'] = df['duration'].cumsum() - df['duration']

    notes = df.iloc[:, 0].tolist()
    df['Note'] = notes
    df['pitch']=notes_to_midi_pitches(notes)
    df = df[df["pitch"] != -1]
    df['velocity'] = 100
    df['instrument'] = 'flute'
    return df, df_rests, rest_indexes_list


def get_features_from_annotation(df_annotation, feature_rate, visualize=False):
    f_pitch = df_to_pitch_features(df_annotation, feature_rate=feature_rate)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    if visualize:
        plot_chromagram(f_chroma_quantized, title='Quantized chroma features - Annotation', Fs=feature_rate, figsize=(9, 3))
    f_pitch_onset = df_to_pitch_onset_features(df_annotation)
    f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset,
                                            feature_rate=feature_rate,
                                            feature_sequence_length=f_chroma_quantized.shape[1],
                                            visualize=visualize)
    
    return f_chroma_quantized, f_DLNCO

###################################

def detect_initial_onset(audio: np.ndarray, sr: int, threshold: float =0.005, 
                         frame_size: int =2048, hop_length: int=512) -> float:
    """
    Detect the first significant onset in an audio signal using RMS energy.

    Args:
        audio :
            Audio time series.
        sr : int
            Sampling rate of the audio signal.
        threshold :
            RMS energy threshold above which a frame is considered active.
            Default is 0.005.
        frame_size :
            Frame length for RMS computation. Default is 2048.
        hop_length :
            Hop length between successive frames. Default is 512.

    Returns:
        Time (in seconds) of the first detected onset. Returns 0.0 if no onset is found.
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    for i, r in enumerate(rms):
        if r > threshold:
            return times[i]
    return 0.0

#################################################

def move_column(df, col_name, new_pos):
    cols = list(df.columns)
    cols.remove(col_name)
    cols.insert(new_pos, col_name)
    return df[cols]


def adjust_silences(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust onset and offset times for silence events in a musical alignment.

    This function redistributes the timing of consecutive silence events
    (rows where 'Note' == '0') so that their durations match the proportions
    specified by 'Score_duration', while fitting exactly between surrounding
    sounding notes.

    Args:
        df_all :
            DataFrame containing aligned musical events with at least the columns:
            - 'Note'
            - 'Onset (ms)'
            - 'Offset (ms)'
            - 'Score_duration'

    Returns:
        Updated DataFrame where silence segments have adjusted onset and offset times.
        The column 'Time_duration (ms)' is recomputed accordingly.

    Notes
    -----
    - Consecutive silences are treated as groups.
    - Each group is stretched or compressed to fit between neighboring notes.
    - Within each group, durations are distributed proportionally to
      'Score_duration'.
    - If total score duration is zero, durations are evenly distributed.
    """
    silence_mask = (df_all['Note'].astype(str) == '0')
    silence_indices = df_all[silence_mask].index
    groups = []
    current_group = []

    for i in silence_indices:
        if not current_group:
            current_group = [i]
        elif i == current_group[-1] + 1:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    if current_group:
        groups.append(current_group)

    for group in groups:
        start = df_all.loc[group[0] - 1, 'Offset (ms)'] if group[0] > 0 else 0
        end = df_all.loc[group[-1] + 1, 'Onset (ms)'] if group[-1] + 1 < len(df_all) else start
        duration = end - start

        score_durations = df_all.loc[group, 'Score_duration'].values
        score_total = score_durations.sum()

        onset = start
        for idx, sdur in zip(group, score_durations):
            frac = (sdur / score_total) if score_total > 0 else 1 / len(group)
            seg_dur = duration * frac
            df_all.at[idx, 'Onset (ms)'] = onset
            df_all.at[idx, 'Offset (ms)'] = onset + seg_dur
            onset += seg_dur

    df_all['Time_duration (ms)'] = df_all['Offset (ms)'] - df_all['Onset (ms)']
    return df_all
