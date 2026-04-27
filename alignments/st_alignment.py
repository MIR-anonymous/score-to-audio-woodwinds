#!/usr/bin/env python
# coding: utf-8

from common_variables_functions import *
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NHARMONICS=10
DECAY_POWER=1.25

#####################################################

def read_audio_and_spectrogram(filename: str, sr: int=SR, n_fft: int=NFFT, n_hop: int=HOP_LENGTH)-> tuple[np.ndarray, int, np.ndarray]:
    """Read an audio file, compute spectrogram
    
    Args:
      - filename: The path to the audio file.
      - sr: The sample rate.
      - n_fft: The FFT size.
      - n_hop: The hop size.

    Returns:
      A tuple containing the signal, sample rate and frequency spectrogram.
    """
    y, sr = librosa.load(filename, sr=sr)

    # Compute linear frequency spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=n_hop)
    S = np.abs(stft)
    return y, sr, S


def generate_flute_harmonic_template_discrete(performer: str, note:str, y: np.ndarray, 
                                              sr: int=SR, n_fft: int=NFFT, n_harmonics: int=NHARMONICS, decay_power: int=DECAY_POWER)->np.ndarray:
    """
    Produces the frequency discrete harmonic template of a note for the given performer.

    Args:
        - performer : performer name
        - note: The name of the note (e.g., 'C4').
        - y: signal
        - sr: sample rate of the signal (Hz). Determines the frequency resolution
        - n_fft: The number of FFT bins.
        - n_harmonics: The maximum number of harmonics to consider.
        - decay_power: The power of the decreasing amplitude harmonic template.

    Returns:
        A NumPy array representing the harmonic template in the frequency domain with discrete bins.
    """
    if note=='0':
        return 0.1*np.ones(n_fft // 2 + 1)

    f_0 = librosa.note_to_hz(note)
    A_0 = 1  # Amplitude of the fundamental

    # Generate harmonic frequencies up to the maximum frequency
    f_n = f_0 * np.arange(1, n_harmonics + 1)
    f_n = f_n[f_n<=sr]

    # Create the frequency template with discrete bins
    T_n = np.zeros(n_fft // 2 + 1)  # Frequency domain template
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for f in f_n:
        # Find the closest bin index to the harmonic frequency
        index = np.argmin(np.abs(frequencies - f))
        if index < len(T_n):
            T_n[index] = A_0 / f**decay_power

    return T_n/np.linalg.norm(T_n)


def generate_note_likelihood_matrix(performer: str, y: np.ndarray, sr: int, S: np.ndarray)->np.ndarray:
    """Generate a note likelihood matrix for a given audio signal.

    Each element of the matrix represents the likelihood (or similarity)
    between a discrete harmonic template of a note played by a specific performer
    and the spectral content of the audio signal at a given time frame.

    Args:
        - performer: Identifier or type of the performer (e.g., musician name or model).
        - y :signal
        - sr : Sampling rate of the audio signal in Hz.
        - S : spectral representation (e.g., a spectrogram) of `y`

    Returns:
        A matrix of shape (number_of_notes, number_of_frames),
        where each row corresponds to a note in range MIDI [24,96[
        and each column corresponds to a time frame in the audio.
        Each element contains the likelihood score between the note template
        and the audio spectrum at that frame.

    Notes:
        - The likelihood is computed as the dot product between the harmonic template
          and the spectral vector at each time frame.
    """
    midi_start, midi_end = 24, 96
    n_pitches = midi_end - midi_start 
    note_likelihood_matrix = np.zeros((n_pitches, S.shape[1]))
    
    for i, midi_pitch in enumerate(range(midi_start, midi_end)):
        note_name = librosa.midi_to_note(midi_pitch)
        T_note = generate_flute_harmonic_template_discrete(performer, note_name, y, sr)
        if T_note.shape[0] != S.shape[0]:
            raise ValueError(f"Mismatch dimension template ({T_note.shape[0]}) vs spectrogram ({S.shape[0]})")
        for t in range(S.shape[1]):
            note_likelihood_matrix[i, t] = np.dot(T_note, S[:, t])
    
    return note_likelihood_matrix


def pitches_to_midi_notes(pitches: list[str|None])->list[int|None]:
    """Returns a sequence of chroma values from a sequence of pitches names.

    Args:
        - pitches: sequence of pitches names and rests(None) according to the score and the performance

    Returns:
        sequence of chroma values(12) and rest(None)
    """
    res=[]
    for p in pitches:
        if p == '0':
            res.append(None)
        else:
            res.append(librosa.note_to_midi(p)%12)
    return res


def get_features_from_audio(performer, fantasia):
    ext ='.m4a' 
    audiofilename = f'{AUDIOS}{performer}/{performer}_{fantasia}{ext}'
    if audiofilename:
        y, sr, S = read_audio_and_spectrogram(audiofilename, n_hop=HOP_LENGTH)
        
    note_likelihood_matrix = generate_note_likelihood_matrix(performer, y, sr, S)
    chroma_likelihood_matrix= transform_pitch_72_to_chroma_12(note_likelihood_matrix)
    f_chroma_quantized = quantize_chroma(f_chroma=chroma_likelihood_matrix)
    return f_chroma_quantized 
    

def pitch_sequence_to_binary_matrix(midi_pitch_sequence: list[int|None]) -> np.ndarray:
    """
    Convert a sequence of MIDI pitches into a binary pitch activation matrix over 72 semitone bins (C1 to C7).

    Args:
        - midi_pitch_sequence : Sequence of MIDI note numbers (integers in [24, 96]) or None for rests

    Returns:
        Binary matrix of shape (72, N), where N is the length of the input sequence.
        Each column corresponds to a frame, and each row to a semitone from MIDI 24 (C1) to MIDI 96 (C7).
        A value of 1 indicates the presence of the corresponding pitch at that frame.
    """
    matrix = np.zeros((72, len(midi_pitch_sequence)), dtype=int)
    for t, pitch in enumerate(midi_pitch_sequence):
        if pitch is not None and 24 <= pitch < 96:
            matrix[pitch - 24, t] = 1
    return matrix


def transform_pitch_72_to_chroma_12(pitch_matrix_72: np.ndarray) -> np.ndarray:
    """
    Convert a pitch activation map over 72 semitone bins (covering 6 octaves) into a 12-bin chroma representation.

    Args:
        - pitch_matrix_72 : Array of shape (72, n_frames), representing energy or activation 
                                      for each semitone (e.g., from C1 to C7) at each time frame.


    Returns:
        np.ndarray: Array of shape (12, n_frames), representing the chroma (pitch classes modulo octave), 
                    with either binary or weighted values.
    """
    n_frames = pitch_matrix_72.shape[1]
    chroma_12 = np.zeros((12, n_frames))

    for i in range(12):
        bins_indices = list(range(i, 72, 12))
        chroma_12[i, :] = np.sum(pitch_matrix_72[bins_indices, :], axis=0)
    return chroma_12


def chroma_sequence_to_matrix(midi_chromas_sequence: list[int|None])->np.ndarray:
    """Returns a binary chroma matrix from a chroma sequence

    Args:
        - midi_chromas_sequence: sequence of chroma values(12) and rest(None) according to the score and the performance

    Returns:
        binary chroma matrix of shape (12, N) where N can be the length of the midi_chromas_sequence
        or a stretch value corresponding to the number of frames
    """
    chroma_matrix = np.zeros((12, len(midi_chromas_sequence)))
    for t, pitch_class in enumerate(midi_chromas_sequence):
        if pitch_class is not None and 0 <= pitch_class < 12:
            chroma_matrix[pitch_class, t] = 1
    return chroma_matrix



def main(performer: str, fantasia: int):
    """Produce alignment

    Args:
        performer: performer name
        fantasia: fantasia number
    """
    feature_rate = SR/HOP_LENGTH
    step_weights = np.array([1.5, 1.5, 2.0])
    threshold_rec = 10 ** 6
    
    # loading the recording
    ext ='.m4a' 
    audiopath = f'{AUDIOS}{performer}/{performer}_{fantasia}{ext}'
    audio, sr = librosa.load(audiopath, sr=SR)
    
    #Loading the .csv annotation file
    df_annotation, df_rests, rest_indexes_list = produce_df_from_unfolded_scores(performer, fantasia)
    
    initial_onset = detect_initial_onset(audio, sr)
    df_annotation['start'] += initial_onset
    df_annotation['end'] = df_annotation['start'] + df_annotation['duration']
    
    
    f_chroma_quantized_annotation, _ = get_features_from_annotation(df_annotation, 
                                                                                     feature_rate)
    
    f_chroma_quantized_audio = get_features_from_audio(performer, fantasia)
    
    # Finding optimal shift of chroma vectors
    f_cens_1hz_audio = quantized_chroma_to_CENS(f_chroma_quantized_audio, 201, 50, feature_rate)[0]
    f_cens_1hz_annotation = quantized_chroma_to_CENS(f_chroma_quantized_annotation, 201, 50, feature_rate)[0]
    opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_audio, f_cens_1hz_annotation)
    # print('Pitch shift between the audio recording and score, determined by DTW:', opt_chroma_shift, 'bins')
    f_chroma_quantized_annotation = shift_chroma_vectors(f_chroma_quantized_annotation, opt_chroma_shift)

    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_quantized_audio,
                            f_chroma2=f_chroma_quantized_annotation,
                            input_feature_rate=feature_rate, 
                            step_weights=step_weights, 
                            threshold_rec=threshold_rec, 
                            verbose=False)
    
    wp = make_path_strictly_monotonic(wp)
    
    df_annotation_warped = df_annotation.copy(deep=True)
    df_annotation_warped["end"] = df_annotation_warped["start"] + df_annotation_warped["duration"]
    df_annotation_warped[['start', 'end']] = scipy.interpolate.interp1d(wp[1] / feature_rate, 
                            wp[0] / feature_rate, kind='linear', fill_value="extrapolate")(df_annotation[['start', 'end']])
    df_annotation_warped["duration"] = df_annotation_warped["end"] - df_annotation_warped["start"]
    df_annotation_warped.drop(columns=['velocity', 'pitch', 'instrument'], inplace=True)
    df_annotation_warped = df_annotation_warped.rename(columns={
                    'start': 'Onset (ms)',
                    'end': 'Offset (ms)',
                    'duration': 'Time_duration (ms)'
                    })
    cols_to_multiply = ['Onset (ms)', 'Offset (ms)', 'Time_duration (ms)']
    df_annotation_warped [cols_to_multiply] = df_annotation_warped [cols_to_multiply] * 1000
    
    # Add df_rests keeping original indexes
    df_annotation_warped = df_annotation_warped.copy()
    df_rests = df_rests.copy()
    df_annotation_warped['Note'] = df_annotation_warped.get('Note', None)
    df_rests['Note'] = '0'
    df_all = pd.concat([df_annotation_warped, df_rests])
    df_all = df_all.sort_index().reset_index(drop=True)
    required_cols = ['Onset (ms)', 'Offset (ms)', 'Score_duration', 'Note']
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df_all = adjust_silences(df_all)
    
    col = ['Note','Onset (ms)','Score_duration',
                    'Time-Signature','Movement','Measure','Repeated']

    df_all = df_all[[c for c in col if c in df_all.columns]]
       
    # Save output alignment
    output_dir = f"{ALIGNMENTS_ST}{performer}"
    os.makedirs(output_dir, exist_ok=True)
  
    df_all.to_csv(f"{output_dir}/st_{fantasia}.csv", index=False)
    

if __name__ == '__main__':
    for performer in ALL:
        for fantasia in range(1, 13):
            main(performer, fantasia)