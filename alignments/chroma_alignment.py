from common_variables_functions import *
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_features_from_audio(audio, tuning_offset, Fs, feature_rate, visualize=False):
        f_pitch = audio_to_pitch_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, feature_rate=feature_rate, verbose=visualize)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        if visualize:
            plot_chromagram(f_chroma_quantized, title='Quantized chroma features - Audio', Fs=feature_rate, figsize=(9,3))

        f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize)
        f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset, feature_rate=feature_rate, feature_sequence_length=f_chroma_quantized.shape[1], visualize=visualize)
        return f_chroma_quantized, f_DLNCO


def main(performer: str, fantasia: int):
    """Main function 

    Args:
        - performer: name of the performer
        - fantasia: fantasia number (1-12)
    """

    feature_rate = SR/HOP_LENGTH
    step_weights = np.array([1.5, 1.5, 2.0])
    threshold_rec = 10 ** 6
    
    # loading the recording
    ext='.m4a'
    
    audiopath= f'{AUDIOS}{performer}/{performer}_{fantasia}{ext}'
    audio, sr = librosa.load(audiopath, sr=SR)
    
    #Loading the .csv annotation file
    df_annotation, df_rests, rest_indexes_list = produce_df_from_unfolded_scores(performer, fantasia)
    
    initial_onset = detect_initial_onset(audio, sr)

    df_annotation['start'] += initial_onset
    df_annotation['end'] = df_annotation['start'] + df_annotation['duration']
    
    #estimate tuning
    tuning_offset = estimate_tuning(audio, SR)
    # Computing quantized chroma and DLNCO features
    f_chroma_quantized_audio, _ = get_features_from_audio(audio, tuning_offset, 
                                                                    SR, feature_rate, visualize=False)
    
    f_chroma_quantized_annotation, _ = get_features_from_annotation(df_annotation, feature_rate)
    
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
    output_dir = f"{ALIGNMENTS_CHROMA}{performer}"
    os.makedirs(output_dir, exist_ok=True)
    df_all.to_csv(f"{output_dir}/chroma_f{fantasia}.csv", index=False)

if __name__ == '__main__':
    for performer in ALL:
        for fantasia in range(1, 13):
            main(performer, fantasia)