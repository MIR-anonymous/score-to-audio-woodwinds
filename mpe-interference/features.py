import numpy as np
import librosa


def hcqt(audio, sr_audio, param_hcqt):
    """
        Compute Harmonic CQT

        Parameters
        ----------
        audio_v : np.array
            mono audio signal as np.array
        sr_audio : int
            sampling rate of the audio signal
        param_hcqt : dict
            configuration parameters for the Harmonic CQT

        Returns

        -------
        data_3m : np.array
            3D array of Harmonic CQT
        time_sec_v : np.array
            corresponding time [in sec] of analysis windows
        frequency_hz_v : np.array
            corresponding frequency [in Hz] of CQT channels
    """
    fmin = float(librosa.note_to_hz(param_hcqt.min_note))
    tuning_est = librosa.estimate_tuning(y=audio)
    fmin_tuned = fmin * 2**(tuning_est / param_hcqt.bins_per_octave)

    hcqt_list = []
    min_time_frames = float('inf')
    for idx, h in enumerate(param_hcqt.harmonics):
        A_m = np.abs(librosa.cqt(y=audio, sr=sr_audio,
                                fmin=h*fmin_tuned,
                                hop_length=param_hcqt.hop_length,
                                bins_per_octave=param_hcqt.bins_per_octave,
                                n_bins=param_hcqt.n_bins,
                                tuning=0.0,
                                ))
        hcqt_list.append(A_m)
        min_time_frames = min(min_time_frames, A_m.shape[1])
    
    # Trim all HCQT arrays to the smallest frame length
    hcqt_list = [A[:, :min_time_frames] for A in hcqt_list]
    data_3m = np.stack(hcqt_list, axis=0)

    n_times = data_3m.shape[2]
    time_sec_v = librosa.frames_to_time(np.arange(n_times),
                                            sr=sr_audio,
                                            hop_length=param_hcqt.hop_length)
    
    frequency_hz_v = librosa.cqt_frequencies(n_bins=param_hcqt.n_bins,
                                                    fmin=fmin_tuned,
                                                    bins_per_octave=param_hcqt.bins_per_octave)

    return data_3m, time_sec_v, frequency_hz_v