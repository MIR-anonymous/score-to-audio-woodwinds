from predictor import Predictor
import numpy as np
audiofilename=''
predictor = Predictor(device='cuda')
prediction, times = predictor.predict(audiofilename)
np.savez(f'{audiofilename}.npz', prediction=prediction, times=times)