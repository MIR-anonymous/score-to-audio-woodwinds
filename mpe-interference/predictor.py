from types import SimpleNamespace
import torch
import numpy as np
import librosa
from features import hcqt as get_hcqt
from training import PitchLigthning



class Predictor():
	def __init__(self, device='mps', monophonic=False):
		"""Predictor class for the multi-pitch estimation model.
		Parameters
		----------
		device : str
			Device to run the model on. Options are 'cpu', 'cuda' or 'mps'.
		"""
		self.device = device
		self.monophonic = monophonic
		self.sr_audio = 22050
		self.param_hcqt = SimpleNamespace(
			min_note='C1',
			harmonics=[0.5, 1, 2, 3, 4, 5],
			hop_length=512,
			n_bins=216, # 6 octaves with 36 bins per octave
			bins_per_octave=36 # 3 bins per semitone in 12 tone equal temperament
		)
		# load the model
		lightningModule = PitchLigthning.load_from_checkpoint('src/model.ckpt', map_location=self.device,  
														weights_only=False)
		self.model = lightningModule.model
		self.model.eval()

	
	def predict(self, audio_file):
		"""Predict the multi-pitch of an audio file.
		Parameters
		----------
		audio_file : str
			Path to the audio file.
			
		Returns
		----------
		prediction : np.ndarray
			Predicted multi-pitch. Shape is (72, timeFrames) where 72 is the number of pitches.
		times : np.ndarray
			Corresponding time stamps.
		frequencies : np.ndarray
			Corresponding frequencies."""
		# load audio
		audio, sr = librosa.load(audio_file, sr=self.sr_audio, mono=True)
		# audio = audio[:30*sr] # cut audio to 30 seconds?
		# compute hcqt
		hcqt, times, frequencies = get_hcqt(audio, self.sr_audio, self.param_hcqt)
		# turn to tensor add batch dimension
		hcqt = torch.tensor(hcqt, dtype=torch.float32, device=self.device)
		X = hcqt.unsqueeze(0)
		# predict
		with torch.no_grad():
				y = self.model(X)
				y = torch.sigmoid(y) 
		# convert to numpy and return
		prediction = y.squeeze(0).squeeze(0).cpu().detach().numpy()
		
		if self.monophonic:
			monophonic_indices = np.argmax(prediction, axis=0)
			pitch_indices = np.arange(prediction.shape[1])
			prediction_monophonic = np.zeros_like(prediction)
			prediction_monophonic[monophonic_indices, pitch_indices] = prediction[monophonic_indices, pitch_indices]
			prediction = prediction_monophonic

		return prediction, times



if __name__ == "__main__":
	device = 'mps'
	predictor = Predictor(device)
	audio_file = 'data/test.wav'
	prediction, times = predictor.predict(audio_file)
	np.savez('data/test_transcription.npz', prediction=prediction, times=times)