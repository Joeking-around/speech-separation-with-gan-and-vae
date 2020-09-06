from numpy import *
import os
import librosa
import matplotlib.pyplot as plt

def eval_sisnr(s_hat,s):
	if s_hat.shape[0]==s.shape[0]:
		eps = finfo(float32).eps;
		s_target=(dot(s_hat,s)/dot(s,s))*s
		s_error=s_hat-s_target
		sisnr=10*log10(max(dot(s_target,s_target),eps)/max(dot(s_error,s_error),eps))
		return sisnr
	else:
		print('They need to have same dimension')
		return None

def evaluation(s1,s2,mixture,folder):
	# s1,s2 is numpy.ndarray mixture is Mixture()

	if os.path.exists(folder+'male_estimated.wav'):
		os.remove ( folder+'male_estimated.wav')
	if os.path.exists (folder+'female_estimated.wav' ):
		os.remove ( folder+'female_estimated.wav' )
	if os.path.exists(folder+'mix.wav'):
		os.remove ( folder+'mix.wav' )
	if os.path.exists(folder+'male_origin.wav'):
		os.remove ( folder+'male_origin.wav' )
	if os.path.exists ( folder+'female_origin.wav' ):
		os.remove ( folder+'female_origin.wav' )
	male_sisdr=eval_sisnr ( s1, mixture.wav1 )
	female_sisdr=eval_sisnr( s2, mixture.wav2 )
	male_o_sisdr=eval_sisnr(mixture.wav1+mixture.wav2,mixture.wav1)
	female_o_sisdr=eval_sisnr (  mixture.wav1 + mixture.wav2,mixture.wav2 )
	
	with open(folder+'evaluation_results.txt','w') as f:
		seq=['male_sisdr:'+str(male_sisdr)+'\n','sisdr between mixture and male sound:'+str(male_o_sisdr)+'\n',
		     'female_sisdr:'+str(female_sisdr)+'\n','sisdr between mixture and female sound:'+str(female_o_sisdr)+'\n']
		f.writelines(seq)
	
	librosa.output.write_wav(folder+'mix.wav',mixture.wav1 + mixture.wav2,16000)
	librosa.output.write_wav ( folder + 'male_origin.wav', mixture.wav1, 16000 )
	librosa.output.write_wav ( folder + 'female_origin.wav', mixture.wav2, 16000 )
	librosa.output.write_wav(folder+'male_estimated.wav',s1,16000)
	librosa.output.write_wav (  folder+'female_estimated.wav' ,s2,16000)

	print ( 'male_sisdr:',male_sisdr )
	print('sisdr between mixture and male sound:',male_o_sisdr)
	print ( 'female_sisdr:',female_sisdr )
	print ( 'sisdr between mixture and female sound:', female_o_sisdr )

