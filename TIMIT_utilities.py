import glob
import os

import librosa
import numpy as np
from sphfile import SPHFile
import tensorflow as tf

class Mixture():
	def __init__(self,wav1,wav2,mag1,mag2,mag_mix,phase_mix):
		self.wav1=wav1
		self.wav2=wav2
		self.mag1=mag1
		self.mag2=mag2
		self.mag_mix=mag_mix
		self.phase_mix=phase_mix
	
	def __len__(self):
		return self.mag_mix.shape[0]

def timit_sph2wav(path):
	# to transform the dataset from sph file to .wav file
	if os.path.exists ( path + r'TIMIT/done.txt' ):
		print ( 'The dataset has been already preprocessed' )
	else:
		sph_files = glob.glob ( path + r'TIMIT/*/*/*/*.WAV' )
		for file_path in sph_files:
			sph = SPHFile ( file_path )
			sph.write_wav ( filename=file_path.replace ( '.WAV', 'copy.WAV' ) )
			os.remove ( file_path )
			os.rename ( file_path.replace ( '.WAV', 'copy.WAV' ), file_path )
		with open ( path + r'TIMIT/done.txt', 'w' ) as f:
			f.write ( 'The dataset has been preprocessed' )
		print ( 'Done' )
	return None


def load_timit_data(path, dr=None, malefolder=None, femalefolder=None):
	if dr is None:
		dr = np.random.randint ( 1, 9 )
	path += r'TIMIT/TRAIN/DR%d' % dr
	if malefolder is None:
		malefolder = np.random.choice ( [name for name in os.listdir ( path ) if name[0] == 'M'] )
	if femalefolder is None:
		femalefolder = np.random.choice ( [name for name in os.listdir ( path ) if name[0] == 'F'] )
	
	# file with similar length as test pair
	malesounds = [librosa.core.load ( path + r'/' + malefolder + r'/' + file, sr=None )[0] for file in
	              os.listdir ( path + r'/' + malefolder ) if 'WAV' in file]
	femalesounds = [librosa.core.load ( path + r'/' + femalefolder + r'/' + file, sr=None )[0] for file in
	                os.listdir ( path + r'/' + femalefolder ) if 'WAV' in file]
	length_male, length_female = list ( map ( lambda x: x.shape[0] , malesounds )), list ( map ( lambda x: x.shape[0] ,
	                                                                                       femalesounds ))
	i = np.argmin ( np.array ( [[abs ( i1 - i2 ) for i1 in length_male] for i2 in length_female] ) )
	maxl = max ( length_male[i % 10], length_female[i // 10] )
	ts1=np.pad ( malesounds[i % 10], (0, maxl - length_male [ i % 10 ]), 'constant')
	ts2=np.pad ( femalesounds[i // 10], (0, maxl - length_female[i // 10]), 'constant')
	# the rest of them used as training set
	malesounds.pop ( i % 10 )
	femalesounds.pop ( i // 10 )
	tr1=np.concatenate ( malesounds )
	tr2=np.concatenate ( femalesounds )
	# 这两行代码是把训练集截断成一样的长度但是我感觉没有必要
	# tr1 = tr1[0:min ( tr1.shape[0], tr2.shape[0] )]
	# tr2 = tr2[0:min ( tr1.shape[0], tr2.shape[0] )]
	
	# 这里函数首先把信号长度截断到1024的整数倍 然后前后各补1024个零
	sr = 16000
	
	tr1 = tr1 / np.std ( tr1 )
	tr2 = tr2/ np.std ( tr2 )
	ts1 = ts1 / np.std ( ts1 )
	ts2 = ts2 / np.std ( ts2 )

	sz = 1024
	def zp(x):
		return np.hstack ( (np.zeros ( sz ), x[0:int ( sz * np.floor ( x.shape[0] / sz ) )], np.zeros ( sz )) )
	
	tr1 = zp ( tr1[0:int ( sz * np.floor ( tr1.shape[0] / sz ) )] )
	tr2 = zp ( tr2[0:int ( sz * np.floor ( tr2.shape[0] / sz ) )] )
	ts1 = zp ( ts1[0:int ( sz * np.floor ( ts1.shape[0] / sz ) )] )
	ts2 = zp ( ts2[0:int ( sz * np.floor ( ts2.shape[0] / sz ) )] )
	data_info=('_'.join ( ['DR' + str ( dr ), malefolder, femalefolder] ))
	return tr1,tr2,ts1,ts2,data_info

def process_timit_data(tr1,tr2,ts1,ts2,batchsize):
	sz=1024
	spec1=librosa.stft(tr1,n_fft=sz,hop_length=256).transpose()
	spec2=librosa.stft(tr2,n_fft=sz,hop_length=256).transpose()
	mag1,phase1=np.abs(spec1),np.angle(spec1)
	mag2,phase2=np.abs(spec2),np.angle(spec2)
	
	spec_mix=librosa.stft(ts1+ts2,n_fft=1024,hop_length=256).transpose()
	
	mag_mix,phase_mix=np.abs(spec_mix),np.angle(spec_mix)
	spec_test1=librosa.stft(ts1,n_fft=1024,hop_length=256).transpose()
	spec_test2 = librosa.stft (ts2, n_fft=1024, hop_length=256 ).transpose()
	mag_test1,mag_test2=np.abs(spec_test1),np.abs(spec_test2)
	
	# form dataset
	train_db1=tf.data.Dataset.from_tensor_slices(mag1).shuffle(10000).batch(batchsize)
	train_db2=tf.data.Dataset.from_tensor_slices(mag2).shuffle(10000).batch(batchsize)
	mixture=Mixture(ts1,ts2,mag_test1,mag_test2,mag_mix,phase_mix)
	
	
	train_dbs=[train_db1,train_db2]
	return train_dbs,mixture
	
# if __name__ == '__main__':
	# path1=r'e:/TIMIT/TRAIN/DR1/MCPM0/SA1.WAV'
	# path2 = r'e:/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV'
	# male,fs1=lr.core.load(path1,sr=None)
	# female,fs2 = lr.core.load( path2,sr=None )
	# # playsound.playsound(path1)
	# # playsound.playsound(path2)
	#
	# mix=(male[:female.shape[0]]+female)
	# # mix=male
	# output_path=r'E:\UniversityOfSurrey\general_project\projectaudio\mixture.wav'
	# lr.output.write_wav(output_path,mix,fs1)
	# print(male)
	# playsound.playsound(output_path)

	
	# batchsize = 32
	
	# path = r'e:/'
	# timit_sph2wav ( path )
	# tr1, tr2, ts1, ts2, malefolder, femalefolder, dr = load_timit_data ( path, dr=1, malefolder='MCPM0',
	#                                                                      femalefolder='FCJF0' )
	# train_dbs, mixture = process_timit_data ( tr1, tr2, ts1, ts2, batchsize )
	# print(mixture.mag_mix)

