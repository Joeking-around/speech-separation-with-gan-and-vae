import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import datasets,optimizers,losses
import matplotlib.pyplot as plt
from TIMIT_utilities import *
from utilities import *
from network_utilities import *
import time

# tensorboard --logdir E:/python_project/Generalproject

if __name__ == '__main__':
	# transform timit .sph file into .wav format
	train_epochs=6000
	sep_epochs=20000
	batchsize = 64
	path = r'e:/' # where your timit dataset folder locates
	timit_sph2wav ( path )
	runs=6
	for run in range(runs):
		tr1, tr2, ts1, ts2,data_info= load_timit_data ( path )
		# tr1, tr2, ts1, ts2, data_info = load_timit_data ( path,dr=3,malefolder='MDJM0',femalefolder='FGCS0' )
		train_dbs, mixture = process_timit_data ( tr1, tr2, ts1, ts2,batchsize )
		model_type='VAE'
		# model_type='GAN' if run < (runs//2) else 'VAE'
		if model_type=='VAE':
			save_path,result_folder =VAE_train ( train_dbs, epochs=train_epochs,batchsize=batchsize, optimizer='Adam', L1=100,L2=513,z_dim=20,learning_rate=0.001,data_info=data_info )
			VAE_male=VAE(L1=100,L2=513,z_dim=20)
			VAE_male.build ( input_shape=(batchsize, 513) )
			VAE_female = VAE ( L1=100, L2=513, z_dim=20 )
			VAE_female.build ( input_shape=(batchsize, 513) )
			VAE_male.load_weights('./save/male_VAE/')
			VAE_female.load_weights ( './save/female_VAE/' )
			models=VAE_male,VAE_female
			#
			# VAE_test(models=models,source_number=1,mixture=mixture,epochs=20000,optimizer='Adam',learning_rate=0.001)
			s1,s2=VAE_separation ( models=models, mixture=mixture, epochs=sep_epochs, optimizer='Adam',learning_rate=0.001,result_folder=result_folder )
			evaluation(s1=s1,s2=s2,mixture=mixture,folder=result_folder)
		elif model_type=='GAN':
			save_path,result_folder=GAN_train(train_dbs,epochs=train_epochs,learning_rate=0.0001,batchsize=batchsize,h_dimG=513,L1G=100,L2G=513,L1D=90,optimizer='RMSprop',data_info=data_info )
			generator_male=Generator(L1=100,L2=513)
			generator_male.build(input_shape=(batchsize,513))
			generator_male.load_weights('./save/male_G/')
			generator_female=Generator(L1=100,L2=513)
			generator_female.build(input_shape=(batchsize,513))
			generator_female.load_weights('./save/female_G/')
			discriminator_male=Discriminator(L1=90)
			discriminator_male.build ( input_shape=(batchsize, 513) )
			discriminator_male.load_weights('./save/male_D/')
			discriminator_female=Discriminator(L1=90)
			discriminator_female.build(input_shape=(batchsize,513))
			discriminator_female.load_weights('./save/female_D/')
			generators=generator_male,generator_female
			discriminators=discriminator_male,discriminator_female

			# test_generator(generator_female)
			# GAN_test(generators,source_number=1,mixture=mixture,epochs=2,optimizer='RMSprop',learning_rate=0.001)
			s1,s2=GAN_separation(generators=generators,discriminators=discriminators,mixture=mixture,epochs=sep_epochs,optimizer='Adam',learning_rate=0.001,result_folder=result_folder)
			evaluation(s1=s1,s2=s2,mixture=mixture,folder=result_folder)

#
#
# path = r'e:/' # where your timit dataset folder locates
# tr1, tr2, ts1, ts2, data_info = load_timit_data ( path,dr=3,malefolder='MDJM0',femalefolder='FGCS0' )
# figure = plt.figure (dpi=300)
# ax1 = plt.subplot2grid ( (2, 4), (0, 0) )
# ax1.specgram(ts1,Fs=16000,NFFT=1024,noverlap=256)
# ax2 = plt.subplot2grid ( (2, 4), (0, 1) )
# ax2.specgram(ts2,Fs=16000,NFFT=1024,noverlap=256)
# ax3 = plt.subplot2grid ( (2, 4), (1, 0) )
# ax4 = plt.subplot2grid ( (2, 4), (1, 1) )
# ax5 = plt.subplot2grid ( (2, 4), (0, 2), colspan=2, rowspan=2 )
# plt.savefig()
# plt.show()