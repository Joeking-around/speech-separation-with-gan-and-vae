import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers,optimizers,losses,Sequential,initializers
import os
from utilities import *
import librosa

class Generator(keras.Model):
	def __init__(self,**kwargs):
		super().__init__()
		# should be L1=100 L2=513
		self.L1=kwargs['L1']
		self.L2=kwargs['L2']
		self.l1=layers.Dense(self.L1)
		self.l2=layers.Dense(self.L2)
		
	def call(self,input):
		output=tf.nn.softplus(self.l2(tf.nn.softplus(self.l1(input))))
		return output
	
class Discriminator(keras.Model):
	def __init__(self,**kwargs):
		super().__init__()
		# shoule be L1=90
		self.L1=kwargs['L1']
		
		self.l1=layers.Dense(self.L1)
		self.l2=layers.Dense(1)
		
	def call(self,input):
		# Standard GAN
		# output=tf.nn.sigmoid(self.l2(tf.nn.tanh(self.l1(input))))
		# Wasserstein GAN
		output = self.l2 ( tf.nn.tanh ( self.l1 ( input ) ) )
		return output

class VAE(keras.Model):
	def __init__(self,**kwargs):
		super().__init__()
		# should be L1=100 L2=513 z_dim=20
		self.L1=kwargs['L1']
		self.L2=kwargs['L2']
		self.z_dim=kwargs['z_dim']
		# encoder
		self.fc1=layers.Dense(self.L1)
		self.fc21=layers.Dense(self.z_dim,kernel_initializer=initializers.zeros())
		self.fc22=layers.Dense(self.z_dim,kernel_initializer=initializers.zeros())
		# decoder
		self.fc3=layers.Dense(self.L2)
	def encoder(self,input):
		h=tf.nn.relu(self.fc1(input))
		# mean and log variance
		mu=self.fc21(h)
		log_var=self.fc22(h)
		return mu,log_var
	
	def reparameterization(self,mu,log_var):
		eps=tf.random.normal(log_var.shape)
		std=tf.exp(log_var)**0.5
		z=mu+std*eps
		return z
	
	def decoder(self,z):
		output=tf.nn.softplus(self.fc3(z))
		return output
	
	def call(self, inputs, training=None, mask=None):
		mu,log_var=self.encoder(inputs)
		z=self.reparameterization(mu,log_var)
		out=self.decoder(z)
		return out,mu,log_var
	

def VAE_train(train_dbs,epochs,**kwargs):
	L1=kwargs['L1']
	L2=kwargs['L2']
	z_dim=kwargs['z_dim']
	learning_rate=kwargs['learning_rate']
	batchsize=kwargs['batchsize']
	data_info=kwargs['data_info']
	if kwargs['optimizer']=='Adam':
		optimizer=optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999)
	elif kwargs['optimizer']=='RMSprop':
		optimizer=optimizers.RMSprop(learning_rate=learning_rate)
	result_folder='./VAE_result/'+data_info+'/'
	log_dir = result_folder + 'tensorboard/'
	summary_writer = tf.summary.create_file_writer ( log_dir )
	if not os.path.exists(result_folder):
		os.makedirs(result_folder)
	if not os.path.exists(r'./save'):
		os.makedirs(r'./save')
	save_path={}

	for train_db in train_dbs:
		model = VAE ( L1=L1, L2=L2, z_dim=z_dim )
		
		for epoch in range ( epochs ):
			for step, x in enumerate ( train_db ):
				with tf.GradientTape () as tape:
					out, mu, log_var = model ( x )
					
					rec_loss = losses.mean_squared_error ( x, out )
					# print ( 'rec_loss=', rec_loss )
					rec_loss = tf.reduce_sum ( rec_loss ) / x.shape[0]
					kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp ( log_var ))
					kl_div = tf.reduce_sum ( kl_div ) / x.shape[0]
					
					loss = rec_loss + kl_div
				grads = tape.gradient ( loss, model.trainable_variables )
				
				optimizer.apply_gradients ( zip ( grads, model.trainable_variables ) )
			with summary_writer.as_default():
				tf.summary.scalar ( 'male_loss' if train_db==train_dbs[0] else 'female_loss', float ( loss ), step=epoch)
				tf.summary.scalar ( 'male_rec_loss' if train_db==train_dbs[0] else 'female_rec_loss', float ( rec_loss ), step=epoch )
				tf.summary.scalar ( 'male_kl-div' if train_db==train_dbs[0] else 'female_kl-div', float ( kl_div ), step=epoch )
			if epoch%50==0:
				print ( epoch, 'rec_loss=', float ( rec_loss ), 'kl=', float ( kl_div ), 'loss=', float ( loss ) )
		# save model
		tag='male_VAE' if train_db==train_dbs[0] else 'female_VAE'
		save_path[tag] =r'./save/' + tag + '/'
		model.save_weights(r'./save/'+tag+'/',overwrite=True)
	return save_path,result_folder

def GAN_train(train_dbs,epochs,**kwargs):
	h_dimG=kwargs['h_dimG']
	L1G=kwargs['L1G']
	L2G=kwargs['L2G']
	L1D=kwargs['L1D']
	learning_rate = kwargs['learning_rate']
	batchsize = kwargs['batchsize']
	data_info = kwargs['data_info']
	result_folder = './GAN_result/' + data_info + '/'
	log_dir=result_folder+'tensorboard/'
	summary_writer = tf.summary.create_file_writer ( log_dir )
	if not os.path.exists(result_folder):
		os.makedirs(result_folder)
	if kwargs['optimizer']=='Adam':
		G_optimizer=optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999)
		D_optimizer = optimizers.Adam ( learning_rate=learning_rate, beta_1=0.9, beta_2=0.999 )
	elif kwargs['optimizer']=='RMSprop':
		G_optimizer=optimizers.RMSprop(learning_rate=learning_rate)
		D_optimizer = optimizers.RMSprop ( learning_rate=learning_rate )
	if not os.path.exists(r'./save'):
		os.makedirs(r'./save')

	for train_db in train_dbs:
		generator=Generator(L1=L1G,L2=L2G)
		discriminator=Discriminator(L1=L1D)
		save_path={}
		for epoch in range(epochs):
			for step,batch_x in enumerate(train_db):
				batch_h = tf.random.normal ( [batch_x.shape[0], h_dimG] )
				# sample hidden variable
				with tf.GradientTape() as tape:
					fake = generator ( batch_h )
					d_fake = discriminator ( fake )
					d_real = discriminator ( batch_x )
					# standard GAN
					# d_loss_fake = losses.binary_crossentropy ( tf.zeros_like ( d_fake ), d_fake, from_logits=True )
					# d_loss_real=losses.binary_crossentropy(tf.ones_like(d_real),d_real,from_logits=True)
					# loss=tf.reduce_mean(d_loss_fake+d_loss_real)
					# Wasserstein GAN
					d_loss = d_fake - d_real
					d_loss=tf.reduce_mean(d_loss)
					# Wgan-gp
					# alpha = tf.random.uniform ( shape=[batch_x.shape[0], 1], minval=0, maxval=1 )
					# differences = fake - batch_x
					# interpolates = batch_x + alpha * differences
					# gradients = tf.gradients ( discriminator ( interpolates ), [interpolates] )[0]
					# slopes = tf.sqrt ( tf.reduce_sum ( tf.square ( gradients ), axis=1 ) )
					# gradient_penalty = tf.reduce_mean ( (slopes - 1.) ** 2 )
					# LAMBDA = 10
					# d_loss += LAMBDA * gradient_penalty
				grads = tape.gradient ( d_loss, discriminator.trainable_variables )
				D_optimizer.apply_gradients ( zip ( grads, discriminator.trainable_variables))
				for w in discriminator.trainable_variables:
					w = tf.clip_by_value ( w, clip_value_min=-0.01, clip_value_max=0.01 )
				if step%5==0:
					batch_h = tf.random.normal ( [batch_x.shape[0], h_dimG] )
					with tf.GradientTape () as tape:
						fake = generator ( batch_h )
						d_fake = discriminator ( fake )
						# standard GAN
						# d_loss_fake = losses.binary_crossentropy ( tf.ones_like ( d_fake ), d_fake, from_logits=True )
						# loss=tf.reduce_mean(d_loss_fake)
						# Wasserstein GAN
						g_loss = -d_fake
						g_loss=tf.reduce_mean(g_loss)
					grads=tape.gradient(g_loss,generator.trainable_variables)
					G_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
			# if epoch%5==0:
			with summary_writer.as_default():
				tf.summary.scalar ( 'male_g_loss' if train_db==train_dbs[0] else 'female_g_loss' , float ( g_loss ), step=epoch)
				tf.summary.scalar ( 'male_d_loss' if train_db==train_dbs[0] else 'female_d_loss' , float ( d_loss ), step=epoch )
			print(epoch,'d_loss=',float(d_loss),'g_loss=',float(g_loss))
		tagG='male_G' if train_db==train_dbs[0] else 'female_G'
		tagD='male_D' if train_db==train_dbs[0] else 'female_D'
		save_path[tagG]='./save/'+tagG+'/'
		generator.save_weights('./save/'+tagG+'/')
		save_path[tagD]='./save/' + tagD + '/'
		discriminator.save_weights ( './save/' + tagD + '/' )
	
	return save_path,result_folder

def VAE_test(models,source_number,mixture,epochs=10,**kwargs):
	model=models[0] if source_number==1 else models[1]
	target=mixture.mag1 if source_number==1 else mixture.mag2
	learning_rate = kwargs['learning_rate']
	if kwargs['optimizer']=='Adam':
		optimizer=optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999)
	elif kwargs['optimizer']=='RMSprop':
		optimizer=optimizers.RMSprop(learning_rate=learning_rate)
	
	z = tf.Variable ( tf.random.normal([len(target),model.z_dim]), trainable=True )
	for epoch in range(epochs):
		with tf.GradientTape() as tape:
			out = model.decoder ( z )
			rec_loss=losses.mean_squared_error(target,out)
			loss=tf.reduce_mean(rec_loss)
		grads=tape.gradient(loss, z)
		# print('grads',grads)
		optimizer.apply_gradients( [(grads,z)] )
		if epoch % 50 == 0:
			print('epoch:',epoch,'loss:',float(loss))
		
	
	return None

def GAN_test(generators,source_number,mixture,epochs=10,**kwargs):
	generator=generators[0] if source_number==1 else generators[1]
	target=mixture.mag1 if source_number==1 else mixture.mag2
	
	learning_rate = kwargs['learning_rate']
	if kwargs['optimizer'] == 'Adam':
		optimizer = optimizers.Adam ( learning_rate=learning_rate, beta_1=0.9, beta_2=0.999 )
	elif kwargs['optimizer'] == 'RMSprop':
		optimizer = optimizers.RMSprop ( learning_rate=learning_rate )
	
	h = tf.Variable ( tf.random.normal ( [len (target ), 513] ), trainable=True )
	for epoch in range(epochs):
		with tf.GradientTape() as tape:
			out = generator ( h )
			rec_loss=losses.mean_squared_error(target,out)
			loss=tf.reduce_mean(rec_loss)
		grads=tape.gradient(loss, h)
		optimizer.apply_gradients( [(grads,h)] )
		if epoch%50==0:
			print('epoch:',epoch,'loss:',float(loss))
	
	return None

def VAE_separation(models,mixture,epochs,**kwargs):
	learning_rate = kwargs['learning_rate']
	if kwargs['optimizer'] == 'Adam':
		optimizer = optimizers.Adam ( learning_rate=learning_rate, beta_1=0.9, beta_2=0.999 )
	elif kwargs['optimizer'] == 'RMSprop':
		optimizer = optimizers.RMSprop ( learning_rate=learning_rate )
	result_folder=kwargs['result_folder']
	log_dir = result_folder + 'tensorboard/'
	summary_writer = tf.summary.create_file_writer ( log_dir )
	VAE_male,VAE_female=models
	VAE_male.trainable=False
	VAE_female.trainable=False
	target=mixture.mag_mix
	z1 = tf.Variable ( tf.random.normal ( [len ( target ), VAE_male.z_dim] ), trainable=True )
	z2 = tf.Variable ( tf.random.normal ( [len ( target ), VAE_female.z_dim] ), trainable=True )
	
	for epoch in range(epochs):
		with tf.GradientTape() as tape:
			out1=VAE_male.decoder(z1)
			out2=VAE_female.decoder(z2)
			loss=losses.mean_squared_error(target,out1+out2)
			loss=tf.reduce_mean(loss)
			smooth_loss = np.abs ( out1[1:] - out1[:-1] ) + np.abs ( out2[1:] - out2[:-1] )
			smooth_loss = tf.reduce_mean ( smooth_loss )
			loss= loss+0.1*smooth_loss
		grads=tape.gradient(loss,[z1,z2])
		optimizer.apply_gradients(zip(grads,[z1,z2]))
		with summary_writer.as_default ():
			tf.summary.scalar ( 'rec_loss', float ( loss ), step=epoch )
			tf.summary.scalar ( 'smooth_loss', float ( smooth_loss ), step=epoch )
		if epoch%50==0:
			print ( 'epoch:', epoch, 'loss:', float ( loss ) ,'smooth_loss:',float(smooth_loss))

	out1=out1.numpy()
	out2=out2.numpy()
	eps = finfo(float32).eps
	estimated_source_male=out1/((out1+out2)+eps)*mixture.mag_mix*np.exp(1j*mixture.phase_mix)
	estimated_source_female = out2/((out1+out2)+eps)*mixture.mag_mix*np.exp(1j*mixture.phase_mix)
	s1=librosa.istft(estimated_source_male.transpose(),hop_length=256)
	s2= librosa.istft ( estimated_source_female.transpose (), hop_length=256 )
	
	return s1,s2
def GAN_separation(generators,discriminators,mixture,epochs,**kwargs):
	learning_rate = kwargs['learning_rate']
	if kwargs['optimizer'] == 'Adam':
		optimizer = optimizers.Adam ( learning_rate=learning_rate, beta_1=0.9, beta_2=0.999 )
	elif kwargs['optimizer'] == 'RMSprop':
		optimizer = optimizers.RMSprop ( learning_rate=learning_rate )
	result_folder = kwargs['result_folder']
	log_dir = result_folder + 'tensorboard/'
	summary_writer = tf.summary.create_file_writer ( log_dir )
	generator_male, generator_female = generators
	discriminator_male, discriminator_female = discriminators
	generator_male.trainable=False
	generator_female.trainable=False
	discriminator_male.trainable = False
	discriminator_female.trainable = False
	target=mixture.mag_mix
	h1 = tf.Variable ( tf.random.normal ( [len ( target ), 513] ), trainable=True )
	h2 = tf.Variable ( tf.random.normal ( [len ( target ), 513] ), trainable=True )
	
	for epoch in range(epochs):
		with tf.GradientTape() as tape:
			out1=generator_male(h1)
			out2=generator_female(h2)
			score_m=discriminator_male(out1)
			score_f=discriminator_female(out2)
			rec_loss=losses.mean_squared_error(mixture.mag_mix,out1+out2)
			rec_loss=tf.reduce_mean(rec_loss)
			score_loss=-(score_f+score_m)
			score_loss=tf.reduce_mean(score_loss)
			smooth_loss=np.abs(out1[1:]-out1[:-1])+np.abs(out2[1:]-out2[:-1])
			smooth_loss=tf.reduce_mean(smooth_loss)
	
			loss= rec_loss+0.1*score_loss+0.1*smooth_loss
			
		grads=tape.gradient(loss,[h1,h2])
		optimizer.apply_gradients(zip(grads,[h1,h2]))
		with summary_writer.as_default ():
			tf.summary.scalar ( 'rec_loss', float ( loss ), step=epoch )
			tf.summary.scalar ( 'score_loss', float ( score_loss ), step=epoch )
			tf.summary.scalar ( 'smooth_loss', float ( smooth_loss ), step=epoch )
		if epoch%50==0:
			print ( 'epoch:', epoch, 'loss:', float ( loss ),'rec_loss:',float(rec_loss),'score_loss:',float(score_loss),'smooth_loss:',float(smooth_loss) )
	
	out1 = out1.numpy ()
	out2 = out2.numpy ()
	eps = finfo ( float32 ).eps
	estimated_source_male = out1 / ((out1 + out2) + eps) * mixture.mag_mix * np.exp ( 1j * mixture.phase_mix )
	estimated_source_female = out2 / ((out1 + out2) + eps) * mixture.mag_mix * np.exp ( 1j * mixture.phase_mix )
	s1 = librosa.istft ( estimated_source_male.transpose (), hop_length=256 )
	s2 = librosa.istft ( estimated_source_female.transpose (), hop_length=256 )
	return s1,s2

def test_generator(generator,**kwargs):
	
	if os.path.exists('./test.wav'):
		os.remove('./test.wav')
	h1 = tf.Variable ( tf.random.normal ( [180, 513] ), trainable=True )
	fake=generator(h1)
	fake=fake.numpy()
	fake_signal=librosa.istft(fake.transpose(),hop_length=256)
	librosa.output.write_wav('./test.wav',fake_signal,16000)
	return None