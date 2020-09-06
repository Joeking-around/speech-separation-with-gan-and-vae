import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets,losses,optimizers,layers

class VAE(keras.Model):
	def __init__(self):
		super().__init__()
		self.fc1=layers.Dense(128)
		self.fc2=layers.Dense(20)
		self.fc3 = layers.Dense ( 20)
		self.fc4 = layers.Dense ( 128 )
		self.fc5 = layers.Dense ( 784)
	
	def encoder(self, x):
		h = tf.nn.relu ( self.fc1 ( x ) )
		mu = self.fc2 ( h )
		log_var = self.fc3 ( h )
		
		return mu, log_var
	
	def decoder(self, z):
		out = tf.nn.relu ( self.fc4 ( z ) )
		out = self.fc5 ( out )
		# 返回图片数据，784 向量
		return out
	def reparameterize(self, mu, log_var):
		eps = tf.random.normal ( log_var.shape )
		# 计算标准差
		std = tf.exp ( log_var ) ** 0.5
		# reparameterize 技巧
		z = mu + std * eps
		return z
	def call(self, inputs, training=None):
		mu, log_var = self.encoder ( inputs )
		# reparameterization trick
		z = self.reparameterize ( mu, log_var )
		# 通过解码器生成
		x_hat = self.decoder ( z )
		# 返回生成样本，及其均值与方差
		return x_hat, mu, log_var

	
if __name__=='__main__':
	(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
	x_train, x_test = x_train.astype ( np.float32 ) / 255., x_test.astype ( np.float32) / 255.
	train_db = tf.data.Dataset.from_tensor_slices ( x_train )
	train_db = train_db.shuffle ( 64 * 5 ).batch ( 64 )
	# 构建测试集对象
	test_db = tf.data.Dataset.from_tensor_slices ( x_test )
	test_db = test_db.batch ( 64)
	# 只需要通过图
	model = VAE ()
	model.build ( input_shape=(4, 784) )
	# 优化器
	optimizer = tf.optimizers.Adam ( 0.001 )
	
	for epoch in range ( 100 ):  # 训练 100 个 Epoch
		for step, x in enumerate ( train_db ):
			x = tf.reshape ( x, [-1, 784] )
	# 构建梯度记录器
			with tf.GradientTape () as tape:
	# 前向计算
				x_rec_logits, mu, log_var = model ( x )
	# 损失计算
				rec_loss = tf.nn.sigmoid_cross_entropy_with_logits ( labels=x, logits = x_rec_logits)
				rec_loss = tf.reduce_sum ( rec_loss ) / x.shape[0]
				kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp ( log_var ))
				kl_div = tf.reduce_sum ( kl_div ) / x.shape[0]
	# 合并误差项
				loss = rec_loss + 1. * kl_div
	# 自动求导
			grads = tape.gradient ( loss, model.trainable_variables )
			optimizer.apply_gradients ( zip ( grads, model.trainable_variables ) )
			if step % 100 == 0:
				print ( epoch, step, 'kl div:', float ( kl_div ), 'rec loss:', float (rec_loss ) )
	
	# z = tf.random.normal ( (32, 20) )
	# logits = model.decoder ( z )  # 仅通过解码器生成图片
	# x_hat = tf.sigmoid ( logits )  # 转换为像素范围
	# x_hat = tf.reshape ( x_hat, [-1, 28, 28] ).numpy () * 255.
	# x_hat = x_hat.astype ( np.uint8 )
	# save_image( x_hat, 'vae_images/epoch_%d_sampled.png' % epoch )