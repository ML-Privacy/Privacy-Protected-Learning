import sys
import argparse
import tensorflow as tf
import numpy
import Models
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import csv

from Params import Params

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='set data type as mnist or cifar10')
parser.add_argument('--iterations', help='Set the number of DLG iterations')
parser.add_argument('--img', help='Set the number of images for the DLG')
parser.add_argument('--type', help='Set the type of DLG test')
parser.add_argument('--seed', help='enable random seed or stick with constant')
args = parser.parse_args()


# # CDGD model instantiation (not functional)
# model_name = 'CNN'
# identical = True
# initer = 'glorot_uniform'

# # nb_agents = 1
# nb_agents = 3

# big_k=1
# maxLam=0.01
# topology = 'full'
# always_update = False
# lr = 0.01

# params = Params(nb_agents, big_k, always_update=always_update, topology=topology, maxLam=maxLam)

# Generating test data for checking deep leakage

if args.data:
	global data_type
	data_type = args.data

if args.iterations:
	itn = args.iterations
else:
	itn = 30000

if args.img:
	num_img = args.img
else:
	num_img = 1

if args.type:
	global test_type
	test_type = args.type
else:
	raise ValueError('Must specify the type as base or new')

print(data_type, '\t', test_type)

if args.data == 'mnist':
	file_n = 'mnist/mnist_DLG_'+test_type+'.csv'
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	# train_images, test_images = train_images / 255.0, test_images / 255.0
	train_images, test_images = train_images.reshape(train_images.shape[0], 28, 28, 1), test_images.reshape(test_images.shape[0], 28, 28, 1)
	# train_images, test_images = train_images.reshape(-1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)
	train_images, test_images = train_images / 255, test_images / 255
	nb_classes = len(numpy.unique(train_labels))
	train_labels = tf.one_hot(train_labels, 10)
	width, height, channel = 28, 28, 1

elif args.data == 'cifar10':
	file_n = 'cifar10/cifar10_DLG_'+test_type+'.csv'
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
	train_images, test_images = train_images / 255.0, test_images / 255.0
	nb_classes = len(numpy.unique(train_labels))
	# train_labels = train_labels.flatten()
	train_labels = tf.one_hot(train_labels, 10)
	# test_labels = tf.one_hot(train_labels, 10)
	width, height, channel = 32, 32, 3

else:
	raise ValueError('Data set must be mnist of cifar10')

img_dim = (width, height, channel)

x_input = tf.keras.Input(shape=img_dim)
x_sig = x_input#tf.keras.layers.Activation('sigmoid')(x_input)
x1 = tf.keras.layers.Conv2D(32, kernel_size=5, activation='sigmoid', padding="same")(x_sig)
x = tf.keras.layers.Conv2D(32, kernel_size=5, activation='sigmoid', padding="same")(x1)
x = tf.keras.layers.Conv2D(32, kernel_size=5, activation='sigmoid', padding="same")(x)
x = tf.keras.layers.Add()([x, x1])
x2 = tf.keras.layers.Conv2D(64, kernel_size=5, activation='sigmoid', padding="same")(x1)
x = tf.keras.layers.Conv2D(64, kernel_size=5, activation='sigmoid', padding="same")(x)
x = tf.keras.layers.Conv2D(64, kernel_size=5, activation='sigmoid', padding="same")(x)
x = tf.keras.layers.Add()([x, x2])
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='sigmoid')(x)
x = tf.keras.layers.Dense(10)(x)
model = tf.keras.models.Model(inputs=[x_input], outputs=[x])

# Clipping of gradient
def clipGrad(grad, c):
    std = tf.math.reduce_std(grad)
    return tf.clip_by_value(grad, -c*std,c*std, name=None)

# Computes the maximum of gradient
def getST(grad):
    return tf.math.reduce_max(tf.math.abs(grad))

# Computed the ternation of gradient 
def tern(grad, st, stMultiplier):
    if stMultiplier == 0:
        return grad

    bernoulli = tf.math.abs(grad) / (st * stMultiplier)
    dist = tfp.distributions.Bernoulli(probs=bernoulli, dtype=tf.float32)
    bt = dist.sample()
    return tf.math.multiply(tf.math.sign(grad), bt) * st * stMultiplier

# Computes the test
def experiment(img = 25,
			   num_img = 1,
			   iterations = 5000,
			   olr = 0.1,
			   momentum=0.8,
			   plotName='CDGDDLG_Test/Cifar_',
			   clipC=2.5,
			   stMultiplier=1,
			   file=None,
			   test_type= None):
	
	lr = olr
	
	gt_x = tf.constant(train_images[img:img+num_img])
	gt_y = tf.constant(train_labels[img:img+num_img])

	# Standard no additions
	x = []
	with tf.GradientTape() as tape:
		gt_out = model(gt_x, training=True)
		# gt_out = tf.reshape(gt_out, (num_img, 1, 10))

		if data_type == 'mnist':
			gt_out = tf.reshape(gt_out, (num_img, 10)) # Mnist

		elif data_type == 'cifar10':
			gt_out = tf.reshape(gt_out, (num_img, 1, 10))

		loss = tf.keras.losses.categorical_crossentropy(gt_y, gt_out) # Calculate losses between true and predicted labels
		gt_gradient = tape.gradient(loss, model.trainable_variables) # Target, sources. Computes gradient using operations recorded

	if test_type == 'new':

		for i in range(len(gt_gradient)):
			shape = gt_gradient[i].get_shape().as_list()
			epsilon = 0.5 / tf.math.pow(tf.cast(i + 1, tf.float32) * 0.007 + 1, 0.7) # randint(2, 250) 
			
			alpha = .5 / tf.math.pow(tf.cast(i + 1, tf.float32) * 0.007 + 1, 0.3) * numpy.ones(shape)

			b_temp = numpy.random.rand(len(gt_gradient), len(gt_gradient))

			b_v = b_temp/b_temp.sum(axis=0)[None,:]
			b = b_v[i, i]

			# b = 1
			# w = .1

			# b = 0.5
			
			# if  True:#i % 2 is not 0:
			# 	alpha = numpy.random.random_sample(shape)
			# 	# alpha = tf.linalg.tensor_diag(tf.random.uniform(shape=[1]))
			# 	b_v = 1
			# Test if leak with variable - grad
			# x.append(model.trainable_variables[i] - gt_gradient[i])

			# Test is multiplying by rand helps
			x.append(b * alpha * gt_gradient[i])

			# Full model
			# x.append((1-epsilon + epsilon * w) * model.trainable_variables[i] - b_v * alpha * gt_gradient[i])
			# x.append(w * model.trainable_variables[i] - b_v * alpha * gt_gradient[i])
			
		gt_gradient = x
		# gt_gradient = epsilon * x
	
		for i in range(len(gt_gradient)):
			gt_gradient[i] = clipGrad(gt_gradient[i], clipC)
			st = getST(gt_gradient[i])
			gt_gradient[i] = tern(gt_gradient[i], st, stMultiplier)

	@tf.function
	def train(opt, input, grad):
		with tf.GradientTape(persistent=True) as tape: # More than one function call allowed
			tape.watch(input)
			with tf.GradientTape() as tape2:
				dummy_out = model(input, training=True) # Initialize dummy variables
				
				# dummy_out = tf.reshape(dummy_out, (num_img, 1, 10)) #cifar10

				if data_type == 'mnist':
					dummy_out = tf.reshape(dummy_out, (num_img, 10)) # Initialize dummy outputs mnist
				
				elif data_type =='cifar10':
					dummy_out = tf.reshape(dummy_out, (num_img, 1, 10)) #cifar10

				loss = tf.keras.losses.categorical_crossentropy(output, dummy_out) # Determine losses based on (True, predicted labels)
				dummy_gradient = tape2.gradient(loss, model.trainable_variables) # (Target, Sources) to compute gradient

			if test_type == 'new':
				if (stMultiplier != 0):
					for i in range(len(dummy_gradient)):
						dummy_gradient[i] = clipGrad(dummy_gradient[i], clipC)
						st = getST(dummy_gradient[i])
						dummy_gradient[i] = tern(dummy_gradient[i], st, stMultiplier)


			loss = []
			rloss = 0
			for g1, g2 in zip(grad, dummy_gradient): 
				t = tf.keras.losses.MSE(g1, g2) # Computes the difference between gradients
				loss.append(t)
				rloss += tf.math.reduce_sum(t)
		gradientsI = tape.gradient(loss, input) # Input gradients
		gradientsO = tape.gradient(loss, output) # Output gradients
		opt.apply_gradients(zip([gradientsI], [input])) # (Processed gradients, var_list)
		opt.apply_gradients(zip([gradientsO], [output]))
		return rloss

	# Save gradient Data function
	def savePlot():
		if num_img == 1:
			width = 1
			height = 1
		elif num_img == 2:
			width = 1
			height = 2
		elif num_img == 4:
			width = 2
			height = 2
		elif num_img == 8:
			width = 2
			height = 4
		elif num_img == 16:
			width = 4
			height = 4
		elif num_img == 32:
			width = 4
			height = 8

		plt.figure(figsize=(width * 4 + 2, height * 2))

		for x in range(height):
			for y in range(width):
				# plt.subplot(height, width * 2 + 1, x * (width * 2 + 1) + y + 1)
				# plt.imshow(train_images[img + width * x * 2 + y], cmap=plt.cm.binary)
				# plt.subplot(height, width * 2 + 1, x * (width * 2 + 1) + y + width + 2)
				# plt.imshow(input.numpy()[width * x + y], cmap=plt.cm.binary)

				if data_type == 'mnist':
					plt.subplot(height, width * 2 + 1, x * (width * 2 + 1) + y + 1)
					plt.imshow(tf.reshape(train_images[img + width * x * 2 + y], (28,28)), cmap=plt.cm.binary)
					plt.subplot(height, width * 2 + 1, x * (width * 2 + 1) + y + width + 2)
					plt.imshow(tf.reshape(input.numpy()[width * x + y],(28,28)), cmap=plt.cm.binary)

				else:
					plt.subplot(height, width * 2 + 1, x * (width * 2 + 1) + y + 1)
					plt.imshow(train_images[img + width * x * 2 + y], cmap=plt.cm.binary)
					plt.subplot(height, width * 2 + 1, x * (width * 2 + 1) + y + width + 2)
					plt.imshow(input.numpy()[width * x + y], cmap=plt.cm.binary)

		print(plotName+str(num_img) + '_'+str(i))
		plt.savefig(plotName+str(num_img) + '_'+str(i))
		plt.close()

	#opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
	opt = tf.keras.optimizers.Adam(learning_rate=lr)

	input = tf.Variable(tf.random.uniform((num_img, width, height, channel)))
	# output = tf.Variable(tf.random.normal((num_img, 1, 10))) # for cifar10

	if data_type == 'mnist':
		output = tf.Variable(tf.random.normal((num_img, 10))) # For mnist

	elif data_type == 'cifar10':
		output = tf.Variable(tf.random.normal((num_img, 1, 10))) # for cifar10

	oloss = -1

	oil = tf.math.reduce_sum(tf.keras.losses.MSE(gt_x, input))
	ool = tf.math.reduce_sum(tf.keras.losses.MSE(gt_y, output))
	
	iData = []
	lData = []
	mData = []
	bData = []

	for i in range(iterations):
		rloss = train(opt, input, gt_gradient)
		m_l = tf.math.reduce_sum(tf.keras.losses.MSE(gt_x, input)).numpy()

		if i % 100 == 0:
			print(i, rloss.numpy(), '    \t\t', tf.math.reduce_sum(tf.keras.losses.MSE(gt_x, input)).numpy())

		if i % (iterations // 5000) == 0:
		# if i % (iterations // 5) == 0:
				
			iData.append(i)
			lData.append(rloss.numpy())
			mData.append(m_l)

		if i == 5000 and False:
			lr = 0.01

		# Uncomment for saving images
		if i % (iterations // 10) == 0:
		 	savePlot()

	print('\n')
	print(oil)
	print(ool)
	print(tf.math.reduce_sum(tf.keras.losses.MSE(gt_x, input)))
	print(tf.math.reduce_sum(tf.keras.losses.MSE(gt_y, output)))
	print(output)
	print(gt_y)

	if file != None:
		file.writerow([plotName])
		file.writerow(iData)
		file.writerow(lData)
		file.writerow(mData)
		file.writerow(bData)

	std = tf.math.reduce_std(input)
	mean = tf.reduce_mean(input)
	min = tf.math.reduce_min(input)
	max = tf.reduce_max(input)

	new_input = tf.math.abs((input - mean) / std)

	savePlot()

if args.seed:
	Seed = numpy.random.randint(0, 40000)
else:
	Seed = 32946

with open(file_n, mode='a') as csv_file:
	
	file = csv.writer(csv_file, lineterminator = '\n')

	experiment(img = Seed,
			num_img = num_img,
			iterations = itn,
			olr = 0.001, # olr is learning rate
			plotName=args.data+'/_'+test_type,
			clipC=2.5,
			stMultiplier=1,
			file=file,
			test_type=test_type)
