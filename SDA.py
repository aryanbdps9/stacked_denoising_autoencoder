from PIL import Image
import tensorflow as tf
import numpy as np
import random
import os
import platform
import glob
import matplotlib
import matplotlib.pyplot as plt
lambda_g = float(raw_input('lambda:'))
beta_g = float(raw_input('beta:'))
lambda1 = lambda_g
beta1 = beta_g
lambda2 = lambda_g
beta2 = beta_g
lambda0 = lambda_g
beta0 = beta_g
learning_rate_g = float(raw_input('alpha:'))
learning_rate = learning_rate_g / 10
max_learning_rate_g = 5E3
mini_batch_size = 4096
n_epochs = int(raw_input('n_epochs:'))
n_epochs0 = int(raw_input('n_epochs0:'))
blur_strength = 0.2
salt_prob = 0.02
patchify_stride = 4
patchify_test_stride=1
patchify_mode_train = 'salt'
patchify_mode_test = 'salt'
patch_nr_g = int(raw_input('patch_size:'))
patch_nc_g = patch_nr_g
inp_addr_g = 'bwimgs/*'
inp_addr_ng= 'cwimgs/*'
inp_addr_test_g = 'bwimgst/*'
inp_addr_test_ng = 'cwimgst/*'
momentum_g = 0.5
p_g = float(raw_input('rho:'))
K = 2

print("""[lambda_g, beta_g, lambda1, beta1, lambda2, beta2, lambda0, beta0, learning_rate_g, mini_batch_size, n_epochs, n_epochs0, blur_strength, patchify_stride,
       patch_nr_g, patch_nc_g, inp_addr_g, momentum_g, p_g, K]\n""", [lambda_g, beta_g, lambda1, beta1, lambda2, beta2, lambda0, beta0, learning_rate_g, mini_batch_size, n_epochs, n_epochs0, blur_strength, patchify_stride,
                                                                 patch_nr_g, patch_nc_g, inp_addr_g, momentum_g, p_g, K])
# this partitions dataset. Each row of dataset_x is falttened patch.
# dataset_x is noisy and y is clean
def partition_dataset(dataset_x, dataset_y, size=mini_batch_size):
	dataset_x = dataset_x
	dataset_y = dataset_y
	# dataset_x = dataset_x
	# dataset_y = dataset_y
	print("dataset_x.shape: ", np.shape(dataset_x))
	r, c = np.shape(dataset_x)
	permm = np.random.permutation(r)
	# permm = np.arange(0,r)
	res_x = [dataset_x[permm[i:i+size]] for i in range(0, r, size)]
	res_y = [dataset_y[permm[i:i+size]] for i in range(0, r, size)]
	# return [res_x, res_y]
	return zip(res_x, res_y)

def plot_npa(mat, outfilename='patches/tepat.png', flatt=False):
	if (flatt):
		
		os.system('mkdir -p patches/')
	imgg = Image.fromarray(mat)
	imgg.save(outfilename)
	if (mat.size() <= 4096):
		print(mat)
	else:
		print("mat size:", mat.size())




def patchify(inp_img, patch_nr, patch_nc, stride=1, get_noisy=True, noisetype='gauss'): #inp_img is a np array
	r,c = np.shape(inp_img)
	res_clean = []
	res_noisy = []
	for i in range(0, r - patch_nr + stride + 1, stride):
		# if ()
		if (i+patch_nr-1 < r):
			ii = i
		else:
			ii = r - patch_nr
		for j in range(0, c - patch_nc + stride + 1, stride):
			if (j+patch_nc-1 < c):
				jj = j
			else:
				jj = c - patch_nc
			# print("ii, jj = ", ii, jj,)
			temp_patch = inp_img[ii:ii+patch_nr, jj:jj+patch_nc].reshape((1,patch_nc*patch_nr))
			# print("temp_patch.shape: ", np.shape(temp_patch))
			if (get_noisy):
				if (noisetype =='gauss'):
					bad_patch = noise_maker(temp_patch)
				else:
					bad_patch = salty(temp_patch)
				res_noisy.append(bad_patch)
				
			res_clean.append(temp_patch)
	if (get_noisy):
		return [np.concatenate(res_noisy, axis=0), np.concatenate(res_clean, axis=0)]
	else:
		return np.concatenate(res_clean, axis=0)
	# return [np.array(res_noisy), np.array(res_clean)]


def depatchify(image_patches, patch_nr, patch_nc, r, c, stride=1):
	res = np.zeros((r, c))
	# resd = np.zeros((r,c))
	countt = np.zeros((r,c))

	cntr = -1
	for i in range(0, r - patch_nr + stride + 1, stride):
		if (i+patch_nr-1 < r):
			ii = i
		else:
			ii = r - patch_nr
		for j in range(0, c - patch_nc + stride + 1, stride):
			cntr += 1
			if (j+patch_nc-1 < c):
				jj = j
			else:
				jj = c - patch_nc
			# print("ii, jj = ", ii, jj,)
			temp_patch = image_patches[cntr, :].reshape(patch_nr, patch_nc)
			# td_patch = dirty_patches[i, :].reshape(patch_nr, patch_nc)
			# temp_patch = inp_img[ii:ii+patch_nr, jj:jj+patch_nc].flatten().reshape((1,patch_nc*patch_nr))
			# print("temp_patch.shape: ", np.shape(temp_patch))
			# bad_patch = noise_maker(temp_patch)
			# print("ii, jj = ", ii, jj, "\ntemp_patch = ", temp_patch)
			res[ii:ii+patch_nr, jj:jj+patch_nc] = temp_patch + res[ii:ii+patch_nr, jj:jj+patch_nc]
			# resd[ii:ii+patch_nr, jj:jj+patch_nc] = td_patch + resd[ii:ii+patch_nr, jj:jj+patch_nc]
			countt[ii:ii + patch_nr, jj:jj + patch_nc] += 1
	
	resb = np.divide(res, countt)
	# resd = np.divide(resd, countt)
	return resb

# def depatchify(image_patches, patch_nr, patch_nc, r, c, stride=1):
# 	res = np.zeros((r, c))
# 	# resd = np.zeros((r,c))
# 	countt = np.zeros((r,c))
# 	res = [[[] for j in range(c)] for i in range(r)]
	

# 	cntr = -1
# 	for i in range(0, r - patch_nr + stride + 1, stride):
# 		if (i+patch_nr-1 < r):
# 			ii = i
# 		else:
# 			ii = r - patch_nr
# 		for j in range(0, c - patch_nc + stride + 1, stride):
# 			cntr += 1
# 			if (j+patch_nc-1 < c):
# 				jj = j
# 			else:
# 				jj = c - patch_nc
# 			# print("ii, jj = ", ii, jj,)
# 			temp_patch = image_patches[cntr, :].reshape(patch_nr, patch_nc)
# 			# td_patch = dirty_patches[i, :].reshape(patch_nr, patch_nc)
# 			# temp_patch = inp_img[ii:ii+patch_nr, jj:jj+patch_nc].flatten().reshape((1,patch_nc*patch_nr))
# 			# print("temp_patch.shape: ", np.shape(temp_patch))
# 			# bad_patch = noise_maker(temp_patch)
# 			# print("ii, jj = ", ii, jj, "\ntemp_patch = ", temp_patch)
# 			res[ii:ii+patch_nr, jj:jj+patch_nc] = temp_patch + res[ii:ii+patch_nr, jj:jj+patch_nc]
# 			# resd[ii:ii+patch_nr, jj:jj+patch_nc] = td_patch + resd[ii:ii+patch_nr, jj:jj+patch_nc]
# 			countt[ii:ii + patch_nr, jj:jj + patch_nc] += 1
	
# 	resb = np.divide(res, countt)
# 	# resd = np.divide(resd, countt)
# 	return resb

# this returns test_data. Each row of rv is a patch of noisy test data;
# images is a list of 2d matrices(normalised); each elem of indices denotes the starting index
# of patches in 1st rv. rv -> return value


def test_patchify(inp_addr=inp_addr_test_g, inp_addrn=inp_addr_test_ng, lim=3):
	filelist = sorted(glob.glob(inp_addr), key=str.lower) # clean
	filelist2 = sorted(glob.glob(inp_addrn), key=str.lower)# dirty
	indicess = []
	lengths = []
	x = []
	y = []
	stack_size = 0
	images = []
	cntr = 0
	for filn, film_dirty in zip(filelist, filelist2):
		cntr += 1
		img = Image.open(filn)
		imgd= Image.open(film_dirty)
		img.load()
		imgd.load()
		data = np.asarray(img, dtype="int32") / 256.0
		dirty_data = np.asarray(img, dtype='int32') / 256.0
		images.append(data)
		_, batch_x = patchify(dirty_data, patch_nr_g, patch_nc_g, patchify_test_stride, get_noisy=True, noisetype=patchify_mode_test)
		indicess.append(stack_size)
		lengths.append(np.shape(batch_x)[0])
		stack_size += np.shape(batch_x)[0]
		# print("batch.shape", np.shape(batch_x))
		x.append(batch_x)
		if (cntr >= lim):
			break
		# y.append(batch_y)
	# if max_num is None:
	return [np.concatenate(x, axis=0), images, indicess, lengths]

def salty(imgl1):
	r, c = np.shape(imgl1)
	imgl = np.zeros((r,c))
	for i in range(r):
		unif = np.random.uniform(0.0,1.0, (1,c))
		zeee = -1*((unif < (salt_prob / 2.0)) - 1)
		onnn = (unif >= (salt_prob / 2.0)) * (unif < salt_prob)
		imgl[i,:] = np.minimum(np.maximum(imgl[i,:], onnn), zeee)
	return imgl

def noise_maker(imgl1):
	r, c = np.shape(imgl1)
	imgl = np.zeros((r,c))
	for i in range(r):
		imgl[i, :] = imgl1[i,:] + blur_strength*np.random.randn(1, c)
	
	return imgl

def get_all_patches_flat(inp_addr=inp_addr_g, max_num=None):
	filelist = glob.glob(inp_addr)
	x = []
	y = []
	for filn in filelist:
		img = Image.open(filn)
		img.load()
		data = np.asarray(img, dtype="float32") / 256.0
		r,c = np.shape(data)
		batch_x, batch_y = patchify(data, patch_nr_g, patch_nc_g, patchify_stride, noisetype=patchify_mode_train)
		data_img = Image.fromarray(((depatchify(batch_x, patch_nr_g, patch_nc_g, r, c, patchify_stride) /1.0)*256.0).astype('int32'))
		# data_img.show()
		# input()

		# print("batch_x.shape", np.shape(batch_x))
		x.append(batch_x)
		y.append(batch_y)
		# print("dim of batch_y, concat:", np.shape(batch_y), np.shape(np.concatenate(y, axis=0)))
		# print("rrmse: ", rrmse_calc(np.concatenate(y, axis=0), batch_y))
	resx = np.concatenate(x, axis=0)
	resy = np.concatenate(y, axis=0)
	r, _ = np.shape(resx)
	permm = np.random.permutation(r)
	# permm = np.arange(0,r)
	resx = resx[permm,:]
	resy = resy[permm,:]
	# res_x = [dataset_x[permm[i:i+size]] for i in range(0, r, size)]
	# res_y = [dataset_y[permm[i:i+size]] for i in range(0, r, size)]
	r,_ = np.shape(resx)
	return [resx, resy]

def rrmse_calc(img1, img2):
	diff = img1 - img2
	diff = diff ** 2
	num = np.sum(diff)
	den = np.sum(img2**2)
	return np.sqrt(num/den)

def test_patchify_nd_depatchify():
	filelist = glob.glob(inp_addr_g)
	x = []
	y = []
	for filn in filelist:
		img = Image.open(filn)
		img.load()
		data = np.asarray(img, dtype="int32") / 256.0
		r,c = np.shape(data)
		batch_x, batch_y = patchify(data, patch_nr_g, patch_nc_g, patchify_stride)
		recons_img = depatchify(batch_y, patch_nr_g, patch_nc_g, r, c, patchify_stride)
		# print("sum of diffs's square = ", np.sum((recons_img - data) ** 2))
		# print("batch_x.shape", np.shape(batch_x))
		x.append(batch_x)
		y.append(batch_y)
	# if max_num is None:
	# 	return [np.concatenate(x, axis=0), np.concatenate(y, axis=0)]
	# else:
	# 	return [np.concatenate(x, axis=0), np.concatenate(y, axis=0)]  # :P

def get_all_patches_flat2(inp_addr=inp_addr_g, inp_addrn=inp_addr_ng, max_num=None):
	filelist = sorted(glob.glob(inp_addr), key=str.lower)
	x = []
	y = []
	for filn in filelist:
		img = Image.open(filn)
		img.load()
		data = np.asarray(img, dtype="float32") / 256.0
		r,c = np.shape(data)
		_, batch_y = patchify(data, patch_nr_g, patch_nc_g, patchify_stride, noisetype=patchify_mode_train)
		y.append(batch_y)
		# print("dim of batch_y, concat:", np.shape(batch_y), np.shape(np.concatenate(y, axis=0)))
		# print("rrmse: ", rrmse_calc(np.concatenate(y, axis=0), batch_y))
	filelist = sorted(glob.glob(inp_addrn), key=str.lower)
	for filn in filelist:
		img = Image.open(filn)
		img.load()
		data = np.asarray(img, dtype="float32") / 256.0
		r,c = np.shape(data)
		_, batch_y = patchify(data, patch_nr_g, patch_nc_g, patchify_stride)
		x.append(batch_y)

	resx = np.concatenate(x, axis=0)
	resy = np.concatenate(y, axis=0)
	r, _ = np.shape(resx)
	permm = np.random.permutation(r)
	# permm = np.arange(0,r)
	resx = resx[permm,:]
	resy = resy[permm,:]
	r,_ = np.shape(resx)
	return [resx, resy]


# training_data_noisy, training_data_clean = pair_maker('bwimgs/*.bmp')
training_data_noisy, training_data_clean = get_all_patches_flat2()
# noisy_img = depatchify(training_data_noisy, patch_nr_g, patch_nc_g, 512, 512, patchify_stride)
# clean_img = depatchify(training_data_clean, patch_nr_g, patch_nc_g, 512, 512, patchify_stride)
# img1212 = Image.open('bwimgs/1.bmp')
# img1212.load()
# datajhgj = np.asarray(img1212, dtype="float32") / 256.0
# print("rrmse: ", rrmse_calc(noisy_img, datajhgj))

# i_img = Image.fromarray((noisy_img*256.0).astype('int32'))
# i_img.show()
# diff1212 = np.sum((clean_img - datajhgj) ** 2)
# print("diff1212:", diff1212)
# input()
# print("rrmse_init = ", rrmse_calc(noisy_img, clean_img))
# print('%'*70)
print("training_data_noisy's shape, training_data_clean's shape: ", np.shape(training_data_noisy), np.shape(training_data_clean))

# def save_image( npdata, outfilename ) :
# 	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
# 	img.save( outfilename )


# dummy 1 layer net to test data loading and if it works
learning_rate_plac = learning_rate_plac = tf.placeholder(tf.float32, shape=[])
########################DEFINING FIRST DA##########################################################################
input_size1 = patch_nr_g * patch_nc_g
layer_size1 = K*input_size1
xinitializer = tf.contrib.layers.xavier_initializer()
b1 = tf.Variable(xinitializer([1, layer_size1]), name='b1')

w1 = tf.Variable(xinitializer([input_size1, layer_size1]), name='w1')
b_1 = tf.Variable(xinitializer([1, input_size1]), name='b_1')
# w_1 = tf.Variable(xinitializer([layer_size1, input_size1]), name='w_1')

inp1 = tf.placeholder(tf.float32, [None, input_size1], name='inp1')
correct_output1 = tf.placeholder(tf.float32, [None, input_size1], name='correct_output1')

hidden_1 = tf.sigmoid(tf.add(tf.matmul(inp1, w1), b1), name='hidden_1')
result1 = tf.sigmoid(tf.add(tf.matmul(hidden_1, tf.transpose(w1)), b_1), name='result1')
p = tf.placeholder(tf.float32, name='p')
recon1 = tf.reduce_mean(tf.square(correct_output1 - result1), name='recon1')
# _cost = tf.reduce_mean(tf.square(correct_output - result))# + lambdaa * (tf.square(tf.norm(w)) + tf.square(tf.norm(w1)))
p1_hat = tf.reduce_mean(hidden_1, axis=0)
klD1 = tf.reduce_sum(tf.add(p*tf.log(p/p1_hat), (1-p)*tf.log((1-p)/(1-p1_hat))), name='klD1')
# klD1 = tf.reduce_sum(tf.add(p*tf.log(p/tf.reduce_mean(hidden_1)),(1-p)*tf.log((1-p)/(1-tf.reduce_mean(hidden_1)))), name='klD1')
# regularizer1 = tf.add(tf.reduce_sum(tf.square(w1)),tf.reduce_sum(tf.square(w_1)))
regularizer1 = tf.add(tf.square(tf.norm(w1, ord='fro', axis=(0,1))), tf.square(tf.norm(w1, ord='fro',  axis=(0,1))), name='regularizer1')
# regularizer1 = tf.add(tf.square(tf.norm(w1, ord='fro', axis=(0,1))), tf.square(tf.norm(w_1, ord='fro',  axis=(0,1))), name='regularizer1')
loss1 = tf.add(recon1,(beta1*klD1+lambda1*regularizer1), name='loss1')
# train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_plac).minimize(_cost)
train_step1 = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_plac).minimize(loss1, name='train_step1')
# train_step1 = tf.train.MomentumOptimizer(learning_rate = learning_rate_plac, momentum=momentum_g).minimize(loss1, name='train_step1')
#####################################DEFINED FIRST DA(hidden_1)###################################################
######################################DEFINING SECOND DA #########################################################
input_size2 = layer_size1
layer_size2 = K*input_size2

b2 = tf.Variable(xinitializer([1, layer_size2]), name='b2')
w2 = tf.Variable(xinitializer([input_size2, layer_size2]), name='w2')
b_2 = tf.Variable(xinitializer([1, input_size2]), name='b_2')
# w_2 = tf.Variable(xinitializer([layer_size2, input_size2]), name='w_2')

inp2 = tf.placeholder(tf.float32, [None, input_size2], name='inp2')
correct_output2 = tf.placeholder(tf.float32, [None, input_size2], name='correct_output2')
hidden_2 = tf.sigmoid(tf.add(tf.matmul(inp2, w2), b2), name='hidden_2')
result2 = tf.sigmoid(tf.add(tf.matmul(hidden_2, tf.transpose(w2)), b_2), name='result2')
p2 = tf.placeholder(tf.float32, name='p2')
recon2 = tf.reduce_mean(tf.square(correct_output2 - result2), name='recon2')
p2_hat = tf.reduce_mean(hidden_2, axis=0)
klD2 = tf.reduce_sum(tf.add(p2*tf.log(p2/p2_hat), (1-p2)*tf.log((1-p2)/(1-p2_hat))), name='klD2')

# klD2 = tf.reduce_sum(tf.add(p2*tf.log(p2/tf.reduce_mean(hidden_2)),(1-p2)*tf.log((1-p2)/(1-tf.reduce_mean(hidden_2)))), name='klD2')
# regularizer2 = tf.add(tf.reduce_sum(tf.square(w2)),tf.reduce_sum(tf.square(w_2)))
regularizer2 = tf.add(tf.square(tf.norm(w2, ord='fro', axis=(0,1))), tf.square(tf.norm(w2, ord='fro', axis=(0,1))), name='regularizer2')
loss2 = tf.add(recon2, (beta2*klD2+lambda2*regularizer2), name='loss2')
# GradientDescentOptimizer
train_step2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_plac).minimize(loss2, name='train_step2')
# train_step2 = tf.train.MomentumOptimizer(learning_rate=learning_rate_plac, momentum=momentum_g).minimize(loss2, name='train_step2')
#######################################DEFINED SECOND DA(hidden_2) ####################################################
######################################DEFINING THE BIG TED MOSBY#######################################################
input_size0 = patch_nr_g * patch_nc_g


b0_1 = tf.Variable(xinitializer([1, input_size1]), name='b0_1')
# w0_1 = tf.Variable(xinitializer([layer_size1, input_size1]), name='w0_1')
b0_2 = tf.Variable(xinitializer([1, input_size2]), name='b0_2')
# w0_2 = tf.Variable(xinitializer([layer_size2, input_size2]), name='w0_2')

inp0 = tf.placeholder(tf.float32, [None, input_size0], name='inp0')
correct_output0 = tf.placeholder(tf.float32, [None, input_size0], name='correct_output0')
hidden0_1	= tf.sigmoid(tf.add(tf.matmul(inp0,      w1),  b1),  name='hidden0_1')
hidden0_2	= tf.sigmoid(tf.add(tf.matmul(hidden0_1, w2),  b2),  name='hidden0_2')
hidden0_3	= tf.sigmoid(tf.add(tf.matmul(hidden0_2, tf.transpose(w2)), b0_2), name='hidden0_3')
result0		= tf.sigmoid(tf.add(tf.matmul(hidden0_3, tf.transpose(w1)), b0_1), name='result0')
# hidden0_3	= tf.sigmoid(tf.add(tf.matmul(hidden0_2, w0_2), b0_2), name='hidden0_3')
# result0		= tf.sigmoid(tf.add(tf.matmul(hidden0_3, w0_1), b0_1), name='result0')
# hidden0_3	= tf.sigmoid(tf.add(tf.matmul(hidden0_2, w_2), b_2), name='hidden0_3')
# result0		= tf.sigmoid(tf.add(tf.matmul(hidden0_3, w_1), b_1), name='result0')

recon0		= tf.reduce_mean(tf.square(correct_output0 - result0), name='recon0')
regularizer0 = tf.add(tf.square(tf.norm(w1, ord='fro', axis=(0, 1))), (tf.square(tf.norm(w2, ord='fro', axis=(0, 1)))
	 + tf.square(tf.norm(w2, ord='fro', axis=(0, 1))) + tf.square(tf.norm(w1, ord='fro', axis=(0, 1)))), name='regularizer0')
loss0 = tf.add(recon0, lambda0*regularizer0, name='loss0')

train_step0 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_plac).minimize(loss0, name='train_step0')
# train_step0 = tf.train.MomentumOptimizer(learning_rate=learning_rate_plac, momentum=momentum_g).minimize(loss0, name='train_step0')
#######################################DEFINED THE BIG TED MOSBY#######################################################



###################################3#
_init = tf.global_variables_initializer()

# train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate_plac, momentum = momentum_g).minimize(_cost)
# train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_plac).minimize(_cost)
# train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(_cost)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(result,axis=1),tf.argmax(correct_output,axis=1)),tf.float32))
accuracy1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(correct_output1 - result1), axis=1) / (tf.reduce_sum(tf.square(correct_output1), axis=1)+1E-8), name='accuracy1'))
accuracy2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(
	correct_output2 - result2), axis=1) / (tf.reduce_sum(tf.square(correct_output2), axis=1) + 1E-8)), name='accuracy2')
accuracy0 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(correct_output0 - result0), axis=1) / (tf.reduce_sum(tf.square(correct_output0), axis=1)+1E-8), name='accuracy0'))

# Get training, validation and test data matrices
# 		i_tr,o_tr,i_va,o_va,i_te,o_te=ml.get_matrices()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver()

# sess = tf.Session()
print("initialize_all_variables.............")
# sess.run(tf.initialize_all_variables())
sess.run(_init)
print("initinit#"*12)

# initialize writer for using TensorBoard
# tf.summary.scalar("Training Accuracy", accuracy)
tf.summary.scalar("Cost", loss1)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs", graph=sess.graph)
# define batch
cntr = 0
prev_cost = 10**10
# saver.restore(sess, "ck/model.ckpt")
for epoch_no in range(n_epochs):
	print("epoch1#" + str(epoch_no) + '*' * 20)
	av_cost = 0
	av_accu = 0
	cntr = 0
	for batch, batch_output in partition_dataset(training_data_noisy, training_data_clean, mini_batch_size):
		# print("batch.shape: ", np.shape(batch))
		_, summary = sess.run([train_step1, summary_op], {
								inp1: batch, correct_output1: batch_output, learning_rate_plac: learning_rate, p:p_g})
		writer.add_summary(summary, epoch_no)
		# print("\n2K Mini-batch = {0}. Accuracy = {1}\r".format(cntr,sess.run(accuracy,{inp:batch, correct_output:batch_output})),end='',flush=True)
		accu_, klD1_, regularizer1_, costt = sess.run([accuracy1, klD1, regularizer1, loss1], {
								inp1: batch, correct_output1: batch_output, learning_rate_plac: learning_rate, p:p_g})
		av_cost += costt
		av_accu += accu_
		cntr += 1
	av_accu /= cntr
	av_cost /= cntr
	if (av_cost > prev_cost):
		learning_rate = min(learning_rate / 1.8, max_learning_rate_g)
	else:
		learning_rate = min(learning_rate * 1.1, max_learning_rate_g)
	prev_cost = av_cost
		# print("Accuracy = ", sess.run([accuracy, _cost],{inp:batch, correct_output:batch_output}))
	print("accuracy, klD1, regularizer1, costt, learning_rate:",
		av_accu, klD1_*beta1, regularizer1_*lambda1, av_cost, learning_rate)

# b1_npa, w1_npa, b_1_npa, w_1_npa = sess.run([b1, w1, b_1, w_1])
print("jon don")

# input_size2 = layer_size1
# layer_size2 = K*input_size2

# b2 = tf.Variable(xinitializer([1, layer_size2]), name='b2')
# # b2 = tf.get_variable(initializer=xinitializer([1, layer_size2]), name='b2', shape=(1, layer_size2))

# w2 = tf.Variable(xinitializer([input_size2, layer_size2]), name='w2')
# b_2 = tf.Variable(xinitializer([1, input_size2]), name='b_2')
# w_2 = tf.Variable(xinitializer([layer_size2, input_size2]), name='w_2')

# inp2 = tf.placeholder(tf.float32, [None, input_size2], name='inp2')
# correct_output2 = tf.placeholder(tf.float32, [None, input_size2], name='correct_output2')
# lambda2 = .5
# beta2 = .5
# hidden_2 = tf.sigmoid(tf.add(tf.matmul(inp2, w2), b2), name='hidden_2')
# result2 = tf.sigmoid(tf.add(tf.matmul(hidden_2, w_2), b_2), name='result2')
# p2 = tf.placeholder(tf.float32, name='p2')
# recon2 = tf.reduce_mean(tf.square(correct_output2 - result2), name='recon2')
# klD2 = tf.reduce_sum(tf.add(p2*tf.log(p2/tf.reduce_mean(hidden_2)),(1-p2)*tf.log((1-p2)/(1-tf.reduce_mean(hidden_2)))), name='klD2')
# # regularizer2 = tf.add(tf.reduce_sum(tf.square(w2)),tf.reduce_sum(tf.square(w_2)))
# regularizer2 = tf.square(tf.norm(w2, ord='fro', axis=(0,1))) + tf.square(tf.norm(w_2, ord='fro', axis=(0,1)), name='regularizer2')
# loss2 = tf.add(recon2, (beta2*klD2+lambda2*regularizer2), name='loss2')
# train_step2 = tf.train.MomentumOptimizer(learning_rate=learning_rate_plac, momentum=momentum_g).minimize(loss2, name='train_step2')

# _init = tf.global_variables_initializer()
if (True):
	print(sess.run(tf.report_uninitialized_variables()))
	print('$'* 50)

	# sess.run(tf.initialize_variables(list(tf.get_variable(name)
	#                                       for name in sess.run(tf.report_uninitialized_variables()))))
	# _initt = tf.global_variables_initializer()
	# print(sess.run(tf.report_uninitialized_variables()))
	print('$' * 50)

	h_x_list = sess.run(hidden_1, {inp1:training_data_noisy})
	h_y_list = sess.run(hidden_1, {inp1:training_data_clean})

	# accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(
	# 	correct_output2 - result2), axis=1) / tf.reduce_sum(tf.square(correct_output2), axis=1)))

	tf.summary.scalar("Cost", loss2)
	summary_op = tf.summary.merge_all()
	writer = tf.summary.FileWriter("./logs", graph=sess.graph)
	# define batch
	cntr = 0
	prev_cost = 10**10
	# saver.restore(sess, "ck/model.ckpt")
	learning_rate = learning_rate_g
	for epoch_no in range(n_epochs):
		print("epoch2#" + str(epoch_no) + '*' * 20)
		av_cost=0
		av_accu = 0
		cntr = 0
		for batch, batch_output in partition_dataset(h_x_list, h_y_list, mini_batch_size):
			# print("batch.shape: ", np.shape(batch))
			sess.run([train_step2], {
									inp2: batch, correct_output2: batch_output, learning_rate_plac: learning_rate, p2:p_g})
			# writer.add_summary(summary, epoch_no)
			# print("\n2K Mini-batch = {0}. Accuracy = {1}\r".format(cntr,sess.run(accuracy,{inp:batch, correct_output:batch_output})),end='',flush=True)
			accu_, klD2_, regularizer2_, costt = sess.run([accuracy2, klD2, regularizer2, loss2], {
									inp2: batch, correct_output2: batch_output, learning_rate_plac: learning_rate, p2:p_g})
			cntr += 1
			av_accu += accu_
			av_cost += costt
		av_accu /= cntr
		av_cost /= cntr
		if (av_cost > prev_cost):
			learning_rate = min(learning_rate / 1.8, max_learning_rate_g)
		else:
			learning_rate = min(learning_rate * 1.1, max_learning_rate_g)
		prev_cost = av_cost
			# print("Accuracy = ", sess.run([accuracy, _cost],{inp:batch, correct_output:batch_output}))
		print("accuracy, klD2, regularizer1, costt, learning_rate:",
			av_accu, klD2_*beta2, regularizer2_*lambda2, av_cost, learning_rate)

	h_x_list = None
	h_y_list = None

	cntr = 0
	prev_cost = 10 *10
	# saver.restore(sess, "ck/model.ckpt")
	learning_rate = learning_rate_g
	for epoch_no in range(n_epochs0):
		print("epoch0#" + str(epoch_no) + '*' * 20)
		av_cost = 0
		cntr = 0
		av_accu = 0
		for batch, batch_output in partition_dataset(training_data_noisy, training_data_clean, mini_batch_size):
			# print("batch.shape: ", np.shape(batch))
			sess.run([train_step0], {
						inp0: batch, correct_output0: batch_output, learning_rate_plac: learning_rate})
			# writer.add_summary(summary, epoch_no)
			# print("\n2K Mini-batch = {0}. Accuracy = {1}\r".format(cntr,sess.run(accuracy,{inp:batch, correct_output:batch_output})),end='',flush=True)
			accu_, regularizer0_, costt = sess.run([accuracy0, regularizer0, loss0], {
						inp0: batch, correct_output0: batch_output, learning_rate_plac: learning_rate})
			cntr += 1
			av_cost += costt
			av_accu += accu_
		av_cost /= cntr
		av_accu /= cntr
		if (av_cost > prev_cost):
			learning_rate = min(learning_rate / 1.8, max_learning_rate_g)
		else:
			learning_rate = min(learning_rate * 1.1, max_learning_rate_g)
		prev_cost = av_cost
			# print("Accuracy = ", sess.run([accuracy, _cost],{inp:batch, correct_output:batch_output}))
		print("accuracy, regularizer0, costt, learning_rate:",
					av_accu, regularizer0_*lambda0, av_cost, learning_rate)





print("laugherere"+'#'*20)
save_path= saver.save(sess, "ck/model.ckpt")
# patchify(inp_img, patch_nr, patch_nc, stride=1, get_noisy=True)

# test_data_xmat, test_clean_imgs, indices, lengths = test_patchify()
# net_out = sess.run(result0, {inp0: test_data_xmat})
# net_out_int= (net_out * 255.0).astype('int32')
# depat_net_float = depatchify(net_out, patch_nr_g, patch_nc_g, 512, 512, stride=patchify_stride)
# depat_net_int = (depat_net_float * 255.0).astype('int32')
# # net_out = sess.run(result0, {inp0:test_data_xmat})
# resc_test_data_xmat = (test_data_xmat * 255.0).astype('int32')
# resc_test_clean_imgs = (patchify(test_clean_imgs[0], patch_nr_g, patch_nc_g, patchify_stride, False) * 255.0).astype('int32')
rrmses = []

filelist1 = sorted(glob.glob(inp_addr_test_g), key=str.lower)
filelist2 = sorted(glob.glob(inp_addr_test_ng), key=str.lower)
ind = -1
for filn, film_dirty in zip(filelist1, filelist2):
	ind += 1
	print(film_dirty)
	img = Image.open(filn)
	imgd= Image.open(film_dirty)
	img.load()
	imgd.load()
	print("imgs loaded")
	data = np.asarray(img,dtype='int32') / 256.0
	dirty_data = np.asarray(imgd,dtype='int32')/256.0
	r, c = np.shape(dirty_data)
	batch_x = patchify(dirty_data,  patch_nr_g, patch_nc_g,
	                   patchify_test_stride, get_noisy=False, noisetype=patchify_mode_test)
	img_mat = sess.run(result0, {inp0:batch_x})
	# net_out_int = (img_mat * 255.0).astype('int32')
	temp_img = depatchify(img_mat, patch_nr_g, patch_nc_g,
	                      r, c, stride=patchify_test_stride)
	scales_temp_img = (temp_img * 255.0).astype('int32')
	scales_temp_img_n = (dirty_data * 255.0).astype('int32')
	print("showing.....................")
	plt.imsave('outs/'+str(ind)+'.png', scales_temp_img, cmap=matplotlib.cm.gray)
	plt.imsave('dirts/'+str(ind)+'.png', scales_temp_img_n, cmap=matplotlib.cm.gray)
	
	print(scales_temp_img)
	# input()
	rrmse = rrmse_calc(temp_img, data)
	rrmses.append(rrmse)


# accumulator = 0
# for ind in range(len(test_clean_imgs)):
# 	img_start_ind = indices[ind]
# 	img_end_ind = img_start_ind + lengths[ind]
# 	img_mat = net_out[img_start_ind:img_end_ind, :]
# 	noisy_img_mat = test_data_xmat[img_start_ind:img_end_ind,:]
# 	r,c = np.shape(test_clean_imgs[ind])
# 	temp_img = depatchify(img_mat, patch_nr_g, patch_nc_g, r, c, stride=patchify_test_stride)
# 	noisy_temp_img = depatchify(noisy_img_mat, patch_nr_g, patch_nc_g, r, c, stride=patchify_test_stride)
# 	# immin = temp_img.min()
# 	# immax = temp_img.max()
# 	scales_temp_img = (temp_img  * 255.0).astype('int32')
# 	scales_temp_img_n=(noisy_temp_img * 255.0).astype('int32')
# 	# (temp_img * 255.0).astype('int32')

# 	print("showing.....................")
# 	plt.imsave('outs/'+str(ind)+'.png', scales_temp_img, cmap=matplotlib.cm.gray)
# 	plt.imsave('dirts/'+str(ind)+'.png', scales_temp_img_n, cmap=matplotlib.cm.gray)
	
# 	print(scales_temp_img)
# 	# input()
# 	rrmse = rrmse_calc(temp_img, test_clean_imgs[ind])
# 	rrmses.append(rrmse)
aver_rrmse = np.mean(rrmses)
print("\nTraining data accuracy = ", aver_rrmse)
# print("\nTraining data accuracy = {0}".format(sess.run(accuracy,{inp1:training_data_noisy, correct_output1:training_data_clean})))
# sess.close()
# def show_patch(r):
# 	print(test_data_xmat[r,:])
# 	print(net_out[r,:])
# 	img1 = Image.fromarray(resc_test_data_xmat[r,:].reshape(patch_nr_g, patch_nc_g))
# 	img2 = Image.fromarray(resc_test_clean_imgs[r, :].reshape(patch_nr_g, patch_nc_g))
# 	img3 = Image.fromarray(net_out_int[r, :].reshape(patch_nr_g, patch_nc_g))
# 	img1.show('dirty')
# 	img2.show('clean')
# 	img3.show('output')

# def plot_img():
# 	immy = Image.fromarray(depat_net_int)
# 	immy.show()
