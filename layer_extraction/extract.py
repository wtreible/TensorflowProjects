
# boilerplate code
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf

model_fn = 'output_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
#tf.import_graph_def(graph_def)
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def)

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

def showarray(a, fmt='png'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    im = PIL.Image.fromarray(a)
    im.save(f, fmt)
    Image(data=f.getvalue())
    im.show(command='xdg-open')

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    print(type(t_score), type(t_grad), type(t_input))
    img = img0.copy()
    for i in range(iter_n):
        # '_' was 'score'
	g, _ = sess.run([t_grad, t_score], {t_input:img})
        
	# normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
        #print(score, end = ' ')
    clear_output()
    showarray(visstd(img))

layer = 'mixed_10/conv/Conv2D'
def go(layer):
	render_naive(T(layer)[:,:,:,0])

