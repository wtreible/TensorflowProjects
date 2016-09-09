import sys, os
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

modelFullPath = 'output_graph.pb'
labelsFullPath = 'output_labels.txt'


def create_graph():
	"""Creates a graph from saved GraphDef file and returns a saver."""
	# Creates graph from saved graph_def.pb.
	with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_dir):
	answer = None

	image_data = []
	for im_file in os.listdir(image_dir):
		im = os.path.join(image_dir, im_file)
		if os.path.isfile(im):
			if not tf.gfile.Exists(im):
				tf.logging.fatal('File does not exist %s', im)
				return answer
			image_data.append([im, tf.gfile.FastGFile(im, 'rb').read()])
		
	# Creates graph from saved GraphDef.
	create_graph()

	spores_list = []
	with tf.Session() as sess:

		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		with open(labelsFullPath, 'rb') as f:
			lines = f.readlines()
                        labels = [str(w).replace("\n", "") for w in lines]
			print "\n===========================\nResults:\n"
			for idata in image_data:
				predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': idata[1]})
				predictions = np.squeeze(predictions)

				top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

				#lines = f.readlines()
				#labels = [str(w).replace("\n", "") for w in lines]
				print "Image:", idata[0]
				for node_id in top_k:
					human_string = labels[node_id]
					score = predictions[node_id]
					print('%s (score = %.5f)' % (human_string, score))				

				answer = labels[top_k[0]]
				if answer == 'spore':
					spores_list.append((idata[0], predictions[top_k[0]]))

	with open('spores_list.txt', 'w+') as fp:
		[fp.write(img + ' ' + str(score) + '\n') for img, score in spores_list]


if __name__ == '__main__':
	image_dir = sys.argv[1]
	if os.path.isdir(image_dir):
		print "Image directory:", image_dir
		run_inference_on_image(image_dir)
	else:
		print "Image directory doesn't exist... exiting..."
