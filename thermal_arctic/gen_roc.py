import sys
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

labels = ['bird', 'bear', 'wave']

def create_graph(modelFullPath):
	"""Creates a graph from saved GraphDef file and returns a saver."""
	# Creates graph from saved graph_def.pb.
	with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


def gen_roc_data(graph_path, file_list_path, image_folder_path):
	answer = None

	image_data = []
	with open(file_list_path, 'r+') as fp:
		next(fp) # Skip first line
		for line in fp:
			l = line.split()
			img_name = l[0]
			img_label = labels(int(l[1]))
			img_path = os.path.join(image_folder_path, image_label, img_name)
			if not tf.gfile.Exists(img_path):
				tf.logging.fatal('File does not exist %s', img_path)
                        	return answer
			image_data.append([img_name, img_label, tf.gfile.FastGFile(img_path, 'rb').read()])
		
	# Creates graph from saved GraphDef.
	create_graph(graph_path)

	with tf.Session() as sess:

		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		
		with open('roc_data.txt', 'w+') as fp:

			print "\n\n\nResults:\n"
			for idata in image_data:
				predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': idata[2]})
				predictions = np.squeeze(predictions)

				top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
	
				output_line = idata[0]
				for label_id in range(len(labels)):
					node_id = top_k.index(label_id)
					output_line += predictions[node_id] + ','
				
				print output_line

				output_line += '\n'
				fp.write(output_line)		
				
				answer = labels[top_k[0]]
	return answer


if __name__ == '__main__':
	graph_path = sys.argv[1]
	file_list_path = sys.argv[2]
	image_folder_path = sys.argv[3]	
	print "Path to Graph:", graph_path
	print "Path to text file:", file_list_path
	print "Path to parent images folder", image_folder_path
	
	gen_roc_data(graph_path, file_list_path, image_folder_path)


