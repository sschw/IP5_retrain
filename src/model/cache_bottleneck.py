"""Cache the bottleneck values of the given image files and the SavedModel.

The SavedModel is used to generate the bottleneck values for the new images with
the new classification. 
Bottleneck is an inofficial term for the values right before the softmax
function.
These are used for retraining the model so we have to cache them. Otherwise
we would have to recalculate it every iteration.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
from google.protobuf import text_format
import model

FLAGS = tf.app.flags.FLAGS

def assemble_example(value, label):
    return_example = tf.train.Example(features=tf.train.Features(feature={
        "bottleneck_tensor_value": tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return return_example

def create_file_list():
    filelist = {}
    for category in ['train', 'test', 'validation']:
        with open(os.path.join(FLAGS.retrain_processed_data_dir, category, 'files.txt')) as f:
            filelist[category] = {}
            for l in f:
                lsplit = l.strip().split(" ")
                if lsplit[1] not in filelist[category]:
                    filelist[category][lsplit[1]] = []
                filelist[category][lsplit[1]].append(lsplit[0])
    return filelist
    
def read_png(path):
    file_contents = tf.read_file(path)
    image = tf.image.decode_png(file_contents, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image

def convert_bottlenecks_to_tfrecords(reset_cache = False):
    with tf.Session(graph=tf.Graph()) as sess:
        
        meta = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "../../models/1")
        
        bottleneck_tensor = tf.get_default_graph().get_tensor_by_name(model.BOTTLENECK_TENSOR_NAME)
        input_tensor = tf.get_default_graph().get_tensor_by_name(model.BOTTLENECK_INPUT_TENSOR_NAME)
        
        filelist = create_file_list()
        
        for category in ['train', 'test', 'validation']:
            if not os.path.exists(os.path.join(FLAGS.retrain_bottleneck_data_dir, category)):
                os.makedirs(os.path.join(FLAGS.retrain_bottleneck_data_dir, category))
            
            for label in filelist[category]:
                if reset_cache or not os.path.isfile(os.path.join(FLAGS.retrain_bottleneck_data_dir, category, label)):
                    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.retrain_bottleneck_data_dir, category, label))
                    for entry in filelist[category][label]:
                        sample = sess.run(read_png(os.path.join(os.pardir, os.pardir, entry)))
                        label_id = int(label)
                        bottleneck_tensor_value = sess.run(bottleneck_tensor, {input_tensor: [sample]})
                        bottleneck_tensor_value = np.squeeze(bottleneck_tensor_value)
                        
                        example = assemble_example(bottleneck_tensor_value, label_id)
                        writer.write(example.SerializeToString())
                    writer.close()

        

if __name__ == "__main__":
    convert_bottlenecks_to_tfrecords()