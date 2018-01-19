"""Export ip5wke model given existing training checkpoints.

The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

import os.path

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
                            
tf.app.flags.DEFINE_integer('is_training', False,
                            """Is training or not for batch norm""")
                            
tf.app.flags.DEFINE_float('dropout_keep_probability', 1.0,
                            """How many nodes to keep during dropout""")

NUM_TOP_CLASSES = 3

def export():
    with tf.Session() as sess:
      meta = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "../../models/1")
      bottleneck_graph_def = tf.graph_util.convert_variables_to_constants(sess, meta.graph_def, ["local5/local5"])
    tf.reset_default_graph()
        
    #bottleneck_graph_def = tf.graph_util.extract_sub_graph(meta.graph_def, dest_nodes=["local5/local5"])
    serialized_tf_example, jpegs, logits = tf.import_graph_def(graph_def=bottleneck_graph_def, name="", return_elements=[model.INPUT_TENSOR_NAME, model.JPEGS_TENSOR_NAME, model.BOTTLENECK_TENSOR_NAME])
    
    #serialized_tf_example = tf.get_default_graph().get_tensor_by_name(INPUT_TENSOR_NAME)
    #jpegs = tf.get_default_graph().get_tensor_by_name(JPEGS_TENSOR_NAME)
    
    #logits = tf.get_default_graph().get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
    with tf.variable_scope('retrain') as scope:
      logits = model.softmax(logits, model.num_of_classes())
    logits = tf.nn.softmax(logits)

    # Transform output to topK result.
    values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)

    # Create a constant string Tensor where the i'th element is
    # the human readable class description for the i'th index.
    class_descriptions = []
    for s in range(model.num_of_classes()):
      class_descriptions.append(str(s))
    class_tensor = tf.constant(class_descriptions)

    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        class_tensor, default_value="UNKNOWN")

    classes = table.lookup(tf.cast(indices, dtype=tf.int64))
  
    with tf.Session() as sess:
      variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='retrain')
      
      saver = tf.train.Saver(variables_to_restore)
        
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded softmax from %s at step=%s.',
            ckpt.model_checkpoint_path, global_step)
      else:
        print('No checkpoint file found at %s', FLAGS.checkpoint_dir)
        return

      # Export inference model.
      output_path = os.path.join(
          compat.as_bytes(FLAGS.output_dir),
          compat.as_bytes(str(FLAGS.model_version)))
      print('Exporting trained model to', output_path)
      builder = saved_model_builder.SavedModelBuilder(output_path)

      # Build the signature_def_map.
      classify_inputs_tensor_info = utils.build_tensor_info(
          serialized_tf_example)
      classes_output_tensor_info = utils.build_tensor_info(classes)
      scores_output_tensor_info = utils.build_tensor_info(values)

      classification_signature = signature_def_utils.build_signature_def(
          inputs={
              signature_constants.CLASSIFY_INPUTS: classify_inputs_tensor_info
          },
          outputs={
              signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classes_output_tensor_info,
              signature_constants.CLASSIFY_OUTPUT_SCORES:
                  scores_output_tensor_info
          },
          method_name=signature_constants.CLASSIFY_METHOD_NAME)

      predict_inputs_tensor_info = utils.build_tensor_info(jpegs)
      prediction_signature = signature_def_utils.build_signature_def(
          inputs={'images': predict_inputs_tensor_info},
          outputs={
              'classes': classes_output_tensor_info,
              'scores': scores_output_tensor_info
          },
          method_name=signature_constants.PREDICT_METHOD_NAME)

      legacy_init_op = tf.group(
          tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  classification_signature,
          },
          legacy_init_op=legacy_init_op)

      builder.save()
      print( 'Successfully exported model to %s', FLAGS.output_dir)

def main(unused_argv=None):
  if tf.gfile.Exists(FLAGS.output_dir):
      tf.gfile.DeleteRecursively(FLAGS.output_dir)
  tf.gfile.MakeDirs(FLAGS.output_dir)
  export()


if __name__ == '__main__':
  tf.app.run()
