import bottle
import base64

from grpc.beta import implementations
import json
import tensorflow as tf

import predict_pb2
import prediction_service_pb2

from random import *
import os

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "ip5wke", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 100.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS

DIR_NEW_WORKPIECES = 'new_workpieces'
DIR_WORKPIECE_IDS  = 'workpiece_ids'


class Inference:
    def __init__(self, host, port):
        serv_host = FLAGS.host
        serv_port = FLAGS.port
        model_name = FLAGS.model_name
        model_version = FLAGS.model_version
        self.request_timeout = FLAGS.request_timeout

        # Create gRPC client and request
        channel = implementations.insecure_channel(serv_host, serv_port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = 'predict_images'

        if model_version > 0:
            self.request.model_spec.version.value = model_version

        self._host = host
        self._port = port
	bottle.BaseRequest.MEMFILE_MAX = 1000000
        self._app = bottle.Bottle()
        self._route()

    def _route(self):
        self._app.route('/recognize_workpiece', method="POST", callback=self.recognize_workpiece)
        self._app.route('/new_workpiece_id', method="GET", callback=self.new_workpiece_id)
        self._app.route('/add_workpiece_image', method="POST", callback=self.add_workpiece_image)

    def start(self):
        self._app.run(host=self._host, port=self._port)

    def recognize_workpiece(self):
        # REST endpoint for prediction, takes base64 encoded jpg and returns the top 3 predicted classes with class probabilities and images

        file_data = base64.b64decode(bottle.request.json['image'])

        with open('current.jpg', 'wb') as f:
            f.write(file_data)
            f.close()

        self.request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(file_data, shape=[1]))

        # Send request
        answer = self.stub.Predict(self.request, self.request_timeout)
        scores = [answer.outputs['scores'].float_val[0],
                             answer.outputs['scores'].float_val[1],
                             answer.outputs['scores'].float_val[2]]
        classes = [answer.outputs['classes'].string_val[0].decode("utf-8"),
                             answer.outputs['classes'].string_val[1].decode("utf-8"),
                             answer.outputs['classes'].string_val[2].decode("utf-8")]
	images = []

	for i in range(0, 3):
	    workpiece_id = str(classes[i])
	    filename = workpiece_id + (".PNG" if os.path.isfile(DIR_WORKPIECE_IDS + "/" + workpiece_id + ".PNG") else ".jpg")
	    f = open(DIR_WORKPIECE_IDS + '/' + filename, 'rb')
	    images.append(base64.b64encode(f.read()))
	    f.close()

        return {"scores": scores, "classes": classes, "images": images}

    def new_workpiece_id(self):
        # REST endpoint for getting a new workpiece id
	print("new workpiece id requested")
	id_files = os.listdir(DIR_WORKPIECE_IDS)
	ids = map(lambda filename: int(os.path.splitext(filename)[0]), id_files) # files are named <id>.[PNG|JPG]
	ids = sorted(ids)
	new_id = ids[-1]+1 # could also use random id: randint(1000, 1000000)
	print("new workpiece id = " + str(new_id))

        return {"workpieceId": new_id}

    def add_workpiece_image(self):
        # REST endpoint for adding an image for a new workpiece
	print("adding image for new workpiece")
	workpiece_id = bottle.request.json['workpieceId']
	
        image_number = bottle.request.json['imageNumber']
	image = bottle.request.json['image']
    	directory = DIR_NEW_WORKPIECES + '/' + str(workpiece_id) + '/'
    	if not os.path.exists(directory):
	    os.makedirs(directory)
   	with open(directory + str(image_number) + '.jpg', 'wb') as f:
    	    f.write(base64.b64decode(image))
    	    f.close()


	id_file_name = DIR_WORKPIECE_IDS + '/' + str(workpiece_id) + '.jpg'
	if not os.path.isfile(id_file_name): # if it is the first file uploaded with this id: also save it as id file.
	    print("adding " + id_file_name)
	    with open(id_file_name, 'wb') as id_file:
	        id_file.write(base64.b64decode(image))
    	        id_file.close()

        return {"workpieceId": workpiece_id}



if __name__ == '__main__':
    # start server
    server = Inference(host='0.0.0.0', port=8888)
    server.start()