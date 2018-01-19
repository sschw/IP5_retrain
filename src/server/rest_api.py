import bottle
import base64

from grpc.beta import implementations
import json
import tensorflow as tf

import predict_pb2
import prediction_service_pb2

from random import *
import os
import subprocess

import sys
sys.path.append('../data/')

import crop_obj

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "ip5wke", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 100.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS

DIR_NEW_WORKPIECES = '../../data/retrain/raw/'
DIR_WORKPIECE_IDS  = '../../data/retrain/processed/'
TRANSFER_LEARNING_SCRIPT = '../../start_retrain.sh'


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
        self._app.route('/initiate_transfer_learning', method="POST", callback=self.initiate_transfer_learning)

    def start(self):
        self._app.run(host=self._host, port=self._port)

    def recognize_workpiece(self):
        # REST endpoint for prediction, takes base64 encoded jpg and returns the top 3 predicted classes with class probabilities and images

        file_data = base64.b64decode(bottle.request.json['image'])

        with open('current.jpg', 'wb') as f:
            f.write(file_data)
        
        # crop the file if possible.
        # if the object can't be found, leave it untouched
        try:
            crop_obj.scale_and_resize_from_imagedata(imread('current.jpg'), 'current.jpg')
            with open('current.jpg', 'rb') as f:
                file_data = f.read()
        except:
            pass

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
            filename = os.listdir(DIR_WORKPIECE_IDS + workpiece_id)[0]
            f = open(DIR_WORKPIECE_IDS + '/' + workpiece_id + '/' + filename, 'rb')
            images.append(base64.b64encode(f.read()))
            f.close()

        return {"scores": scores, "classes": classes, "images": images}

    def new_workpiece_id(self):
        # REST endpoint for getting a new workpiece id
        print("new workpiece id requested")
        dir_list = [d for d in os.listdir(DIR_NEW_WORKPIECES) 
                      if os.path.isdir(os.path.join(DIR_NEW_WORKPIECES, d))]
        dir_int_list = [int(d) for d in dir_list]
        new_id = max(dir_int_list) + 1
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

        return {"workpieceId": workpiece_id}
    
    def initiate_transfer_learning(self):
        # REST endpoint to start the transfer learning
        print("starting script " + TRANSFER_LEARNING_SCRIPT)
        subprocess.call([TRANSFER_LEARNING_SCRIPT])



if __name__ == '__main__':
    # start server
    server = Inference(host='0.0.0.0', port=8888)
    server.start()