ip5wke_retrain
==============================

Image Recognition of Machine Parts, based on the ip5wke model using Tensorflow to retrain the last two layers.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── retrain
    │       ├── processed  <- The the preprocessed data set for calculating the bottleneck values
    │       ├── raw        <- The original, immutable data dump.
    │       └── tfrecords  <- The bottleneck values for each class
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── 1              <- The model that will be take as input for retraining the model
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │
    └── src                <- Source code for use in this project.
        ├── data           <- Scripts to generate data
        │   ├── crop_obj.py                <- Crops the image on the object using thresholding
        │   ├── make_dataset.py            <- Creates the dataset from raw images
        │   └── prepare_dataset.py         <- Splits the dataset into train, validation and testing sets (80%, 10%, 10%)
        │
        ├── models         <- Scripts to train models and then use trained models to make predictions
        │   ├── cache_bottleneck.py       <- calculates the bottleneck values for all processed images
        │   ├── model.py                  <- contains model definition
        │   ├── replace_softmax.py        <- uses the newly trained model and the old model and combine them for production use
        │   ├── retrain_model.py          <- retrains the model
        │   └── test_retrain_model.py     <- tests and evaluates a retrained model on validation or test set
        │
        └── server         <- Scripts to run the rest api
            ├── model_pb2.py              <- needed for google RPC to work
            ├── predict_pb2.py            <- needed for google RPC to work
            ├── prediction_service_pb2.py <- needed for google RPC to work
            └── rest_api.py               <- Contains the REST API for the mobile App
