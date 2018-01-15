#!/bin/sh
# Prepares the data, retrains the model and exports the model

echo "Prepare the data"
python ./src/data/make_dataset.py ./data/retrain/raw  ./data/retrain/processed

echo "Start transfer learning"
cd ./src/model

echo "Start creating bottleneck values"
python ./cache_bottleneck.py

echo "Start retrain of last layers"
python ./retrain_model.py

echo "Serve the model by replacing the current models last layers"
python ./replace_softmax.py

# rm -rf ./models/1
# mv /tmp/ip5wke_retrain_output/1 ./models/1
