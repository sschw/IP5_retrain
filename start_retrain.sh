#!/bin/sh
# Prepares the data, retrains the model and exports the model

# Ensure that we are at the correct path
cd "$(dirname "$0")"

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

echo "Remove old model"
# mv ./models/1 ./old_models/"$(date +"%Y%m%d_%H%M%S")"
# or 
rm -rf ./models/1

echo "Add new model"
mv /tmp/ip5wke_retrain_output/1 ./models/1
