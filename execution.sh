#!/bin/sh
export CUDA_HOME=/opt/cuda-10.1.168_418_67/

export CUDNN_HOME=/opt/cuDNN-cuDNN-7.6.0.64_9.2/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

export NUM_GPUS=$1

echo $NUM_GPUS

cd . .
#export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
source activate diffuse

#python scripts/image_train.py

#python scripts/train_autoencoder.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name autoencoder

#python scripts/train_baseline_classifier.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name baseline_classifier

#python scripts/saliency_maps.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name create_saliency

python scripts/image_sample_dif-fuse.py --model_path results/test/model054000.pt