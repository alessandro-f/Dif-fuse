# Dif-fuse
Pytorch repository for the paper “Diffusion Models for Counterfactual Generation
and Anomaly Detection in Brain Images”. An official version will be released soon.


To pre-process BraTS data, after placing the NIfTI files in the folder *data/brats2021*, you can run:
```
python scripts/brats-preprocess.py 
```
To train a diffusion model on healthy samples:
```
python scripts/image_train.py 
```
To train an autoencoder to reconstruct the images:
```
python scripts/train_autoencoder.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name autoencoder
```
To train a baseline classifier to classify the images:
```
python scripts/train_baseline_classifier.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name baseline_classifier
```
To generate the saliency maps with ACAT (Adversarial Counterfactual Attention), 
employing the classifier and autoencoder trained in the previous steps:
```
python scripts/saliency_maps.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name create_saliency
```
To generate healthy samples with Dif-fuse, you can run the following code, making sure to set the correct path to
the model that you wish to use:
```
python scripts/image_sample_dif-fuse.py --model_path results/test/model054000.pt 
```
