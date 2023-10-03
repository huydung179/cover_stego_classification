# Technical exercise

## Environment setup

I used PyTorch 2.0.1+cu118 to develop this project. I assume that you have correctly installed at least one Pytorch version on your system-site-packages and also added the CUDA path to your `PATH` environment variable. To install dependencies, I used the buildin `venv` module as follows:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Classification Training
To train the classification model, you can run the following command:
```bash
python train.py --data_dir [your_data_dir] --save_path [your_save_path] --batch_size [your_batch_size] --epochs [your_epochs] --lr [your_learning_rate]
```

## Classification Evaluation
To evaluate the model, you can run the following command:
```bash
python evaluate.py --data_dir [your_data_dir] --model_path [your_model_path] --batch_size [your_batch_size]
```

## Reconstruction Training
To train the reconstruction model, you can run the following command:
```bash
python train_auto_encoder.py --data_dir [your_data_dir] --save_path [your_save_path] --batch_size [your_batch_size] --epochs [your_epochs] --lr [your_learning_rate] --encoded_space_dim [your_encoded_space_dim]
```

## Reconstruction Evaluation
To evaluate the model, you can run the following command:
```bash
python evaluation_reconstruction.py --data_dir [your_data_dir] --model_path [your_model_path] --batch_size [your_batch_size] --encoded_space_dim [your_encoded_space_dim]
```