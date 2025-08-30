# Dental Tooth Detection with YOLOv8

## Project Overview
This project uses the powerful YOLOv8 object detection model to identify and classify different types of dental teeth from images. It covers the entire process—from preparing and organizing the dataset, training the model with advanced deep learning techniques, to evaluating its performance and making predictions with clear visual results. The goal is to create a practical and accurate tool that can help dentists and researchers analyze teeth more efficiently and effectively.

## Project Structure

├── ToothNumber_TaskDataset/
├── dataset_split/
├── scripts/
│   ├── train.py
│   └── ToothDetection.ipynb
├── runs/
├── data.yaml
├── yolov8m.pt
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


## Environment Setup
To set up your environment, install the required Python packages using:

pip install -r requirements.txt

The key dependencies include:
- ultralytics (YOLOv8 implementation)  
- split-folders (for dataset splitting)  
- pyyaml (to handle YAML config files)  
- torch (PyTorch, for deep learning framework)  

## Usage Instructions

### Dataset
The repository includes both the original dataset (`ToothNumber_TaskDataset`) and the processed dataset splits (`dataset_split/`).  
The `data.yaml` config file points to `dataset_split/` folders for training, validation, and testing.

### Training

- **Recommended:** Run the provided Google Colab notebook in `scripts/` to leverage GPU acceleration for faster training.  
- **Alternative:** Run the training script locally (`scripts/train.py`) if you have access to a GPU. CPU-only training is possible but will be significantly slower.


### Evaluation and Prediction

- Evaluation and prediction are integrated into the training scripts and notebook workflows.  
- Outputs, logs, and model checkpoints are saved under the `runs/` directory after training.

## File and Folder Descriptions

| Name                        | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `ToothNumber_TaskDataset/`  | Original dataset with all raw images and annotations         |
| `dataset_split/`            | Dataset split into training, validation, and test sets       |
| `data.yaml`                 | YAML file specifying dataset paths, class count, and names   |
| `scripts/`                  | Contains all code files and notebooks for training and testing |
| `yolov8m.pt`                | Trained YOLOv8 model weights                                  |
| `runs/`                     | Automatically generated folder with training outputs and logs |
| `.gitignore`                | Lists files and folders excluded from Git tracking           |
| `requirements.txt`          | Specifies Python packages and versions needed                 |

## Notes

- The dataset size is small (~500 images), so both original and split datasets are included in the repo for completeness and ease of use.  
- For best performance, use GPU-enabled environments like Google Colab for training.  
- Adjust paths in `data.yaml` if you reorganize dataset folders.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or further information, please contact:   
[Chinnavontari Abhinaya]  
[abhinayareddy1358@gmail.com]  
