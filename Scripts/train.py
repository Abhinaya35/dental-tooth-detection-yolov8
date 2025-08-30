import os
import yaml
import splitfolders
from ultralytics import YOLO

def install_requirements():
    
    print("Make sure ultralytics and split-folders are installed in your environment.")

def prepare_data(extract_path: str, split_output: str):
    import os

    # Split dataset into train/val/test if not already done
    if not os.path.exists(split_output):
        print("Splitting dataset into train/val/test...")
        splitfolders.ratio(
            extract_path,
            output=split_output,
            seed=42,
            ratio=(.8, .1, .1)
        )
        print("Dataset split done.")
    else:
        print("Dataset already split.")


def create_data_yaml(yaml_path: str):
    yaml_content = {
        'train': 'dataset_split/train/images',
        'val': 'dataset_split/val/images',
        'test': 'dataset_split/test/images',
        'nc': 32,
        'names': {
            0: "Canine (13)",
            1: "Canine (23)",
            2: "Canine (33)",
            3: "Canine (43)",
            4: "Central Incisor (21)",
            5: "Central Incisor (41)",
            6: "Central Incisor (31)",
            7: "Central Incisor (11)",
            8: "First Molar (16)",
            9: "First Molar (26)",
            10: "First Molar (36)",
            11: "First Molar (46)",
            12: "First Premolar (14)",
            13: "First Premolar (34)",
            14: "First Premolar (44)",
            15: "First Premolar (24)",
            16: "Lateral Incisor (22)",
            17: "Lateral Incisor (32)",
            18: "Lateral Incisor (42)",
            19: "Lateral Incisor (12)",
            20: "Second Molar (17)",
            21: "Second Molar (27)",
            22: "Second Molar (37)",
            23: "Second Molar (47)",
            24: "Second Premolar (15)",
            25: "Second Premolar (25)",
            26: "Second Premolar (35)",
            27: "Second Premolar (45)",
            28: "Third Molar (18)",
            29: "Third Molar (28)",
            30: "Third Molar (38)",
            31: "Third Molar (48)"
        }
    }
    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)
    print(f"Created {yaml_path} file.")

def train_model():
    model = YOLO('yolov8m.pt')
    model.train(
        data='data.yaml',
        imgsz=640,
        epochs=120,
        lr0=0.001,
        batch=8,
        patience=20
    )
    print("Training completed.")

def evaluate_model():
    # Using CLI equivalent in Python or from ultralytics you can also do:
    os.system('yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml')
    print("Evaluation finished.")

def predict_and_save():
    # Run prediction and save using YOLO CLI
    os.system('yolo detect predict model=runs/detect/train/weights/best.pt source=dataset_split/test/images save=True')
    print("Prediction completed and results saved using CLI.")

    # Additional post-processing prediction using Ultralytics API for more control
    model = YOLO('runs/detect/train/weights/best.pt')
    results = model.predict(source='dataset_split/test/images', conf=0.6, iou=0.45, save=True)
    print("Post-processing prediction done using Ultralytics Python API.")

def main():
    dataset_source_path = 'tooth_detection_yolo/ToothNumber_TaskDataset' 
    split_output = 'dataset_split'
    yaml_path = 'data.yaml'

    prepare_data(dataset_source_path, split_output)  
    create_data_yaml(yaml_path)
    train_model()
    evaluate_model()
    predict_and_save()

if __name__ == "__main__":
    main()
