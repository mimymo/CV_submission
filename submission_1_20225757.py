import os
import cv2
import yaml
import torch
import random
import numpy as np
from PIL import Image
from datetime import datetime
from models import YOLOv8n

def submission_1_20225757(yaml_path, output_json_path):
    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    data_config = load_yaml_config(yaml_path)
    model_name = 'yolo11n'
    ex_dict = {}
    epochs = 20
    batch_size = 2
    optimizer = 'AdamW'
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    
    ###### can be modified (Only Models, which can't be modified in demo) ######
    dropout       = 0.12             
    multi_scale   = True             
    aug_args = dict(
        hsv_h=0.02,  hsv_s=0.55, hsv_v=0.45,      

        degrees=6.0, translate=0.08, scale=0.25, shear=0.5,
        perspective=0.0,

        mixup=0.20,            
        mosaic=0.40,             
        copy_paste=0.15,         

        # 反転
        flipud=0.05,
        fliplr=0.30,

        # Random Erasing
        erasing=0.1      
    )
   
    from ultralytics import YOLO
    Experiments_Time = datetime.now().strftime("%y%m%d_%H%M%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    control_random_seed(42)

  
    model = YOLO(f"{model_name}.yaml", verbose=False)
    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        amp=True,              
        optimizer=optimizer,
        lr0=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dropout=dropout,
        multi_scale=multi_scale,
        close_mosaic=8,
        device=device,
        project=output_dir,
        name=Experiments_Time,
        pretrained=False,           
        cos_lr=True,                
        verbose=False,
        **aug_args,                 
    )

    param_cnt = sum(p.numel() for p in model.model.parameters())
    assert param_cnt < 4_000_000, f"Parameter limit exceeded: {param_cnt/1e6:.2f} M"

    best_pt = os.path.join(output_dir, Experiments_Time, "weights", "best.pt")
    model   = YOLO(best_pt, verbose=False)

    test_images  = get_test_images(data_config)
    results_dict = detect_and_save_bboxes(model, test_images) 
    save_results_to_file(results_dict, output_json_path)

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_test_images(config):
    test_path = config['test']
    root_path = config['path']

    test_path = os.path.join(root_path, test_path)
    
    if os.path.isdir(test_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths
def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False 

def detect_and_save_bboxes(model, image_paths):
    results_dict = {}

    for img_path in image_paths:
        results = model(img_path, verbose=False, task='detect',
                        augment=True,          
                          conf=0.10, iou=0.5)    
        img_results = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict

def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")