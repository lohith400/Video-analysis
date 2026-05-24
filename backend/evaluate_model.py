from ultralytics import YOLO

def main():
    model_path = 'runs/detect/indian_plate_v1/weights/best.pt'
    print(f"Loading best model from {model_path} for validation...")
    model = YOLO(model_path)
    
    metrics = model.val(data='plate_dataset/data.yaml')
    
    print('\n' + '=' * 40)
    print("      INDIAN PLATE DETECTION METRICS      ")
    print('=' * 40)
    print(f'mAP50:      {metrics.box.map50:.4f}')
    print(f'mAP50-95:   {metrics.box.map:.4f}')
    print(f'Precision:  {metrics.box.mp:.4f}')
    print(f'Recall:     {metrics.box.mr:.4f}')
    print('=' * 40)
    print("Use these exact numbers for your MCA Semester I research paper!")

if __name__ == "__main__":
    main()
