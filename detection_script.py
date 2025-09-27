print(" BUILDING DETECTION SCRIPT WITH  MY IMPROVED MODEL")
print("=" * 50)

# Load My BEST trained model (15 epochs)
best_model_path = Path("/content/training_results/glove_detection_yolov8s_improved/weights/best.pt")
trained_model = YOLO(str(best_model_path))

print(f" Loaded improved model: {best_model_path}")
print(f" Performance: 59.6% mAP50, 73.3% Precision")

# Create output directories for assessment
OUTPUT_DIR = Path("/content/output")
LOGS_DIR = Path("/content/logs")
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

print(f" Output directories created:")
print(f"   Images: {OUTPUT_DIR}")
print(f"   Logs: {LOGS_DIR}")

# Detection functions
def save_detection_json(filename, detections, output_path):
    """Save detection results to JSON file as required by assessment"""
    result = {
        "filename": filename,
        "detections": detections
    }

    json_path = output_path / f"{Path(filename).stem}.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    return json_path

def process_detections(results, confidence_threshold=0.5):
    """Convert YOLO results to assessment format"""
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                xyxy = boxes.xyxy[i].tolist()

                if conf >= confidence_threshold:
                    # Map class to assessment labels
                    label = "Wearing_Gloves" if cls == 1 else "No_Gloves"

                    detection = {
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [round(x, 1) for x in xyxy]  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)

    return detections

print(" Detection functions ready!")
print("\n Ready to test detection on sample images...")