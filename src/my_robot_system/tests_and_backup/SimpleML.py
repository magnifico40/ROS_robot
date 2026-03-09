import torch
import torchvision
from torchvision.models.detection import FasterRCNN
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
import numpy as np
import cv2
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(current_dir, '..', 'models', 'best_model_map_small.pth'))

if not os.path.exists(MODEL_PATH):
    print(f"Błąd: Plik nie istnieje pod ścieżką: {MODEL_PATH}")
else:
    print(f"Model znaleziony: {MODEL_PATH}")



#MODEL_PATH = 'best_model_map_small.pth'
IMG_SIZE = 608
CONFIDENCE_THRESHOLD = 0.3

CLASSES = [
    'right turn', 'left turn', 'puddle', 'street vendor', 'obstacle',
    'bad road', 'garbage bin', 'chair', 'pothole', 'car', 'motorcycle',
    'pedestrian', 'fence', 'gate barrier', 'roadblock', 'door', 'tree',
    'plant pot', 'drain', 'stair', 'pole', 'zebra cross'
]

PARTICULAR_CLASSES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 20]
FILTERED_CLASSES = [CLASSES[i] for i in range(len(CLASSES)) if i in PARTICULAR_CLASSES]
CLASSES = FILTERED_CLASSES

def preprocess_image(image):
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()

def get_model_jetson(num_classes, img_size=608):
    backbone = mobilenet_backbone(
        backbone_name="mobilenet_v3_small",
        pretrained=False, #pretrained=True,
        trainable_layers=5,
        fpn=True
    )
    backbone.out_channels = 256

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size)
        features = backbone(dummy_input)
        feature_map_names = list(features.keys())
        num_feature_maps = len(feature_map_names)

    if img_size >= 600:
        available_sizes = [16, 32, 64, 128, 256, 512]
    else:
        available_sizes = [8, 16, 32, 64, 128, 256]

    selected_sizes = tuple((s,) for s in available_sizes[:num_feature_maps])
    selected_aspect_ratios = ((0.5, 1.0, 2.0),) * num_feature_maps

    anchor_generator = AnchorGenerator(
        sizes=selected_sizes,
        aspect_ratios=selected_aspect_ratios
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=feature_map_names,
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_score_thresh=0.001,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
        rpn_score_thresh=0.0
    )

    return model


def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=2,
):
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def remap_state_dict_keys(state_dict):
    """
    Przemapowuje klucze state_dict z nowszego TorchVision (struktura 0.0.weight)
    na format starszego TorchVision używanego na Jetson Nano (struktura 0.weight).
    
    Problem: Model wytrenowany na nowszym PyTorch/TorchVision ma klucze np.:
      - backbone.fpn.inner_blocks.0.0.weight
    Ale Jetson Nano z Python 3.6 i starszym TorchVision oczekuje:
      - backbone.fpn.inner_blocks.0.weight
    """
    new_state_dict = {}
    
    # Mapowanie prefixów: stary_format -> nowy_format
    mappings = {
        # FPN inner_blocks (konwolucje 1x1 do redukcji kanałów)
        'backbone.fpn.inner_blocks.0.0': 'backbone.fpn.inner_blocks.0',
        'backbone.fpn.inner_blocks.1.0': 'backbone.fpn.inner_blocks.1',
        # FPN layer_blocks (konwolucje 3x3 do wygładzania)
        'backbone.fpn.layer_blocks.0.0': 'backbone.fpn.layer_blocks.0',
        'backbone.fpn.layer_blocks.1.0': 'backbone.fpn.layer_blocks.1',
        # RPN head conv (konwolucja w głowicy RPN)
        'rpn.head.conv.0.0': 'rpn.head.conv',
    }
    
    remapped_count = 0
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in mappings.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix)
                remapped_count += 1
                break
        new_state_dict[new_key] = value
    
    if remapped_count > 0:
        print(f"Przemapowano {remapped_count} kluczy dla kompatybilności wersji TorchVision")
    
    return new_state_dict


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")

NUM_CLASSES = len(CLASSES) + 1
model = get_model_jetson(NUM_CLASSES, img_size=IMG_SIZE)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Przemapuj klucze dla kompatybilności wersji TorchVision
state_dict = remap_state_dict_keys(state_dict)
model.load_state_dict(state_dict)  # Teraz powinno działać bez strict=False!

model.to(device)
model.eval()


cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera error")
    exit(1)

print("Camera ready. Press 'q' to quit.")

prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        img_tensor = preprocess_image(frame_rgb).to(device).unsqueeze(0)

        with torch.no_grad():
            start_infer = time.time()
            prediction = model(img_tensor)[0]
            infer_time = (time.time() - start_infer) * 1000

        detections = 0
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > CONFIDENCE_THRESHOLD:
                detections += 1

                box = box.cpu().numpy().astype(int)
                x_min, y_min, x_max, y_max = box

                label_idx = int(label.cpu()) - 1
                class_name = CLASSES[label_idx] if 0 <= label_idx < len(CLASSES) else f"ID: {label_idx}"

                text = f"{class_name}: {score:.2f}"

                if 'pedestrian' in class_name or 'car' in class_name or 'motorcycle' in class_name:
                    color = (0, 0, 255)
                elif 'pothole' in class_name or 'bad road' in class_name or 'obstacle' in class_name:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), color, 2)

                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame_resized,
                              (x_min, y_min - text_height - 10),
                              (x_min + text_width, y_min),
                              color, -1)
                cv2.putText(frame_resized, text, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.rectangle(frame_resized, (5, 5), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame_resized, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_resized, f"Inference: {infer_time:.1f}ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_resized, f"Detections: {detections}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Detection', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    cap.release()
    cv2.destroyAllWindows()