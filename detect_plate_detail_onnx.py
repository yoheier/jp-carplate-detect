import time

import argparse
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch
from utils import non_max_suppression
from label_definitions import namesA  # クラス名を定義したファイルを必ず用意

def preprocess(image, input_size):
    """
    入力画像を ONNX モデル用に前処理
    """
    original_h, original_w, _ = image.shape
    scale = min(input_size[1] / original_h, input_size[0] / original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((input_size[1], input_size[0], 3), 128, dtype=np.float32)

    pad_x = (input_size[0] - new_w) // 2
    pad_y = (input_size[1] - new_h) // 2
    padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image

    img = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # 正規化
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # バッチ次元追加
    return img.astype(np.float32), scale, pad_x, pad_y

def safe_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    安全な NMS 実行（次元エラー防止）
    """
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)
    return non_max_suppression(prediction, conf_thres, iou_thres)

def postprocess(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y):
    """
    出力後処理＋ NMS
    """
    outputs_tensor = torch.tensor(outputs[0])  # [1, N, num_classes + 5]
    print(f"[DEBUG] outputs_tensor shape: {outputs_tensor.shape}")
    outputs_tensor = outputs_tensor.squeeze(0)  # [N, num_classes + 5]

    results = safe_non_max_suppression(outputs_tensor, conf_thres=conf_threshold, iou_thres=iou_threshold)

    detections = []
    for result in results:
        if result is None or result.size(0) == 0:
            continue
        for *box, conf, cls in result:
            x1, y1, x2, y2 = box
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
    return detections

def draw_boxes(image, detections):
    """
    バウンディングボックス描画
    """
    for (x1, y1, x2, y2, conf, cls) in detections:
        class_name = namesA[cls] if cls < len(namesA) else f"id:{cls}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 5
        cv2.rectangle(image,
                      (label_x, label_y - label_size[1] - 5),
                      (label_x + label_size[0], label_y),
                      (0, 255, 0), -1)
        cv2.putText(image, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    return image

def run_inference(onnx_path, input_path, conf_threshold, iou_threshold, output_path, input_size=(640, 640)):
    """
    推論メイン処理
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"[ERROR] Failed to load image: {input_path}")
        return

    input_data, scale, pad_x, pad_y = preprocess(image, input_size)

    # 推論時間計測
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed_time = time.time() - start_time
    print(f"[TIME] ModelA (plate detail) inference time: {elapsed_time:.3f} sec")
    print(f"[INFO] Model outputs received. Output count: {len(outputs)}")

    detections = postprocess(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y)
    print(f"[INFO] Postprocessing completed. {len(detections)} characters detected.")

    image = draw_boxes(image, detections)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) / f"output_{Path(input_path).name}"
    cv2.imwrite(str(output_file), image)
    print(f"[INFO] Result saved to {output_file}")

    # 認識結果まとめて表示
    if detections:
        print("[RESULT] Detected characters:")
        for _, _, _, _, conf, cls in detections:
            class_name = namesA[cls] if cls < len(namesA) else f"id:{cls}"
            print(f" - {class_name} (confidence: {conf:.2f})")
    else:
        print("[RESULT] No characters detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX plate detail (character) detection test script")
    parser.add_argument('onnx_model_path', type=str, help="Path to the ONNX model file")
    parser.add_argument('input_path', type=str, help="Path to the input image")
    parser.add_argument('--output_path', type=str, default="./results", help="Directory to save outputs")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--iou', type=float, default=0.45, help="IOU threshold for NMS")
    args = parser.parse_args()

    run_inference(args.onnx_model_path, args.input_path, args.conf, args.iou, args.output_path)
