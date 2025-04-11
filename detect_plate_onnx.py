import time

import argparse
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch
from utils import non_max_suppression

def safe_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)
    return non_max_suppression(prediction, conf_thres, iou_thres)

def preprocess(image, input_size):
    """
    画像をモデル入力サイズに前処理
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
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # バッチ次元を追加
    return img.astype(np.float32), scale, pad_x, pad_y


def postprocess(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y):
    """
    ONNX モデル出力を後処理して、NMS 付きでバウンディングボックス取得
    """
    outputs_tensor = torch.tensor(outputs[0])  # [1, N, 85]
    print(f"[DEBUG] outputs_tensor shape: {outputs_tensor.shape}")
    outputs_tensor = outputs_tensor.squeeze(0)  # [N, 85]

    results = safe_non_max_suppression(outputs_tensor, conf_thres=conf_threshold, iou_thres=iou_threshold)

    bboxes = []
    for result in results:
        if result is None or result.size(0) == 0:
            continue
        for *box, conf, cls in result:
            x1, y1, x2, y2 = box
            # 元画像座標へ戻す
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            bboxes.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
    return bboxes


def draw_boxes(image, bboxes):
    """
    バウンディングボックス描画
    """
    for (x1, y1, x2, y2, conf, cls) in bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"plate {conf:.2f}"
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


def save_cropped_plates(image, bboxes, output_path):
    """
    検出されたプレート領域を個別保存
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, (x1, y1, x2, y2, conf, cls) in enumerate(bboxes):
        cropped_plate = image[y1:y2, x1:x2]
        plate_file = output_path / f"output_plate_{idx + 1}.jpg"
        cv2.imwrite(str(plate_file), cropped_plate)
        print(f"[INFO] Cropped plate saved to {plate_file}")


def run_inference(onnx_path, input_path, conf_threshold, iou_threshold, output_path, input_size=(640, 640)):
    """
    推論実行メイン処理
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_path = Path(input_path)
    if input_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        print("[ERROR] Unsupported file type. Please provide an image file.")
        return

    image = cv2.imread(str(input_path))
    input_data, scale, pad_x, pad_y = preprocess(image, input_size)

    # 推論時間計測
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed_time = time.time() - start_time
    print(f"[TIME] ModelB (plate detect) inference time: {elapsed_time:.3f} sec")
    print(f"[INFO] Model outputs received. Output count: {len(outputs)}")

    bboxes = postprocess(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y)
    print(f"[INFO] Postprocessing completed. {len(bboxes)} plates detected.")

    image = draw_boxes(image, bboxes)

    save_cropped_plates(image, bboxes, output_path)

    output_file = Path(output_path) / f"output_{input_path.name}"
    cv2.imwrite(str(output_file), image)
    print(f"[INFO] Result saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX plate detection test script")
    parser.add_argument('onnx_model_path', type=str, help="Path to the ONNX model file")
    parser.add_argument('input_path', type=str, help="Path to the input image")
    parser.add_argument('--output_path', type=str, default="./results", help="Directory to save outputs")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--iou', type=float, default=0.45, help="IOU threshold for NMS")
    args = parser.parse_args()

    run_inference(args.onnx_model_path, args.input_path, args.conf, args.iou, args.output_path)
