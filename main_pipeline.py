import time
import os

import argparse
import cv2
from pathlib import Path
import onnxruntime as ort
from label_definitions import namesA
from detect_plate_onnx import preprocess as preprocess_plate, postprocess as postprocess_plate
from detect_plate_detail_onnx import preprocess as preprocess_detail, postprocess as postprocess_detail, safe_non_max_suppression
from utils import draw_japanese_labels, sort_plate_characters, select_best_hiragana 
from label_definitions import namesA
from sklearn.cluster import KMeans 
import numpy as np


# パス設定

output_dir = Path("./results")
output_dir.mkdir(parents=True, exist_ok=True)

# 環境変数からしきい値取得（デフォルト値あり）
MODEL_A_CONF_THRESHOLD = float(os.getenv("MODEL_A_CONF_THRESHOLD", 0.45))
MODEL_A_IOU_THRESHOLD = float(os.getenv("MODEL_A_IOU_THRESHOLD", 0.3))
MODEL_B_CONF_THRESHOLD = float(os.getenv("MODEL_B_CONF_THRESHOLD", 0.5))
MODEL_B_IOU_THRESHOLD = float(os.getenv("MODEL_B_IOU_THRESHOLD", 0.45))


# 環境変数からモデルパス取得（デフォルト値あり）
MODEL_A_PATH = os.getenv("MODEL_A_PATH", "./yolov5s_carplate_detail_ModelA.onnx")
MODEL_B_PATH = os.getenv("MODEL_B_PATH", "./yolov5s_carplate_ditect_ModelB.onnx")

def load_onnx_model(onnx_path):
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    print(f"[INFO] Loaded ONNX model: {onnx_path}")
    print(f"[INFO] Execution Providers: {session.get_providers()}")
    return session

def detect_plate(session, image, conf_threshold=MODEL_B_CONF_THRESHOLD,
                  iou_threshold=MODEL_B_IOU_THRESHOLD, input_size=(640, 640)):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_data, scale, pad_x, pad_y = preprocess_plate(image, input_size)

    # 推論時間計測
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed_time = time.time() - start_time
    print(f"[TIME] ModelB (plate detect) inference time: {elapsed_time:.3f} sec")

    bboxes = postprocess_plate(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y)
    return bboxes

def detect_plate_details(session, plate_image, conf_threshold=MODEL_A_CONF_THRESHOLD,
                          iou_threshold=MODEL_A_IOU_THRESHOLD, input_size=(640, 640)):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_data, scale, pad_x, pad_y = preprocess_detail(plate_image, input_size)

    # 推論時間計測
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed_time = time.time() - start_time
    print(f"[TIME] ModelA (plate detail) inference time: {elapsed_time:.3f} sec")

    # NMS付きでバウンディングボックス抽出
    detections = postprocess_detail(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y)
    return detections

def warp_bbox_coords(bbox_list, M):
    """
    射影変換行列Mをバウンディングボックスに適用し、変換後の座標を返す。
    bbox_list: [(x1, y1, x2, y2, ...), ...]
    """
    warped_bboxes = []

    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox[:4]
        points = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, M).reshape(-1, 2)

        # 変換後の外接矩形に変換
        x_coords = transformed[:, 0]
        y_coords = transformed[:, 1]
        x1n, y1n = np.min(x_coords), np.min(y_coords)
        x2n, y2n = np.max(x_coords), np.max(y_coords)

        warped_bboxes.append((int(x1n), int(y1n), int(x2n), int(y2n), *bbox[4:]))

    return warped_bboxes

def draw_result(original_image, plate_bbox, plate_details):
    x1, y1, x2, y2, conf, cls = plate_bbox
    # Plate 枠
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Plate details
    for detail in plate_details:
        dx1, dy1, dx2, dy2, d_conf, d_cls = detail
        # 元画像の Plate 座標にオフセット
        gx1, gy1 = x1 + dx1, y1 + dy1
        gx2, gy2 = x1 + dx2, y1 + dy2
        class_name = namesA[d_cls] if d_cls < len(namesA) else f"id:{d_cls}"
        label = f"{class_name} {d_conf:.2f}"

        cv2.rectangle(original_image, (gx1, gy1), (gx2, gy2), (0, 0, 255), 1)
        cv2.putText(original_image, label, (gx1, gy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return original_image

def simple_trapezoid_correct(image, x1, y1, x2, y2, out_width=640, out_height=224):
    top_left     = [x1, y1]
    top_right    = [x2, y1]
    bottom_left  = [x1, y2]
    bottom_right = [x2, y2]

    pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    pts_dst = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (out_width, out_height))
    return warped, M

def process_frame(frame, sessionA, sessionB):
    print("[INFO] Detecting plates...")
    plate_text = ""  
    plate_bboxes = detect_plate(sessionB, frame)

    if not plate_bboxes:
        print("[INFO] No plates detected.")
        return frame, ""

    for plate_bbox in plate_bboxes:
        x1, y1, x2, y2, _, _ = plate_bbox
        plate_crop, M  = simple_trapezoid_correct(frame, x1, y1, x2, y2, out_width=640, out_height=224)

        if plate_crop.size == 0:
            print("[WARN] Empty crop, skipping...")
            continue

        print("[INFO] Detecting plate details...")
        plate_details = detect_plate_details(sessionA, plate_crop)

        if not plate_details:
            print("[INFO] No plate details detected, skipping drawing bbox.")
            #continue  # 追加: 詳細がない場合はスキップ

        # 座標補正: plate_crop 基準 → 元画像基準
        M_inv = np.linalg.inv(M)
        adjusted_plate_details = warp_bbox_coords(plate_details, M_inv)

        def cluster_and_sort(details):
            if len(details) <= 1:
                return details  # 1つならそのまま

            # y中心座標を計算
            y_centers = np.array([ (d[1] + d[3]) / 2 for d in details ]).reshape(-1, 1)

            # KMeansクラスタリング (クラスター数=2 前提)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(y_centers)
            labels = kmeans.labels_

            # クラスターごとにソート
            clustered = {0: [], 1: []}
            for label, detail in zip(labels, details):
                clustered[label].append(detail)

            # y 軸小さい方が上段
            if np.mean([ (d[1] + d[3]) / 2 for d in clustered[0] ]) > np.mean([ (d[1] + d[3]) / 2 for d in clustered[1] ]):
                top_line = clustered[1]
                bottom_line = clustered[0]
            else:
                top_line = clustered[0]
                bottom_line = clustered[1]

            # 各行で x 座標昇順
            top_line.sort(key=lambda d: (d[0] + d[2]) / 2)
            bottom_line.sort(key=lambda d: (d[0] + d[2]) / 2)

            return top_line + bottom_line

        # 検出結果を並べ替え
        sorted_details = cluster_and_sort(adjusted_plate_details)

        # ★ ひらがなフィルタ処理追加
        hiragana_class_ids = list(range(11, 57))  # クラス 11〜56 がひらがな
        image_width = frame.shape[1]

        best_hiragana = select_best_hiragana(adjusted_plate_details, hiragana_class_ids, image_width)

        # debug out
        for item in sorted_details:
            print(f"[DEBUG] item: {item}, len:{len(item)}")


        # 詳細なデバッグ出力
        print("[DEBUG] Plate details after sorting:")
        for idx, (dx1, dy1, dx2, dy2, conf, cls) in enumerate(sorted_details):
            label = namesA[cls] if cls < len(namesA) else f"id:{cls}"
            print(f" [{idx}] {label}: conf={conf:.2f}, bbox=({dx1}, {dy1}, {dx2}, {dy2})")

        # Plate text の組み立て
        plate_text = ''
        for _, _, _, _, _, cls in sorted_details:
            if cls in hiragana_class_ids:
                # ひらがなクラスならベストなものだけ追加
                if best_hiragana and cls == best_hiragana[5]:
                    plate_text += namesA[cls]
            else:
                plate_text += namesA[cls]
        print(f"[RESULT] Plate text: {plate_text}")

        # プレートバウンディングボックス（緑の太線）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 結果を描画
        frame = draw_japanese_labels(
            frame,
            adjusted_plate_details,
            label_map=namesA,
            font_path="./fonts/meiryo.ttc",
            plate_text=plate_text,  # plate_text を渡す
            plate_bbox=(x1, y1, x2, y2)
        )  
    return frame, plate_text

def load_models():
    print("[INFO] Loading models...")
    print(f"[CONFIG] MODEL_B_PATH: {MODEL_B_PATH}")
    print(f"[CONFIG] MODEL_A_PATH: {MODEL_A_PATH}")
    print(f"[CONFIG] MODEL_B_CONF_THRESHOLD: {MODEL_B_CONF_THRESHOLD}, MODEL_B_IOU_THRESHOLD: {MODEL_B_IOU_THRESHOLD}")
    print(f"[CONFIG] MODEL_A_CONF_THRESHOLD: {MODEL_A_CONF_THRESHOLD}, MODEL_A_IOU_THRESHOLD: {MODEL_A_IOU_THRESHOLD}")
    sessionB = load_onnx_model(MODEL_B_PATH)
    sessionA = load_onnx_model(MODEL_A_PATH)
    return sessionA, sessionB

#------ USBカメラを使ったストリーム作成モード --------------#
def camera_loop(sessionA, sessionB):
    cap = cv2.VideoCapture(0)  # 0: USBカメラ / PiCamera2は後ほど
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed, skipping...")
            continue

        inference_start = time.time()
        result_frame = process_frame(frame, sessionA, sessionB)
        inference_time = time.time() - inference_start
        print(f"[TIME] Total inference time: {inference_time:.3f} sec")

        # FPS計算
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # FPS 表示
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Carplate Detection", result_frame)

        # 'q' で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera loop terminated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Japanese carplate detection pipeline")
    parser.add_argument("--mode", choices=["image", "camera"], default="image",
                        help="Execution mode: image or camera (default: image)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to the input image when in image mode")
    return parser.parse_args()


def main(mode: str, image_path: str = None):
    total_start_time = time.time()
    print("[INFO] Loading models...")
    model_load_start = time.time()
    sessionA, sessionB = load_models()
    model_load_time = time.time() - model_load_start
    print(f"[TIME] Model loading time: {model_load_time:.3f} sec")

    if mode == "image":
        if not image_path:
            print("[ERROR] --image オプションで画像パスを指定してください")
            return

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return

        inference_start = time.time()
        result_image, p_text = process_frame(original_image, sessionA, sessionB)
        inference_time = time.time() - inference_start
        print(f"[TIME] Total inference time: {inference_time:.3f} sec")

        output_file = output_dir / "pipeline_result.jpg"
        cv2.imwrite(str(output_file), result_image)
        print(f"[INFO] Final result saved to {output_file}")

    elif mode == "camera":
        camera_loop(sessionA, sessionB)

if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode, image_path=args.image)

