import time
import cv2
from pathlib import Path
import onnxruntime as ort
from label_definitions import namesA
from detect_plate_onnx import preprocess as preprocess_plate, postprocess as postprocess_plate
from detect_plate_detail_onnx import preprocess as preprocess_detail, postprocess as postprocess_detail, safe_non_max_suppression
from utils import draw_japanese_labels, sort_plate_characters, select_best_hiragana 
from label_definitions import namesA


# パス設定
image_path = "./RX-8_Plate.jpg"
modelB_path = "./yolov5s_carplate_ditect_ModelB.onnx"
modelA_path = "./yolov5s_carplate_detail_ModelA.onnx"
output_dir = Path("./results")
output_dir.mkdir(parents=True, exist_ok=True)

def load_onnx_model(onnx_path):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def detect_plate(session, image, conf_threshold=0.5, iou_threshold=0.45, input_size=(640, 640)):
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

def detect_plate_details(session, plate_image, conf_threshold=0.55, iou_threshold=0.3, input_size=(640, 640)):
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

def process_frame(frame, sessionA, sessionB):
    print("[INFO] Detecting plates...")
    plate_text = ""  
    plate_bboxes = detect_plate(sessionB, frame)

    if not plate_bboxes:
        print("[INFO] No plates detected.")
        return

    for plate_bbox in plate_bboxes:
        x1, y1, x2, y2, _, _ = plate_bbox
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            print("[WARN] Empty crop, skipping...")
            continue

        print("[INFO] Detecting plate details...")
        plate_details = detect_plate_details(sessionA, plate_crop)

        if not plate_details:
            print("[INFO] No plate details detected, skipping drawing bbox.")
            #continue  # 追加: 詳細がない場合はスキップ

        # 座標補正: plate_crop 基準 → 元画像基準
        adjusted_plate_details = []
        for detail in plate_details:
            dx1, dy1, dx2, dy2, conf, cls = detail
            adjusted_plate_details.append((
                dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1, conf, cls
            ))
        # 検出結果を並べ替え
        sorted_details = sort_plate_characters(adjusted_plate_details)

        # ★ ひらがなフィルタ処理追加
        hiragana_class_ids = list(range(11, 57))  # クラス 11〜56 がひらがな
        image_width = frame.shape[1]

        best_hiragana = select_best_hiragana(adjusted_plate_details, hiragana_class_ids, image_width)

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

def get_frame_from_image():
    # 入力画像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return
    return original_image

def load_models():
    print("[INFO] Loading models...")
    sessionB = load_onnx_model(modelB_path)
    sessionA = load_onnx_model(modelA_path)
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

        result_frame, _ = process_frame(frame, sessionA, sessionB)

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



def main(mode="image"):
    # モデルロード
    print("[INFO] Loading models...")
    sessionA, sessionB = load_models()

    # 入力画像
    if mode == "image":
        original_image = get_frame_from_image()

        # 1フレーム処理
        if original_image is not None:
            result_image, _ = process_frame(original_image, sessionA, sessionB)     

        output_file = output_dir / f"pipeline_result.jpg"
        cv2.imwrite(str(output_file), result_image)
        print(f"[INFO] Final result saved to {output_file}")

    elif mode == "camera":
        camera_loop(sessionA, sessionB)

if __name__ == "__main__":
    main(mode="image")
