import cv2
import pytesseract
import numpy as np
import os
import time
import onnxruntime as ort

# 自作モジュールの前処理・後処理 (YOLO用)
from detect_plate_onnx import preprocess as preprocess_plate, postprocess as postprocess_plate

# --- パス設定 ---
image_path = "./obihiro-test.jpg"
modelB_path = "./yolov5s_carplate_ditect_ModelB.onnx"

# --- 環境変数からしきい値取得（デフォルト値あり）---
MODEL_B_CONF_THRESHOLD = float(os.getenv("MODEL_B_CONF_THRESHOLD", 0.5))
MODEL_B_IOU_THRESHOLD = float(os.getenv("MODEL_B_IOU_THRESHOLD", 0.45))
MODEL_B_PATH = os.getenv("MODEL_B_PATH", modelB_path)

# --- Tesseract バイナリパス設定 (環境に応じて変更) ---
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def load_onnx_model(onnx_path):
    """
    ONNXモデルをロードして InferenceSession を返すシンプルな関数
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[INFO] Loaded ONNX model: {onnx_path}")
    return session

def detect_plate(session, image, conf_threshold=MODEL_B_CONF_THRESHOLD,
                 iou_threshold=MODEL_B_IOU_THRESHOLD, input_size=(640, 640)):
    """
    YOLOv5s (ONNX) でナンバープレートを検出し、バウンディングボックスを返す関数。
    戻り値: [ [x1, y1, x2, y2, conf, class], ... ] のリスト
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 前処理
    input_data, scale, pad_x, pad_y = preprocess_plate(image, input_size)

    # 推論時間計測
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed_time = time.time() - start_time
    print(f"[TIME] ModelB (plate detect) inference time: {elapsed_time:.3f} sec")

    # 後処理で [x1, y1, x2, y2, conf, cls_id] を返す
    bboxes = postprocess_plate(outputs, conf_threshold, iou_threshold, scale, pad_x, pad_y)
    return bboxes

def simple_trapezoid_correct(image, x1, y1, x2, y2, out_width=300, out_height=150):
    """
    軽度に斜めのプレートを想定し、(x1, y1, x2, y2) の領域を台形補正して切り出す。
    out_width, out_height: 補正後の出力画像サイズ
    """
    # 軸平行なBBの4点を一旦取得 (上: y1, 下: y2)
    top_left     = [x1, y1]
    top_right    = [x2, y1]
    bottom_left  = [x1, y2]
    bottom_right = [x2, y2]

    # 射影変換に使う配列
    pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    pts_dst = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype=np.float32)

    # 変換行列を計算
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (out_width, out_height))
    return warped

def binarize_plate(plate_image):
    """
    プレート画像を二値化する簡易関数。
    ホワイトバランスや色差をある程度軽減する目的。
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # Otsuの自動閾値
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw

def ocr_plate_multiline(plate_image):
    """
    Tesseract で複数行（2行）想定の日本ナンバープレートをOCRする関数。
    plate_image: 切り出したプレート画像 (numpy配列)
    戻り値: OCRで抽出したテキスト文字列（改行含む可能性あり）
    """

    # plate_image が1チャネルかどうかチェック
    if len(plate_image.shape) == 2:
        # すでに1ch: そのまま使う
        gray = plate_image
    else:
        # カラーならGRAYSCALE化
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    config = "--psm 6 --oem 2 -l jpn"
    text = pytesseract.image_to_string(gray, config=config)
    return text

def ocr_with_confidence(img):
    """
    TesseractでOCRし、各行ごとのtextとconfidenceを取得する例。
    """
    # lang='jpn' など、必要に応じて設定
    config = "--psm 6 --oem 1 -l jpn"
    
    # TSV形式を辞書で取得
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    # data は keys=['level', 'page_num', 'block_num', 'par_num',
    # 'line_num', 'word_num', 'left', 'top', 'width', 'height',
    # 'conf', 'text'] など
    # 'conf' が confidence (0〜100, -1 は無効) 
    # 'text' が認識文字列

    results = []
    for i, conf in enumerate(data["conf"]):
        # conf=-1 は無視対象（認識失敗など）
        if conf == '-1':
            continue
        text = data["text"][i]
        # 空文字はスキップ
        if not text.strip():
            continue

        # 必要に応じて行/単語座標なども取得可能
        left   = data["left"][i]
        top    = data["top"][i]
        width  = data["width"][i]
        height = data["height"][i]

        # リストに (テキスト, 信頼度, 座標) としてまとめる
        results.append({
            "text": text,
            "conf": float(conf),
            "bbox": (left, top, width, height)
        })

    return results

if __name__ == "__main__":
    # 1. モデルロード
    sessionB = load_onnx_model(MODEL_B_PATH)

    # 2. 画像ロード
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        exit()

    # 3. Plate検出
    bboxes = detect_plate(sessionB, image)
    if not bboxes:
        print("[INFO] No plates detected.")
        exit()

    # 4. 検出された各 Plate の領域に対して処理 (複数想定)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf, cls_id = bbox

        # 軽度斜めを想定した台形補正 → warped
        t0 = time.time()
        plate_warped = simple_trapezoid_correct(
            image, x1, y1, x2, y2, out_width=500, out_height=250
        )
        warp_time = time.time() - t0
        print(f"[TIME] Warping (plate {i}): {warp_time:.3f} sec")

        if plate_warped.size == 0:
            print(f"[WARN] Warped plate is empty at index {i}, skipping...")
            continue

        # 二値化
        plate_bw = binarize_plate(plate_warped)
        if plate_bw.size == 0:
            print(f"[WARN] Binarized plate is empty at index {i}, skipping...")
            continue

        # OCRの処理時間を計測
        ocr_start = time.time()
        #ocr_text = ocr_plate_multiline(plate_bw)  # 2行想定で試す
        data = ocr_with_confidence(plate_bw)
        ocr_elapsed = time.time() - ocr_start
        for item in data:
            print(f"text='{item['text']}', conf={item['conf']:.2f}, bbox={item['bbox']}")

        #print(f"[TIME] OCR (plate {i}): {ocr_elapsed:.3f} sec")

        # 結果表示
        #print(f"[RESULT] Plate {i} text:\n{ocr_text.strip()}\n")

        # 必要に応じて表示 (GUI)
        # cv2.imshow(f"Plate {i} (warped)", plate_warped)
        # cv2.imshow(f"Plate {i} (bw)", plate_bw)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()
