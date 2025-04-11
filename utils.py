import time

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def draw_japanese_labels(cv2_image, detections, label_map, font_path="fonts/NotoSansCJK-Regular.ttc", font_size=24, plate_text=None, plate_bbox=None):
    """
    OpenCV画像に日本語ラベルを描画する

    Args:
        cv2_image: OpenCVの画像 (BGR)
        detections: [(x1, y1, x2, y2, conf, cls), ...]
        label_map: クラスIDと日本語ラベルの辞書
        font_path: フォントファイルパス
        font_size: フォントサイズ

    Returns:
        OpenCVの画像 (BGR)
    """
    # OpenCV → PIL 変換
    image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # フォント読み込み
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[WARNING] フォント読み込み失敗: {e}、デフォルトフォントで描画します")
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if isinstance(label_map, list):
            label = label_map[cls] if cls < len(label_map) else f"id:{cls}"
        else:
            label = label_map.get(cls, f"id:{cls}")

        # バウンディングボックス
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # ラベル文字列
        text = f"{label} {conf:.2f}"

        # テキスト背景
        # テキストサイズを取得（Pillow v10以降対応）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # テキスト背景
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")

        # テキスト描画
        draw.text((x1, y1 - text_height), text, font=font, fill="white")

        if plate_text and plate_bbox:
            # プレートバウンディングボックスの左上に文字列描画
            bx1, by1, _, _ = plate_bbox
            draw.text((bx1, by1 - font_size * 2), plate_text, font=font, fill="blue")

    # PIL → OpenCV 戻し
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def sort_plate_characters(detections):
    """
    ナンバープレート内の文字検出結果を、上段・下段に分けて並び替える
    """
    if not detections:
        return []

    # 各検出の中央y座標を計算
    centers = [((d[1] + d[3]) / 2) for d in detections]
    median_y = np.median(centers)

    # 上段・下段に分割
    upper_line = [d for d, cy in zip(detections, centers) if cy < median_y]
    lower_line = [d for d, cy in zip(detections, centers) if cy >= median_y]

    # 各段をx1（左端）の小さい順に並び替え
    upper_sorted = sorted(upper_line, key=lambda d: d[0])
    lower_sorted = sorted(lower_line, key=lambda d: d[0])

    return upper_sorted + lower_sorted


def select_best_hiragana(detections, hiragana_class_ids, image_width):
    """
    検出されたひらがなの中からベストを選択
    - 信頼度スコア
    - 面積
    - x軸中央寄り
    を複合評価

    Args:
        detections: [(x1, y1, x2, y2, conf, cls), ...]
        hiragana_class_ids: ひらがなクラスIDのリスト
        image_width: 画像の横幅（中央寄り評価用）

    Returns:
        選ばれたひらがな or None
    """
    hiragana_detections = [d for d in detections if d[5] in hiragana_class_ids]
    if not hiragana_detections:
        return None

    center_x = image_width / 2

    def score_fn(det):
        x1, y1, x2, y2, conf, cls = det
        width = x2 - x1
        height = y2 - y1
        area = width * height
        bbox_center_x = (x1 + x2) / 2
        center_distance = abs(bbox_center_x - center_x)

        # スコア優先、次に小さい面積、最後に中心寄り
        return (
            conf,                 # 高スコア優先
            -area,                # 小さい面積優先
            -center_distance      # 中央寄り優先
        )

    # ベストなひらがなを選択
    best_hiragana = max(hiragana_detections, key=score_fn)
    return best_hiragana
