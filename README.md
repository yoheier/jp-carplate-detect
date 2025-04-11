# jp-carplate-detect

Raspberry Pi 4B で動作する日本ナンバープレート検出・認識パイプライン。  
ONNX Runtime (DNNL サポート) を用いて高速に推論します。

## Setup

### 1. Python 仮想環境の構築

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Git LFS (Large File Storage)

ONNX モデルやフォント等の大容量ファイルは Git LFS で管理しています。

```bash
git lfs install
git lfs pull
```

### 3. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

> **Note:**  
> `onnxruntime` は Raspberry Pi 4B 上でビルド済みの `.whl` ファイルを使用しています。  
> CPU 最適化版 (DNNL 対応) であるため、リポジトリ内の `.whl` をそのまま使用します。

## Usage

### 静止画モードでテスト実行

```bash
python main_pipeline.py --mode image
```

- 入力: `./RX-8_Plate.jpg`
- 出力: `./results/pipeline_result.jpg`
- 認識されたプレート文字列と FPS ログが表示されます。

### USB カメラを使用したストリーム処理

```bash
python main_pipeline.py --mode camera
```

- 入力: USB カメラ (デバイスID: `0`)
- ウィンドウ表示あり
- `q` キーで終了します
- 認識結果とリアルタイム FPS がオーバーレイ表示されます

## Notes

- PiCamera2 対応は今後の拡張予定です。
- 推論速度のボトルネックは現在「モデルB (plate detect)」です。改善検討中です。
- モデル改善や INT8 / FP16 最適化のため、計測ログが自動出力されるようになっています。
