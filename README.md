# Interaction2024

このプロジェクトは、[インタラクション2024](https://www.interaction-ipsj.org/)の[デモセッション](https://www.interaction-ipsj.org/2024/program/#interactive1)（1B-45）で発表した内容「VR 環境下における身体背面部へのタップ操作による入力領域拡張」の実装です。

## 概要

本研究では、VR空間内でのタッチインタラクションを全身に拡張するシステムを提案しています。Nintendo Switch用コントローラーであるJoy-Conを利用し、VRヘッドセットと組み合わせることで、ユーザの身体の様々な部位へのタッチ操作を可能にします。

## 主な機能

1. Joy-Conからの加速度・ジャイロセンサーデータの取得
2. VRヘッドセットの位置・姿勢情報の利用
3. 機械学習モデルによる手の位置の推論とタップ検出
4. 身体をタップして遊ぶリズムゲームアプリケーション

## システム構成

- Python: センサーデータの処理、機械学習モデルの実装　（バックエンド）
- Unity: VR環境の構築、ユーザインタフェースの実装（クライアント、リズムゲーム）　

## セットアップ

1. Pythonの依存ライブラリをインストール:
```
pip install -r requirements.txt
```

2. Unityプロジェクトを開き、必要なアセットをインポート

3. Joy-ConをPCに接続

4. Pythonスクリプトを実行してセンサーデータの取得を開始:
```
python Python/socket_pipeline.py
```

5. Meta Quest 2/3を装着し、リストバンドに装着したJoy-Conを手首につける

6. UnityでVRアプリケーションを実行（Sceneフォルダ内のTitleシーンを実行）

## 参考文献

- [インタラクション2024 論文「VR 環境下における身体背面部へのタップ操作による入力領域拡張」](https://www.interaction-ipsj.org/proceedings/2024/data/pdf/1B-45.pdf)

