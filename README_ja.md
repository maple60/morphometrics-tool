# Leaf Shape Analysis Tool

[English README here (README.md)](README.md)

napariを基盤としたGUIツールで、葉の輪郭抽出からEFD解析までを再現可能な形で一貫して実行できます。

## 環境構築

本プロジェクトは `[uv](https://docs.astral.sh/uv/)` を用いて環境を管理しています。
`uv` は `[pip](https://pip.pypa.io/en/stable/)` / `[venv](https://docs.python.org/3/library/venv.html)` / `[poetry](https://python-poetry.org/)` を統合した高速な Python パッケージマネージャです。

### 前提

- Python3 以上がインストールされていること
- uv がインストールされていること

uv のインストール方法を以下に簡単に記載いたします。 

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # curl
wget -qO- https://astral.sh/uv/install.sh | sh # wget
```

最新の情報や詳細は [公式ドキュメント](https://docs.astral.sh/uv/getting-started/installation/) をご参照ください。

### 仮想環境の作成と依存関係の同期

Windows:

```powershell
uv venv
venv\Scripts\Activate.ps1
uv sync
```

macOS / Linux:

```bash
uv venv
source .venv/bin/activate     # macOS / Linux
uv sync
```

これにより、`.venv` フォルダに仮想環境が作成され、`pyproject.toml` に記載された依存関係がインストールされます。

## 起動

```powershell
uv run python -m leaf_shape_tool.main
```

## 既知の問題

現状のバージョンにおける既知の問題を以下に記載します。

### 新しい画像を読み込んだ時などに動かなくなる

napariのバグに起因する問題で、特定の操作を行った後にGUIが応答しなくなることがあります。
この場合、napariを再起動することで解決します。

再現性が低いのですが、レイヤ消去後に新しい画像を読み込んだ時に多い印象です。

以下のようなエラーメッセージになります。

```bash
OSError: exception: access violation reading 0x0000000000000028
```

OpenGL の描画中に無効なメモリアクセスが起きて落ちているようです。
Python 側で対処するのは難しいため、napari のアップデートを待つ必要があります。
