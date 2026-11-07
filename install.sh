#!/bin/sh
# Vellum installer · 依赖检查 → 编译 OCR 引擎 → 链接命令 → 访达快速操作
#   sh install.sh               安装 / 升级
#   sh install.sh --uninstall   卸载
# 只写用户目录（~/.local/bin、~/Library/Services），不需要 sudo。
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$HOME/.local/bin/vellum"
SERVICES="$HOME/Library/Services"
QA_REVIEW="Vellum · 扫描并调整"
QA_SCAN="Vellum · 扫描并导出 PDF"
# 旧版本（docscan 1.0 / 1.1）的名字，升级时顺手清掉
OLD_QA="本地扫描为 PDF|本地扫描（选模式）|DocScan · 扫描为 PDF|DocScan · 选模式扫描|DocScan · 提取文字|Vellum · 扫描为 PDF|Vellum · 选模式扫描|Vellum · 提取文字|Vellum · 检查并调整|Vellum · OCR 识别并复制"

remove_actions() {
  for n in "$QA_REVIEW" "$QA_SCAN"; do rm -rf "$SERVICES/$n.workflow"; done
  echo "$OLD_QA" | tr '|' '\n' | while read -r n; do rm -rf "$SERVICES/$n.workflow"; done
}

if [ "$1" = "--uninstall" ]; then
  rm -f "$BIN" "$HOME/.local/bin/docscan"
  remove_actions
  /System/Library/CoreServices/pbs -update 2>/dev/null || true
  echo "✓ Vellum 已卸载（日志 ~/Library/Logs/vellum.log 保留）"
  exit 0
fi

echo "▸ [1/5] Python 依赖"
if ! /usr/bin/python3 -c "import cv2, numpy, PIL, fontTools, scipy" 2>/dev/null; then
  echo "        安装 opencv-python-headless numpy pillow fonttools（用户级）"
  /usr/bin/python3 -m pip install --user -q opencv-python-headless numpy pillow fonttools scipy
fi
/usr/bin/python3 -c "import cv2, numpy, PIL, fontTools; print('        OpenCV', cv2.__version__, '· NumPy', numpy.__version__, '· Pillow', PIL.__version__, '· fontTools', fontTools.version)"

echo "▸ [2/5] OCR 引擎（macOS Vision）"
if command -v swiftc >/dev/null 2>&1; then
  if [ ! -x "$DIR/bin/vellum-ocr" ] || [ "$DIR/tools/ocr.swift" -nt "$DIR/bin/vellum-ocr" ]; then
    echo "        swiftc 编译中（约 1 分钟）…"
    swiftc -O -swift-version 5 "$DIR/tools/ocr.swift" -o "$DIR/bin/vellum-ocr"
  fi
  echo "        ✓ $DIR/bin/vellum-ocr"
else
  echo "        ⚠ 没有 swiftc，OCR 不可用。装 Xcode 命令行工具后重跑：xcode-select --install"
fi

echo "▸ [3/5] 检查窗口 Vellum.app（SwiftUI · Liquid Glass）"
if command -v swiftc >/dev/null 2>&1; then
  APP=$(/usr/bin/python3 -c "import sys; sys.path.insert(0, '$DIR'); import vellum; print(vellum.review_app() or '')" 2>&1 | tail -1)
  echo "        ✓ $APP"
fi

echo "▸ [4/5] 命令 → $BIN（docscan 保留为别名）"
chmod +x "$DIR/bin/vellum" "$DIR/vellum.py"
mkdir -p "$HOME/.local/bin"
ln -sf "$DIR/bin/vellum" "$BIN"
ln -sf "$DIR/bin/vellum" "$HOME/.local/bin/docscan"

echo "▸ [5/5] 访达快速操作"
mkdir -p "$SERVICES"
remove_actions
/usr/bin/python3 "$DIR/tools/make_quick_actions.py" "$SERVICES" "$BIN" "$QA_REVIEW" "$QA_SCAN"
/System/Library/CoreServices/pbs -update 2>/dev/null || true

cat <<MSG
✓ Vellum $(/usr/bin/python3 "$DIR/vellum.py" --version | cut -d' ' -f2) 就绪

  访达   选中照片 → 右键 → 快速操作 → $QA_REVIEW（推荐）/ $QA_SCAN
  终端   vellum 照片.jpg            vellum -h 看全部参数
  日志   ~/Library/Logs/vellum.log
MSG
