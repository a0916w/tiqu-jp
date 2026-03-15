#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# tiqu-jp 一键部署脚本 — Ubuntu 22.04 + NVIDIA GPU
# ═══════════════════════════════════════════════════════════════
#
# 用法:
#   chmod +x setup.sh
#   ./setup.sh
#
# 适配环境:
#   - Ubuntu 22.04.5 LTS (Docker)
#   - NVIDIA RTX A5000 24GB VRAM
#   - Python 3.11.11
#   - CUDA 驱动 565.57 (无需 nvcc)
#
# ═══════════════════════════════════════════════════════════════

set -e  # 遇到错误立即退出

echo "═══════════════════════════════════════════════════"
echo "🚀 tiqu-jp 环境部署开始"
echo "═══════════════════════════════════════════════════"

# ─── 1. 系统依赖 ───────────────────────────────────────────
echo ""
echo "📦 [1/6] 安装系统依赖 (ffmpeg)..."

if command -v ffmpeg &>/dev/null; then
    echo "   ✅ ffmpeg 已安装: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "   安装 ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg
    echo "   ✅ ffmpeg 安装完成"
fi

# ─── 2. Python 检查 ────────────────────────────────────────
echo ""
echo "🐍 [2/6] 检查 Python 环境..."

PYTHON_VERSION=$(python3 --version 2>&1)
PIP_VERSION=$(pip3 --version 2>&1)
echo "   Python: $PYTHON_VERSION"
echo "   pip: $PIP_VERSION"

# ─── 3. PyTorch + CUDA ────────────────────────────────────
echo ""
echo "🔥 [3/6] 安装 PyTorch (CUDA 12.4)..."

# 检查是否已有 CUDA 版 PyTorch
CUDA_CHECK=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "missing")

if [ "$CUDA_CHECK" = "cuda" ]; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "   ✅ PyTorch $TORCH_VER 已安装 (CUDA)"
    echo "   ✅ GPU: $GPU_NAME"
else
    echo "   安装 PyTorch + CUDA 12.4..."
    pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo "   验证 CUDA..."
    python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用！'
print(f'   ✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda}')
print(f'   ✅ GPU: {torch.cuda.get_device_name(0)}')
p=torch.cuda.get_device_properties(0); vram=getattr(p,'total_memory',getattr(p,'total_mem',0))
print(f'   ✅ VRAM: {vram/(1024**3):.1f}GB')
"
fi

# ─── 4. Python 依赖 ───────────────────────────────────────
echo ""
echo "📚 [4/6] 安装 Python 依赖..."

pip3 install --no-cache-dir -r requirements.txt

# ─── 5. 预下载模型 ────────────────────────────────────────
echo ""
echo "🌐 [5/6] 预下载模型..."

# Silero VAD 模型
python3 -c "
import torch
try:
    model, _ = torch.hub.load('snakers4/silero-vad', model='silero_vad', trust_repo=True)
    print('   ✅ Silero VAD 模型已缓存')
except Exception as e:
    print(f'   ⚠️  Silero VAD 下载失败: {e}')
    print('   （非必需，但强烈推荐。可稍后手动安装。）')
"

# ─── 6. faster-whisper 验证 ──────────────────────────────
echo ""
echo "⚡ [6/6] 验证 faster-whisper 后端..."

python3 -c "
try:
    import faster_whisper
    print(f'   ✅ faster-whisper {faster_whisper.__version__} 已安装')
except ImportError:
    print('   ⚠️  faster-whisper 未安装')
    print('   请执行: pip install faster-whisper')

try:
    import stable_whisper
    # 检查 stable-ts 是否支持 faster-whisper
    if hasattr(stable_whisper, 'load_faster_whisper'):
        print('   ✅ stable-ts 支持 faster-whisper 后端')
    else:
        print('   ⚠️  stable-ts 版本过旧，请升级: pip install -U stable-ts')
except ImportError:
    print('   ❌ stable-ts 未安装')
"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ 部署完成！验证环境..."
echo "═══════════════════════════════════════════════════"
echo ""

# 最终验证
python3 -c "
import sys
import torch
import stable_whisper

print('🔍 环境验证:')
print(f'   Python:           {sys.version.split()[0]}')
print(f'   PyTorch:          {torch.__version__}')
print(f'   CUDA:             {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'   GPU:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    _p=torch.cuda.get_device_properties(0); _v=getattr(_p,'total_memory',getattr(_p,'total_mem',0))
    print(f'   VRAM:          {_v/(1024**3):.1f}GB')
else:
    print('   VRAM:          N/A')
print(f'   stable-ts:        {stable_whisper.__version__}')

try:
    import faster_whisper
    print(f'   faster-whisper:   {faster_whisper.__version__} ⚡')
except ImportError:
    print(f'   faster-whisper:   ❌ 未安装')

try:
    import transformers
    print(f'   transformers:     {transformers.__version__} (kotoba-whisper)')
except ImportError:
    print(f'   transformers:     ❌ (kotoba-whisper 需要)')

try:
    import demucs
    print(f'   demucs:           ✅')
except ImportError:
    print(f'   demucs:           ❌ 未安装')

try:
    import tqdm
    print(f'   tqdm:             ✅')
except ImportError:
    print(f'   tqdm:             ❌ (可选)')

try:
    import yaml
    print(f'   pyyaml:           ✅')
except ImportError:
    print(f'   pyyaml:           ❌ (可选)')

# Silero VAD 检查
try:
    model, _ = torch.hub.load('snakers4/silero-vad', model='silero_vad', trust_repo=True)
    print(f'   Silero VAD:       ✅')
except:
    print(f'   Silero VAD:       ⚠️ 未缓存（可选）')

print()
print('🎬 使用方法:')
print('   python3 tiqu.py video.mp4                              # 默认 (faster-whisper)')
print('   python3 tiqu.py video.mp4 --backend kotoba-whisper     # 日本語特化模型')
print('   python3 tiqu.py video.mp4 --backend whisper            # 原版 Whisper')
print('   python3 tiqu.py video.mp4 --compute-type int8_float16  # INT8 量化加速')
print('   python3 tiqu.py video.mp4 --cache-dir ./cache          # 断点续传')
print('   python3 tiqu.py video.mp4 --config tiqu-config.yaml    # 配置文件')
print('   python3 tiqu.py video.mp4 --format vtt srt --verbose   # 多格式+详细日志')
print('   python3 tiqu.py *.mp4 --output-dir ./subs              # 批量处理')
"
