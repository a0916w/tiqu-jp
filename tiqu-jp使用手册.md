# tiqu-jp 日本語字幕提取 — 使用手册

> 适配环境：Ubuntu 22.04 + NVIDIA RTX A5000 (24GB VRAM) + Python 3.11

---

## 一、部署

### 1.1 一键部署

```bash
git clone git@github.com:a0916w/tiqu-jp.git
cd tiqu-jp
chmod +x setup.sh
./setup.sh
```

脚本会自动完成：

| 步骤 | 内容 |
|------|------|
| 1 | 安装 ffmpeg |
| 2 | 检查 Python 环境 |
| 3 | 安装 PyTorch + CUDA 12.4 |
| 4 | 安装 Python 依赖（stable-ts / faster-whisper / transformers / demucs / tqdm / pyyaml） |
| 5 | 预下载 Silero VAD 模型 |
| 6 | 验证各后端 |

### 1.2 手动部署

```bash
# 1. 安装 PyTorch（CUDA 12.4）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 ffmpeg
apt-get install -y ffmpeg
```

---

## 二、三个转录后端

| 后端 | 命令 | 说明 | 推荐场景 |
|------|------|------|----------|
| **faster-whisper** | `--backend faster-whisper` | CTranslate2 加速，速度快，显存省 | **默认**，通用场景 |
| **kotoba-whisper** | `--backend kotoba-whisper` | 日本語特化蒸馏模型（kotoba-tech） | 日语精度优先 |
| **whisper** | `--backend whisper` | OpenAI 原版 Whisper | 兼容 / 对比测试 |

### 2.1 faster-whisper（默认）

```bash
python3 tiqu.py video.mp4
# 等价于
python3 tiqu.py video.mp4 --backend faster-whisper --model large-v3
```

- 基于 CTranslate2，速度 2~4 倍，显存减半
- 通过 stable-ts 封装，支持精确时间戳 + refine 微调
- 支持 `--compute-type float16 / int8_float16 / int8`

### 2.2 kotoba-whisper（日語特化）

```bash
python3 tiqu.py video.mp4 --backend kotoba-whisper
```

- 模型：`kotoba-tech/kotoba-whisper-v2.2`（HuggingFace，首次自动下载）
- 基于 distil-whisper 架构，专门针对日语微调
- 通过 transformers pipeline 调用，支持 GPU fp16
- 可指定其他模型：`--kotoba-model kotoba-tech/kotoba-whisper-v2.0`

### 2.3 whisper（原版）

```bash
python3 tiqu.py video.mp4 --backend whisper
```

- OpenAI 原版 Whisper，通过 stable-ts 封装
- 支持 fp16 加速（CUDA），refine 时间戳微调

---

## 三、基本用法

### 3.1 单文件

```bash
python3 tiqu.py video.mp4
```

### 3.2 批量处理

```bash
python3 tiqu.py *.mp4 --output-dir ./subs
```

### 3.3 指定输出格式

```bash
python3 tiqu.py video.mp4 --format vtt srt json
```

### 3.4 使用配置文件

```bash
python3 tiqu.py video.mp4 --config config_default.yaml
```

### 3.5 使用纠错词典

```bash
python3 tiqu.py video.mp4 --corrections corrections.yaml
```

### 3.6 启用断点续传

```bash
python3 tiqu.py video.mp4 --cache-dir ./cache
```

---

## 四、处理流程

```
视频文件
  │
  ▼
① ffmpeg 音频提取（16kHz mono WAV）
  │
  ▼
② Demucs 人声分离（htdemucs_ft，GPU 加速）
  │
  ▼
③ ffmpeg 语音增强（带通滤波 80-7500Hz + 动态压缩 + EBU R128 响度标准化）
  │
  ▼
③.5 Silero VAD 语音活动检测（独立标注哪里有人说话）
  │
  ▼
④ ASR 转录（faster-whisper / kotoba-whisper / whisper 三选一）
  │
  ▼
⑤ 质量检查（幻觉检测 / 重复过滤 / 日文验证 / 语气词过滤）
  │
  ▼
⑤.5 VAD 交叉验证 + 幻觉链删除 + 遗漏区域二次转录
  │
  ▼
⑥ 后处理（断句 / 智能合并 / 时间轴修正 / 纠错词典 / 回声去重）
  │
  ▼
VTT / SRT / JSON
```

---

## 五、CLI 参数速查

### 输出

| 参数 | 说明 | 默认 |
|------|------|------|
| `--output-dir` / `-o` | 输出目录 | 视频所在目录 |
| `--format` / `-f` | 输出格式 `vtt` `srt` `json` | `vtt` |

### 后端 & 模型

| 参数 | 说明 | 默认 |
|------|------|------|
| `--backend` | `faster-whisper` / `kotoba-whisper` / `whisper` | `faster-whisper` |
| `--compute-type` | `float16` / `int8_float16` / `int8` / `float32` | `float16` |
| `--kotoba-model` | kotoba-whisper HuggingFace 模型 ID | `kotoba-tech/kotoba-whisper-v2.2` |
| `--model` / `-m` | Whisper 模型大小 | `large-v3` |
| `--device` | `auto` / `cuda` / `mps` / `cpu` | `auto` |
| `--beam-size` | Beam search 宽度 | `10` |

### Demucs 人声分离

| 参数 | 说明 | 默认 |
|------|------|------|
| `--skip-demucs` | 跳过人声分离（源音频已干净时使用） | 不跳过 |
| `--demucs-model` | `htdemucs` / `htdemucs_ft` / `htdemucs_6s` / `mdx_extra` | `htdemucs_ft` |
| `--demucs-shifts` | 随机偏移次数（越高越好但越慢，GPU 推荐 2~5） | `2` |
| `--demucs-segment` | 分段长度（秒），显存不足时设 `40`~`60` | 不分段 |

### 功能开关

| 参数 | 说明 | 默认 |
|------|------|------|
| `--no-refine` | 跳过 stable-ts 时间戳微调 | 启用 |
| `--no-fp16` | 禁用 fp16（仅原版 Whisper） | 启用 |
| `--no-vad` | 跳过 Silero VAD 交叉验证 | 启用 |
| `--no-strip-fillers` | 保留语气词 | 自动去除 |
| `--no-retranscribe` | 禁用遗漏区域二次转录 | 启用 |

### 其他

| 参数 | 说明 | 默认 |
|------|------|------|
| `--corrections` | ASR 纠错词典 YAML 路径 | 无 |
| `--cache-dir` | 缓存目录（启用断点续传） | 无 |
| `--config` / `-c` | YAML 配置文件路径 | 无 |
| `--keep-temp` | 保留临时文件（调试用） | 不保留 |
| `--verbose` / `-v` | 详细日志 | 关闭 |

> **优先级**：CLI 参数 > YAML 配置文件 > 代码默认值

---

## 六、配置文件 (YAML)

默认配置文件 `config_default.yaml`：

```yaml
# --- 基础设置 ---
language: "ja"
model_name: "large-v3"
backend: "faster-whisper"           # faster-whisper | kotoba-whisper | whisper
compute_type: "float16"             # faster-whisper 专用
fp16: true
kotoba_model: "kotoba-tech/kotoba-whisper-v2.2"  # kotoba-whisper 专用

# --- Whisper 转录参数 ---
beam_size: 10
condition_on_previous_text: false   # 防止错误传播跳段
word_timestamps: true
no_speech_threshold: 0.6
compression_ratio_threshold: 2.8    # 日文比英文高（英文默认 2.4）
logprob_threshold: -1.0
max_chars_per_second: 15.0
temperature: [0.0, 0.2, 0.4, 0.6]  # 去掉 0.8/1.0，高温几乎全是幻觉
suppress_silence: true
vad_onset: 0.35
vad_offset: 0.2
initial_prompt: "ああ、そうなんですか。へえ、それはすごいですね。うん、分かった。じゃあ、やってみよう。"
# ★ initial_prompt 只能写纯对话，绝不能写指令性语句，否则 Whisper 会原样输出到字幕里

# --- Demucs 人声分离 ---
demucs_shifts: 2
demucs_segment: null      # 显存不够时设 40~60
demucs_overlap: 0.25

# --- Silero VAD ---
vad_filter: true
vad_threshold: 0.5
vad_min_speech_ms: 250
vad_min_silence_ms: 100

# --- 遗漏区域二次转录 ---
retranscribe_missed: true
retranscribe_beam_size: 8
retranscribe_min_duration: 0.5

# --- 后处理 ---
min_segment_duration: 0.3
max_segment_duration: 8.0
min_display_duration: 0.5
strip_filler_words: true

# --- 输出 ---
output_formats:
  - vtt
  - srt
  - json
```

---

## 七、ASR 纠错词典

文件格式（YAML）：

```yaml
corrections:
  # 格式：ASR误识别 → 正确文本（全局替换）
  
  # 角色名
  かりちゃん: あかりちゃん
  アカリちゃん: あかりちゃん

  # 专有名词
  ハンマーユージロ: 範馬勇次郎
  ハンマー友人: 範馬勇次郎

  # 听错的词
  市場子女: 淑女
  アメリカのタイプ: あかりちゃんのタイプ
```

用法：

```bash
# 建议流程：先不带词典跑一遍，检查字幕中反复出现的误识别，再编辑词典重新跑
python3 tiqu.py video.mp4 --corrections corrections.yaml
```

---

## 八、常用场景

### 8.1 最高质量 — faster-whisper（推荐）

```bash
python3 tiqu.py video.mp4 \
  --config config_default.yaml \
  --corrections corrections.yaml \
  --format vtt srt \
  --cache-dir ./cache \
  --verbose
```

### 8.2 日語特化 — kotoba-whisper

```bash
python3 tiqu.py video.mp4 \
  --backend kotoba-whisper \
  --corrections corrections.yaml \
  --format vtt srt \
  --verbose
```

### 8.3 对比三个后端

```bash
# 先跑三次（用缓存复用前面的步骤）
python3 tiqu.py video.mp4 --backend faster-whisper --cache-dir ./cache -o ./out-fw
python3 tiqu.py video.mp4 --backend kotoba-whisper --cache-dir ./cache -o ./out-kotoba
python3 tiqu.py video.mp4 --backend whisper        --cache-dir ./cache -o ./out-whisper
# 然后比较三份字幕的覆盖率报告和实际效果
```

### 8.4 快速出稿（牺牲一点精度换速度）

```bash
python3 tiqu.py video.mp4 \
  --compute-type int8_float16 \
  --beam-size 5 \
  --no-retranscribe \
  --no-refine
```

### 8.5 源音频很干净（无 BGM）

```bash
python3 tiqu.py video.mp4 --skip-demucs
```

### 8.6 调试模式

```bash
python3 tiqu.py video.mp4 --keep-temp --verbose
# 会保留中间文件：raw_audio.wav / vocals.wav / enhanced_audio.wav
# 可逐一检查每步的音频质量
```

### 8.7 显存不足（< 8GB VRAM）

```bash
python3 tiqu.py video.mp4 \
  --demucs-segment 40 \
  --compute-type int8 \
  --model medium
```

---

## 九、输出说明

### 覆盖率报告

每次处理完成后会输出覆盖率报告：

```
📊 字幕覆盖率报告：
• 音频总时长：600.9s（10.0min）
• VAD 语音时长：471.5s（78.5%）
• 字幕总时长：435.7s（72.5%）
• 字幕段数：155 段

✅ 精确度（Precision）：89.1% ← 字幕≈真实语音
✅ 召回率（Recall）：82.4% ← 语音被字幕覆盖
```

| 指标 | 含义 | 正常范围 |
|------|------|----------|
| 精确度 | 字幕段与真实语音的重叠率 | > 85% |
| 召回率 | VAD 语音被字幕覆盖的比例 | > 80% |

### 语气词处理

| 自动删除 | 保留 |
|----------|------|
| あ、あっ、あー | うん（是） |
| え、えっ、えー | ええ（是） |
| お、おっ、おー | はい（是） |
| ん、んー | ううん（不） |
| えっと、うーん | いいえ（不） |
| まあ、ふーん | いや（不） |
| はぁ、そうそうそう | そう（对） |
| はいはいはい | あの（那个） |

### 幻觉过滤

自动过滤以下类型：

- **Prompt 回吐**：`日本語の会話を正確に書き起こします` 等指令性文本
- **YouTube 幻觉**：`ご視聴ありがとうございました`、`チャンネル登録`
- **重复串**：同一短语重复 4+ 次
- **非日文**：纯英文段、纯汉字段（中文幻觉）
- **静默幻觉链**：连续 2+ 段 VAD 语音重叠 < 5% → 整条链删除

---

## 十、依赖清单

| 包 | 用途 | 版本 |
|----|------|------|
| torch + torchaudio | 深度学习基础 | ≥ 2.1 |
| stable-ts | Whisper 精确时间戳 | ≥ 2.16 |
| faster-whisper | CTranslate2 加速转录 | ≥ 1.0 |
| openai-whisper | Whisper 原版（备用后端） | ≥ 20231117 |
| transformers + accelerate | kotoba-whisper 后端 | ≥ 4.39 |
| demucs | 人声分离 | ≥ 4.0.1 |
| tqdm | 进度条 | ≥ 4.60 |
| pyyaml | YAML 配置支持 | ≥ 6.0 |
| ffmpeg | 音频提取 + 增强 | 系统安装 |
| Silero VAD | 语音活动检测 | torch.hub 自动下载 |

---

## 十一、文件结构

```
tiqu-jp/
├── tiqu.py                  # 主程序
├── config_default.yaml      # 默认配置
├── corrections_example.yaml # 纠错词典示例
├── requirements.txt         # Python 依赖
├── setup.sh                 # 一键部署脚本
├── tiqu-jp使用手册.md        # 本文档
└── .gitignore
```
