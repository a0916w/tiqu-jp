#!/usr/bin/env python3
"""
tiqu.py — 日本語字幕提取工具
视频 → ffmpeg音频提取 → Demucs人声分离 → 语音增强 → stable-ts転録 → 質量検査 → VTT/SRT出力

后端支持:
  - faster-whisper (默认) — CTranslate2 加速，速度 2~4 倍，显存减半
  - whisper (原版)        — OpenAI 原版 Whisper

特性:
  - Silero VAD 交叉验证（过滤幻觉 + 检测遗漏 + 二次转录）
  - 日本語特化（语气词去除、幻觉过滤、标点规范化）
  - YAML 配置文件支持
  - 断点续传 / 缓存中间结果
  - 容错降级（各步骤失败自动降级处理）
  - 批量处理进度条
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── 可选依赖（软导入，缺失时自动降级） ──────────────────────────
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _progress(iterable, **kwargs):
    """tqdm 进度条封装，未安装 tqdm 时透明回退"""
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """全局配置参数"""

    # --- 转录后端 ---
    backend: str = "faster-whisper"   # "faster-whisper" (推荐) | "whisper" (原版)
    compute_type: str = "float16"     # faster-whisper 精度: float16 / int8_float16 / int8 / float32

    # --- Whisper / stable-ts ---
    model: str = "large-v3"
    language: str = "ja"
    beam_size: int = 10
    suppress_silence: bool = True
    condition_on_previous_text: bool = False  # ★ False 防止错误传播导致整段跳过，大幅提升召回率
    word_timestamps: bool = True
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    no_speech_threshold: float = 0.7          # 放宽：保留更多边界段（0.6→0.7）
    compression_ratio_threshold: float = 2.8   # 日文重复模式多，放宽（英文默认 2.4）
    logprob_threshold: float = -1.0
    temperature: tuple = (0.0, 0.2, 0.4, 0.6)  # 去掉 0.8/1.0 — 高温产出几乎全是幻觉
    initial_prompt: str = (
        # ★ 日本語特化 prompt — 对话风格引导 Whisper 进入"日语会话"模式
        # 包含语气词（えっと/そうですね）和常见敬体，减少中文/英文幻觉
        "こんにちは、よろしくお願いします。"
        "えっと、そうですね、それでは始めましょう。"
        "日本語の会話を正確に書き起こします。"
        "句読点は「、」「。」を使い、「？」「！」も適宜使います。"
    )
    refine_timestamps: bool = True  # 是否使用 stable-ts refine 微调时间戳
    fp16: bool = True               # 仅原版 Whisper 使用，faster-whisper 用 compute_type

    # --- 遗漏区域二次转录 ---
    retranscribe_missed: bool = True    # 对 VAD 检出但 Whisper 遗漏的区域自动二次转录
    retranscribe_beam_size: int = 8     # 二次转录 beam（5→8，更彻底，显存充裕时值得）
    retranscribe_min_duration: float = 0.5  # 只重新转录时长 > 此值的遗漏区域（0.8→0.5 捕捉短遗漏）

    # --- Demucs ---
    demucs_model: str = "htdemucs_ft"
    skip_demucs: bool = False
    demucs_shifts: int = 1          # 随机偏移次数（越高质量越好，速度越慢。GPU 充裕可设 2~5）
    demucs_overlap: float = 0.25    # 分段处理的重叠比例
    demucs_segment: Optional[int] = None  # 分段长度（秒），None=不分段。显存不够时设 40~60

    # --- ffmpeg 语音增强 ---
    highpass_freq: int = 80          # Hz — 去除低频隆隆声，保留男声基频
    lowpass_freq: int = 7500         # Hz — 保留齿擦音，去除高频噪音
    target_lufs: int = -16           # EBU R128 标准
    sample_rate: int = 16000         # Whisper 要求 16kHz
    demucs_sample_rate: int = 44100  # Demucs 训练时的采样率

    # --- 质量检查（日本語特化） ---
    # 日本語は1文字あたりの情報量が多い（漢字+假名混合）
    # 一般会話: 5~10 文字/秒, アナウンサー: ~13 文字/秒, 上限 15 で幻覚を検出
    max_chars_per_second: float = 15.0   # 日文每秒最大字符数（超过此值基本是幻觉）
    min_segment_duration: float = 0.3    # 最短段时长 (秒)
    max_segment_duration: float = 8.0    # 最长段时长 (秒)
    min_display_duration: float = 0.5    # 最短显示保底 (秒)
    merge_gap_threshold: float = 0.3     # 相邻段间隔 < 此值则合并（如果合并后不超长）
    repeat_threshold: int = 2            # 连续重复 N 段标记为幻觉
    hallucination_patterns: list = field(default_factory=lambda: [
        # ── 通用幻觉 ──
        r"(.{2,})\1{3,}",                   # 重复字符串 ≥4 次（如「すすすす」）
        r"[♪♫🎵🎶♩♬]+",                      # 音乐符号
        r"^[\s　]*$",                         # 空白段
        # ── 日本語 Whisper 典型幻觉 ──
        r"ご視聴ありがとうございました",        # YouTube 结尾
        r"チャンネル登録",                     # YouTube 幻觉
        r"お疲れ様でした。$",                  # 尾部幻觉
        r"次回もお楽しみに",                   # 预告幻觉
        r"ご覧いただきありがとうございます",     # 感谢观看
        r"最後までご覧",                       # 最后观看
        r"お見逃しなく",                       # 请勿错过
        r"字幕[:：・]",                        # 字幕归属文本
        r"翻訳[:：・]",                        # 翻译归属文本
        r"提供[:：・]",                        # 赞助归属
        r"^(ん|え|あ|う|お|は){5,}$",          # 无意义语气词重复
        r"^[\(（【].+[\)）】]$",               # 纯注释段「(笑)」「（拍手）」
        r"^[a-zA-Z0-9\s\.\,\!\?\-]+$",       # 纯英文/数字段（非日语内容）
        r"^[\u4e00-\u9fff]+$",                # 纯汉字无假名（可能是中文幻觉）
    ])

    # --- 日语语气词（フィラー）处理 ---
    strip_fillers: bool = True          # 自动去除 あー/えっと/うーん 等无意义语气词

    # --- Silero VAD 交叉验证 ---
    vad_filter: bool = True             # 启用 Silero VAD 独立验证
    vad_min_speech_ratio: float = 0.15  # 段内至少 15% 时间有语音才保留（0.3→0.15 减少误删）
    vad_threshold: float = 0.5          # VAD 语音检测灵敏度（0~1，越低越灵敏）
    vad_min_speech_ms: int = 250        # VAD 最短语音段 (ms)
    vad_min_silence_ms: int = 100       # VAD 最短静音段 (ms)

    # --- 输出 ---
    output_formats: list = field(default_factory=lambda: ["vtt"])
    keep_temp: bool = False

    # --- 设备 ---
    device: str = "auto"      # auto / cuda / mps / cpu（用于 Whisper）
    demucs_device: str = ""   # 留空 = 跟随 auto 检测结果（Demucs 可以用 MPS）

    # --- 缓存（断点续传） ---
    cache_dir: Optional[str] = None  # 设置后启用缓存，跑过的步骤不再重复


# ═══════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("tiqu")


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

@contextmanager
def timed_step(step_name: str):
    """计时上下文管理器，统一日志输出"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"   ✅ 完成 ({elapsed:.1f}s)")


def detect_devices() -> tuple[str, str]:
    """自动检测最佳计算设备，分别返回 (whisper_device, demucs_device)

    Whisper (stable-ts) 的 DTW 对齐不兼容 MPS float64，必须用 CPU。
    Demucs 完全兼容 MPS，可以加速 2~3 倍。
    """
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
        logger.info(f"✅ 检测到 CUDA GPU: {name} ({vram:.1f}GB VRAM)")
        # 显示 CUDA 版本信息
        logger.info(f"   CUDA 版本: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        return "cuda", "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("✅ 检测到 Apple Silicon — Whisper 用 CPU（DTW 不兼容 MPS），Demucs 用 MPS 加速")
        return "cpu", "mps"
    else:
        logger.info("⚠️  未检测到 GPU，将使用 CPU")
        return "cpu", "cpu"


def run_ffmpeg(args: list[str], desc: str = "") -> subprocess.CompletedProcess:
    """执行 ffmpeg 命令"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    logger.debug(f"执行: ffmpeg {' '.join(args)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg 失败 [{desc}]: {result.stderr}")
        raise RuntimeError(f"ffmpeg 执行失败: {desc}\n{result.stderr}")
    return result


def format_time_vtt(seconds: float) -> str:
    """秒 → VTT 时间格式 HH:MM:SS.mmm"""
    seconds = max(0.0, seconds)  # 防止负数
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_time_srt(seconds: float) -> str:
    """秒 → SRT 时间格式 HH:MM:SS,mmm"""
    return format_time_vtt(seconds).replace(".", ",")


def _compute_cache_key(video_path: str, config: Config) -> str:
    """根据视频文件 + 关键配置参数计算缓存键

    配置变化（如换模型/后端）会生成新的缓存目录，避免旧缓存污染。
    """
    stat = os.stat(video_path)
    key_parts = [
        os.path.abspath(video_path),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        config.demucs_model,
        str(config.demucs_shifts),
        str(config.skip_demucs),
        config.model,
        config.backend,
        config.compute_type,
        str(config.beam_size),
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════
# Step 1: ffmpeg 音频提取
# ═══════════════════════════════════════════════════════════════════

def extract_audio(video_path: str, output_path: str, config: Config,
                  target_sr: Optional[int] = None) -> str:
    """从视频中提取音频为 WAV

    Args:
        target_sr: 目标采样率。None = 使用 config.sample_rate (16kHz)
    """
    sr = target_sr or config.sample_rate
    logger.info(f"📼 Step 1 — 提取音频 ({sr}Hz)...")

    with timed_step("音频提取"):
        run_ffmpeg([
            "-i", video_path,
            "-vn",                    # 不要视频
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", str(sr),
            "-ac", "1",               # 单声道
            output_path,
        ], "音频提取")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"   文件大小: {size_mb:.1f}MB")
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Step 2: Demucs 人声分离
# ═══════════════════════════════════════════════════════════════════

def separate_vocals(audio_path: str, output_path: str, config: Config) -> str:
    """使用 Demucs Python API 分离人声（避免子进程开销，支持 CUDA/MPS 加速）"""
    if config.skip_demucs:
        logger.info("⏭️  Step 2 — 跳过人声分离（--skip-demucs）")
        return audio_path

    logger.info(f"🎤 Step 2 — Demucs 人声分离 (模型: {config.demucs_model}, 设备: {config.demucs_device})...")

    with timed_step("Demucs 人声分离"):
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        # 加载模型
        device = torch.device(config.demucs_device)
        model = get_model(config.demucs_model)
        model.to(device)

        # CUDA: 显示显存占用
        if config.demucs_device == "cuda":
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"   Demucs 模型已加载，显存占用: {mem_mb:.0f}MB")

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)

        # Demucs 需要特定采样率 (通常 44100Hz)
        if sr != model.samplerate:
            logger.info(f"   重采样: {sr}Hz → {model.samplerate}Hz")
            waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)

        # 确保是双声道（Demucs 期望双声道输入）
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # 添加 batch 维度: (channels, samples) → (1, channels, samples)
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        waveform = waveform.unsqueeze(0).to(device)

        # 分离（传入 GPU 优化参数）
        apply_kwargs = dict(
            progress=True,
            device=device,
            shifts=config.demucs_shifts,
            overlap=config.demucs_overlap,
        )
        # 可选：分段处理（长音频或显存不足时）
        if config.demucs_segment is not None:
            apply_kwargs["segment"] = config.demucs_segment
            logger.info(f"   分段处理: segment={config.demucs_segment}s, overlap={config.demucs_overlap}")
        if config.demucs_shifts > 1:
            logger.info(f"   使用 {config.demucs_shifts} 次随机偏移增强质量")

        logger.info("   分离中...")
        with torch.no_grad():
            sources = apply_model(model, waveform, **apply_kwargs)

        # 提取人声轨道
        # sources shape: (1, num_sources, channels, samples)
        vocal_idx = model.sources.index("vocals")
        vocals = sources[0, vocal_idx]  # (channels, samples)

        # 反标准化
        vocals = vocals * ref.std() + ref.mean()

        # 取单声道
        vocals = vocals.mean(dim=0, keepdim=True).cpu()

        # 保存
        torchaudio.save(output_path, vocals, model.samplerate)

        # ★ 释放 Demucs 模型和显存，给后续 Whisper 腾出空间
        del model, sources, waveform
        if config.demucs_device == "cuda":
            torch.cuda.empty_cache()
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"   Demucs 模型已释放，剩余显存占用: {mem_mb:.0f}MB")

    return output_path


# ═══════════════════════════════════════════════════════════════════
# Step 3: ffmpeg 语音增强 + 降采样至 16kHz
# ═══════════════════════════════════════════════════════════════════

def enhance_audio(input_path: str, output_path: str, config: Config) -> str:
    """带通滤波 + 动态压缩 + 响度标准化 + 降采样至 16kHz

    Demucs 输出为 44.1kHz，这里同时做增强和降采样，一步到位给 Whisper。
    """
    logger.info("🔊 Step 3 — 语音增强 + 降采样至 16kHz...")

    with timed_step("语音增强"):
        # 构建 ffmpeg 音频滤镜链
        filters = [
            # 高通滤波 — 去除低频隆隆声
            f"highpass=f={config.highpass_freq}:poles=2",
            # 低通滤波 — 去除高频噪音（保留齿擦音）
            f"lowpass=f={config.lowpass_freq}:poles=2",
            # 动态压缩 — 缩小动态范围，让轻声更清晰
            "compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-45|-27/-25|0/-10:gain=5",
            # EBU R128 响度标准化
            f"loudnorm=I={config.target_lufs}:TP=-1.5:LRA=11",
        ]
        filter_chain = ",".join(filters)

        run_ffmpeg([
            "-i", input_path,
            "-af", filter_chain,
            "-ar", str(config.sample_rate),  # 降采样至 16kHz
            "-ac", "1",
            "-acodec", "pcm_s16le",
            output_path,
        ], "语音增强")

    return output_path


# ═══════════════════════════════════════════════════════════════════
# Silero VAD — 独立语音活动检测（与 Whisper 交叉验证）
# ═══════════════════════════════════════════════════════════════════

_vad_model = None


def _load_vad_model():
    """加载 Silero VAD 模型（自动缓存，~2MB）"""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    import torch
    try:
        _vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
        )
        logger.info("   ✅ Silero VAD 模型已加载")
        return _vad_model
    except Exception as e:
        logger.warning(f"⚠️  Silero VAD 加载失败: {e}")
        logger.warning("   将跳过 VAD 验证。可能需要网络下载模型（首次约 2MB）。")
        return None


def get_speech_regions(audio_path: str, config: Config) -> list[tuple[float, float]]:
    """使用 Silero VAD 检测音频中的所有语音区域

    Returns:
        list of (start_sec, end_sec) — 每个元素代表一段连续语音
    """
    import torch
    import torchaudio

    model = _load_vad_model()
    if model is None:
        return []

    # 加载 16kHz 音频
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    # 确保单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # (samples,)

    # Silero VAD 要求 16kHz，返回 sample 级别的时间戳
    speech_timestamps = _get_speech_timestamps_safe(
        wav, model,
        sampling_rate=16000,
        threshold=config.vad_threshold,
        min_speech_duration_ms=config.vad_min_speech_ms,
        min_silence_duration_ms=config.vad_min_silence_ms,
    )

    regions = [
        (ts['start'] / 16000.0, ts['end'] / 16000.0)
        for ts in speech_timestamps
    ]

    total_speech = sum(e - s for s, e in regions)
    audio_dur = len(wav) / 16000.0
    logger.info(f"   VAD 检测到 {len(regions)} 个语音区域，"
                f"语音 {total_speech:.1f}s / 总时长 {audio_dur:.1f}s "
                f"({total_speech / max(audio_dur, 0.01) * 100:.1f}%)")

    return regions


def _get_speech_timestamps_safe(wav, model, **kwargs) -> list[dict]:
    """安全调用 Silero VAD 的 get_speech_timestamps"""
    try:
        # Silero VAD 新版本内置方法
        from torch.hub import load as _hub_load
        # 使用 model 自带的工具函数
        import torch
        # 尝试新 API (silero-vad >= 5.0)
        if hasattr(model, 'get_speech_timestamps'):
            return model.get_speech_timestamps(wav, **kwargs)
        # 旧 API — 从 utils 中获取
        _, utils = torch.hub.load(
            'snakers4/silero-vad', model='silero_vad', trust_repo=True
        )
        get_speech_timestamps = utils[0]
        return get_speech_timestamps(wav, model, **kwargs)
    except Exception as e:
        logger.warning(f"   VAD 检测失败: {e}")
        return []


def compute_speech_overlap(seg_start: float, seg_end: float,
                           speech_regions: list[tuple[float, float]]) -> float:
    """计算一个时间段与语音区域列表的重叠总时长"""
    overlap = 0.0
    for vad_start, vad_end in speech_regions:
        o_start = max(seg_start, vad_start)
        o_end = min(seg_end, vad_end)
        if o_end > o_start:
            overlap += o_end - o_start
    return overlap


def vad_validate(segments: list, speech_regions: list[tuple[float, float]],
                 config: Config) -> list:
    """用 Silero VAD 结果交叉验证 Whisper 字幕段

    双重保障：
    1. 过滤幻觉 — Whisper 输出字幕但 VAD 认为无语音 → 删除
    2. 检测遗漏 — VAD 检测到语音但 Whisper 没有对应字幕 → 警告
    """
    if not speech_regions:
        logger.info("   ⏭️ 无 VAD 数据，跳过交叉验证")
        return segments

    logger.info("🔍 VAD 交叉验证字幕段...")

    validated = []
    vad_filtered_count = 0
    vad_filtered_examples = []

    for seg in segments:
        duration = seg.end - seg.start
        if duration <= 0:
            continue

        overlap = compute_speech_overlap(seg.start, seg.end, speech_regions)
        ratio = overlap / duration

        if ratio >= config.vad_min_speech_ratio:
            validated.append(seg)
        else:
            vad_filtered_count += 1
            if len(vad_filtered_examples) < 5:
                vad_filtered_examples.append(
                    f"     {format_time_vtt(seg.start)}→{format_time_vtt(seg.end)} "
                    f"[speech={ratio:.0%}] {seg.text[:40]}"
                )

    if vad_filtered_count:
        logger.info(f"   🗑️  VAD 过滤了 {vad_filtered_count} 个无语音段（幻觉）")
        for ex in vad_filtered_examples:
            logger.info(ex)

    # === 遗漏检测 ===
    missed = detect_missed_speech(validated, speech_regions)
    if missed:
        logger.warning(f"   ⚠️  检测到 {len(missed)} 个可能遗漏的语音区域：")
        for start, end in missed[:10]:
            logger.warning(
                f"      {format_time_vtt(start)} → {format_time_vtt(end)} "
                f"({end - start:.1f}s) — 无对应字幕！"
            )
        if len(missed) > 10:
            logger.warning(f"      ... 还有 {len(missed) - 10} 个")
    else:
        logger.info("   ✅ 未检测到遗漏的语音区域")

    logger.info(f"   验证后保留: {len(validated)}/{len(segments)} 段")
    return validated


def detect_missed_speech(segments: list, speech_regions: list[tuple[float, float]],
                         min_duration: float = 0.8) -> list[tuple[float, float]]:
    """检测 VAD 有语音但 Whisper 没有字幕覆盖的区域

    Args:
        min_duration: 只报告超过此时长的遗漏（避免短语气词噪音）
    """
    missed = []

    for vad_start, vad_end in speech_regions:
        vad_duration = vad_end - vad_start
        if vad_duration < min_duration:
            continue

        # 计算该语音区域被字幕覆盖的比例
        coverage = 0.0
        for seg in segments:
            o_start = max(vad_start, seg.start)
            o_end = min(vad_end, seg.end)
            if o_end > o_start:
                coverage += o_end - o_start

        coverage_ratio = coverage / vad_duration
        if coverage_ratio < 0.5:  # 不到 50% 被覆盖 → 可能遗漏（0.3→0.5 更敏感）
            missed.append((vad_start, vad_end))

    return missed


def get_audio_duration(audio_path: str) -> float:
    """获取音频文件时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def print_coverage_report(segments: list, speech_regions: list[tuple[float, float]],
                          audio_duration: float):
    """输出字幕覆盖率报告 — Precision & Recall"""
    if not segments or audio_duration <= 0:
        return

    total_subtitle = sum(s.end - s.start for s in segments)
    total_speech = sum(e - s for s, e in speech_regions) if speech_regions else 0

    logger.info("")
    logger.info("═" * 55)
    logger.info("📊 字幕覆盖率报告")
    logger.info(f"   音频总时长:       {audio_duration:.1f}s ({audio_duration / 60:.1f}min)")

    if speech_regions:
        logger.info(f"   VAD 语音时长:     {total_speech:.1f}s "
                     f"({total_speech / audio_duration * 100:.1f}%)")

    logger.info(f"   字幕总时长:       {total_subtitle:.1f}s "
                 f"({total_subtitle / audio_duration * 100:.1f}%)")
    logger.info(f"   字幕段数:         {len(segments)}")

    if total_speech > 0 and speech_regions:
        # Precision: 字幕时间中，有多少落在真实语音区域
        subtitle_in_speech = 0.0
        for seg in segments:
            subtitle_in_speech += compute_speech_overlap(
                seg.start, seg.end, speech_regions
            )
        precision = subtitle_in_speech / max(total_subtitle, 0.01)

        # Recall: 真实语音时间中，有多少被字幕覆盖
        speech_covered = 0.0
        for vad_s, vad_e in speech_regions:
            for seg in segments:
                o_s = max(vad_s, seg.start)
                o_e = min(vad_e, seg.end)
                if o_e > o_s:
                    speech_covered += o_e - o_s
        recall = speech_covered / max(total_speech, 0.01)

        logger.info(f"   ─────────────────────────────────────")
        logger.info(f"   精确度 (Precision): {precision:.1%}  ← 字幕≈真实语音")
        logger.info(f"   召回率 (Recall):    {recall:.1%}  ← 语音被字幕覆盖")

        # 质量评级
        if recall >= 0.95 and precision >= 0.85:
            logger.info(f"   评级: 🟢 优秀")
        elif recall >= 0.85 and precision >= 0.70:
            logger.info(f"   评级: 🟡 良好")
        else:
            logger.info(f"   评级: 🔴 需检查")

        if recall < 0.85:
            logger.warning(f"   ⚠️  召回率偏低！有语音未被提取。")
            logger.warning(f"      建议：1) 使用 --keep-temp 检查中间音频  "
                           f"2) 确认 Demucs 分离质量  3) 尝试 --backend whisper 对比")
        if precision < 0.70:
            logger.warning(f"   ⚠️  精确度偏低！可能有幻觉字幕。")
            logger.warning(f"      建议：1) 确认 Demucs 分离质量  2) 检查 BGM 泄漏")

    logger.info("═" * 55)


# ═══════════════════════════════════════════════════════════════════
# Step 4: stable-ts 转录（支持 faster-whisper / 原版 Whisper）
# ═══════════════════════════════════════════════════════════════════

# 全局模型缓存，避免批量处理时重复加载
_whisper_model = None
_whisper_model_key = None
_whisper_backend = None    # 实际使用的后端（可能因加载失败而回退）


def _get_whisper_model(config: Config):
    """获取或缓存 Whisper 模型（批量处理只加载一次）

    支持两种后端：
    - faster-whisper: CTranslate2 加速，速度 2~4×，显存减半
    - whisper: OpenAI 原版 Whisper
    """
    global _whisper_model, _whisper_model_key, _whisper_backend
    import stable_whisper

    model_key = f"{config.backend}:{config.model}:{config.compute_type}"
    if _whisper_model is not None and _whisper_model_key == model_key:
        logger.info(f"   使用已缓存的模型 ({_whisper_backend})")
        return _whisper_model

    if config.backend == "faster-whisper":
        # ★ faster-whisper 后端：CTranslate2 加速
        compute_type = config.compute_type
        # CPU 不支持 float16，自动回退
        if config.device not in ("cuda",):
            if compute_type == "float16":
                compute_type = "int8"
                logger.info(f"   设备 {config.device} 不支持 float16，回退到 {compute_type}")

        logger.info(f"   加载 faster-whisper {config.model} (compute_type={compute_type})...")
        try:
            _whisper_model = stable_whisper.load_faster_whisper(
                config.model, device=config.device, compute_type=compute_type
            )
            _whisper_backend = "faster-whisper"
            _whisper_model_key = model_key
            logger.info(f"   ✅ faster-whisper 模型已加载")
            return _whisper_model
        except Exception as e:
            logger.warning(f"   ⚠️ faster-whisper 加载失败: {e}")
            logger.warning(f"   自动回退到原版 Whisper...")
            # 继续走 whisper 分支

    # 原版 Whisper 后端
    logger.info(f"   加载 Whisper {config.model}...")
    _whisper_model = stable_whisper.load_model(config.model, device=config.device)
    _whisper_backend = "whisper"
    _whisper_model_key = model_key
    return _whisper_model


def transcribe_audio(audio_path: str, config: Config):
    """使用 stable-ts (Whisper / faster-whisper) 进行日语转录"""
    # 确定精度标签
    if _whisper_backend == "faster-whisper" or config.backend == "faster-whisper":
        fp_label = f"compute_type={config.compute_type}"
    else:
        use_fp16 = config.fp16 and config.device == "cuda"
        fp_label = "fp16" if use_fp16 else "fp32"

    logger.info(f"📝 Step 4 — 转录 (后端: {config.backend}, 模型: {config.model}, "
                f"设备: {config.device}, {fp_label})...")

    with timed_step("转录"):
        model = _get_whisper_model(config)
        actual_backend = _whisper_backend or config.backend

        if actual_backend == "faster-whisper":
            # ═══ faster-whisper 路径 ═══
            transcribe_kwargs = dict(
                audio=audio_path,
                language=config.language,
                beam_size=config.beam_size,
                condition_on_previous_text=config.condition_on_previous_text,
                word_timestamps=config.word_timestamps,
                initial_prompt=config.initial_prompt,
                no_speech_threshold=config.no_speech_threshold,
                compression_ratio_threshold=config.compression_ratio_threshold,
                temperature=config.temperature,
                suppress_silence=config.suppress_silence,
                vad=True,
                vad_threshold=config.vad_onset,
            )
            logger.info("   转录中 (faster-whisper)...")
            result = model.transcribe_stable(**transcribe_kwargs)
        else:
            # ═══ 原版 Whisper 路径 ═══
            use_fp16 = config.fp16 and config.device == "cuda"
            transcribe_kwargs = dict(
                audio=audio_path,
                language=config.language,
                beam_size=config.beam_size,
                condition_on_previous_text=config.condition_on_previous_text,
                word_timestamps=config.word_timestamps,
                suppress_silence=config.suppress_silence,
                initial_prompt=config.initial_prompt,
                no_speech_threshold=config.no_speech_threshold,
                compression_ratio_threshold=config.compression_ratio_threshold,
                temperature=config.temperature,
                fp16=use_fp16,
                vad=True,
                vad_threshold=config.vad_onset,
            )
            logger.info("   转录中 (whisper)...")
            result = model.transcribe(**transcribe_kwargs)

        # 可选：使用 stable-ts 的 refine 功能微调时间戳
        if config.refine_timestamps:
            try:
                logger.info("   微调时间戳 (refine)...")
                model.refine(audio_path, result)
            except Exception as e:
                logger.warning(f"   ⚠️ refine 失败，使用原始时间戳: {e}")
        else:
            logger.info("   跳过 refine（--no-refine）")

    n_segments = len(result.segments)
    logger.info(f"   识别出 {n_segments} 段字幕")

    # CUDA: 显示峰值显存
    if config.device == "cuda":
        import torch
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"   转录峰值显存: {peak_mb:.0f}MB")

    return result


# ═══════════════════════════════════════════════════════════════════
# Step 5: 字幕质量检查
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """字幕段"""
    start: float
    end: float
    text: str
    confidence: float = 1.0       # 平均对数概率
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0


# ═══════════════════════════════════════════════════════════════════
# 日本語フィラー（语气词）検出 — あ/え/うん 等自动去除
# ═══════════════════════════════════════════════════════════════════

# 语气词单元正则（长模式在前防止短模式贪婪匹配）
_JP_FILLER = (
    r'えーっと'                # えーっと
    r'|えっと[ー]?'            # えっと、えっとー
    r'|えーと'                 # えーと
    r'|あの[ーう]?ね?'         # あの、あのー、あのう、あのね
    r'|うーん+'                # うーん
    r'|そうそう(?:そう)*'      # そうそう、そうそうそう（重复形のみ、単独「そう」は有意味）
    r'|はいはい(?:はい)*'      # はいはい、はいはいはい（重複のみ、単独「はい」は有意味）
    r'|なんか'                 # なんか
    r'|ほら(?:ほら)*'          # ほら、ほらほら
    r'|こう'                   # こう
    r'|[あぁ]+[ーっ]*'         # あ、ああ、あー、あっ
    r'|[えぇ]+[ーっ]*'         # え、ええ、えー、えっ
    r'|[うぅ]+[ーんっ]*'       # う、うー、うん、うっ
    r'|[おぉ]+[ーっ]*'         # お、おー、おっ
    r'|[んン]+[ーっ]*'         # ん、んー
    r'|はぁ[ーっ]*'            # はぁ、はぁー（溜息、単独「は」は除外）
    r'|ふ[ーんっ]+'            # ふーん、ふっ（ふ単独は除外）
    r'|まぁ?[ーあぁ]+'         # まあ、まー、まぁ
    r'|ねぇ?[ーえぇ]+'         # ねー、ねえ（ね単独は除外）
)

_JP_FILLER_SEP = r'[\s　、。]*'

# 整段纯语气词判定
_JP_FILLER_ONLY_RE = re.compile(
    rf'^{_JP_FILLER_SEP}(?:{_JP_FILLER})'
    rf'(?:{_JP_FILLER_SEP}(?:{_JP_FILLER}))*'
    rf'{_JP_FILLER_SEP}$'
)

# 段首语气词（用于 strip）
_JP_FILLER_PREFIX_RE = re.compile(
    rf'^(?:(?:{_JP_FILLER}){_JP_FILLER_SEP})+'
)

# 段尾语气词（用于 strip）
_JP_FILLER_SUFFIX_RE = re.compile(
    rf'(?:{_JP_FILLER_SEP}(?:{_JP_FILLER}))+{_JP_FILLER_SEP}$'
)


def is_filler_only(text: str) -> bool:
    """检查整段是否只由无意义语气词构成

    例: 「あー」「えっと」「うーん、あー」 → True
        「あー、今日は天気がいいですね」    → False
        「はい」（有意义的应答）             → False
        「そう」（有意义的应答）             → False
    """
    return bool(_JP_FILLER_ONLY_RE.match(text))


def strip_filler_words(text: str) -> str:
    """去除段首段尾的语气词，保留中间有意义的内容

    例: 「えっと、今日は天気がいいですね」→「今日は天気がいいですね」
        「今日は天気がいいですね、あー」  →「今日は天気がいいですね」
        「えっと、あの、やっぱりいいです」→「やっぱりいいです」
    """
    # 去段首语气词
    text = _JP_FILLER_PREFIX_RE.sub('', text)
    # 去段尾语气词
    text = _JP_FILLER_SUFFIX_RE.sub('', text)
    # 清理残余标点
    return text.strip('、。 　')


def extract_segments(result) -> list[Segment]:
    """从 stable-ts 结果中提取段列表"""
    segments = []
    for seg in result.segments:
        segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            confidence=getattr(seg, "avg_logprob", 0.0),
            no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
            compression_ratio=getattr(seg, "compression_ratio", 1.0),
        ))
    return segments


def is_japanese_text(text: str, min_ratio: float = 0.1) -> bool:
    """检查文本是否包含足够的日文字符（平假名/片假名/汉字）

    日语句子通常混合平假名 + 汉字 + 片假名。
    纯英文、纯中文（无假名）、纯韩文 → 判定为非日语。

    Args:
        text: 待检查文本
        min_ratio: 日文字符（假名+汉字）最低占比，去除标点后计算
    """
    if not text:
        return False

    # 跳过标点和空格，只看实质字符
    skip_chars = set(' \t\n　、。！？「」『』（）・ーー〜～…─—―,.:;!?')
    jp_count = 0
    total = 0

    for char in text:
        if char in skip_chars:
            continue
        total += 1
        # 平假名 U+3040-U+309F
        # 片假名 U+30A0-U+30FF
        # 汉字   U+4E00-U+9FFF (CJK Unified)
        # 汉字扩展 U+3400-U+4DBF (CJK Extension A)
        # 半角片假名 U+FF65-U+FF9F
        if ('\u3040' <= char <= '\u309f' or   # 平假名
            '\u30a0' <= char <= '\u30ff' or   # 片假名
            '\u4e00' <= char <= '\u9fff' or   # 汉字
            '\u3400' <= char <= '\u4dbf' or   # 汉字扩展
            '\uff65' <= char <= '\uff9f'):     # 半角片假名
            jp_count += 1

    if total == 0:
        return False
    return (jp_count / total) >= min_ratio


def has_kana(text: str) -> bool:
    """检查文本是否包含至少一个假名字符（平假名或片假名）

    日语句子几乎一定有假名；纯汉字无假名很可能是中文幻觉。
    """
    for char in text:
        if ('\u3040' <= char <= '\u309f' or   # 平假名
            '\u30a0' <= char <= '\u30ff' or   # 片假名
            '\uff65' <= char <= '\uff9f'):     # 半角片假名
            return True
    return False


def quality_check(segments: list[Segment], config: Config) -> list[Segment]:
    """多维度字幕质量检查（针对日語优化）"""
    logger.info("🔍 Step 5 — 字幕质量检查...")

    original_count = len(segments)
    filtered = []
    removed_reasons = {
        "空白段": 0,
        "无语音概率过高": 0,
        "压缩比过高": 0,
        "对数概率过低": 0,
        "字符速率异常": 0,
        "零时长": 0,
        "重复文本": 0,
        "幻觉模式": 0,
        "非日文内容": 0,
        "纯语气词": 0,
    }

    # --- 编译幻觉正则 ---
    hallucination_re = [re.compile(p) for p in config.hallucination_patterns]

    # --- 逐段检查 ---
    for i, seg in enumerate(segments):
        text = seg.text.strip()

        # 1. 空白段
        if not text:
            removed_reasons["空白段"] += 1
            continue

        # 2. 零时长
        duration = seg.end - seg.start
        if duration <= 0:
            removed_reasons["零时长"] += 1
            continue

        # 3. 无语音概率
        if seg.no_speech_prob > config.no_speech_threshold:
            removed_reasons["无语音概率过高"] += 1
            logger.debug(f"   过滤 [no_speech={seg.no_speech_prob:.2f}]: {text[:30]}")
            continue

        # 4. 压缩比（幻觉指标）
        if seg.compression_ratio > config.compression_ratio_threshold:
            removed_reasons["压缩比过高"] += 1
            logger.debug(f"   过滤 [compression={seg.compression_ratio:.2f}]: {text[:30]}")
            continue

        # 5. 对数概率
        if seg.confidence < config.logprob_threshold:
            removed_reasons["对数概率过低"] += 1
            logger.debug(f"   过滤 [logprob={seg.confidence:.2f}]: {text[:30]}")
            continue

        # 6. 字符速率异常
        char_count = len(text)
        chars_per_sec = char_count / max(duration, 0.01)
        if chars_per_sec > config.max_chars_per_second:
            removed_reasons["字符速率异常"] += 1
            logger.debug(f"   过滤 [cps={chars_per_sec:.1f}]: {text[:30]}")
            continue

        # 7. 幻觉模式匹配
        is_hallucination = False
        for pattern in hallucination_re:
            if pattern.search(text):
                is_hallucination = True
                break
        if is_hallucination:
            removed_reasons["幻觉模式"] += 1
            logger.debug(f"   过滤 [幻觉模式]: {text[:30]}")
            continue

        # 8. 日文内容验证 — 纯英文/纯中文（无假名）/乱码 → 非日语
        if not is_japanese_text(text):
            removed_reasons["非日文内容"] += 1
            logger.debug(f"   过滤 [非日文]: {text[:30]}")
            continue

        # 9. 纯语气词段 — あー/えっと/うーん 等无意义内容
        if config.strip_fillers and is_filler_only(text):
            removed_reasons["纯语气词"] += 1
            logger.debug(f"   过滤 [语气词]: {text[:30]}")
            continue

        filtered.append(seg)

    # --- 重复文本检测 ---
    final = []
    repeat_count = 1
    for i, seg in enumerate(filtered):
        if i > 0 and seg.text == filtered[i - 1].text:
            repeat_count += 1
            if repeat_count > config.repeat_threshold:
                removed_reasons["重复文本"] += 1
                continue
        else:
            repeat_count = 1
        final.append(seg)

    # --- 汇总 ---
    removed_total = original_count - len(final)
    logger.info(f"   原始段数: {original_count}, 保留: {len(final)}, 过滤: {removed_total}")
    for reason, count in removed_reasons.items():
        if count > 0:
            logger.info(f"   - {reason}: {count} 段")

    return final


# ═══════════════════════════════════════════════════════════════════
# Step 6: 后处理
# ═══════════════════════════════════════════════════════════════════

def postprocess(segments: list[Segment], config: Config) -> list[Segment]:
    """后处理：合并短段 → 拆分长段 → 最短保底 → 时间重叠修复 → 标点规范化"""
    logger.info("✂️  Step 6 — 后处理...")
    before_count = len(segments)

    # 1. 合并过短的相邻段（间隔小于阈值且合并后不超长）
    segments = merge_short_segments(segments, config)
    logger.debug(f"   合并短段后: {len(segments)} 段")

    # 2. 拆分过长段
    result = []
    for seg in segments:
        if (seg.end - seg.start) > config.max_segment_duration:
            result.extend(split_long_segment(seg, config))
        else:
            result.append(seg)
    logger.debug(f"   拆分长段后: {len(result)} 段")

    # 3. 最短显示保底
    for seg in result:
        if (seg.end - seg.start) < config.min_display_duration:
            seg.end = seg.start + config.min_display_duration

    # 4. 修复时间重叠（确保段与段之间不重叠）
    for i in range(1, len(result)):
        if result[i].start < result[i - 1].end:
            # 取中间点作为分界
            mid = (result[i].start + result[i - 1].end) / 2
            result[i - 1].end = mid
            result[i].start = mid

    # 5. 日文标点规范化
    for seg in result:
        seg.text = normalize_japanese_punctuation(seg.text)

    # 6. 去除段首段尾语气词（えっと、あのー 等）
    if config.strip_fillers:
        filler_stripped = 0
        cleaned_result = []
        for seg in result:
            cleaned = strip_filler_words(seg.text)
            if cleaned:
                if cleaned != seg.text:
                    filler_stripped += 1
                seg.text = cleaned
                cleaned_result.append(seg)
            else:
                filler_stripped += 1  # 全部是语气词，删除整段
        if filler_stripped:
            logger.info(f"   🗑️ 语气词清理: {filler_stripped} 段受影响")
        result = cleaned_result

    logger.info(f"   ✅ 后处理完成: {before_count} → {len(result)} 段")
    return result


def merge_short_segments(segments: list[Segment], config: Config) -> list[Segment]:
    """合并相邻的短段：如果两段间隔小且合并后不超长，则合并"""
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start - prev.end
        combined_duration = seg.end - prev.start
        combined_text_len = len(prev.text) + len(seg.text)

        # 合并条件：间隔小 + 合并后不超长 + 合并后文本不太长
        if (gap < config.merge_gap_threshold
                and combined_duration <= config.max_segment_duration
                and combined_text_len <= 40):
            # 合并
            prev.end = seg.end
            prev.text = prev.text + seg.text
        else:
            merged.append(seg)

    return merged


def split_long_segment(seg: Segment, config: Config) -> list[Segment]:
    """将过长的段按日文标点智能拆分

    优先级：句号(。！？) > 逗号/引号(、」』) > 接续助词(て/で/が/けど) > 字符等分
    """
    text = seg.text
    duration = seg.end - seg.start

    # 优先级1：按句末标点拆分（最自然的断句点）
    split_points = [i + 1 for i, c in enumerate(text) if c in "。！？…"]

    # 优先级2：逗号、引号闭合、括号闭合
    if not split_points:
        split_points = [i + 1 for i, c in enumerate(text) if c in "、；，」』）～"]

    # 优先级3：日语接续助词之后（て/で/が/けど/から/ので — 自然的从句边界）
    if not split_points:
        jp_clause_re = re.compile(r'(?:て|で|が|けど|から|ので|けれど|のに|ながら)(?=.)')
        for m in jp_clause_re.finditer(text):
            split_points.append(m.end())

    # 优先级4：都没有则字符等分
    if not split_points:
        n_parts = max(2, int(len(text) / 20))  # 每 ~20 字一段
        chunk_size = len(text) // n_parts
        split_points = [chunk_size * i for i in range(1, n_parts)]

    # 创建子段
    parts = []
    prev_idx = 0
    for sp in split_points:
        part = text[prev_idx:sp].strip()
        if part:
            parts.append(part)
        prev_idx = sp
    remainder = text[prev_idx:].strip()
    if remainder:
        parts.append(remainder)

    if len(parts) <= 1:
        return [seg]

    # 按字符比例分配时间
    total_chars = sum(len(p) for p in parts)
    sub_segments = []
    current_time = seg.start

    for part_text in parts:
        ratio = len(part_text) / max(total_chars, 1)
        part_duration = duration * ratio
        sub_segments.append(Segment(
            start=round(current_time, 3),
            end=round(current_time + part_duration, 3),
            text=part_text,
        ))
        current_time += part_duration

    return sub_segments


def normalize_japanese_punctuation(text: str) -> str:
    """日文標点規範化 — 针对日语字幕输出优化"""

    # 1. 半角标点 → 全角（日文排版标准）
    text = text.translate(str.maketrans(
        '!?,.():;[]{}',
        '！？、。（）：；「」｛｝',
    ))

    # 2. 统一波浪号（〜 和 ～ 混用问题）
    text = text.replace('〜', '～')

    # 3. 统一长音符
    text = text.replace('ｰ', 'ー')

    # 4. 统一省略号
    text = text.replace('...', '…')
    text = text.replace('。。。', '…')
    text = text.replace('・・・', '…')

    # 5. 半角片假名 → 全角（ｱ→ア, ｲ→イ 等）
    text = _halfwidth_katakana_to_fullwidth(text)

    # 6. 去除多余空格（日文不需要空格）
    text = re.sub(r'[\s　]+', '', text)

    # 7. 清理首尾多余标点
    text = text.strip('、。')

    return text


def _halfwidth_katakana_to_fullwidth(text: str) -> str:
    """半角片假名 → 全角片假名"""
    # U+FF65-FF9F → 全角对照表
    _hw_to_fw = str.maketrans(
        'ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ',
        'ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン゛゜',
    )
    return text.translate(_hw_to_fw)


# ═══════════════════════════════════════════════════════════════════
# 遗漏区域二次转录
# ═══════════════════════════════════════════════════════════════════

def _merge_nearby_regions(regions: list[tuple[float, float]],
                          max_gap: float = 2.0) -> list[tuple[float, float]]:
    """合并相邻的遗漏区域（gap < max_gap 秒时合并为一段）"""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged = [list(sorted_regions[0])]

    for start, end in sorted_regions[1:]:
        prev = merged[-1]
        if start - prev[1] <= max_gap:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def retranscribe_missed_regions(
    audio_path: str,
    missed_regions: list[tuple[float, float]],
    config: Config,
    temp_dir: str,
) -> list[Segment]:
    """对遗漏的语音区域进行二次转录

    策略与第一次不同：
    - condition_on_previous_text=False（避免错误传播）
    - 较小 beam_size（更宽容，减少静音跳过）
    - 较窄 temperature（减少幻觉）
    - 略高 no_speech_threshold（更严格过滤）
    """
    if not missed_regions:
        return []

    # 合并相邻区域减少转录次数
    merged = _merge_nearby_regions(missed_regions, max_gap=2.0)
    logger.info(f"🔄 二次转录: {len(missed_regions)} 个遗漏区域 → 合并为 {len(merged)} 个片段")

    new_segments = []
    model = _get_whisper_model(config)
    actual_backend = _whisper_backend or config.backend

    for i, (start, end) in enumerate(_progress(merged, desc="   🔄 二次转录", unit="段")):
        # 添加上下文填充 (0.3s 每侧)
        pad = 0.3
        padded_start = max(0, start - pad)
        padded_end = end + pad
        duration = padded_end - padded_start

        if duration < 0.5:
            continue

        # 提取音频片段
        clip_path = os.path.join(temp_dir, f"retranscribe_{i}.wav")
        try:
            run_ffmpeg([
                "-ss", f"{padded_start:.3f}",
                "-to", f"{padded_end:.3f}",
                "-i", audio_path,
                "-ar", str(config.sample_rate),
                "-ac", "1",
                "-acodec", "pcm_s16le",
                clip_path,
            ], f"提取遗漏区域 #{i}")
        except Exception as e:
            logger.debug(f"   ffmpeg 提取遗漏区域失败: {e}")
            continue

        # 用不同参数转录
        try:
            # ★ 二次转录参数策略：更宽容，更不容易跳过语音
            retranscribe_kwargs = dict(
                language=config.language,
                beam_size=config.retranscribe_beam_size,
                condition_on_previous_text=False,       # ★ 无上下文依赖
                word_timestamps=config.word_timestamps,
                initial_prompt=config.initial_prompt,
                no_speech_threshold=0.8,                # ★ 更宽容（不轻易判定为"无语音"）
                temperature=(0.0, 0.2, 0.4),            # ★ 多温度尝试
                suppress_silence=config.suppress_silence,
                vad=True,
                vad_threshold=config.vad_onset,
            )

            if actual_backend == "faster-whisper":
                result = model.transcribe_stable(clip_path, **retranscribe_kwargs)
            else:
                use_fp16 = config.fp16 and config.device == "cuda"
                retranscribe_kwargs["fp16"] = use_fp16
                result = model.transcribe(clip_path, **retranscribe_kwargs)

            # 提取有效段并调整时间戳
            for seg in result.segments:
                text = seg.text.strip()
                if not text:
                    continue
                if not is_japanese_text(text):
                    continue
                if config.strip_fillers and is_filler_only(text):
                    continue

                new_segments.append(Segment(
                    start=round(seg.start + padded_start, 3),
                    end=round(seg.end + padded_start, 3),
                    text=text,
                    confidence=getattr(seg, "avg_logprob", 0.0),
                    no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
                    compression_ratio=getattr(seg, "compression_ratio", 1.0),
                ))

        except Exception as e:
            logger.debug(f"   区域 {padded_start:.1f}-{padded_end:.1f}s 二次转录失败: {e}")
            continue

    if new_segments:
        logger.info(f"   ✅ 二次转录恢复了 {len(new_segments)} 段字幕")
    else:
        logger.info("   ℹ️  二次转录未产出新字幕（遗漏区域可能为背景噪音）")

    return new_segments


# ═══════════════════════════════════════════════════════════════════
# 缓存工具
# ═══════════════════════════════════════════════════════════════════

def _save_segments_cache(segments: list[Segment], path: str):
    """序列化 Segment 列表到 JSON"""
    data = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "confidence": s.confidence,
            "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio,
        }
        for s in segments
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_segments_cache(path: str) -> list[Segment]:
    """从 JSON 反序列化 Segment 列表"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Segment(**d) for d in data]


def _save_regions_cache(regions: list[tuple[float, float]], path: str):
    """序列化语音区域到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regions, f)


def _load_regions_cache(path: str) -> list[tuple[float, float]]:
    """从 JSON 反序列化语音区域"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(r) for r in data]


# ═══════════════════════════════════════════════════════════════════
# 输出格式
# ═══════════════════════════════════════════════════════════════════

def write_vtt(segments: list[Segment], output_path: str):
    """输出 WebVTT 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = format_time_vtt(seg.start)
            end = format_time_vtt(seg.end)
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg.text}\n\n")
    logger.info(f"   📄 VTT: {output_path}")


def write_srt(segments: list[Segment], output_path: str):
    """输出 SRT 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_time_srt(seg.start)
            end = format_time_srt(seg.end)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg.text}\n\n")
    logger.info(f"   📄 SRT: {output_path}")


def write_json(segments: list[Segment], output_path: str):
    """输出 JSON 格式（方便后续处理）"""
    data = [
        {
            "index": i,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "duration": round(seg.end - seg.start, 3),
            "text": seg.text,
        }
        for i, seg in enumerate(segments, 1)
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"   📄 JSON: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def process_video(video_path: str, config: Config, output_dir: Optional[str] = None):
    """处理单个视频文件的完整流程

    包含：缓存检查 → 容错降级 → 二次转录 → 覆盖率报告
    """
    video_path = os.path.abspath(video_path)
    video_name = Path(video_path).stem

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"文件不存在: {video_path}")

    # 输出目录
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    # ─── 工作目录：缓存模式 vs 临时模式 ───
    use_cache = config.cache_dir is not None
    if use_cache:
        cache_key = _compute_cache_key(video_path, config)
        temp_dir = os.path.join(config.cache_dir, cache_key)
        os.makedirs(temp_dir, exist_ok=True)
        # 写入元数据
        meta = {
            "video": video_path,
            "video_name": video_name,
            "cache_key": cache_key,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": config.backend,
            "model": config.model,
        }
        with open(os.path.join(temp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"🎬 开始处理: {Path(video_path).name}")
        logger.info(f"   💾 缓存目录: {temp_dir}")
    else:
        temp_dir = tempfile.mkdtemp(prefix="tiqu_")
        logger.info(f"🎬 开始处理: {Path(video_path).name}")
        logger.info(f"   临时目录: {temp_dir}")

    total_start = time.time()

    try:
        # ─── 自动检测设备 ───
        if config.device == "auto":
            config.device, auto_demucs = detect_devices()
            if not config.demucs_device:
                config.demucs_device = auto_demucs
        if not config.demucs_device:
            config.demucs_device = config.device

        # ═══════════════════════════════════════════════════════════
        # Step 1: 音频提取
        # ═══════════════════════════════════════════════════════════
        if config.skip_demucs:
            raw_audio = os.path.join(temp_dir, "raw_audio.wav")
            if os.path.exists(raw_audio):
                logger.info("📼 Step 1 — ✅ 使用缓存")
            else:
                extract_audio(video_path, raw_audio, config, target_sr=config.sample_rate)
        else:
            raw_audio = os.path.join(temp_dir, "raw_audio_44k.wav")
            if os.path.exists(raw_audio):
                logger.info("📼 Step 1 — ✅ 使用缓存")
            else:
                extract_audio(video_path, raw_audio, config, target_sr=config.demucs_sample_rate)

        # ═══════════════════════════════════════════════════════════
        # Step 2: Demucs 人声分离（容错：失败则用原始音频）
        # ═══════════════════════════════════════════════════════════
        vocals_audio = os.path.join(temp_dir, "vocals.wav")
        if os.path.exists(vocals_audio):
            logger.info("🎤 Step 2 — ✅ 使用缓存")
        else:
            try:
                separate_vocals(raw_audio, vocals_audio, config)
                # skip_demucs 时 separate_vocals 返回原路径，需复制
                if config.skip_demucs and raw_audio != vocals_audio:
                    if not os.path.exists(vocals_audio):
                        shutil.copy2(raw_audio, vocals_audio)
            except Exception as e:
                logger.warning(f"⚠️  Demucs 人声分离失败: {e}")
                logger.warning("   🔄 降级方案: 使用原始音频（无人声分离）")
                shutil.copy2(raw_audio, vocals_audio)

        # ═══════════════════════════════════════════════════════════
        # Step 3: 语音增强（容错：失败则用人声音频）
        # ═══════════════════════════════════════════════════════════
        enhanced_audio = os.path.join(temp_dir, "enhanced.wav")
        if os.path.exists(enhanced_audio):
            logger.info("🔊 Step 3 — ✅ 使用缓存")
        else:
            try:
                enhance_audio(vocals_audio, enhanced_audio, config)
            except Exception as e:
                logger.warning(f"⚠️  语音增强失败: {e}")
                logger.warning("   🔄 降级方案: 使用原始人声音频")
                # 需要降采样到 16kHz
                try:
                    run_ffmpeg([
                        "-i", vocals_audio,
                        "-ar", str(config.sample_rate),
                        "-ac", "1",
                        "-acodec", "pcm_s16le",
                        enhanced_audio,
                    ], "降采样到 16kHz")
                except Exception:
                    shutil.copy2(vocals_audio, enhanced_audio)

        # ═══════════════════════════════════════════════════════════
        # Step 3.5: VAD 语音区域检测（容错：失败则跳过）
        # ═══════════════════════════════════════════════════════════
        speech_regions = []
        audio_duration = get_audio_duration(enhanced_audio)
        regions_cache_path = os.path.join(temp_dir, "speech_regions.json")

        if config.vad_filter:
            if os.path.exists(regions_cache_path):
                logger.info("🎯 VAD — ✅ 使用缓存")
                try:
                    speech_regions = _load_regions_cache(regions_cache_path)
                except Exception:
                    speech_regions = []
            else:
                logger.info("🎯 VAD — Silero 语音区域检测...")
                try:
                    with timed_step("Silero VAD"):
                        speech_regions = get_speech_regions(enhanced_audio, config)
                    if speech_regions and use_cache:
                        _save_regions_cache(speech_regions, regions_cache_path)
                except Exception as e:
                    logger.warning(f"⚠️  VAD 检测失败: {e}")
                    logger.warning("   🔄 降级方案: 跳过 VAD 交叉验证")

        # ═══════════════════════════════════════════════════════════
        # Step 4: 转录
        # ═══════════════════════════════════════════════════════════
        segments_cache_path = os.path.join(temp_dir, "raw_segments.json")

        if os.path.exists(segments_cache_path):
            logger.info("📝 Step 4 — ✅ 使用缓存的转录结果")
            segments = _load_segments_cache(segments_cache_path)
            logger.info(f"   加载了 {len(segments)} 段字幕")
        else:
            result = transcribe_audio(enhanced_audio, config)
            segments = extract_segments(result)
            # 保存到缓存
            if use_cache:
                _save_segments_cache(segments, segments_cache_path)

        # ═══════════════════════════════════════════════════════════
        # Step 5: 质量检查
        # ═══════════════════════════════════════════════════════════
        segments = quality_check(segments, config)

        # ═══════════════════════════════════════════════════════════
        # Step 5.5: VAD 交叉验证 + 二次转录遗漏区域
        # ═══════════════════════════════════════════════════════════
        if config.vad_filter and speech_regions:
            segments = vad_validate(segments, speech_regions, config)

            # ★ 二次转录遗漏区域
            if config.retranscribe_missed:
                missed = detect_missed_speech(
                    segments, speech_regions,
                    min_duration=config.retranscribe_min_duration,
                )
                if missed:
                    try:
                        new_segs = retranscribe_missed_regions(
                            enhanced_audio, missed, config, temp_dir
                        )
                        if new_segs:
                            segments.extend(new_segs)
                            segments.sort(key=lambda s: s.start)
                            logger.info(f"   合并后总计: {len(segments)} 段")
                    except Exception as e:
                        logger.warning(f"⚠️  二次转录失败: {e}")
                        logger.warning("   🔄 降级方案: 仅使用第一次转录结果")

        # ═══════════════════════════════════════════════════════════
        # Step 6: 后处理 + 输出
        # ═══════════════════════════════════════════════════════════
        segments = postprocess(segments, config)

        if not segments:
            logger.warning("⚠️  未识别到任何有效字幕段！")
            return

        # 写入文件
        logger.info("💾 写入字幕文件...")
        writers = {
            "vtt": write_vtt,
            "srt": write_srt,
            "json": write_json,
        }
        for fmt in config.output_formats:
            if fmt in writers:
                out_path = os.path.join(output_dir, f"{video_name}.{fmt}")
                writers[fmt](segments, out_path)

        # 覆盖率报告
        print_coverage_report(segments, speech_regions, audio_duration)

        total_elapsed = time.time() - total_start
        logger.info(f"🎉 全部完成！耗时 {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
        logger.info(f"   输出目录: {output_dir}")

        # CUDA: 显示峰值显存使用
        if config.device == "cuda":
            import torch
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"   GPU 峰值显存: {peak_mb:.0f}MB")

    finally:
        # 清理临时文件（缓存模式不清理）
        if not use_cache and not config.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("   已清理临时文件")
        elif config.keep_temp and not use_cache:
            logger.info(f"   保留临时文件: {temp_dir}")


# ═══════════════════════════════════════════════════════════════════
# YAML 配置文件
# ═══════════════════════════════════════════════════════════════════

def _load_config_yaml(path: str) -> dict:
    """加载 YAML 配置文件

    示例 tiqu-config.yaml:
        model: large-v3
        backend: faster-whisper
        compute_type: float16
        beam_size: 10
        demucs_shifts: 2
        strip_fillers: true
        output_formats: [vtt, srt]
    """
    try:
        import yaml
    except ImportError:
        logger.error("❌ 加载 YAML 配置需要 pyyaml: pip install pyyaml")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        logger.error(f"❌ 配置文件格式错误（应为键值对）: {path}")
        sys.exit(1)

    logger.info(f"📋 已加载配置: {path} ({len(data)} 项)")
    return data


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 tiqu — 日本語字幕提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tiqu.py video.mp4                                  # 基本用法
  python tiqu.py video.mp4 --backend whisper                # 使用原版 Whisper
  python tiqu.py video.mp4 --compute-type int8_float16      # INT8 量化加速
  python tiqu.py video.mp4 --format vtt srt --verbose       # 多格式 + 详细日志
  python tiqu.py video.mp4 --skip-demucs --keep-temp        # 跳过分离 + 保留临时文件
  python tiqu.py video.mp4 --cache-dir ./cache              # 启用断点续传
  python tiqu.py video.mp4 --config tiqu-config.yaml        # 使用配置文件
  python tiqu.py *.mp4 --output-dir ./subtitles             # 批量处理
        """,
    )

    # --- 位置参数 ---
    parser.add_argument(
        "videos",
        nargs="+",
        help="输入视频文件路径（支持多个）",
    )

    # --- 输出 ---
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="输出目录（默认为视频所在目录）",
    )
    parser.add_argument(
        "--format", "-f",
        nargs="+",
        choices=["vtt", "srt", "json"],
        help="输出格式 (默认: vtt)",
    )

    # --- 后端 ---
    parser.add_argument(
        "--backend",
        choices=["faster-whisper", "whisper"],
        help="转录后端 (默认: faster-whisper)",
    )
    parser.add_argument(
        "--compute-type",
        choices=["float16", "int8_float16", "int8", "float32"],
        help="faster-whisper 计算精度 (默认: float16)",
    )

    # --- 模型 ---
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
        help="Whisper 模型 (默认: large-v3)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        help="计算设备 (默认: auto)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="Beam search 宽度 (默认: 10)",
    )

    # --- Demucs ---
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        default=None,
        help="跳过 Demucs 人声分离（音源已很干净时使用）",
    )
    parser.add_argument(
        "--demucs-model",
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
        help="Demucs 模型 (默认: htdemucs_ft)",
    )
    parser.add_argument(
        "--demucs-shifts",
        type=int,
        help="Demucs 随机偏移次数（越高质量越好但越慢，默认: 1，推荐 GPU: 2~5）",
    )
    parser.add_argument(
        "--demucs-segment",
        type=int,
        help="Demucs 分段处理长度（秒），显存不足时设 40~60",
    )

    # --- 功能开关 ---
    parser.add_argument(
        "--no-refine",
        action="store_true",
        default=None,
        help="跳过 stable-ts refine 时间戳微调（加快速度）",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        default=None,
        help="禁用 fp16 加速（仅原版 Whisper 后端）",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        default=None,
        help="跳过 Silero VAD 交叉验证",
    )
    parser.add_argument(
        "--no-strip-fillers",
        action="store_true",
        default=None,
        help="保留语气词（あー/えっと/うーん 等）",
    )
    parser.add_argument(
        "--no-retranscribe",
        action="store_true",
        default=None,
        help="禁用遗漏区域二次转录",
    )

    # --- 缓存 / 断点续传 ---
    parser.add_argument(
        "--cache-dir",
        help="缓存目录（启用断点续传，已完成的步骤不再重复）",
    )

    # --- 配置文件 ---
    parser.add_argument(
        "--config", "-c",
        help="YAML 配置文件路径（CLI 参数优先级高于配置文件）",
    )

    # --- 其他 ---
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=None,
        help="保留临时文件（用于调试）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="显示详细日志",
    )

    return parser.parse_args()


def _build_config(args) -> Config:
    """从 CLI 参数 + YAML 配置构建 Config

    优先级: CLI 显式参数 > YAML 配置 > Config 默认值
    (CLI 参数默认为 None，仅非 None 时才覆盖)
    """
    # 1. 基础 Config（默认值）
    config = Config()

    # 2. YAML 覆盖默认值
    if args.config:
        yaml_data = _load_config_yaml(args.config)
        for key, value in yaml_data.items():
            if hasattr(config, key):
                # tuple 类型特殊处理（YAML 读出来是 list）
                if key == "temperature" and isinstance(value, list):
                    value = tuple(value)
                setattr(config, key, value)
            else:
                logger.warning(f"   ⚠️ 配置文件中的未知字段: {key}")

    # 3. CLI 显式参数覆盖（仅非 None 值）
    if args.backend is not None:
        config.backend = args.backend
    if args.compute_type is not None:
        config.compute_type = args.compute_type
    if args.model is not None:
        config.model = args.model
    if args.device is not None:
        config.device = args.device
    if args.beam_size is not None:
        config.beam_size = args.beam_size
    if args.format is not None:
        config.output_formats = args.format
    if args.skip_demucs is not None:
        config.skip_demucs = args.skip_demucs
    if args.demucs_model is not None:
        config.demucs_model = args.demucs_model
    if args.demucs_shifts is not None:
        config.demucs_shifts = args.demucs_shifts
    if args.demucs_segment is not None:
        config.demucs_segment = args.demucs_segment
    if args.no_refine is not None:
        config.refine_timestamps = not args.no_refine
    if args.no_fp16 is not None:
        config.fp16 = not args.no_fp16
    if args.no_vad is not None:
        config.vad_filter = not args.no_vad
    if args.no_strip_fillers is not None:
        config.strip_fillers = not args.no_strip_fillers
    if args.no_retranscribe is not None:
        config.retranscribe_missed = not args.no_retranscribe
    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir
    if args.keep_temp is not None:
        config.keep_temp = args.keep_temp

    return config


def _print_system_info(config: Config):
    """显示系统和环境信息，便于调试"""
    import platform
    import torch

    logger.info("═" * 60)
    logger.info("📊 系统信息")
    logger.info(f"   OS: {platform.system()} {platform.release()}")
    logger.info(f"   Python: {platform.python_version()}")
    logger.info(f"   PyTorch: {torch.__version__}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"   GPU: {props.name}")
        _vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        logger.info(f"   VRAM: {_vram / (1024**3):.1f}GB")
        logger.info(f"   CUDA: {torch.version.cuda}")
    logger.info(f"   后端: {config.backend} | 模型: {config.model}")
    if config.backend == "faster-whisper":
        logger.info(f"   compute_type: {config.compute_type}")
    else:
        logger.info(f"   fp16: {config.fp16 and config.device == 'cuda'}")
    logger.info(f"   Whisper 设备: {config.device}")
    logger.info(f"   Demucs 设备: {config.demucs_device or '(跟随 Whisper)'}")
    logger.info(f"   Demucs shifts: {config.demucs_shifts}")
    logger.info(f"   VAD 验证: {'✅ 启用' if config.vad_filter else '❌ 禁用'}")
    logger.info(f"   二次转录: {'✅ 启用' if config.retranscribe_missed else '❌ 禁用'}")
    logger.info(f"   缓存: {config.cache_dir or '❌ 未启用'}")
    logger.info(f"   tqdm 进度条: {'✅' if _tqdm else '❌ (pip install tqdm)'}")
    logger.info("═" * 60)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # 检查 ffmpeg
    if shutil.which("ffmpeg") is None:
        import platform
        if platform.system() == "Darwin":
            hint = "brew install ffmpeg"
        else:
            hint = "apt-get install -y ffmpeg"
        logger.error(f"❌ 未找到 ffmpeg，请先安装: {hint}")
        sys.exit(1)

    # 构建配置（YAML + CLI 合并）
    config = _build_config(args)

    # 显示系统信息
    _print_system_info(config)

    # 处理每个视频（多文件时带进度条）
    videos = args.videos
    if len(videos) > 1:
        logger.info(f"📁 批量处理: {len(videos)} 个文件")

    video_iter = _progress(
        videos,
        desc="📁 批量处理",
        unit="个",
    ) if len(videos) > 1 else videos

    for video_path in video_iter:
        try:
            process_video(video_path, config, args.output_dir)
        except Exception as e:
            logger.error(f"❌ 处理失败 [{video_path}]: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
