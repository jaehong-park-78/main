"""
FWRZIP v2.1 — FWR 기반 적응형 압축기 (최종 통합본)

기능:
- FWR 5축 (F·W·C·η·T·R) 기반 압축
- 텍스트 구조 인식 (JSON, LOG, CSV, 반복 텍스트)
- Dictionary / Delta / RLE 전처리
- 엔트로피 기반 압축 생략 (이미 압축된 데이터)
- R (Resonance) 점수로 최적 전략 선택
- PyQt5 GUI
- CLI 지원
"""

import sys
import os
import json
import zlib
import base64
import hashlib
import math
import re
import time
import argparse
from collections import Counter
from typing import Any, Dict, Optional, Union, List, Tuple

# PyQt5는 GUI 모드에서만 import
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar,
        QMessageBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QTextCursor
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# ============================================================
# FWRZIP v2.1 코어 엔진
# ============================================================

class FWRZIP:
    VERSION = "FWRZIP v2.1"
    FORMAT = "fwrzip-2"

    def __init__(self, history_limit: int = 32, verbose: bool = True):
        self.version = self.VERSION
        self.history = []
        self.history_limit = max(1, history_limit)
        self.verbose = verbose
        self.dictionary_max_patterns = 64
        self.dictionary_max_pattern_length = 8

    # ============================================================
    # 공통 유틸리티
    # ============================================================

    @staticmethod
    def _to_bytes(data) -> bytes:
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, bytes):
            return data
        if isinstance(data, (bytearray, memoryview)):
            return bytes(data)
        raise TypeError("data must be str, bytes, bytearray, or memoryview")

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _entropy(data: bytes) -> float:
        if not data:
            return 0.0
        counter = Counter(data)
        length = len(data)
        entropy = 0.0
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _repeat_ratio(data: bytes) -> float:
        if len(data) < 2:
            return 0.0
        repeat_count = sum(1 for i in range(len(data)-1) if data[i] == data[i+1])
        return repeat_count / (len(data) - 1)

    @staticmethod
    def _structural_repeat_ratio(data: bytes) -> float:
        if len(data) < 6:
            return 0.0
        trigrams = Counter(data[i:i+3] for i in range(len(data)-2))
        if not trigrams:
            return 0.0
        max_count = max(trigrams.values())
        if max_count < 2:
            return 0.0
        return max_count / (len(data) - 2)

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @staticmethod
    def _looks_like_utf8_text(data: bytes) -> bool:
        try:
            text = data.decode("utf-8")
            return "\x00" not in text
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _decode_utf8(data: bytes) -> Optional[str]:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _draw_bar(value: float, max_value: float, width: int = 30) -> str:
        if max_value == 0:
            return "░" * width
        bar_len = int((value / max_value) * width)
        return '█' * bar_len + '░' * (width - bar_len)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ============================================================
    # F — Flow: 데이터 분석
    # ============================================================

    def analyze_data(self, data: bytes) -> Dict[str, Any]:
        is_text = self._looks_like_utf8_text(data)
        entropy = self._entropy(data)
        repeat_ratio = self._repeat_ratio(data)
        structural_repeat = self._structural_repeat_ratio(data)

        pattern_score = max(
            (1.0 - entropy / 8.0) * 0.4,
            repeat_ratio * 0.3,
            structural_repeat * 0.3
        )
        pattern_score = min(1.0, pattern_score)

        result = {
            "type": "text" if is_text else "binary",
            "subtype": "plain",
            "size": len(data),
            "entropy": round(entropy, 6),
            "repeat_ratio": round(repeat_ratio, 6),
            "structural_repeat": round(structural_repeat, 6),
            "pattern_score": round(pattern_score, 6),
            "features": {},
            "compressible": True
        }

        # 이미 압축된 데이터 판별 (엔트로피 > 7.5, 바이너리)
        if not is_text and entropy > 7.5:
            result["compressible"] = False
            result["subtype"] = "binary_compressed"
            return result

        if not is_text:
            result["subtype"] = "binary"
            return result

        text = data.decode("utf-8")
        stripped = text.strip()
        lines = [line for line in text.splitlines() if line.strip()]
        result["features"]["line_count"] = len(lines)

        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(text)
                result["subtype"] = "json"
                result["features"]["is_json"] = True
                return result
            except Exception:
                pass

        if len(lines) >= 3:
            for delim in [",", "\t", "|", ";"]:
                count = sum(line.count(delim) for line in lines)
                if count >= len(lines) * 1.5:
                    result["subtype"] = "csv"
                    result["features"]["delimiter"] = delim
                    result["features"]["line_count"] = len(lines)
                    return result

        if len(lines) >= 3:
            ts_count = 0
            for line in lines:
                if re.search(r"\b\d{2}:\d{2}:\d{2}\b", line):
                    ts_count += 1
                elif re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", line):
                    ts_count += 1
            if ts_count / len(lines) >= 0.5:
                result["subtype"] = "log"
                result["features"]["timestamp_ratio"] = ts_count / len(lines)
                return result

        if repeat_ratio >= 0.30 or structural_repeat >= 0.30:
            result["subtype"] = "repetitive"
            return result

        result["subtype"] = "plain"
        return result

    # ============================================================
    # W — Wave: 전처리 전략
    # ============================================================

    @staticmethod
    def rle_encode(data: bytes) -> bytes:
        if not data:
            return b""
        output = bytearray()
        i = 0
        while i < len(data):
            value = data[i]
            j = i + 1
            while j < len(data) and data[j] == value:
                j += 1
            count = j - i

            if value == 0xFF:
                if count == 1:
                    output.extend(b"\xFF\x00\xFF")
                else:
                    while count > 0:
                        chunk = min(count, 0xFFFFFFFF)
                        output.extend(b"\xFF\x01")
                        output.append(value)
                        output.extend(chunk.to_bytes(4, "big"))
                        count -= chunk
            elif count >= 3:
                output.extend(b"\xFF\x01")
                output.append(value)
                output.extend(count.to_bytes(4, "big"))
            else:
                output.extend(bytes([value]) * count)
            i = j
        return bytes(output)

    @staticmethod
    def rle_decode(data: bytes) -> bytes:
        output = bytearray()
        i = 0
        while i < len(data):
            value = data[i]
            i += 1
            if value != 0xFF:
                output.append(value)
                continue
            if i >= len(data):
                raise ValueError("Invalid RLE data")
            tag = data[i]
            i += 1
            if tag == 0:
                if i >= len(data):
                    raise ValueError("Invalid RLE literal")
                output.append(data[i])
                i += 1
            elif tag == 1:
                if i + 5 > len(data):
                    raise ValueError("Invalid RLE run")
                val = data[i]
                count = int.from_bytes(data[i+1:i+5], "big")
                i += 5
                if count <= 0:
                    raise ValueError("Invalid RLE count")
                output.extend(bytes([val]) * count)
            else:
                raise ValueError("Invalid RLE tag")
        return bytes(output)

    def build_dictionary(self, data: bytes) -> List[bytes]:
        if len(data) < 6:
            return []

        counter = Counter()
        max_len = min(self.dictionary_max_pattern_length, len(data))
        step = 1 if len(data) < 100000 else 2

        for length in range(3, max_len + 1):
            for i in range(0, len(data) - length + 1, step):
                pattern = data[i:i+length]
                counter[pattern] += 1

        candidates = []
        for pattern, count in counter.items():
            if count < 2:
                continue
            saving = (len(pattern) - 3) * count
            if saving > 0:
                candidates.append((saving, count, len(pattern), pattern))

        candidates.sort(reverse=True)
        selected = []
        for _, _, _, pattern in candidates:
            if pattern not in selected:
                selected.append(pattern)
            if len(selected) >= self.dictionary_max_patterns:
                break

        return selected

    @staticmethod
    def dictionary_encode(data: bytes, patterns: List[bytes]) -> bytes:
        if not patterns:
            return data
        patterns = sorted(patterns, key=len, reverse=True)
        output = bytearray()
        i = 0
        while i < len(data):
            matched = False
            for idx, pattern in enumerate(patterns):
                if data.startswith(pattern, i):
                    output.extend(b"\xFF\x02")
                    output.append(idx)
                    i += len(pattern)
                    matched = True
                    break
            if matched:
                continue
            if data[i] == 0xFF:
                output.extend(b"\xFF\x00\xFF")
            else:
                output.append(data[i])
            i += 1
        return bytes(output)

    @staticmethod
    def dictionary_decode(data: bytes, patterns: List[bytes]) -> bytes:
        output = bytearray()
        i = 0
        while i < len(data):
            value = data[i]
            i += 1
            if value != 0xFF:
                output.append(value)
                continue
            if i >= len(data):
                raise ValueError("Invalid dictionary data")
            tag = data[i]
            i += 1
            if tag == 0:
                if i >= len(data):
                    raise ValueError("Invalid dictionary literal")
                output.append(data[i])
                i += 1
            elif tag == 2:
                if i >= len(data):
                    raise ValueError("Invalid dictionary token")
                idx = data[i]
                i += 1
                if idx >= len(patterns):
                    raise ValueError("Invalid dictionary index")
                output.extend(patterns[idx])
            else:
                raise ValueError("Invalid dictionary tag")
        return bytes(output)

    def create_delta(self, base: bytes, new_data: bytes) -> Optional[bytes]:
        if len(base) != len(new_data):
            return None
        output = bytearray()
        i = 0
        while i < len(new_data):
            if base[i] == new_data[i]:
                i += 1
                continue
            start = i
            while i < len(new_data) and base[i] != new_data[i]:
                i += 1
            length = i - start
            xor_data = bytes(a ^ b for a, b in zip(base[start:i], new_data[start:i]))
            output.extend(start.to_bytes(8, "big"))
            output.extend(length.to_bytes(4, "big"))
            output.extend(xor_data)
        return bytes(output)

    @staticmethod
    def apply_delta(base: bytes, delta: bytes) -> bytes:
        output = bytearray(base)
        i = 0
        while i < len(delta):
            if i + 12 > len(delta):
                raise ValueError("Invalid Delta header")
            start = int.from_bytes(delta[i:i+8], "big")
            length = int.from_bytes(delta[i+8:i+12], "big")
            i += 12
            if start + length > len(output) or i + length > len(delta):
                raise ValueError("Invalid Delta bounds")
            xor_data = delta[i:i+length]
            i += length
            for j, val in enumerate(xor_data):
                output[start+j] ^= val
        return bytes(output)

    # ============================================================
    # T — History
    # ============================================================

    def find_history_base(self, data: bytes) -> Optional[Dict]:
        for item in reversed(self.history):
            if item["size"] == len(data):
                return item
        return None

    def record_history(self, data: bytes):
        item = {"fingerprint": self._sha256(data), "size": len(data), "data": data}
        self.history.append(item)
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

    def get_history_score(self, strategy_name: str) -> float:
        if not self.history:
            return 0.0
        total = len(self.history)
        success = sum(1 for h in self.history if h.get("method") == strategy_name)
        return success / total if total > 0 else 0.0

    # ============================================================
    # C — Coherence: 데이터-전략 정합성
    # ============================================================

    def calculate_coherence(self, analysis: Dict, strategy: Dict) -> float:
        subtype = analysis.get("subtype", "plain")
        pattern_score = analysis.get("pattern_score", 0.0)
        structural_repeat = analysis.get("structural_repeat", 0.0)
        strategy_name = strategy.get("name", "")

        score = 0.5

        if subtype == "json":
            if "dict" in strategy_name:
                score += 0.3
        elif subtype == "log":
            if "delta" in strategy_name:
                score += 0.4
            elif "dict" in strategy_name:
                score += 0.2
        elif subtype == "csv":
            if "dict" in strategy_name:
                score += 0.3
        elif subtype == "repetitive":
            if "rle" in strategy_name:
                score += 0.4
            elif "dict" in strategy_name:
                score += 0.2

        if pattern_score > 0.5 and "dict" in strategy_name:
            score += 0.2
        if structural_repeat > 0.3 and ("rle" in strategy_name or "dict" in strategy_name):
            score += 0.2

        return min(1.0, score)

    # ============================================================
    # η — Efficiency: 실제 압축 측정
    # ============================================================

    def measure_efficiency(self, data: bytes, compressed: bytes, metadata: Dict = None) -> float:
        raw_size = len(data)
        comp_size = len(compressed)

        overhead = 0
        if metadata:
            if metadata.get("patterns"):
                overhead += len(metadata["patterns"]) * 12
            if metadata.get("base_fingerprint"):
                overhead += 32
            overhead = int(overhead * 1.33)

        total_size = comp_size + overhead
        ratio = total_size / raw_size if raw_size > 0 else 1.0
        efficiency = 1.0 - ratio
        return max(0.0, min(1.0, efficiency))

    # ============================================================
    # R — Resonance: 통합 점수
    # ============================================================

    def calculate_resonance(self, coherence: float, efficiency: float, history_score: float) -> float:
        history_factor = 1.0 + history_score * 0.2
        resonance = coherence * efficiency * history_factor
        return min(1.0, resonance)

    # ============================================================
    # 전략 생성
    # ============================================================

    def _generate_candidates(self, data: bytes, analysis: Dict) -> List[Dict]:
        candidates = []
        subtype = analysis.get("subtype", "plain")
        pattern_score = analysis.get("pattern_score", 0.0)
        structural_repeat = analysis.get("structural_repeat", 0.0)

        # 항상 포함
        candidates.append({"name": "raw", "preprocess": None, "preprocess_name": None})

        # zlib
        candidates.append({"name": "zlib-3", "preprocess": None, "preprocess_name": "zlib"})
        candidates.append({"name": "zlib-6", "preprocess": None, "preprocess_name": "zlib"})
        candidates.append({"name": "zlib-9", "preprocess": None, "preprocess_name": "zlib"})

        # RLE: 반복 데이터
        if analysis["repeat_ratio"] > 0.2 or structural_repeat > 0.2:
            candidates.append({"name": "rle", "preprocess": self.rle_encode, "preprocess_name": "RLE"})

        # Dictionary: 패턴 점수 높음
        if pattern_score > 0.3:
            patterns = self.build_dictionary(data)
            if patterns:
                candidates.append({
                    "name": f"dict({len(patterns)})",
                    "preprocess": lambda d: self.dictionary_encode(d, patterns),
                    "preprocess_name": "Dictionary",
                    "patterns": patterns
                })

        # Delta: History 유사 데이터
        history_base = self.find_history_base(data)
        if history_base is not None:
            delta_raw = self.create_delta(history_base["data"], data)
            if delta_raw is not None and len(delta_raw) < len(data) * 0.5:
                candidates.append({
                    "name": "delta",
                    "preprocess": lambda d: self.create_delta(history_base["data"], d),
                    "preprocess_name": "Delta",
                    "base_fingerprint": history_base["fingerprint"]
                })

        return candidates

    def _select_zlib_level(self, analysis: Dict, strategy_name: str) -> int:
        if "zlib-3" in strategy_name:
            return 3
        if "zlib-6" in strategy_name:
            return 6
        if "zlib-9" in strategy_name:
            return 9

        entropy = analysis.get("entropy", 8.0)
        if entropy < 4.0:
            return 9
        elif entropy < 6.0:
            return 6
        else:
            return 3

    # ============================================================
    # 통합 압축 (FWR 5축 + R)
    # ============================================================

    def compress(self, data, callback=None) -> Dict[str, Any]:
        start_time = time.time()
        raw = self._to_bytes(data)
        raw_size = len(raw)

        if callback:
            callback("")
            callback("╔══════════════════════════════════════════════════════════════╗")
            callback("║           📦 FWRZIP v2.1 — FWR 압축 시작                   ║")
            callback("╚══════════════════════════════════════════════════════════════╝")
            callback(f"  원본 크기: {self._format_size(raw_size)} ({raw_size:,} bytes)")

        # [F] 분석
        analysis = self.analyze_data(raw)

        if callback:
            callback("")
            callback("  ────── [F] Flow — 데이터 분석 ──────")
            callback(f"    타입        : {analysis['type']}/{analysis['subtype']}")
            callback(f"    엔트로피    : {analysis['entropy']:.4f}")
            callback(f"    반복률      : {analysis['repeat_ratio']:.4f}")
            callback(f"    구조 반복률 : {analysis.get('structural_repeat', 0):.4f}")
            callback(f"    패턴 점수   : {analysis['pattern_score']:.4f}")
            callback(f"    압축 가능   : {'✅' if analysis.get('compressible', True) else '❌'}")

        # 압축 불가능 (이미 압축된 데이터)
        if not analysis.get("compressible", True):
            if callback:
                callback("")
                callback("  ⏭️  이미 압축된 데이터로 판단되어 압축을 생략합니다.")
            return self._create_result(raw, "raw_skip", raw, raw, analysis, start_time)

        # 너무 작은 데이터
        if raw_size < 200:
            if callback:
                callback("")
                callback("  ⏭️  데이터가 너무 작아 압축을 생략합니다.")
            return self._create_result(raw, "raw_skip", raw, raw, analysis, start_time)

        # [W] 전략 생성
        strategies = self._generate_candidates(raw, analysis)

        if callback:
            callback("")
            callback(f"  ────── [W] Wave — 전략 생성 ({len(strategies)}개) ──────")

        # [C] + [η] + [T] → [R] 평가
        if callback:
            callback("")
            callback("  ────── [C] Coherence · [η] Efficiency · [T] History ──────")
            callback("  ────── [R] Resonance — 통합 점수 ──────")
            callback("")

        scored = []

        for strat in strategies:
            name = strat["name"]

            if strat["preprocess"]:
                preprocessed = strat["preprocess"](raw)
            else:
                preprocessed = raw

            if name == "raw":
                compressed = preprocessed
                level = None
            else:
                level = self._select_zlib_level(analysis, name)
                compressed = zlib.compress(preprocessed, level)

            metadata = {}
            if strat.get("patterns"):
                metadata["patterns"] = strat["patterns"]
            if strat.get("base_fingerprint"):
                metadata["base_fingerprint"] = strat["base_fingerprint"]

            coherence = self.calculate_coherence(analysis, strat)
            efficiency = self.measure_efficiency(raw, compressed, metadata)
            history_score = self.get_history_score(name)
            resonance = self.calculate_resonance(coherence, efficiency, history_score)

            scored.append({
                "name": name,
                "level": level,
                "compressed": compressed,
                "size": len(compressed),
                "metadata": metadata,
                "coherence": coherence,
                "efficiency": efficiency,
                "history_score": history_score,
                "resonance": resonance,
                "preprocess_name": strat.get("preprocess_name")
            })

            if callback:
                cb = f"    {name:<18} "
                cb += f"size={self._format_size(len(compressed)):<12} "
                cb += f"C={coherence:.2f} η={efficiency:.2f} "
                cb += f"R={resonance:.2f}"
                callback(cb)

        # 최종 선택 (R 점수 기반)
        best = max(scored, key=lambda x: x["resonance"])

        if callback:
            callback("")
            callback(f"  ✅ 최적 전략: {best['name']}")
            callback(f"     압축 크기: {self._format_size(best['size'])} ({best['size']:,} bytes)")
            callback(f"     R 점수   : {best['resonance']:.4f}")
            callback(f"     C 점수   : {best['coherence']:.4f}")
            callback(f"     η 점수   : {best['efficiency']:.4f}")

        compressed_data = best["compressed"]
        normalized = base64.b64encode(compressed_data).decode("ascii")

        result = self._create_result(
            raw, best["name"], compressed_data, normalized,
            analysis, start_time, best
        )

        self.record_history(raw)

        if callback:
            callback("")
            callback("  ────── 📊 최종 압축 결과 ──────")
            callback(f"    원본 크기   : {self._format_size(raw_size)} ({raw_size:,} bytes)")
            callback(f"    압축 크기   : {self._format_size(len(compressed_data))} ({len(compressed_data):,} bytes)")
            callback(f"    절약 크기   : {self._format_size(raw_size - len(compressed_data))} ({raw_size - len(compressed_data):,} bytes)")
            callback(f"    압축률      : {result['compression_percent']:.2f}%")
            callback(f"    절약률      : {result['saved_percent']:.2f}%")
            callback(f"    소요 시간   : {result['elapsed_time']:.3f}초")
            callback("")
            callback("╔══════════════════════════════════════════════════════════════╗")
            callback("║           ✅ FWRZIP v2.1 압축 완료                          ║")
            callback("╚══════════════════════════════════════════════════════════════╝")

        return result

    def _create_result(self, raw: bytes, method: str, compressed: bytes, normalized: str,
                       analysis: Dict, start_time: float, best: Dict = None) -> Dict:
        raw_size = len(raw)
        comp_size = len(compressed)

        result = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "method": method,
            "original_size": raw_size,
            "compressed_size": comp_size,
            "compression_ratio": comp_size / raw_size if raw_size > 0 else 1.0,
            "compression_percent": (comp_size / raw_size * 100) if raw_size > 0 else 100.0,
            "saved_bytes": raw_size - comp_size,
            "saved_percent": ((raw_size - comp_size) / raw_size * 100) if raw_size > 0 else 0.0,
            "compressed_data": normalized,
            "analysis": analysis,
            "fingerprint": self._sha256(raw),
            "elapsed_time": time.time() - start_time,
            "history_match": self.find_history_base(raw) is not None
        }

        if best:
            result["best_candidate"] = best["name"]
            result["resonance_score"] = best.get("resonance", 0)
            result["coherence_score"] = best.get("coherence", 0)
            result["efficiency_score"] = best.get("efficiency", 0)

        return result

    # ============================================================
    # 압축 해제
    # ============================================================

    def decompress(self, compressed_data, callback=None) -> bytes:
        if isinstance(compressed_data, str):
            try:
                compressed_data = json.loads(compressed_data)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid FWRZIP JSON") from exc

        if not isinstance(compressed_data, dict):
            raise TypeError("compressed_data must be dict")

        if compressed_data.get("format") != self.FORMAT:
            raise ValueError(f"Unsupported format: {compressed_data.get('format')}")

        method = compressed_data.get("method")
        payload = base64.b64decode(compressed_data["compressed_data"].encode("ascii"))

        if callback:
            callback("")
            callback("╔══════════════════════════════════════════════════════════════╗")
            callback("║           📂 FWRZIP v2.1 — 압축 해제 시작                  ║")
            callback("╚══════════════════════════════════════════════════════════════╝")
            callback(f"  방식: {method}")

        if method == "raw_skip":
            if callback:
                callback("  ⏭️  압축 생략된 데이터")
            return payload

        if "dict" in method:
            patterns = compressed_data.get("patterns", [])
            if patterns:
                patterns = [base64.b64decode(p.encode("ascii")) for p in patterns]
                dict_data = zlib.decompress(payload)
                result = self.dictionary_decode(dict_data, patterns)
            else:
                result = zlib.decompress(payload)

        elif method == "rle":
            rle_data = zlib.decompress(payload)
            result = self.rle_decode(rle_data)

        elif method == "delta":
            base_fingerprint = compressed_data.get("base_fingerprint")
            if not base_fingerprint:
                raise ValueError("Delta base fingerprint missing")
            base = None
            for item in reversed(self.history):
                if item["fingerprint"] == base_fingerprint:
                    base = item["data"]
                    break
            if base is None:
                raise ValueError("Delta base not found in history")
            delta = zlib.decompress(payload)
            result = self.apply_delta(base, delta)

        elif method in ("zlib-3", "zlib-6", "zlib-9", "zlib"):
            result = zlib.decompress(payload)

        else:
            result = payload

        # 검증
        expected_size = compressed_data.get("original_size")
        if expected_size is not None and len(result) != expected_size:
            raise ValueError(f"Size mismatch: expected {expected_size}, got {len(result)}")

        expected_hash = compressed_data.get("fingerprint")
        if expected_hash and self._sha256(result) != expected_hash:
            raise ValueError("SHA-256 integrity check failed")

        if callback:
            callback(f"  복원 크기: {self._format_size(len(result))} ({len(result):,} bytes)")
            callback("")
            callback("╔══════════════════════════════════════════════════════════════╗")
            callback("║           ✅ FWRZIP v2.1 압축 해제 완료                    ║")
            callback("╚══════════════════════════════════════════════════════════════╝")

        return result

    # ============================================================
    # 파일 I/O
    # ============================================================

    def compress_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        with open(input_file, "rb") as f:
            data = f.read()

        result = self.compress(data, callback=self._log)

        if output_file is None:
            output_file = input_file + ".fwrz"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_file

    def decompress_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        with open(input_file, "r", encoding="utf-8") as f:
            compressed = json.load(f)

        result = self.decompress(compressed, callback=self._log)

        if output_file is None:
            if input_file.endswith(".fwrz"):
                output_file = input_file[:-5] + "_recovered"
            else:
                output_file = input_file + "_recovered"

        with open(output_file, "wb") as f:
            f.write(result)

        return output_file


# ============================================================
# CLI
# ============================================================

def main_cli():
    parser = argparse.ArgumentParser(
        description="FWRZIP v2.1 — FWR 기반 적응형 압축기",
        epilog="텍스트(JSON/LOG/CSV)에 최적화, 이미지/바이너리는 raw_skip"
    )
    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # compress
    cp = subparsers.add_parser("compress", help="파일 압축")
    cp.add_argument("input", help="입력 파일")
    cp.add_argument("-o", "--output", help="출력 파일 (.fwrz)")
    cp.add_argument("-q", "--quiet", action="store_true", help="출력 최소화")

    # decompress
    dp = subparsers.add_parser("decompress", help="파일 압축 해제")
    dp.add_argument("input", help="입력 파일 (.fwrz)")
    dp.add_argument("-o", "--output", help="출력 파일")
    dp.add_argument("-q", "--quiet", action="store_true", help="출력 최소화")

    args = parser.parse_args()

    fwrzip = FWRZIP(verbose=not getattr(args, 'quiet', False))

    if args.command == "compress":
        fwrzip.compress_file(args.input, args.output)
    elif args.command == "decompress":
        fwrzip.decompress_file(args.input, args.output)
    else:
        parser.print_help()


# ============================================================
# PyQt GUI
# ============================================================

if QT_AVAILABLE:

    class CompressionWorker(QThread):
        log_signal = pyqtSignal(str)
        finished_signal = pyqtSignal(object)
        error_signal = pyqtSignal(str)

        def __init__(self, compressor, mode, data, compressed_data=None):
            super().__init__()
            self.compressor = compressor
            self.mode = mode
            self.data = data
            self.compressed_data = compressed_data
            self._is_running = True

        def stop(self):
            self._is_running = False
            self.quit()
            self.wait()

        def run(self):
            try:
                if self.mode == 'compress':
                    result = self.compressor.compress(self.data, callback=self.log_signal.emit)
                    if self._is_running:
                        self.finished_signal.emit(result)
                else:
                    result = self.compressor.decompress(self.compressed_data, callback=self.log_signal.emit)
                    if self._is_running:
                        self.finished_signal.emit(result)
            except Exception as e:
                if self._is_running:
                    self.error_signal.emit(str(e))

    class FWRZIPWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.compressor = FWRZIP()
            self.current_file = None
            self.worker = None
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("FWRZIP v2.1 — FWR 적응형 압축기")
            self.setGeometry(100, 100, 1000, 700)

            font = QFont("Consolas", 10)
            self.setFont(font)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            file_layout = QHBoxLayout()
            self.file_label = QLabel("선택된 파일: 없음")
            self.file_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
            file_layout.addWidget(self.file_label, 1)

            self.select_btn = QPushButton("📂 파일 선택")
            self.select_btn.clicked.connect(self.select_file)
            file_layout.addWidget(self.select_btn)

            self.compress_btn = QPushButton("📦 FWR 압축")
            self.compress_btn.clicked.connect(self.compress_file)
            self.compress_btn.setEnabled(False)
            file_layout.addWidget(self.compress_btn)

            self.decompress_btn = QPushButton("📂 압축 해제")
            self.decompress_btn.clicked.connect(self.decompress_file)
            self.decompress_btn.setEnabled(False)
            file_layout.addWidget(self.decompress_btn)

            layout.addLayout(file_layout)

            self.progress = QProgressBar()
            self.progress.setVisible(False)
            layout.addWidget(self.progress)

            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setFont(QFont("Consolas", 9))
            self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
            layout.addWidget(self.log_text, 1)

            self.statusBar().showMessage("준비")

        def closeEvent(self, event):
            if self.worker and self.worker.isRunning():
                self.worker.stop()
            event.accept()

        def select_file(self):
            path, _ = QFileDialog.getOpenFileName(self, "파일 선택")
            if path:
                self.current_file = path
                size = os.path.getsize(path)
                self.file_label.setText(f"선택된 파일: {os.path.basename(path)} ({size:,} bytes)")
                self.compress_btn.setEnabled(True)
                self.decompress_btn.setEnabled(True)
                self.log_text.clear()
                self.log_text.append(f"[FWRZIP] 파일 선택: {path} ({size:,} bytes)")

        def compress_file(self):
            if not self.current_file:
                return
            with open(self.current_file, 'rb') as f:
                data = f.read()
            self.log_text.append("[FWRZIP] 압축 시작...")
            self.progress.setVisible(True)
            self.progress.setValue(0)

            self.worker = CompressionWorker(self.compressor, 'compress', data)
            self.worker.log_signal.connect(self.append_log)
            self.worker.finished_signal.connect(self.on_compress_finished)
            self.worker.error_signal.connect(self.on_error)
            self.worker.start()

        def decompress_file(self):
            if not self.current_file:
                return
            try:
                with open(self.current_file, 'r', encoding='utf-8') as f:
                    compressed = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "오류", f"파일을 읽을 수 없습니다: {e}")
                return
            self.log_text.append("[FWRZIP] 압축 해제 시작...")
            self.progress.setVisible(True)
            self.progress.setValue(0)

            self.worker = CompressionWorker(self.compressor, 'decompress', None, compressed)
            self.worker.log_signal.connect(self.append_log)
            self.worker.finished_signal.connect(self.on_decompress_finished)
            self.worker.error_signal.connect(self.on_error)
            self.worker.start()

        def append_log(self, msg):
            self.log_text.append(msg)
            self.log_text.moveCursor(QTextCursor.End)
            QApplication.processEvents()

        def on_compress_finished(self, result):
            self.progress.setVisible(False)
            self.statusBar().showMessage("압축 완료")
            if self.current_file:
                out = self.current_file + '.fwrz'
                reply = QMessageBox.question(self, "저장", f"압축 결과 저장?\n{out}",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    with open(out, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    self.log_text.append(f"[FWRZIP] 저장 완료: {out}")
            self.progress.setValue(100)
            self.worker = None

        def on_decompress_finished(self, result):
            self.progress.setVisible(False)
            self.statusBar().showMessage("압축 해제 완료")
            if self.current_file:
                out = self.current_file
                if out.endswith('.fwrz'):
                    out = out[:-5] + '_recovered'
                else:
                    out = out + '_recovered'
                reply = QMessageBox.question(self, "저장", f"복원 결과 저장?\n{out}",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    with open(out, 'wb') as f:
                        f.write(result)
                    self.log_text.append(f"[FWRZIP] 저장 완료: {out}")
            self.progress.setValue(100)
            self.worker = None

        def on_error(self, msg):
            self.progress.setVisible(False)
            self.statusBar().showMessage("오류 발생")
            self.log_text.append(f"\n[ERROR] {msg}")
            QMessageBox.critical(self, "오류", msg)
            self.worker = None

    def main_gui():
        app = QApplication(sys.argv)
        window = FWRZIPWindow()
        window.show()
        sys.exit(app.exec_())


# ============================================================
# 메인
# ============================================================

def main():
    if len(sys.argv) > 1:
        main_cli()
    else:
        if QT_AVAILABLE:
            main_gui()
        else:
            print("PyQt5가 설치되지 않았습니다. CLI 모드로 실행합니다.")
            print("사용법: python fwrzip.py compress <file>")
            main_cli()


if __name__ == "__main__":
    main()
