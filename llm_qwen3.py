# -*- coding: utf-8 -*-
"""llm_qwen3.py

Qwen3 权重建议器：
  - 根据 prompt_zh_qw.md + 当前 PlanConfig + (case)config.json + opt_history + DVH 摘要
  - 调用本地 transformers 模型生成下一轮权重建议
  - 解析输出，得到 (W_MIN, W_MAX, W_NEEDLE, W_OAR)

说明：
  - 该模块仅负责：构建 prompt / 调用模型 / 解析输出。
  - 不负责运行优化引擎。
  - 解析逻辑做了容错：若解析失败，返回上一轮权重（并给出 warning）。

NOTE(2026-01-26):
  - config.json 里的 BONE.mha 仅用于穿刺避让；剂量优化与权重建议时不应把 BONE 当作 OAR。
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import asdict
from typing import Dict, Optional, Tuple, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd

from optim_engine import Weights, clamp_weight


# =========================
# Helpers
# =========================

# BONE 不作为剂量 OAR（与 optim_engine 保持一致）
EXCLUDED_OAR_STEMS = {"BONE"}  # compare via upper()


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _basename_list(xs: Any) -> List[str]:
    """把路径列表转成 stem（不含扩展名），保持原顺序；不过滤（过滤在上层做）。"""
    if not isinstance(xs, list):
        return []
    out = []
    for p in xs:
        if isinstance(p, str) and p.strip():
            stem = os.path.splitext(os.path.basename(p))[0]
            out.append(stem)
    return out


def _infer_oars_from_dvh_dir(dvh_dir: str) -> List[str]:
    """从 DVH 目录推断 OAR：除 GTV 之外的 csv 文件名（不含扩展名）"""
    names = []
    if not dvh_dir or not os.path.isdir(dvh_dir):
        return names
    for fn in sorted(os.listdir(dvh_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        name = os.path.splitext(fn)[0]
        if name.upper() == "GTV":
            continue
        names.append(name)
    return sorted(set([n for n in names if n]))


def summarize_case_config(config_json_path: str, dvh_dir: str = "") -> str:
    """
    仅输出“剂量优化相关的 OAR 列表”，不要把 config.json 内容一股脑塞进 prompt。
    规则：
      - BONE 仅用于穿刺避让，不计入 dose-relevant OAR（但会单独提示是否存在）
      - OAR 来源：优先 config.json paths.oars，其次用 DVH 目录推断做补充
    """
    if not config_json_path or (not os.path.exists(config_json_path)):
        return "- OAR list (dose-relevant): <missing config.json>"

    cfg = _read_json(config_json_path)
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    # config.json 中显式列出的 oars（可能包含 BONE）
    oars_cfg_all = _basename_list(paths.get("oars", []))

    # DVH 推断的结构（可能包含 BONE）
    oars_dvh_all = _infer_oars_from_dvh_dir(dvh_dir)

    # 合并去重（保持稳定排序）
    oars_all = sorted(set([*(oars_cfg_all or []), *(oars_dvh_all or [])]))

    # 是否存在 BONE（穿刺避让用）
    bone_present = any(n.upper() == "BONE" for n in oars_all)

    # 剂量相关 OAR：排除 BONE
    dose_oars = [n for n in oars_all if n and (n.upper() not in EXCLUDED_OAR_STEMS)]
    has_dose_oar = len(dose_oars) > 0

    lines = []
    lines.append(f"- Has OAR (dose-relevant): {str(has_dose_oar)}")
    lines.append(f"- OAR list (dose-relevant): {', '.join(dose_oars) if dose_oars else '<empty>'}")
    lines.append(f"- BONE present (puncture-avoidance only, NOT dose-optimized): {str(bone_present)}")
    return "\n".join(lines)


def extract_dvh_summary(df: pd.DataFrame, name: str, max_points: int = 10) -> str:
    """抽取典型剂量范围的点（8000-16000 cGy），并限制点数，避免 prompt 过长。"""
    if "Dose" not in df.columns:
        return f"{name} DVH：<missing Dose column>"
    vol_col = " Volume" if " Volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is None:
        return f"{name} DVH：<missing Volume column>"

    key_points = df[df["Dose"].between(8000, 16000)]
    if len(key_points) == 0:
        key_points = df.tail(min(max_points, len(df)))

    # 采样/截断点数
    if len(key_points) > max_points:
        key_points = key_points.iloc[:: max(1, len(key_points) // max_points)].head(max_points)

    summary = f"{name} DVH（剂量/体积%）: \n" + ", ".join(
        [f"{int(row['Dose'])}cGy: {float(row[vol_col]):.2f}%" for _, row in key_points.iterrows()]
    )
    return summary


def format_opt_history(df: pd.DataFrame, tail_k: int = 5) -> str:
    """优化历史摘要：与 optim_engine 导出字段严格一致，并自动展开每个 OAR。"""
    if df is None or len(df) == 0:
        return "- <empty>"

    d = df.copy()

    # 1) 决定“轮次”列名（优先 Iters，其次 Round）
    iter_col = None
    if "Iters" in d.columns:
        iter_col = "Iters"
    elif "Round" in d.columns:
        iter_col = "Round"

    # 2) 去重：同一轮只保留最后一条
    if iter_col is not None:
        d[iter_col] = pd.to_numeric(d[iter_col], errors="coerce")
        d = d.drop_duplicates(subset=[iter_col], keep="last")
        d = d.sort_values(iter_col, kind="stable")
    else:
        d = d.drop_duplicates(keep="last")

    # 3) 取最后 tail_k 轮
    d_tail = d.tail(tail_k)
    cols = set(d_tail.columns)

    def g(row, key, default=0.0):
        return row[key] if key in cols and pd.notna(row[key]) else default

    def _pick(row, keys, default=0.0):
        for k in keys:
            if k in cols and pd.notna(row[k]):
                return row[k]
        return default

    # 自动发现 OAR 名称：基于 "<OAR> V50 (%)"
    oar_names = []
    suffix = " V50 (%)"
    for c in d_tail.columns:
        if isinstance(c, str) and c.endswith(suffix):
            name = c[: -len(suffix)]
            if name.upper() in EXCLUDED_OAR_STEMS:
                continue
            oar_names.append(name)
    oar_names = sorted(set([n for n in oar_names if n]))

    lines = []
    for _, row in d_tail.iterrows():
        iters_val = g(row, iter_col, 0) if iter_col is not None else g(row, "Iters", g(row, "Round", 0))
        try:
            iters = int(iters_val)
        except Exception:
            iters = 0

        v100 = float(_pick(row, ["V100 (%)", "V100"], 0.0))
        v150 = float(_pick(row, ["V150 (%)", "V150"], 0.0))
        v200 = float(_pick(row, ["V200 (%)", "V200"], 0.0))
        d90p = float(_pick(row, ["D90 (%)", "D90"], 0.0))
        n_needles = int(_pick(row, ["#needle", "Needle Number"], 0))
        n_seeds = int(_pick(row, ["#seed", "Seed Number"], 0))

        seg = (
            f"- 第{iters}轮: "
            f"V100 (%)={v100:.2f}, V150 (%)={v150:.2f}, V200 (%)={v200:.2f}, D90 (%)={d90p:.2f}, "
            f"#needle={n_needles}, #seed={n_seeds}; "
        )

        # OAR 指标
        oar_segs = []
        for on in oar_names:
            v50 = float(g(row, f"{on} V50 (%)", 0.0))
            dmean = float(g(row, f"{on} Dmean (Gy)", 0.0))
            dmax = float(g(row, f"{on} Dmax (Gy)", 0.0))
            d1 = float(g(row, f"{on} D1 (Gy)", 0.0))
            d1cc = float(g(row, f"{on} D1cc (Gy)", 0.0))
            d01 = float(g(row, f"{on} D0.1cc (Gy)", 0.0))
            oar_segs.append(
                f"{on}: V50 (%)={v50:.2f}, Dmean (Gy)={dmean:.2f}, Dmax (Gy)={dmax:.2f}, "
                f"D1 (Gy)={d1:.2f}, D1cc (Gy)={d1cc:.2f}, D0.1cc (Gy)={d01:.2f}"
            )
        if oar_segs:
            seg += "OAR[" + " | ".join(oar_segs) + "]; "

        seg += (
            f"权重: W_MIN={float(g(row,'Weight minDose')):.4g}, W_MAX={float(g(row,'Weight maxDose')):.4g}, "
            f"W_NEEDLE={float(g(row,'Weight needle')):.4g}, W_OAR={float(g(row,'Weight OAR')):.4g}"
        )
        lines.append(seg)

    return "\n".join(lines)


# =========================
# Advisor
# =========================

class Qwen3WeightAdvisor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        torch_dtype: str = "float16",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.torch_dtype.lower(), torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # device_map: 让 HF 放到单卡（保持和 demo.py 的简单用法一致）
        device_map_val = self.device
        if isinstance(device_map_val, str) and device_map_val.startswith("cuda:"):
            try:
                device_map_val = int(device_map_val.split(":", 1)[1])
            except Exception:
                device_map_val = self.device

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": device_map_val},
            trust_remote_code=True,
        )
        self._model.eval()

    def build_prompt(
        self,
        prompt_template_path: str,
        plan_config_path: str,
        opt_history_csv: str,
        dvh_dir: str,
        config_json_path: str = "",
        extra_instruction: str = "请基于上述信息，给出下一轮优化权重建议，并严格遵循 markdown 代码格式。",
        max_dvh_structs: int = 8,
    ) -> str:
        prompt_template = _read_text(prompt_template_path)
        plan_config = _read_text(plan_config_path) if (plan_config_path and os.path.exists(plan_config_path)) else "<missing PlanConfig.txt>"
        opt_df = pd.read_csv(opt_history_csv) if os.path.exists(opt_history_csv) else pd.DataFrame()

        # DVH
        dvh_lines = []
        if dvh_dir and os.path.isdir(dvh_dir):
            for fn in sorted(os.listdir(dvh_dir)):
                if not fn.lower().endswith(".csv"):
                    continue
                name = os.path.splitext(fn)[0]
                # filter BONE
                if name.upper() in EXCLUDED_OAR_STEMS:
                    continue
                df = pd.read_csv(os.path.join(dvh_dir, fn))
                dvh_lines.append(extract_dvh_summary(df, name))
        # 截断结构数，避免 prompt 过长
        if len(dvh_lines) > max_dvh_structs:
            dvh_lines = dvh_lines[:max_dvh_structs] + ["<DVH truncated>"]

        # config.json summary
        cfg_summary = summarize_case_config(config_json_path, dvh_dir=dvh_dir) if config_json_path else "- <config.json not provided>"

        prompt = prompt_template.strip() + "\n\n"
        prompt += "## 病例 config.json 信息:\n" + cfg_summary.strip() + "\n\n"
        prompt += "## 当前计划配置（PlanConfig.txt）:\n" + plan_config.strip() + "\n\n"
        prompt += "## 优化历史摘要:\n" + format_opt_history(opt_df) + "\n\n"

        prompt += "## 指标单位说明:\n"
        prompt += "- GTV: V100 (%) / V150 (%) / V200 (%) / D90 (%)。其中 D90(%) = D90(cGy) / Rx(cGy) × 100。\n"
        prompt += "- 资源: #needle, #seed 为整数。\n"
        prompt += "- OAR: V50 (%) 以 0.5×Rx 为阈值；Dmean (Gy) / Dmax (Gy) / D1 (Gy) / D1cc (Gy) / D0.1cc (Gy)。\n\n"
        prompt += "参考列名顺序：V100 (%)\tV150 (%)\tV200 (%)\tD90 (%)\t#needle\t#seed\tV50 (%)\tDmean (Gy)\tDmax (Gy)\tD1 (Gy)\tD1cc (Gy)\tD0.1cc (Gy)\n\n"

        # DVH 信息（短摘要+截断）
        # if dvh_lines:
        #     prompt += "## DVH 信息摘要:\n" + "\n".join(dvh_lines) + "\n\n"

        prompt += extra_instruction
        return prompt

    def call_model(self, prompt_text: str) -> str:
        self._lazy_load()
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]  # 输入 token 数

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
            )

        # 只解码新生成的部分，避免把 prompt 也解码出来污染解析
        gen_tokens = outputs[0][input_len:]
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # -------------------------
    # parsing
    # -------------------------
    _NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    def parse_weights(self, text: str, prev: Weights) -> Tuple[Weights, Dict[str, float]]:
        """从模型输出中解析权重。

        兼容多种写法：
          - markdown 列表 + "a -> b" 或 "a → b" 或 "调整为：a → b"
          - 直接给出 W_MIN=xx 等
          - w_{gtv}^l / w_{gtv}^h / w_n / w_{oar}^h
        """
        out: Dict[str, float] = {}

        # 0) 优先解析 JSON（严格 4 个键）
        # 允许模型输出 ```json {...} ``` 或者普通 {...}
        try:
            m = re.search(r"```json\s*({[\s\S]*?})\s*```", text, flags=re.IGNORECASE)
            json_str = m.group(1) if m else None
            if json_str is None:
                m2 = re.search(r'({\s*"W_MIN"[\s\S]*?})', text)
                json_str = m2.group(1) if m2 else None

            if json_str:
                obj = json.loads(json_str)
                keys = {"W_MIN", "W_MAX", "W_NEEDLE", "W_OAR"}
                if isinstance(obj, dict) and keys.issubset(set(obj.keys())):
                    new = Weights(
                        W_MIN=clamp_weight("W_MIN", float(obj["W_MIN"])),
                        W_MAX=clamp_weight("W_MAX", float(obj["W_MAX"])),
                        W_NEEDLE=clamp_weight("W_NEEDLE", float(obj["W_NEEDLE"])),
                        W_OAR=clamp_weight("W_OAR", float(obj["W_OAR"])),
                    )
                    out = {k: float(v) for k, v in asdict(new).items()}
                    return new, out
        except Exception:
            # JSON 解析失败则继续走后面的正则容错
            pass

        def _first_num(s: str) -> Optional[float]:
            m = re.search(self._NUM, s)
            return float(m.group(0)) if m else None

        def _find_new_value(patterns) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
                if not m:
                    continue
                # 尝试取最后一个数（通常是 new value）
                nums = re.findall(self._NUM, m.group(0))
                if nums:
                    return float(nums[-1])
            return None

        # 1) 直接变量名
        out_wmin = _find_new_value([
            rf"\bW[_ ]?MIN\b\s*[:=]\s*{self._NUM}",
            rf"minDose\s*[:=]\s*{self._NUM}",
        ])
        out_wmax = _find_new_value([
            rf"\bW[_ ]?MAX\b\s*[:=]\s*{self._NUM}",
            rf"maxDose\s*[:=]\s*{self._NUM}",
        ])
        out_wneedle = _find_new_value([
            rf"\bW[_ ]?NEEDLE\b\s*[:=]\s*{self._NUM}",
            rf"\bneedle\b\s*[:=]\s*{self._NUM}",
        ])

        # 2) 公式符号写法（可能包含箭头）
        out_wmin = out_wmin if out_wmin is not None else _find_new_value([
            rf"w_\{{\\text\{{gtv\}}\}}\^l[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}{self._NUM}",
            rf"w_\{{gtv\}}\^l[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}{self._NUM}",
        ])
        out_wmax = out_wmax if out_wmax is not None else _find_new_value([
            rf"w_\{{\\text\{{gtv\}}\}}\^h[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}{self._NUM}",
            rf"w_\{{gtv\}}\^h[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}{self._NUM}",
        ])
        out_wneedle = out_wneedle if out_wneedle is not None else _find_new_value([
            rf"w_n[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}{self._NUM}",
            rf"w_\{{n\}}[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}{self._NUM}",
        ])

        # OAR 可能给多个（w_oar1^h / w_oar2^h），取最大值更保守
        oar_vals = []
        for pat in [
            rf"w_\{{\\text\{{oar\d+\}}\}}\^h[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}({self._NUM})",
            rf"w_\{{\\text\{{oar\}}\}}\^h[\s\S]*?(?:→|->|\\rightarrow|调整为|:).{{0,40}}({self._NUM})",
            rf"\bOAR\b[\s\S]*?(?:权重|weight)[\s\S]*?(?:→|->|=|调整为|:).{{0,40}}({self._NUM})",
            rf"\bW[_ ]?OAR\b\s*[:=]\s*({self._NUM})",
        ]:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                v = _first_num(m.group(1) if m.groups() else m.group(0))
                if v is not None:
                    oar_vals.append(float(v))
        out_woar = max(oar_vals) if oar_vals else None

        # assemble
        new = Weights(
            W_MIN=clamp_weight("W_MIN", out_wmin if out_wmin is not None else prev.W_MIN),
            W_MAX=clamp_weight("W_MAX", out_wmax if out_wmax is not None else prev.W_MAX),
            W_NEEDLE=clamp_weight("W_NEEDLE", out_wneedle if out_wneedle is not None else prev.W_NEEDLE),
            W_OAR=clamp_weight("W_OAR", out_woar if out_woar is not None else prev.W_OAR),
        )

        out = {k: float(v) for k, v in asdict(new).items()}
        return new, out

    def suggest_next_weights(
        self,
        prev: Weights,
        prompt_template_path: str,
        plan_config_path: str,
        opt_history_csv: str,
        dvh_dir: str,
        config_json_path: str = "",
    ) -> Dict[str, object]:
        """返回 dict，包含 raw_output / parsed_weights / prompt（便于落盘审计）。"""
        prompt = self.build_prompt(
            prompt_template_path=prompt_template_path,
            plan_config_path=plan_config_path,
            opt_history_csv=opt_history_csv,
            dvh_dir=dvh_dir,
            config_json_path=config_json_path,
        )
        raw = self.call_model(prompt)
        w_new, parsed = self.parse_weights(raw, prev)
        return {
            "prompt": prompt,
            "raw_output": raw,
            "weights": w_new,
            "weights_dict": parsed,
        }