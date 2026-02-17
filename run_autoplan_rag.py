# -*- coding: utf-8 -*-
"""
run_autoplan_rag.py

Integrates RAG into the main workflow of run_autoplan.py:
- round1 uses base weights
- round2~roundN: after each round
    1) Generate a retrieval query using the current round's PlanConfig / DVH / opt_history
    2) Retrieve knowledge context via RAGContextProvider
    3) Append the retrieved context to the llm_qwen3 prompt,
       then call Qwen3WeightAdvisor to output the next-round weights

Audit outputs (per round):
- rag_query.txt
- rag_context.txt
- llm_prompt.txt
- llm_output.txt
- llm_weights.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from llm_qwen3 import Qwen3WeightAdvisor
from optim_engine import (
    CaseCache,
    Weights,
    build_so,
    ensure_dir,
    prepare_case_cache,
    run_one_round_and_export,
)

from rag_module import RAGConfig, RAGContextProvider


# ----------------------
# small helpers
# ----------------------
def _append_history(out_csv: str, row: Dict) -> None:
    ensure_dir(os.path.dirname(out_csv))
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv, encoding="utf-8-sig")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _list_case_dirs(root_dir: str) -> List[str]:
    return [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def _read_text(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return default


def _tail_opt_history(opt_csv: str, k: int = 3) -> str:
    if not os.path.exists(opt_csv):
        return "<missing opt_history.csv>"
    try:
        df = pd.read_csv(opt_csv, encoding="utf-8-sig")
        if len(df) == 0:
            return "<empty opt_history.csv>"
        tail = df.tail(int(k))

        # Keep it concise: extract key columns if present
        cols_pref = [
            "Iters", "V100", "V150", "V200", "D90",
            "Needle Number", "Seed Number",
            "Weight minDose", "Weight maxDose", "Weight needle", "Weight OAR",
        ]
        cols = [c for c in cols_pref if c in tail.columns]
        if cols:
            tail = tail[cols]
        return tail.to_csv(index=False)
    except Exception as e:
        return f"<failed to read opt_history.csv: {type(e).__name__}: {e}>"


def _make_rag_query(
    case: str,
    round_idx: int,
    plan_config_path: str,
    dvh_dir: str,
    opt_history_csv: str,
    config_json_path: str,
) -> str:
    """
    Retrieval query builder:
    Compress the "task + current metrics + OAR situation" into
    a structured query for knowledge retrieval.
    """

    plan_cfg = _read_text(plan_config_path, default="<missing PlanConfig.txt>")
    opt_tail = _tail_opt_history(opt_history_csv, k=3)

    # Only list DVH structure names to avoid overly long query
    structs = []
    if dvh_dir and os.path.isdir(dvh_dir):
        for fn in sorted(os.listdir(dvh_dir)):
            if fn.lower().endswith(".csv"):
                structs.append(os.path.splitext(fn)[0])
    structs_str = ", ".join(structs) if structs else "<no DVH csv>"

    cfg_json = _read_text(config_json_path, default="<missing config.json>")

    query = (
        f"Brachytherapy 125I seed implantation treatment plan optimization "
        f"penalty weight adjustment experience guidelines publications.\n"
        f"Case={case}, current round={round_idx}.\n"
        f"Current DVH structures: {structs_str}.\n"
        f"Retrieve strategies for achieving adequate GTV coverage "
        f"(V100>=95%, higher D90) while reducing V150/V200 and minimizing "
        f"needle/seed count. When OARs are present, how to adjust weights "
        f"to control OAR dose (V50/Dmean/Dmax/D1/D1cc/D0.1cc, etc.).\n\n"
        f"[Recent 3-round history (csv)]\n{opt_tail}\n"
        f"[PlanConfig excerpt]\n{plan_cfg[:1200]}\n"
        f"[config.json excerpt]\n{cfg_json[:1200]}"
    )
    return query


def suggest_next_weights_with_rag(
    advisor: Qwen3WeightAdvisor,
    prev: Weights,
    prompt_template_path: str,
    plan_config_path: str,
    opt_history_csv: str,
    dvh_dir: str,
    config_json_path: str,
    rag_context_text: str,
) -> Dict[str, object]:
    """
    Without modifying llm_qwen3.py:
    concatenate RAG context to the prompt at the orchestrator level.
    """

    base_prompt = advisor.build_prompt(
        prompt_template_path=prompt_template_path,
        plan_config_path=plan_config_path,
        opt_history_csv=opt_history_csv,
        dvh_dir=dvh_dir,
        config_json_path=config_json_path,
    )

    prompt = (
        base_prompt.strip()
        + "\n\n"
        + "## Retrieved clinical / experiential knowledge (prioritize this; "
        + "if conflict occurs, follow PlanConfig and DVH):\n"
        + rag_context_text.strip()
        + "\n\n"
        + "Before suggesting next-round weights, verify whether the RAG content "
        + "matches this case (OARs, prescription dose, current metrics). "
        + "Ignore irrelevant content."
    )

    raw = advisor.call_model(prompt)
    w_new, parsed = advisor.parse_weights(raw, prev)

    return {
        "prompt": prompt,
        "raw_output": raw,
        "weights": w_new,
        "weights_dict": parsed,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()

    # data & result
    parser.add_argument("--root_dir", type=str, default="/hdd2/xz/llm-autoplan/data/")
    parser.add_argument("--result_root", type=str, default="/hdd2/xz/llm-autoplan/result_rag/")
    parser.add_argument("--so_path", type=str, default="/hdd2/xz/libDoseDll/build/libDoseDll.so")

    # LLM
    parser.add_argument("--model_path", type=str, default="/hdd1/xz/models/DeepSeek-R1-Distill-Qwen-14B/")
    parser.add_argument("--prompt_path", type=str, default="prompt_en_qw.md")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # rounds & base weights
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--base_w_min", type=float, default=20.0)
    parser.add_argument("--base_w_max", type=float, default=0.01)
    parser.add_argument("--base_w_needle", type=float, default=600.0)
    parser.add_argument("--base_w_oar", type=float, default=1.0)

    # rag
    parser.add_argument("--knowledge_base_dir", type=str, default="./knowledge_base")
    parser.add_argument("--chroma_dir", type=str, default="./vector_db")
    parser.add_argument("--embed_model_path", type=str, default="/hdd1/xz/models/BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker_model_path", type=str, default="/hdd1/xz/models/BAAI/bge-reranker-large")
    parser.add_argument("--embed_device", type=str, default="cuda")
    parser.add_argument("--reranker_device", type=str, default="cuda:1")
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--rag_min_score", type=float, default=0.3)
    parser.add_argument("--rag_max_chars", type=int, default=6000)

    parser.add_argument(
        "--rag_mode",
        type=str,
        default="per_round",
        choices=["per_round", "per_case"],
        help="per_round: retrieve each round; per_case: retrieve once per case",
    )

    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--verbose", default=True)

    args = parser.parse_args(argv)

    if not os.path.isdir(args.root_dir):
        raise FileNotFoundError(args.root_dir)
    ensure_dir(args.result_root)

    dll = build_so(args.so_path)

    advisor = Qwen3WeightAdvisor(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    base_w = Weights(
        W_MIN=float(args.base_w_min),
        W_MAX=float(args.base_w_max),
        W_NEEDLE=float(args.base_w_needle),
        W_OAR=float(args.base_w_oar),
    )

    rag_cfg = RAGConfig(
        knowledge_base_dir=args.knowledge_base_dir,
        chroma_dir=args.chroma_dir,
        embed_model_path=args.embed_model_path,
        reranker_model_path=args.reranker_model_path,
        embed_device=args.embed_device,
        reranker_device=args.reranker_device,
        top_k=int(args.rag_top_k),
        min_score_threshold=float(args.rag_min_score),
        max_context_chars=int(args.rag_max_chars),
    )
    rag = RAGContextProvider(rag_cfg)

    global_history = os.path.join(args.result_root, "opt_history.csv")
    case_dirs = _list_case_dirs(args.root_dir)
    if args.max_cases and args.max_cases > 0:
        case_dirs = case_dirs[: int(args.max_cases)]

    print(f"[INFO] Found {len(case_dirs)} case folders under {args.root_dir}")

    ok, fail = 0, 0
    for case_dir in case_dirs:
        case = os.path.basename(case_dir)
        cfg_path = os.path.join(case_dir, "config.json")
        if not os.path.exists(cfg_path):
            print(f"\n[SKIP {case}] no config.json")
            continue

        print("\n" + "=" * 80)
        print(f"[RUN] {case}")

        try:
            cache: CaseCache | None = prepare_case_cache(case_dir, dll, verbose=args.verbose)
            if cache is None:
                continue

            case_result_root = os.path.join(args.result_root, case)
            ensure_dir(case_result_root)
            case_history = os.path.join(case_result_root, "opt_history.csv")

            cached_case_rag_context: Optional[str] = None
            w = base_w

            for r in range(1, int(args.rounds) + 1):
                round_dir = os.path.join(case_result_root, f"round{r}")

                summary = run_one_round_and_export(
                    cache=cache,
                    dll=dll,
                    w=w,
                    round_dir=round_dir,
                    verbose=args.verbose,
                )

                row = {
                    "Iters": int(r),
                    "Case": case,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Prescription Dose(cGy)": float(cache.rx_cgy),
                    "Seed Activity(mCi)": float(cache.seed_activity_mCi),
                    **{k: summary[k] for k in [
                        "Weight minDose", "Weight maxDose", "Weight needle", "Weight OAR",
                        "Needle Number", "Seed Number",
                        "V100", "V150", "V200", "D90",
                        "FinalObjective",
                        "RoundDir",
                    ]},
                }

                _append_history(case_history, row)
                _append_history(global_history, row)

                print(
                    f"[{case}][round{r}] "
                    f"W=({w.W_MIN:.4g},{w.W_MAX:.4g},{w.W_NEEDLE:.4g},{w.W_OAR:.4g}) "
                    f"Needles={row['Needle Number']} Seeds={row['Seed Number']} "
                    f"V100={row['V100']:.2f} D90={row['D90']:.1f} obj={row['FinalObjective']:.3e}"
                )

                if r < int(args.rounds):
                    dvh_dir = os.path.join(round_dir, "DVH")
                    plan_cfg = os.path.join(round_dir, "PlanConfig.txt")

                    if args.rag_mode == "per_case":
                        if cached_case_rag_context is None:
                            q = _make_rag_query(case, r, plan_cfg, dvh_dir, case_history, cfg_path)
                            rag_ctx = rag.build_context_text(q)
                            cached_case_rag_context = rag_ctx
                        else:
                            q = "<per_case mode: reuse cached query/context>"
                            rag_ctx = cached_case_rag_context
                    else:
                        q = _make_rag_query(case, r, plan_cfg, dvh_dir, case_history, cfg_path)
                        rag_ctx = rag.build_context_text(q)

                    with open(os.path.join(round_dir, "rag_query.txt"), "w", encoding="utf-8") as f:
                        f.write(q)
                    with open(os.path.join(round_dir, "rag_context.txt"), "w", encoding="utf-8") as f:
                        f.write(rag_ctx)

                    llm_res = suggest_next_weights_with_rag(
                        advisor=advisor,
                        prev=w,
                        prompt_template_path=args.prompt_path,
                        plan_config_path=plan_cfg,
                        opt_history_csv=case_history,
                        dvh_dir=dvh_dir,
                        config_json_path=cfg_path,
                        rag_context_text=rag_ctx,
                    )

                    with open(os.path.join(round_dir, "llm_prompt.txt"), "w", encoding="utf-8") as f:
                        f.write(llm_res["prompt"])
                    with open(os.path.join(round_dir, "llm_output.txt"), "w", encoding="utf-8") as f:
                        f.write(llm_res["raw_output"])
                    with open(os.path.join(round_dir, "llm_weights.json"), "w", encoding="utf-8") as f:
                        json.dump(llm_res["weights_dict"], f, ensure_ascii=False, indent=2)

                    w = llm_res["weights"]

            ok += 1

        except Exception as e:
            fail += 1
            print(f"[FAIL {case}] {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print(f"[DONE] ok={ok}, fail={fail}")
    print(f"[HISTORY] {global_history}")


if __name__ == "__main__":
    main()
