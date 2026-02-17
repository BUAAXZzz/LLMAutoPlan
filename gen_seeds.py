import os
import numpy as np
import pandas as pd


def compute_seeds_from_needles_csv(
    csv_path: str,
    step_mm: float = 5.0,
    extra_outside: int = 2,   # tv_other 外额外多放几颗
    eps: float = 1e-6
) -> pd.DataFrame:
    """
    从 needles.csv 生成粒子坐标：
    - 粒子以 target 为第一颗
    - 沿 target -> tv_other 方向每 step_mm 放一颗，直到 tv_other（不超过；若刚好落在 tv_other 上则包含）
    - 到 tv_other 后，再沿同方向往外额外放 extra_outside 颗（每 step_mm）
    返回 seeds DataFrame:
      [needle_id, seed_index, dist_from_target_mm, x, y, z]
    """
    df = pd.read_csv(csv_path)

    required = [
        "id",
        "pin_x", "pin_y", "pin_z",
        "target_x", "target_y", "target_z",
        "tv_other_x", "tv_other_y", "tv_other_z",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"needles.csv 缺少列: {c}")

    seeds_rows = []

    for _, row in df.iterrows():
        nid = int(row["id"])

        target = np.array([row["target_x"], row["target_y"], row["target_z"]], dtype=float)
        tv_other = np.array([row["tv_other_x"], row["tv_other_y"], row["tv_other_z"]], dtype=float)

        v = tv_other - target
        L = float(np.linalg.norm(v))

        if not np.isfinite(L) or L < eps:
            continue

        u = v / L  # target -> tv_other（朝针头方向）

        # 0, 5, 10, ... <= L
        max_k_inside = int(np.floor((L + eps) / step_mm))
        dists_inside = [k * step_mm for k in range(max_k_inside + 1)]

        # tv_other 后额外两颗：L+5, L+10 ...
        dists_outside = [L + (k + 1) * step_mm for k in range(int(extra_outside))]

        dists = dists_inside + dists_outside

        for k, d in enumerate(dists):
            p = target + u * d
            seeds_rows.append({
                "needle_id": nid,
                "seed_index": k,
                "dist_from_target_mm": float(d),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
            })

    return pd.DataFrame(seeds_rows)
