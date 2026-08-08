"""srk.py — SRK(Soave-Redlich-Kwong) 상태방정식 기반 밀도 계산 및 전환율 역산.

담당: 공용 (수정 시 상대에게 통보 — CLAUDE.md §3).

역할:
  1) 주어진 조성(몰분율)·T·P에서 혼합물 밀도(kg/m³)를 SRK EOS로 계산한다.
  2) 반응기 입구 몰수와 전환율 X로부터 출구 조성을 만들고, 목표 밀도(AT-1100)를
     재현하는 X를 scipy.optimize.brentq 로 역산한다.

물성상수(MW, Tc, Pc, ω, kij, R)는 전부 config/base.yaml 에서 읽는다.
하드코딩 금지 (AVEVA 제공값, 대체 금지 — CLAUDE.md §2.3 / 지시서 §3.4).

기상(vapor) 상: 반응기 출구 밀도 ~36 kg/m³ @ ~713 K, ~2450 kPa 이므로
SRK 3차식의 최대 실근(가장 큰 Z, 가장 큰 몰부피 = 가장 낮은 밀도)을 취한다.

단위 규약: T[K], P[Pa], R[J/mol/K], V_molar[m³/mol], 밀도[kg/m³].
config 의 Pc 는 kPa 이므로 내부에서 Pa 로 환산한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

# 반응식(쿠멘 알킬화): 프로펜 + 벤젠 → 큐멘 (1:1 → 1 몰).
#   출처: 대회 문제 문서 쿠멘 공정 정의. 전환율 X 는 프로펜 기준(README: X_propene_conversion).
COMPONENTS = ("benzene", "propene", "propane", "cumene")  # 조성/조성벡터의 고정 순서

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "base.yaml"


def load_properties(config_path: Path | str = _CONFIG_PATH) -> dict:
    """config/base.yaml 의 properties 블록을 읽어 SRK 계산에 필요한 상수를 반환.

    반환 dict 는 성분별 벡터(COMPONENTS 순서)로 정리한다. 하드코딩 없음.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["properties"]

    props = {
        "MW_kg_per_mol": np.array(
            [p["MW_kg_per_kmol"][c] for c in COMPONENTS], dtype=np.float64
        )
        / 1000.0,  # kg/kmol == g/mol → kg/mol
        "Tc_K": np.array([p["Tc_K"][c] for c in COMPONENTS], dtype=np.float64),
        "Pc_Pa": np.array([p["Pc_kPa"][c] for c in COMPONENTS], dtype=np.float64)
        * 1000.0,  # kPa → Pa
        "omega": np.array([p["omega"][c] for c in COMPONENTS], dtype=np.float64),
        "R": np.float64(p["R_J_per_mol_K"]),
    }

    # kij 대칭행렬 구성. config 에 명시된 2쌍만 비영, 나머지는 0 (지시서 §3.4).
    idx = {c: i for i, c in enumerate(COMPONENTS)}
    kij = np.zeros((len(COMPONENTS), len(COMPONENTS)), dtype=np.float64)
    for pair, val in p["kij"].items():
        a, b = pair.split("_")
        kij[idx[a], idx[b]] = val
        kij[idx[b], idx[a]] = val
    props["kij"] = kij
    return props


def srk_density(x_mole: np.ndarray, T_K: float, P_Pa: float, props: dict) -> float:
    """SRK EOS 로 혼합물 밀도(kg/m³)를 계산. 기상(최대 Z 근) 기준.

    x_mole : COMPONENTS 순서의 몰분율 벡터(합=1).
    """
    R = props["R"]
    Tc = props["Tc_K"]
    Pc = props["Pc_Pa"]
    omega = props["omega"]
    kij = props["kij"]

    # 순수 성분 파라미터 (Soave 1972)
    m = 0.480 + 1.574 * omega - 0.176 * omega**2
    alpha = (1.0 + m * (1.0 - np.sqrt(T_K / Tc))) ** 2
    a_i = 0.42748 * (R**2) * (Tc**2) / Pc * alpha  # Pa·m⁶/mol²
    b_i = 0.08664 * R * Tc / Pc  # m³/mol

    # 혼합 규칙 (van der Waals one-fluid)
    a_ij = (1.0 - kij) * np.sqrt(np.outer(a_i, a_i))
    a_mix = float(x_mole @ a_ij @ x_mole)
    b_mix = float(x_mole @ b_i)

    # 무차원화 후 3차식: Z³ - Z² + (A - B - B²)Z - A·B = 0
    A = a_mix * P_Pa / (R**2 * T_K**2)
    B = b_mix * P_Pa / (R * T_K)
    coeffs = [1.0, -1.0, (A - B - B**2), -(A * B)]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < 1e-9].real
    real = real[real > B]  # 물리적으로 유효한 근만 (V > b)
    if real.size == 0:
        raise ValueError(f"SRK: 유효한 실근 없음 (T={T_K}, P={P_Pa})")
    Z = real.max()  # 기상: 최대 근

    V_molar = Z * R * T_K / P_Pa  # m³/mol
    MW_mix = float(x_mole @ props["MW_kg_per_mol"])  # kg/mol
    return MW_mix / V_molar  # kg/m³


def outlet_mole_fractions(
    n_propene: float,
    n_propane: float,
    n_benzene: float,
    n_cumene_in: float,
    X: float,
) -> np.ndarray:
    """전환율 X 에서 반응기 출구 몰분율(COMPONENTS 순서)을 계산.

    프로펜+벤젠→큐멘 (1:1→1). X 는 프로펜 기준.
      propene_out = n_propene(1-X),  benzene_out = n_benzene - X·n_propene,
      cumene_out  = n_cumene_in + X·n_propene,  propane 은 불활성(불변).
    """
    propene_out = n_propene * (1.0 - X)
    propane_out = n_propane
    benzene_out = n_benzene - X * n_propene
    cumene_out = n_cumene_in + X * n_propene
    n = np.array([benzene_out, propene_out, propane_out, cumene_out], dtype=np.float64)
    return n / n.sum()


def solve_conversion(
    target_rho: float,
    n_propene: float,
    n_propane: float,
    n_benzene: float,
    n_cumene_in: float,
    T_K: float,
    P_Pa: float,
    props: dict,
    bracket: tuple[float, float] = (1e-6, 0.999999),
) -> float:
    """목표 밀도(AT-1100)를 재현하는 전환율 X 를 brentq 로 역산.

    bracket 실패(부호 동일 등) 시 ValueError 를 그대로 전파한다(호출부에서 NaN 처리).
    """

    def f(X: float) -> float:
        x_mole = outlet_mole_fractions(n_propene, n_propane, n_benzene, n_cumene_in, X)
        return srk_density(x_mole, T_K, P_Pa, props) - target_rho

    lo, hi = bracket
    # brentq 는 f(lo), f(hi) 의 부호가 다를 것을 요구. 다르지 않으면 ValueError.
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))
