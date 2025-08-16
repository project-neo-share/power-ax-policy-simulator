
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="WATTS: What‑if Analysis for Tariffs & DR Simulator", layout="wide")

st.title("WATTS — What‑if Analysis for Tariffs & DR Simulator")
st.caption("Turn incentives into measurable outcomes. What‑if design for TOU/DR/Capacity before rollout.")
# -----------------------------
# Helper functions
# -----------------------------
def logistic(x):
    return 1/(1+np.exp(-x))

def participation_probability(program, segment, incentive_krw_per_kwh, tou_spread_percent, info_clarity, automation_level, risk_aversion):
    """
    Compute participation probability using a simple logistic model.
    All inputs normalized to 0-1 except incentive and tou spread, which we normalize internally.
    """
    inc = np.clip(incentive_krw_per_kwh/200.0, 0, 1)  # assume 0~200 KRW/kWh normalization for demo
    tou = np.clip(tou_spread_percent/100.0, 0, 1)     # 0~100% spread normalization
    clr = np.clip(info_clarity, 0, 1)
    auto = np.clip(automation_level, 0, 1)
    risk = np.clip(risk_aversion, 0, 1)

    # Base bias by segment (rough priors)
    base = {
        "데이터센터": -0.1,
        "집합자원사업자": 0.2,
        "대형수요자": 0.0
    }[segment]

    # Weights differ by program
    if program == "DR":
        w_inc, w_tou, w_clr, w_auto, w_risk = 2.2, 0.2, 0.6, 0.8, 1.0
        prog_bias = 0.1
    elif program == "TOU":
        w_inc, w_tou, w_clr, w_auto, w_risk = 0.5, 1.4, 0.9, 0.6, 0.8
        prog_bias = 0.0
    else:  # 용량시장
        w_inc, w_tou, w_clr, w_auto, w_risk = 1.0, 0.2, 0.5, 0.7, 0.9
        prog_bias = 0.05

    z = base + prog_bias + w_inc*inc + w_tou*tou + w_clr*clr + w_auto*auto - w_risk*risk
    return float(logistic(z))  # 0~1

def compute_kpis(program, segment, baseline_energy_kwh, baseline_peak_kw,
                 incentive_krw_per_kwh, tou_spread_percent, peak_share, info_clarity,
                 automation_level, risk_aversion, carbon_intensity, dr_event_hours):
    """Return dict of KPIs (participation, peak reduction, cost saving, carbon reduction)."""
    p = participation_probability(program, segment, incentive_krw_per_kwh, tou_spread_percent,
                                  info_clarity, automation_level, risk_aversion)

    # Curtailable fraction by segment (demo heuristics)
    curt_frac = {"데이터센터": 0.22, "집합자원사업자": 0.35, "대형수요자": 0.18}[segment]

    # Program-specific effectiveness factors
    if program == "DR":
        eff = 1.0
        peak_red_kw = p * baseline_peak_kw * curt_frac * eff
        curtailed_kwh = peak_red_kw * dr_event_hours
        cost_saving = curtailed_kwh * incentive_krw_per_kwh  # incentive paid per curtailed kWh
        carbon_reduction_kg = curtailed_kwh * carbon_intensity
    elif program == "TOU":
        eff = 0.7
        peak_red_kw = p * baseline_peak_kw * curt_frac * eff  # some peak shift
        # Assume portion of energy shifted/reduced during peak
        peak_energy_kwh = baseline_energy_kwh * peak_share
        # convert spread(%) to KRW/kWh differential using a notional avg price 150 KRW/kWh
        price_diff = (tou_spread_percent/100.0) * 150.0
        shift_ratio = 0.25 + 0.35*automation_level  # more automation -> more shift
        shifted_kwh = peak_energy_kwh * p * shift_ratio
        cost_saving = shifted_kwh * price_diff
        # assume 30% of shifted is actual reduction due to efficiency/avoidance
        reduced_kwh = shifted_kwh * 0.30
        carbon_reduction_kg = reduced_kwh * carbon_intensity
    else:  # 용량시장
        eff = 0.6
        peak_red_kw = p * baseline_peak_kw * curt_frac * eff
        # Assume capacity payment (KRW/kW-month) converted from incentive_krw_per_kwh proxy for demo
        capacity_payment_per_kw = incentive_krw_per_kwh * 10  # heuristic proxy
        cost_saving = peak_red_kw * capacity_payment_per_kw
        # Limited direct energy reduction, assume minor operational improvement
        carbon_reduction_kg = (peak_red_kw * 0.1) * carbon_intensity

    # Budget efficiency proxy: KRW per MW reduced (lower is better)
    mw_reduced = max(peak_red_kw/1000.0, 1e-6)
    krw_per_mw = cost_saving / mw_reduced if mw_reduced > 0 else np.nan

    return {
        "participation_rate": p,                # 0~1
        "peak_reduction_kw": float(peak_red_kw),
        "cost_saving_krw": float(cost_saving),
        "carbon_reduction_ton": float(carbon_reduction_kg/1000.0),
        "krw_per_mw": float(krw_per_mw)
    }

def sensitivity_map(program, segment, baseline_energy_kwh, baseline_peak_kw,
                    peak_share, automation_level, risk_aversion, carbon_intensity, dr_event_hours,
                    inc_grid=(0,200,25), clr_grid=(0,1,0.05)):
    inc_vals = np.linspace(inc_grid[0], inc_grid[1], int(inc_grid[2]))
    clr_vals = np.linspace(clr_grid[0], clr_grid[1], int((clr_grid[1]-clr_grid[0])/clr_grid[2])+1)
    heat = np.zeros((len(clr_vals), len(inc_vals)))
    for i, clr in enumerate(clr_vals):
        for j, inc in enumerate(inc_vals):
            kpi = compute_kpis(program, segment, baseline_energy_kwh, baseline_peak_kw,
                               inc, 30, peak_share, clr, automation_level, risk_aversion,
                               carbon_intensity, dr_event_hours)
            heat[i,j] = kpi["participation_rate"]
    return inc_vals, clr_vals, heat

def make_report_md(inputs, kpis):
    md = f"""# 전력시장 제도 참여 시뮬레이터 – 1페이지 리포트 (Demo)
생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 시나리오 요약
- 프로그램: **{inputs['program']}**
- 참여자 유형: **{inputs['segment']}**
- 기본 사용량: **{inputs['baseline_energy_kwh']:,} kWh/월**
- 피크 수요: **{inputs['baseline_peak_kw']:,} kW**
- DR 인센티브: **{inputs['incentive_krw_per_kwh']:.0f} KRW/kWh**
- TOU 스프레드: **{inputs['tou_spread_percent']:.0f}%**
- 정보 명료도: **{inputs['info_clarity']:.2f}**
- 자동화 수준: **{inputs['automation_level']:.2f}**
- 위험회피 성향: **{inputs['risk_aversion']:.2f}**
- 월간 DR 이벤트 시간: **{inputs['dr_event_hours']:.1f} h**
- 탄소계수: **{inputs['carbon_intensity']:.2f} kgCO2/kWh**

## 핵심 성과지표 (추정치)
- 참여율: **{kpis['participation_rate']*100:.1f}%**
- 피크 저감: **{kpis['peak_reduction_kw']:.1f} kW**
- 비용절감(또는 정산/수익): **{kpis['cost_saving_krw']:.0f} KRW/월**
- 탄소감축: **{kpis['carbon_reduction_ton']:.3f} tonCO2/월**
- 원/MW(효율성 지표, 낮을수록 효율): **{kpis['krw_per_mw']:.0f} KRW/MW**

## 가정 및 한계
- 본 도구는 데모로서 간단한 로지스틱/휴리스틱 모델을 사용합니다.
- 실제 정책 설계에는 KPX/KEPCO 집계데이터, 제도별 정산규정, 설비별 운영제약 반영이 필요합니다.
"""
    return md

# -----------------------------
# UI: Sidebar inputs
# -----------------------------
st.title("전력시장 제도 참여 시뮬레이터 (Demo)")
st.caption("정책설계자 DSS + 사업자 선택지원 UX (데모용 단순모형)")

with st.sidebar:
    st.header("시나리오 입력")
    program = st.selectbox("프로그램", ["DR", "TOU", "용량시장"])
    segment = st.selectbox("참여자 유형", ["데이터센터", "집합자원사업자", "대형수요자"])
    baseline_energy_kwh = st.number_input("월간 사용량 (kWh)", min_value=1000, value=100_000, step=1000)
    baseline_peak_kw = st.number_input("피크 수요 (kW)", min_value=10, value=5000, step=10)
    peak_share = st.slider("피크 시간대 에너지 비중(%)", 10, 80, 35)/100.0

    st.markdown("---")
    st.subheader("정책 파라미터")
    incentive_krw_per_kwh = st.slider("인센티브 (KRW/kWh)", 0, 200, 80)
    tou_spread_percent = st.slider("TOU 스프레드(%)", 0, 100, 30)
    dr_event_hours = st.slider("월간 DR 이벤트 시간 (h)", 0, 40, 8)

    st.markdown("---")
    st.subheader("행동/커뮤니케이션 변수")
    info_clarity = st.slider("정보 명료도 (0~1)", 0.0, 1.0, 0.6, 0.05)
    automation_level = st.slider("자동화 수준 (0~1)", 0.0, 1.0, 0.5, 0.05)
    risk_aversion = st.slider("위험회피 성향 (0~1)", 0.0, 1.0, 0.4, 0.05)

    st.markdown("---")
    carbon_intensity = st.slider("탄소계수 (kgCO2/kWh)", 0.1, 1.0, 0.45, 0.01)

inputs = {
    "program": program,
    "segment": segment,
    "baseline_energy_kwh": int(baseline_energy_kwh),
    "baseline_peak_kw": int(baseline_peak_kw),
    "peak_share": float(peak_share),
    "incentive_krw_per_kwh": float(incentive_krw_per_kwh),
    "tou_spread_percent": float(tou_spread_percent),
    "dr_event_hours": float(dr_event_hours),
    "info_clarity": float(info_clarity),
    "automation_level": float(automation_level),
    "risk_aversion": float(risk_aversion),
    "carbon_intensity": float(carbon_intensity)
}

# -----------------------------
# Compute KPIs
# -----------------------------
kpis = compute_kpis(**inputs)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1.6])

with col1:
    st.subheader("핵심 성과지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("참여율", f"{kpis['participation_rate']*100:.1f}%")
    m2.metric("피크 저감(kW)", f"{kpis['peak_reduction_kw']:.1f}")
    m3.metric("비용절감/정산(원/월)", f"{kpis['cost_saving_krw']:.0f}")
    m4.metric("탄소감축(tCO2/월)", f"{kpis['carbon_reduction_ton']:.3f}")

    st.markdown("**효율성 지표** (낮을수록 효율): **{:,} 원/MW**".format(int(kpis["krw_per_mw"])))

with col2:
    st.subheader("민감도 히트맵 (참여율)")
    inc_vals, clr_vals, heat = sensitivity_map(
        program=inputs["program"],
        segment=inputs["segment"],
        baseline_energy_kwh=inputs["baseline_energy_kwh"],
        baseline_peak_kw=inputs["baseline_peak_kw"],
        peak_share=inputs["peak_share"],
        automation_level=inputs["automation_level"],
        risk_aversion=inputs["risk_aversion"],
        carbon_intensity=inputs["carbon_intensity"],
        dr_event_hours=inputs["dr_event_hours"],
        inc_grid=(0,200,25), clr_grid=(0,1,0.05)
    )

    fig, ax = plt.subplots()
    im = ax.imshow(heat, origin='lower', aspect='auto',
                   extent=[inc_vals.min(), inc_vals.max(), clr_vals.min(), clr_vals.max()])
    ax.set_xlabel("인센티브 (KRW/kWh)")
    ax.set_ylabel("정보 명료도 (0~1)")
    ax.set_title("참여율(0~1) 민감도")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

st.markdown("---")
st.subheader("시나리오 요약 및 내보내기")

# Tabbed details
t1, t2 = st.tabs(["요약 테이블", "가정/한계"])
with t1:
    df = pd.DataFrame([{
        "프로그램": inputs["program"],
        "참여자 유형": inputs["segment"],
        "월간 사용량(kWh)": inputs["baseline_energy_kwh"],
        "피크 수요(kW)": inputs["baseline_peak_kw"],
        "인센티브(KRW/kWh)": inputs["incentive_krw_per_kwh"],
        "TOU 스프레드(%)": inputs["tou_spread_percent"],
        "DR 이벤트 시간(h/월)": inputs["dr_event_hours"],
        "정보 명료도": inputs["info_clarity"],
        "자동화 수준": inputs["automation_level"],
        "위험회피 성향": inputs["risk_aversion"],
        "탄소계수(kgCO2/kWh)": inputs["carbon_intensity"],
        "참여율(%)": round(kpis["participation_rate"]*100,1),
        "피크 저감(kW)": round(kpis["peak_reduction_kw"],1),
        "비용절감/정산(원/월)": round(kpis["cost_saving_krw"],0),
        "탄소감축(tCO2/월)": round(kpis["carbon_reduction_ton"],3),
        "원/MW": round(kpis["krw_per_mw"],0)
    }])
    st.dataframe(df, use_container_width=True)
with t2:
    st.write("""
- 본 데모는 **단순화된 로지스틱/휴리스틱 모델**을 사용합니다.
- 실제 정책 설계에는 **집계·비식별 KPIX/KEPCO 데이터**, 제도별 정산규정, 설비제약, 커뮤니케이션 채널 효과 등이 반영되어야 합니다.
- 모든 예측치는 **가정/범위/신뢰구간**과 함께 사용되어야 하며, **민감도 분석** 결과를 병기해야 합니다.
""")

# Downloads
rep_md = make_report_md(inputs, kpis)
st.download_button("📄 1페이지 리포트(Markdown) 다운로드", data=rep_md.encode("utf-8"),
                   file_name="policy_sim_report.md", mime="text/markdown")
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.download_button("📥 시나리오 결과 CSV 다운로드", data=csv_buf.getvalue().encode("utf-8-sig"),
                   file_name="policy_sim_scenario.csv", mime="text/csv")

st.markdown("---")
st.caption("© Demo – 정책설계자 DSS + 사업자 선택지원 UX. 실제 정책 적용 전 검증 및 데이터 연계 필요.")
