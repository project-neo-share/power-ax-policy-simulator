# app.py
# Author: Prof. Dr. Songhee Kang
# Date: 2025. 08. 17.
# Watts (Demo)
# 정책설계자 DSS + 사업자/가정 선택지원 UX (DR/TOU/용량시장)
# - 일반사용자(Residential) TOU 시뮬레이션 포함

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="WATTS: What-if Analysis for Tariffs and DR Simulator (Demo)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def participation_probability(program: str,
                              segment: str,
                              incentive_krw_per_kwh: float,
                              tou_spread_percent: float,
                              info_clarity: float,
                              automation_level: float,
                              risk_aversion: float) -> float:
    """
    간이 참여확률 모델(로지스틱). 프로그램/세그먼트별 가중치를 달리함.
    - incentive_krw_per_kwh: 0~200 정규화
    - tou_spread_percent: 0~100 정규화
    - 나머지 0~1
    """
    inc = np.clip(incentive_krw_per_kwh / 200.0, 0, 1)
    tou = np.clip(tou_spread_percent / 100.0, 0, 1)
    clr = np.clip(info_clarity, 0, 1)
    auto = np.clip(automation_level, 0, 1)
    risk = np.clip(risk_aversion, 0, 1)

    base = {
        "데이터센터": -0.10,
        "집합자원사업자": 0.20,
        "대형수요자": 0.00,
        "일반사용자": -0.05,  # 가정/소상공인: 전환 비용·불편 고려해 소폭 보수적
    }.get(segment, 0.0)

    if program == "DR":
        # DR: 인센티브, 리스크가 큰 드라이버
        w_inc, w_tou, w_clr, w_auto, w_risk, prog_bias = 2.2, 0.2, 0.6, 0.8, 1.0, 0.10
    elif program == "TOU":
        # TOU: 가격 스프레드와 정보 명료도가 핵심. 일반사용자는 자동화/정보 영향 더 큼
        if segment == "일반사용자":
            w_inc, w_tou, w_clr, w_auto, w_risk, prog_bias = 0.2, 1.5, 1.1, 0.9, 0.9, 0.05
        else:
            w_inc, w_tou, w_clr, w_auto, w_risk, prog_bias = 0.5, 1.4, 0.9, 0.6, 0.8, 0.00
    else:  # 용량시장
        # 용량: 인센티브(용량지급), 리스크 영향. TOU 스프레드는 영향 적음
        w_inc, w_tou, w_clr, w_auto, w_risk, prog_bias = 1.0, 0.2, 0.5, 0.7, 0.9, 0.05

    z = base + prog_bias + w_inc*inc + w_tou*tou + w_clr*clr + w_auto*auto - w_risk*risk
    return float(logistic(z))

def compute_kpis(program: str,
                 segment: str,
                 baseline_energy_kwh: float,
                 baseline_peak_kw: float,
                 incentive_krw_per_kwh: float,
                 tou_spread_percent: float,
                 peak_share: float,
                 info_clarity: float,
                 automation_level: float,
                 risk_aversion: float,
                 carbon_intensity: float,
                 dr_event_hours: float,
                 residential_inputs: dict | None = None) -> dict:
    """
    KPI 계산:
    - participation_rate (0~1), peak_reduction_kw, cost_saving_krw, carbon_reduction_ton, krw_per_mw
    - 일반사용자 × TOU: 가정용 디바이스 로직 반영
    """
    p = participation_probability(program, segment, incentive_krw_per_kwh, tou_spread_percent,
                                  info_clarity, automation_level, risk_aversion)

    curt_frac = {
        "데이터센터": 0.22,
        "집합자원사업자": 0.35,
        "대형수요자": 0.18,
        "일반사용자": 0.10,
    }.get(segment, 0.20)

    # ---- 특별 분기: 일반사용자 × TOU
    if program == "TOU" and segment == "일반사용자" and residential_inputs:
        # 입력값
        ev_kwh_day = residential_inputs.get("ev_kwh_per_day", 0.0)         # EV 충전량 kWh/일
        appl_kwh_day = residential_inputs.get("appliance_kwh_per_day", 0.0) # 세탁/건조/식기 kWh/일
        hvac_shift_ratio = residential_inputs.get("hvac_shift_ratio", 0.0)  # 피크 중 HVAC 시프트 가능 비율(0~1)

        # 월 환산(30일)
        ev_month = ev_kwh_day * 30.0
        appl_month = appl_kwh_day * 30.0

        # 가격차(스프레드) → kWh당 절감효과 (평균 150원/kWh 가정)
        price_diff = (tou_spread_percent / 100.0) * 150.0

        # 자동화↑ → 시프트↑, 정보 명료도↑ → 실행률↑ (보수적 휴리스틱)
        ev_shift = ev_month * (0.50*automation_level + 0.20*info_clarity) * p
        appl_shift = appl_month * (0.40*automation_level + 0.20*info_clarity) * p
        # 피크 중 HVAC 비중 간략 가정: baseline_energy * 35% * hvac_shift_ratio
        hvac_peak_kwh = baseline_energy_kwh * 0.35 * hvac_shift_ratio
        hvac_shift = hvac_peak_kwh * (0.30 + 0.30*automation_level) * p

        shifted_kwh = ev_shift + appl_shift + hvac_shift
        reduced_kwh = shifted_kwh * 0.20  # 시프트 중 20%는 실제 절감(편의/회피)

        cost_saving = shifted_kwh * price_diff
        carbon_reduction_kg = reduced_kwh * carbon_intensity

        # 피크 저감(kW) 근사: 피크 시간대 시프트 kWh / 피크시간 길이(2h 가정),
        # HVAC는 1:1, EV/가전은 피크 일치율 낮게 0.5 가중
        peak_reduction_kw = (hvac_shift + 0.5*(ev_shift + appl_shift)) / 2.0

        mw_reduced = max(peak_reduction_kw / 1000.0, 1e-6)
        krw_per_mw = (cost_saving / mw_reduced) if mw_reduced > 0 else np.nan

        return {
            "participation_rate": p,
            "peak_reduction_kw": float(peak_reduction_kw),
            "cost_saving_krw": float(cost_saving),
            "carbon_reduction_ton": float(carbon_reduction_kg / 1000.0),
            "krw_per_mw": float(krw_per_mw),
        }

    # ---- 일반 분기: DR / TOU(사업자 세그먼트) / 용량시장
    if program == "DR":
        eff = 1.0
        peak_red_kw = p * baseline_peak_kw * curt_frac * eff
        curtailed_kwh = peak_red_kw * dr_event_hours
        cost_saving = curtailed_kwh * incentive_krw_per_kwh
        carbon_reduction_kg = curtailed_kwh * carbon_intensity

    elif program == "TOU":
        eff = 0.7
        peak_red_kw = p * baseline_peak_kw * curt_frac * eff
        peak_energy_kwh = baseline_energy_kwh * peak_share
        price_diff = (tou_spread_percent / 100.0) * 150.0  # 평균 150원/kWh 가정
        shift_ratio = 0.25 + 0.35 * automation_level
        shifted_kwh = peak_energy_kwh * p * shift_ratio
        cost_saving = shifted_kwh * price_diff
        reduced_kwh = shifted_kwh * 0.30
        carbon_reduction_kg = reduced_kwh * carbon_intensity

    else:  # 용량시장
        eff = 0.6
        peak_red_kw = p * baseline_peak_kw * curt_frac * eff
        capacity_payment_per_kw = incentive_krw_per_kwh * 10.0  # 데모 근사
        cost_saving = peak_red_kw * capacity_payment_per_kw
        carbon_reduction_kg = (peak_red_kw * 0.1) * carbon_intensity

    mw_reduced = max(peak_red_kw / 1000.0, 1e-6)
    krw_per_mw = (cost_saving / mw_reduced) if mw_reduced > 0 else np.nan

    return {
        "participation_rate": p,
        "peak_reduction_kw": float(peak_red_kw),
        "cost_saving_krw": float(cost_saving),
        "carbon_reduction_ton": float(carbon_reduction_kg / 1000.0),
        "krw_per_mw": float(krw_per_mw),
    }

def sensitivity_map(program: str,
                    segment: str,
                    baseline_energy_kwh: float,
                    baseline_peak_kw: float,
                    peak_share: float,
                    automation_level: float,
                    risk_aversion: float,
                    carbon_intensity: float,
                    dr_event_hours: float,
                    residential_inputs: dict | None = None,
                    inc_grid=(0, 200, 25),
                    clr_grid=(0, 1, 0.05)):
    """인센티브×정보명료도 변화에 따른 참여율 히트맵 데이터."""
    inc_vals = np.linspace(inc_grid[0], inc_grid[1], int((inc_grid[1]-inc_grid[0]) / inc_grid[2]) + 1)
    clr_vals = np.linspace(clr_grid[0], clr_grid[1], int((clr_grid[1]-clr_grid[0]) / clr_grid[2]) + 1)
    heat = np.zeros((len(clr_vals), len(inc_vals)))
    for i, clr in enumerate(clr_vals):
        for j, inc in enumerate(inc_vals):
            p = participation_probability(program, segment, inc, 30, clr, automation_level, risk_aversion)
            heat[i, j] = p
    return inc_vals, clr_vals, heat

def make_report_md(inputs: dict, kpis: dict) -> str:
    return f"""# DPPS – One-Pager (Demo)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Scenario
- Program: **{inputs['program']}**
- Segment: **{inputs['segment']}**
- Monthly Energy: **{inputs['baseline_energy_kwh']:,} kWh**
- Peak Demand: **{inputs['baseline_peak_kw']:,} kW**
- Incentive: **{inputs['incentive_krw_per_kwh']:.0f} KRW/kWh**
- TOU Spread: **{inputs['tou_spread_percent']:.0f}%**
- Information Clarity: **{inputs['info_clarity']:.2f}**
- Automation Level: **{inputs['automation_level']:.2f}**
- Risk Aversion: **{inputs['risk_aversion']:.2f}**
- DR Event Hours (per month): **{inputs['dr_event_hours']:.1f} h**
- Carbon Intensity: **{inputs['carbon_intensity']:.2f} kgCO2/kWh**

## KPIs (Estimates)
- Participation Rate: **{kpis['participation_rate']*100:.1f}%**
- Peak Reduction: **{kpis['peak_reduction_kw']:.1f} kW**
- Cost Saving / Settlement: **{kpis['cost_saving_krw']:.0f} KRW/month**
- Carbon Reduction: **{kpis['carbon_reduction_ton']:.3f} tCO2/month**
- KRW per MW (lower is better): **{kpis['krw_per_mw']:.0f} KRW/MW**

## Notes & Limitations
- Demo model uses simplified logistic/heuristic assumptions.
- Real policy design should calibrate with aggregated/anonymous KPX/KEPCO data, settlement rules, and operational constraints.
"""

# -----------------------------
# UI
# -----------------------------
st.title("WATTS: What-if Analysis for Tariffs and DR Simulator")
st.caption("정책설계자 DSS + 사업자/가정 선택지원 UX · DR/TOU/용량시장 (Demo). Turn incentives into measurable outcomes. What‑if design for TOU/DR/Capacity before rollout.")

with st.sidebar:
    st.header("시나리오 입력")
    program = st.selectbox("프로그램", ["DR", "TOU", "용량시장"])
    segment = st.selectbox("참여자 유형", ["데이터센터", "집합자원사업자", "대형수요자", "일반사용자"])

    baseline_energy_kwh = st.number_input("월간 사용량 (kWh)", min_value=1000, value=100_000, step=1000)
    baseline_peak_kw = st.number_input("피크 수요 (kW)", min_value=10, value=5000, step=10)
    peak_share = st.slider("피크 시간대 에너지 비중(%)", 10, 80, 35) / 100.0

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

    # 일반사용자 × TOU 추가 입력
    residential_inputs = {}
    if program == "TOU" and segment == "일반사용자":
        st.subheader("가정/소상공인 TOU 디바이스 (선택)")
        residential_inputs["ev_kwh_per_day"] = st.slider("EV 충전량 (kWh/일)", 0, 40, 8)
        residential_inputs["appliance_kwh_per_day"] = st.slider("세탁/건조/식기세척 (kWh/일, 합산)", 0, 10, 3)
        residential_inputs["hvac_shift_ratio"] = st.slider("HVAC 시프트 가능 비율(%)", 0, 100, 20) / 100.0

inputs = {
    "program": program,
    "segment": segment,
    "baseline_energy_kwh": float(baseline_energy_kwh),
    "baseline_peak_kw": float(baseline_peak_kw),
    "peak_share": float(peak_share),
    "incentive_krw_per_kwh": float(incentive_krw_per_kwh),
    "tou_spread_percent": float(tou_spread_percent),
    "dr_event_hours": float(dr_event_hours),
    "info_clarity": float(info_clarity),
    "automation_level": float(automation_level),
    "risk_aversion": float(risk_aversion),
    "carbon_intensity": float(carbon_intensity),
}

kpis = compute_kpis(**inputs, residential_inputs=residential_inputs if residential_inputs else None)

# -----------------------------
# KPI + Sensitivity
# -----------------------------
col1, col2 = st.columns([1.2, 1.6])

with col1:
    st.subheader("핵심 성과지표")
    m1, m2, m3, m4 = st.columns(4)

    m1.markdown(f"<p style='font-size:12px;'>참여율<br><b>{kpis['participation_rate']*100:.1f}%</b></p>", unsafe_allow_html=True)
    m2.markdown(f"<p style='font-size:12px;'>피크 저감(kW)<br><b>{kpis['peak_reduction_kw']:.1f}</b></p>", unsafe_allow_html=True)
    m3.markdown(f"<p style='font-size:12px;'>비용절감/정산(원/월)<br><b>{kpis['cost_saving_krw']:.0f}</b></p>", unsafe_allow_html=True)
    m4.markdown(f"<p style='font-size:12px;'>탄소감축(tCO2/월)<br><b>{kpis['carbon_reduction_ton']:.3f}</b></p>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:12px;'><b>효율성 지표</b> (낮을수록 효율): "
        f"<b>{int(kpis['krw_per_mw']):,} 원/MW</b></p>",
        unsafe_allow_html=True
    )
#with col1:
#    st.subheader("핵심 성과지표")
#    m1, m2, m3, m4 = st.columns(4)
#    m1.metric("참여율", f"{kpis['participation_rate']*100:.1f}%")
#    m2.metric("피크 저감(kW)", f"{kpis['peak_reduction_kw']:.1f}")
#    m3.metric("비용절감/정산(원/월)", f"{kpis['cost_saving_krw']:.0f}")
#    m4.metric("탄소감축(tCO2/월)", f"{kpis['carbon_reduction_ton']:.3f}")
#    st.markdown("**효율성 지표** (낮을수록 효율): **{:,} 원/MW**".format(int(kpis["krw_per_mw"])))

with col2:
    st.subheader("민감도 히트맵 (Participation Rate)")
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
        residential_inputs=residential_inputs if residential_inputs else None,
        inc_grid=(0, 200, 25),
        clr_grid=(0, 1, 0.05),
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[inc_vals.min(), inc_vals.max(), clr_vals.min(), clr_vals.max()],
    )
    ax.set_xlabel("Incentive (KRW/kWh)")
    ax.set_ylabel("Information Clarity (0–1)")
    ax.set_title("Participation Rate Sensitivity")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Participation Rate (0–1)")
    st.pyplot(fig)

# -----------------------------
# Tables & Export
# -----------------------------
st.markdown("---")
st.subheader("시나리오 요약 및 내보내기")

t1, t2 = st.tabs(["요약 테이블", "가정/한계"])
with t1:
    row = {
        "프로그램": inputs["program"],
        "참여자 유형": inputs["segment"],
        "월간 사용량(kWh)": int(inputs["baseline_energy_kwh"]),
        "피크 수요(kW)": int(inputs["baseline_peak_kw"]),
        "인센티브(KRW/kWh)": inputs["incentive_krw_per_kwh"],
        "TOU 스프레드(%)": inputs["tou_spread_percent"],
        "DR 이벤트 시간(h/월)": inputs["dr_event_hours"],
        "정보 명료도": inputs["info_clarity"],
        "자동화 수준": inputs["automation_level"],
        "위험회피 성향": inputs["risk_aversion"],
        "탄소계수(kgCO2/kWh)": inputs["carbon_intensity"],
        "참여율(%)": round(kpis["participation_rate"]*100, 1),
        "피크 저감(kW)": round(kpis["peak_reduction_kw"], 1),
        "비용절감/정산(원/월)": round(kpis["cost_saving_krw"], 0),
        "탄소감축(tCO2/월)": round(kpis["carbon_reduction_ton"], 3),
        "원/MW": round(kpis["krw_per_mw"], 0),
    }
    # 일반사용자 TOU 입력 보조 컬럼
    if program == "TOU" and segment == "일반사용자" and residential_inputs:
        row.update({
            "EV(kWh/일)": residential_inputs["ev_kwh_per_day"],
            "가전합산(kWh/일)": residential_inputs["appliance_kwh_per_day"],
            "HVAC 시프트 비율": residential_inputs["hvac_shift_ratio"],
        })
    df = pd.DataFrame([row])
    st.dataframe(df, use_container_width=True)

with t2:
    st.write("""
- 본 데모는 **간단한 로지스틱/휴리스틱 모델**로 동작합니다.
- 실제 정책 설계에는 **KPX/KEPCO 집계·비식별 데이터**, 제도별 정산규정, 설비 제약, 채널 효과(알림/앱) 등이 반영되어야 합니다.
- 모든 예측치는 **가정/범위/신뢰구간**과 함께 사용되어야 하며, **민감도 분석** 결과를 병기해야 합니다.
""")

rep_md = make_report_md(inputs, kpis)
st.download_button("📄 1페이지 리포트(Markdown) 다운로드", data=rep_md.encode("utf-8"),
                   file_name="policy_sim_report.md", mime="text/markdown")

csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.download_button("📥 시나리오 결과 CSV 다운로드", data=csv_buf.getvalue().encode("utf-8-sig"),
                   file_name="policy_sim_scenario.csv", mime="text/csv")

st.markdown("---")
st.caption("© WATTS Demo – Policy design DSS + Participant choice support. Calibrate with real aggregated/anonymous data before policy use.")
