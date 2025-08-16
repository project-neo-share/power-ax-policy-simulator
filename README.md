# DPPS — Digital Power Policy Simulator
**전력정보화 정책 시뮬레이터 (전력시장 제도 참여 시뮬레이터 · Demo)**

> 정책설계자(DSS)와 시장 참여자(사업자) 모두가 사용할 수 있는 **시나리오 기반 정책 시뮬레이터**입니다.  
> DR/TOU/용량시장 등 新제도 도입 전, **참여율·피크저감·비용효율·탄소감축**을 **정량**으로 미리 검토하고, 도입 후에는 **적중률**을 점검할 수 있도록 설계되었습니다.

---

## 🔎 무엇을 할 수 있나요?
- **정책 파라미터**(인센티브, TOU 스프레드, DR 이벤트 시간 등)를 슬라이더로 조정하면
- **참여율·피크 저감(kW)·비용절감/정산(원)·탄소감축(tCO₂)** KPI가 즉시 갱신됩니다.
- **민감도 히트맵**으로 인센티브·정보명료도 변화에 따른 참여율 변화를 한눈에 확인할 수 있습니다.
- **1페이지 리포트(Markdown)**와 **시나리오 결과 CSV**를 다운로드할 수 있습니다.

> 이 저장소는 **데모(간소화된 로지스틱/휴리스틱 모델)**입니다. 실제 정책 적용 시에는 **KPX/KEPCO 집계·비식별 데이터, 제도별 정산규정, 설비제약**을 반영해 보정해야 합니다.

---

## 🚀 빠른 시작 (Quickstart)

### 1) 파일
- `app.py` — 데모 앱 본체  
- `requirements.txt` — 의존성 목록

### 2) 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3) Streamlit Cloud 배포
- GitHub에 두 파일 업로드 → Streamlit Cloud에서 리포지토리 지정 → 엔트리 파일로 `app.py` 선택

---

## 🧭 사용자 시나리오
### ▸ 정책설계자(전력시장과/KPX)
- DR/TOU/용량시장 **시나리오 슬라이더**로 인센티브·구간·강도 조정
- 대시보드에서 **참여율/피크저감/예산효율(원/MW)/탄소감축** 즉시 확인
- **민감도 히트맵**으로 불참 사유(정보 불명확성 등) 완화 효과 검토
- **국회·감사 보고용 1p**를 바로 저장

### ▸ 시장 참여자(데이터센터/집합자원사업자/대형수요자)
- 기본 사용량·피크·자동화 수준 등을 입력
- 제도 참여 시 **예상 절감액·리스크**를 즉시 확인 → 참여 여부 판단

---

## 🧱 폴더 구조
```
streamlit_demo/
├─ app.py                # Streamlit 앱
├─ requirements.txt      # 의존성
└─ README.md             # 이 문서
```

---

## 🧠 모델 개요(데모)
- **참여확률**: 로지스틱 함수 기반, 프로그램별 가중치(인센티브/TOU/정보명료도/자동화/리스크) + 참여자 유형 바이어스  
- **피크 저감(kW)**: 참여율 × 피크수요 × 절감가능비율(부문별) × 프로그램 효과  
- **비용절감/정산(원)**: 프로그램별 휴리스틱(예: DR=인센티브×절감kWh, TOU=스프레드×시프트kWh)  
- **탄소감축**: 절감/회피 kWh × 탄소계수(kgCO₂/kWh)  
- **효율성 지표**: 원/MW(낮을수록 효율)

> ※ 실제 운영에는 **정산 규칙·약관**, **집계데이터**, **설비 제약**, **정책 커뮤니케이션 채널 효과** 등을 반영해 고도화합니다.

---

## 🏷️ 네이밍
- 권장: **DPPS — Digital Power Policy Simulator**  
  (또는 PRISM, PMADS, WATTS 등 브랜딩 가능)

---

## 🧩 License (Dual)
본 저장소는 **코드**와 **문서/콘텐츠**에 서로 다른 라이선스가 적용되는 **듀얼 라이선스** 정책을 따릅니다.

### 1) Code — MIT License
- © 2025 DPPS Contributors  
- SPDX-License-Identifier: **MIT**  
- 요약: 누구나 사용/복제/수정/배포 가능하며, 저작권 고지와 허가 고지를 유지해야 합니다. 보증은 제공되지 않습니다.

```
MIT License

Copyright (c) 2025 DPPS Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 2) Documentation & Non‑Code Assets — Creative Commons **CC BY 4.0**
- README, 사용자 가이드, UI 카피, 이미지 등 **비코드 콘텐츠**는 **CC BY 4.0**을 적용합니다.  
- 원문: https://creativecommons.org/licenses/by/4.0/  
- 요약: 출처(저작자 및 라이선스 링크)만 표기하면 **공유·변형·배포·영리 이용**이 모두 가능합니다.

**Attribution 예시**
> “DPPS — Digital Power Policy Simulator, © 2025 DPPS Contributors, CC BY 4.0”  
> (원문 저장소 URL 또는 문서 링크 포함)

**주의**  
- 외부 라이브러리/아이콘/이미지 사용 시 **각 자산의 개별 라이선스**를 따릅니다.  
- 실제 서비스 전 **법무 검토**를 권장합니다.

---

## 🙋 문의
- 정책설계/DSS: 전력시장과, KPX 정책연구  
- 모델/데이터: 연구기관·대학(계량/시뮬레이션), KEPCO/KPX 데이터팀  
- UX/배포: 내부 개발 파트너 또는 외부 협력사
