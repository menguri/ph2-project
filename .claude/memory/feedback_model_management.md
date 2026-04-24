---
name: final/webapp vs human-proxy 모델 경로 분리 관리
description: final/ 과 webapp/ 의 "공식" 모델과 human-proxy/ 의 BC eval 대상 모델은 독립적으로 관리해야 함
type: feedback
---

final/ 과 webapp/ 는 공식 배포용 모델 슬롯 (논문/데모 기준 best run). human-proxy/ 는 BC proxy eval 집계용이며, 여기서는 ablation 런·실험적 후보 런도 자유롭게 교체 가능.

**Why:** 사용자는 human-proxy BC proxy 결과표에서 새 후보 런 (예: ablation linear penalty) 을 평가해보는 단계와, 이 후보를 정식으로 공식 best 로 승격하는 단계를 분리하고 싶음. 한번에 둘 다 교체하면 되돌리기 어렵고 공식 버전이 섣불리 바뀜 (2026-04-22 counter_circuit PH2 lyogphwp 교체 시 final/webapp 까지 건드렸다가 되돌린 사례).

**How to apply:**
- 사용자가 "human proxy 표 교체" / "human proxy eval 집계" 류 요청을 하면, **오직 아래만** 건드린다:
  - `human-proxy/results_all_norecompute/<algo>_<layout>/` (BC eval scores)
  - `human-proxy/code/build_full_report.py` 의 FINALS / FINAL_BASELINE dict (SP/XP lookup 대상)
  - `human-proxy/reports/` 재빌드
- **건드리지 말 것**:
  - `final/ph2/<layout>/` 또는 `final/baseline/<layout>/` 의 런 복사/제거
  - `webapp/models/<layout>/<algo>/` 의 체크포인트 교체
  - `human-proxy/sh_scripts/eval_final_v2/v3/all.sh` 의 run path (final/ 를 가리킴)
- 사용자가 명시적으로 "final, webapp 버전으로 바꿔줘" 라고 말할 때만 공식 슬롯 교체 진행 (gamma asymm 2026-04-22 사례처럼).
