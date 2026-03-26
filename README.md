# PH2 Project — Overcooked Zero-Shot Coordination

Overcooked 환경에서 Zero-Shot Coordination (ZSC) 연구를 위한 멀티에이전트 학습 + Human-AI 상호작용 실험 플랫폼.

## 프로젝트 구조

```
ph2-project/
├── baseline/               # Baseline 알고리즘 (SP, E3T, FCP, MEP 등)
├── ph2/                    # PH2 알고리즘 (PH1 + PH2 + CycleTransformer)
├── JaxMARL/                # JAX 기반 MARL 환경 라이브러리 (submodule)
├── webapp/                 # Human-AI 실험용 웹 앱
├── scripts/                # 유틸리티 스크립트
├── CLAUDE.md               # 개발 컨텍스트 (알고리즘 상세, 작업 이력)
└── README.md               # 이 파일
```

## 설치

```bash
# venv 생성 및 활성화
python3 -m venv overcooked_v2
source overcooked_v2/bin/activate

# 의존성 설치
./scripts/bootstrap_venv.sh
```

## 알고리즘 개요

### Baseline (`baseline/`)

| 알고리즘 | 설명 | 실행 스크립트 |
|----------|------|-------------|
| **SP** | Self-Play. 자기 자신과 플레이하며 학습 | `baseline/sh_scripts/run_factory_sp.sh` |
| **E3T** | 파트너 행동 예측 모듈 추가 (PartnerPredictor) | `baseline/sh_scripts/run_factory_e3t.sh` |
| **FCP** | SP population과 매칭하여 학습 (2단계) | `baseline/sh_scripts/run_factory_fcp.sh` |
| **MEP** | Maximum Entropy Population (다양성 극대화) | `baseline/sh_scripts/run_factory_mep.sh` |
| **SA** | State Augmentation | `baseline/sh_scripts/run_factory_sa.sh` |
| **OP** | Other-Play | `baseline/sh_scripts/run_factory_op.sh` |

### PH2 (`ph2/`)

| 알고리즘 | 설명 | 실행 스크립트 |
|----------|------|-------------|
| **PH2** | Specialist(PH1) + Independent 듀얼 에이전트 학습 | `ph2/sh_scripts/run_factory_ph2.sh` |
| **PH2-CT** | PH2 + CycleTransformer (상태/행동 재구성) | `USE_CT=1 bash ph2/sh_scripts/run_factory_ph2.sh` |

## 지원 레이아웃

### OV1 — Full Observation (agent_view_size 없음)

| 레이아웃 | 크기 | 재료 | 특징 |
|----------|------|------|------|
| `cramped_room` | 4×5 | onion | 좁은 공간, 기본 협력 |
| `asymm_advantages` | 5×9 | onion | 비대칭 역할, dual destination |
| `coord_ring` | 5×5 | onion | 순환 구조 |
| `forced_coord` | 5×5 | onion | 벽으로 분리된 강제 협력 |
| `counter_circuit` | 5×8 | onion | 카운터 기반 순환 |

### OV2 — Partial Observation (agent_view_size=2)

| 레이아웃 | 크기 | 재료 | 특징 |
|----------|------|------|------|
| `grounded_coord_simple` | 5×8 | 3종 | 다중 재료, 부분 관측 |
| `grounded_coord_ring` | 9×9 | 3종 | 큰 맵, dual destination |
| `demo_cook_simple` | 5×11 | 3종 | 긴 맵 |
| `demo_cook_wide` | 6×11 | 4종 | 넓은 맵, 4종 재료 |
| `test_time_simple` | 5×8 | 3종 | 테스트용 |
| `test_time_wide` | 7×6 | 4종 | 테스트용 넓은 맵 |

### Dual Destination (ToyCoop) — 별도 환경

Overcooked가 아닌 **5×5 cooperative gridworld**. 두 에이전트가 동시에 서로 다른 목표 지점에 도달해야 하는 coordination 문제.

| 항목 | 내용 |
|------|------|
| 환경 | `JaxMARL/jaxmarl/environments/toy_coop/toy_coop.py` |
| 그리드 | 5×5, 벽 없음 |
| Action | 5개 (right, down, left, up, stay) — interact 없음 |
| 목표 | green goal 2개 + pink goal 2개, 에이전트 2명이 동시에 도달 |
| Observation | full (partial_obs=false) |
| env config | `+env=toy_coop` |

```bash
# Dual Destination 실험
cd baseline && bash sh_scripts/run_factory_sp.sh   # +env=toy_coop
cd ph2 && bash sh_scripts/run_factory_ph2.sh       # +env=toy_coop
```

env config는 `+env=<layout_name>`으로 지정 (예: `+env=cramped_room`, `+env=grounded_coord_simple`, `+env=toy_coop`).

## 실험 실행

### Baseline 실행 예시

```bash
cd baseline

# SP — 전 레이아웃
bash sh_scripts/run_factory_sp.sh

# E3T — 전 레이아웃
bash sh_scripts/run_factory_e3t.sh

# FCP (2단계: SP population 먼저 필요)
# 1단계: SP 학습
bash sh_scripts/run_factory_sp.sh
# 2단계: SP population 복사 → FCP 학습
sh sh_scripts/copy_fcp.sh runs/<sp_run_directory>
bash sh_scripts/run_factory_fcp.sh

# MEP (2단계: SP population 기반)
bash sh_scripts/run_factory_mep.sh
```

### PH2 실행 예시

```bash
cd ph2

# PH2 (E3T + PH1 blocked target)
bash sh_scripts/run_factory_ph2.sh

# PH2 + CycleTransformer
USE_CT=1 bash sh_scripts/run_factory_ph2.sh
```

### 주요 하이퍼파라미터

```
# 모델 구조 (모든 알고리즘 공통)
GRU_HIDDEN_DIM=128, FC_DIM_SIZE=128, CNN_FEATURES=32

# 훈련
LR=2.5e-4, TOTAL_TIMESTEPS=3e7, NUM_ENVS=256, NUM_STEPS=256

# PH2 전용
PH1_OMEGA=10.0, PH1_SIGMA=2.0, PH1_POOL_SIZE=128
PH2_RATIO_STAGE1/2/3=2/1/2, E3T_EPSILON=0.2
```

## 평가 및 시각화

```bash
# 단일 알고리즘 평가 (GIF + 지표)
sh sh_scripts/run_visualize.sh --gpu 0 --dir runs/<run_directory>

# Cross-play 평가 (알고리즘 간 협력)
sh sh_scripts/run_visualize.sh --gpu 0 --dir runs/<cross_directory> --cross

# 지표만 (GIF 없이)
sh sh_scripts/run_visualize.sh --gpu 0 --dir runs/<run_directory> --all --no_viz
```

## 체크포인트 구조

```
runs/<timestamp>_<id>_<layout>_<algo>/
├── run_0/ckpt_final/      # seed 0 최종 체크포인트 (Orbax)
├── run_1/ckpt_final/      # seed 1
├── ...
└── eval/                  # 평가 결과
```

**체크포인트 내용:**
- `config`: 학습 설정 전체
- `params`: 기본 policy params
- `params_spec`: Specialist policy (PH2만)
- `params_ind`: Independent policy (PH2만)

## Human-AI 실험 Webapp

`webapp/` 디렉토리에 별도 README 참고. 주요 기능:

- 웹 브라우저에서 학습된 AI(SP/E3T/FCP/PH2)와 Overcooked 협력 플레이
- 게임 후 주관 평가 설문 (7문항 Likert + 자유 서술)
- JaxMARL-compatible trajectory 저장 (BC 학습용)
- 한국어/English i18n 지원

```bash
cd webapp
JAX_PLATFORMS=cpu uvicorn app.main:app --port 8000 --loop asyncio
```

모델 배치: `webapp/models/{layout}/{algo}/{run}/ckpt_final/`

## 디렉토리별 역할

| 디렉토리 | 역할 | 수정 가능 |
|----------|------|----------|
| `baseline/` | SP, E3T, FCP, MEP 등 기존 알고리즘 | O |
| `ph2/` | PH2 + CycleTransformer | O |
| `JaxMARL/` | 환경 라이브러리 (submodule) | X (upstream 관리) |
| `webapp/` | Human-AI 실험 웹 앱 | O |

## 환경 구분

| 구분 | Observation | agent_view_size | 재료 수 | 대표 레이아웃 |
|------|------------|----------------|---------|-------------|
| **OV1** | Full (전체 그리드) | 없음 | 1 (onion) | cramped_room, coord_ring |
| **OV2** | Partial (5×5 시야) | 2 | 2~4 | grounded_coord_simple, demo_cook_wide |

- OV1: obs shape = `(H, W, 30)` — 전체 맵이 보임
- OV2: obs shape = `(5, 5, C)` — 자기 주변 5×5만 보임, C는 재료 수에 따라 변동
- 동일한 `ActorCriticRNN` 네트워크를 사용하되 obs shape만 다름

## 주의사항

- `baseline/`, `ph2/`, `JaxMARL/`의 기존 코드는 webapp 개발 시 **절대 수정하지 않음**
- 모든 알고리즘은 동일한 CNN+GRU 구조(`ActorCriticRNN`)를 공유 — 체크포인트 호환
- PH2 모델의 cross-play 평가 시 `params_ind` (Independent policy) 사용
