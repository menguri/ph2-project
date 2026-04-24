# PH2 Project — CLAUDE.md

이 파일은 Claude Code가 ph2-project 작업 시 자동으로 읽는 컨텍스트 파일입니다.

---

## 1. 구현된 알고리즘 개요

### 주력 알고리즘: PH2 (`ph2/sh_scripts/run_factory_ph2.sh`)

**PH2 (Phase 2)** 는 Overcooked-V2 환경에서의 멀티에이전트 협력 학습 알고리즘이다.
핵심 아이디어는 **Specialist(PH1)** 와 **Independent(PH2)** 두 종류의 에이전트를 함께 훈련시켜 다양한 파트너와 협력할 수 있는 정책을 학습하는 것이다.

#### 훈련 방식

- **에피소드마다 동적으로 ego/partner 역할 배정**: ego 에이전트만 gradient 업데이트
- **PPO 기반 on-policy 학습**: GAE, clipped surrogate loss, value clipping
- **3단계 매치 스케줄링** (Spec-Spec / Spec-Ind / Ind-Ind):

| 단계         | 범위      | PH2_RATIO | 특징                  |
|------------|---------|-----------|---------------------|
| Stage 1    | 0~33%   | 2         | Specialist 비중 높음    |
| Stage 2    | 33~67%  | 1         | 균형                  |
| Stage 3    | 67~100% | 2         | Specialist 비중 다시 높음 |

#### 주요 하이퍼파라미터 (config/model/rnn.yaml 기준)

```
# 모델 구조
GRU_HIDDEN_DIM=128, FC_DIM_SIZE=128, CNN_FEATURES=32, ACTIVATION=relu

# 훈련
LR=2.5e-4, TOTAL_TIMESTEPS=3e7, GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2
NUM_ENVS=256, NUM_STEPS=256, UPDATE_EPOCHS=4, NUM_MINIBATCHES=64
VF_COEF=0.5, ENT_COEF=0.01, MAX_GRAD_NORM=0.25, ANNEAL_LR=True

# PH1
PH1_BETA=1.0, PH1_OMEGA=10.0, PH1_POOL_SIZE=128, PH1_WARMUP_STEPS=2000000
PH2_RATIO_STAGE1/2/3=2/1/2
E3T_EPSILON=0.05, ACTION_PREDICTION=True

# CT (config/experiment/rnn-ct.yaml)
TRANSFORMER_ACTION=True, TRANSFORMER_WINDOW_SIZE=16
TRANSFORMER_D_C=128, TRANSFORMER_N_HEADS=4, TRANSFORMER_N_LAYERS=1
TRANSFORMER_RECON_COEF=1.0, TRANSFORMER_PRED_COEF=1.0, TRANSFORMER_CYCLE_COEF=0.5
```

#### 총 Loss 구성

```
Total Loss = PPO_actor_loss
           + VF_COEF × value_loss
           - ENT_COEF × entropy
           + pred_loss              (PartnerPredictor, E3T)
           + ct_recon_loss          (CycleTransformer, 활성화 시)
           + ct_action_loss         (CycleTransformer, 활성화 시)
           + ct_cycle_loss          (CycleTransformer, 활성화 시)
```

#### 주요 파일

| 파일 | 역할 |
|------|------|
| `ph2/overcooked_v2_experiments/ppo/ippo_ph2_core.py` | 핵심 훈련 루프 (2600+ lines) |
| `ph2/overcooked_v2_experiments/ppo/ippo_ph2.py` | 체크포인트 관리, 래퍼 |
| `ph2/overcooked_v2_experiments/ppo/models/rnn.py` | ActorCriticRNN, ScannedRNN |
| `ph2/overcooked_v2_experiments/ppo/models/cycle_transformer.py` | CycleTransformer 모듈 |
| `ph2/overcooked_v2_experiments/ppo/models/e3t.py` | PartnerPredictor |
| `ph2/overcooked_v2_experiments/ppo/run.py` | 실험 오케스트레이션 |
| `ph2/overcooked_v2_experiments/ppo/main.py` | Hydra 진입점, wandb 로깅 |

---

## 2. 컴포넌트 구현 현황

### A. Observation Encoder — ✅ 완전 구현
- **위치**: `models/rnn.py` → `encode_obs()`, `encode_blocked()`
- CNN 기반 인코더. global full state → embedding (GRU_HIDDEN_DIM=64)
- PH1 blocked target 전용 별도 CNN 인코더 포함

### B. ScannedRNN (GRU) — ✅ 완전 구현
- **위치**: `models/rnn.py` → `ScannedRNN`
- JAX GRUCell + `jax.lax.scan` 래퍼. done 마스크로 에피소드 경계 처리

### C. Action Prediction / PartnerPredictor (E3T) — ✅ 완전 구현
- **위치**: `models/e3t.py`
- 파트너 행동을 예측하는 보조 네트워크
- 구조: `z (64) → Dense(64) → Dense(64) → Dense(6) → L2-norm → pred_logits`
- GRU hidden state (stop_gradient) 입력 → cross-entropy loss
- 출력(pred_logits)은 policy head 입력에 concat

### D. State/Action Reconstruction — CycleTransformer — ✅ 완전 구현 (2026-03-21 신규)
- **위치**: `models/cycle_transformer.py`, `models/rnn.py` (통합)
- **활성화 조건**: `config/experiment/rnn-ct.yaml` → `TRANSFORMER_ACTION=True`

**구조**:
```
raw obs → ct_obs_encoder(CNN, 고정) → Window Buffer (W=16, D=128)
  ↓
CausalTransformerEncoder (D_c=128, n_heads=4, n_layers=1)
  ↓
StateDecoder  → z_hat   (D=128)
ActionDecoder → a_hat   (A=6)
  ↓
CycleEncoder(sg(z_hat), sg(a_hat)) → C_prime (D=128)
```

**Reconstruction target (state)**:
- **설계 근거**: 각 agent의 GRU hidden state는 자기 자신의 partial obs sequence에서 형성됨 → 그 agent의 full obs를 재구성 타겟으로 쓰는 것이 논리적으로 일관됨
- **OV1**: agent 0의 full obs를 모든 actor에 tile (OV1은 어차피 full observation 환경)
- **OV2**: `_extract_global_full_obs_per_actor()` → 각 actor 자신의 시점에서 `get_obs_default()` full obs
  - agent 0: self=agent0 채널, agent 1: self=agent1 채널 — self/other mapping이 각 actor와 일치
- `traj_batch.global_obs`에 env_step마다 저장되어 `_loss_fn`에서 `ct_state_encoder`에 전달
- ct_obs_encoder와 ct_state_encoder 모두 gradient를 받지 않음 (고정 random projection)

**Policy head 입력 차원**:
- ACTION_PREDICTION 모드: 128(GRU) + 6(pred_logits) = **134 dims**
- CT 모드: 128(GRU) + 128(z_hat_sg) + 6(a_hat_sg) = **262 dims**
- → actor/critic Dense[0] shape이 달라 두 모드 간 체크포인트 직접 로드 불가

**체크포인트 모드 감지**:
- `ippo_ph2.py::detect_checkpoint_ct_mode(params)` 함수 사용
- params 트리에 `cycle_transformer` 키 유무로 판단
- `load_checkpoint()`는 `(config, params)` 반환 → `config["TRANSFORMER_ACTION"]` 바로 사용 가능

**3가지 손실**:
- `L_recon = MSE(z_hat, sg(ct_state_encoder(ego_obs))) × TRANSFORMER_RECON_COEF`
- `L_action = CE(a_hat, partner_action) × TRANSFORMER_PRED_COEF`
- `L_cycle = MSE(C_prime, sg(C)) × TRANSFORMER_CYCLE_COEF`

**Gradient 흐름**:
- `ct_obs_encoder`, `ct_state_encoder`: gradient 없음 (고정 초기값)
- `CausalTransformerEncoder`, Decoders, CycleEncoder: full gradient (CT 손실 3종으로 훈련)
- Policy head 입력 (z_hat, a_hat): stop_gradient (CT는 auxiliary branch)

### E. Policy Head & Value Head — ✅ 완전 구현
- **위치**: `models/rnn.py` → `ActorCriticRNN.__call__()`
- Actor: concat(z, pred_logits/CT출력) → Dense(128) → Dense(6) → Categorical
- Critic: embedding → Dense(128) → Dense(1)

### F. PH1 Blocked-Target — ✅ 완전 구현
- Pool 기반 샘플링 (PH1_POOL_SIZE=128), 멀티-타겟 패널티, beta 스케줄링
- Warmup 기간 (PH1_WARMUP_STEPS) 이후 pool 활성화

### G. Stablock — ✅ 완전 구현
- **위치**: `utils/stablock.py`
- 좌표 기반 blocked state 관리. 에피소드 경계에서 리샘플링

### H. E3T Epsilon-Greedy Mixture — ✅ 완전 구현
- partner 에이전트에만 E3T_EPSILON 확률로 랜덤 행동 적용

---

## 3. 개발 규칙

> **이 규칙들은 항상 준수한다.**

1. **기존 알고리즘 보호**: 새로운 컴포넌트나 알고리즘을 개발할 때, 기존 알고리즘의 학습·평가·과거 체크포인트 로드에 절대 지장을 주어서는 안 된다. config flag로 새 기능을 격리하고 backwards compatibility를 유지한다.

2. **코드 품질**: 코드는 명쾌하고 짜임새 있게 작성한다. 한글 주석을 적극 사용한다. 복잡한 로직, gradient 흐름 분기, 조건부 활성화 등은 반드시 주석으로 설명한다.

3. **작업 이력 관리**: 아래 섹션(4번)을 항상 최신 상태로 유지한다.

---

## 4. 작업 이력 (항상 최신 상태 유지)

### 최근 수정 완료
- **CycleTransformer 신규 구현** (2026-03-21)
  - `models/cycle_transformer.py` 신규 생성
  - `models/rnn.py` — `ActorCriticRNN`에 CycleTransformer 통합 (ct_encode_state, cycle_transformer_forward)
  - `ippo_ph2_core.py` — ct 손실 3종 추가 (recon, action, cycle)
  - `models/model.py` — 모델 초기화 로직 업데이트
- **체크포인트 모드 감지 헬퍼 추가** (2026-03-23)
  - `ippo_ph2.py` — `detect_checkpoint_ct_mode(params)` 추가
  - CT 체크포인트 vs ACTION_PREDICTION 체크포인트 자동 감지

### 최근 수정 완료 (2026-03-23)
- **CT recon target → full global state** (`ippo_ph2_core.py`)
  - `Transition`에 `global_obs` 필드 추가 (CT recon 타겟용 full global state)
  - `_extract_global_full_obs_per_actor()` 신규 추가 (OV2 per-actor full obs)
  - OV1/OV2 감지 변수 `is_partial_obs` 추가 (env kwargs의 `agent_view_size` 유무로 판단)
  - env_step: OV1은 agent0 full obs tile, OV2는 per-actor full obs → `_global_obs`로 Transition에 저장
  - `_loss_fn`: CT recon target을 `traj_batch.obs` → `traj_batch.global_obs`로 변경
  - `ct_state_encoder` dummy init shape: `state_shape` → `ph1_block_shape` (OV2 full obs shape)
- **CT 실행 스크립트 지원** (`run_user_wandb.sh`, `run_factory_ph2.sh`)
  - `run_user_wandb.sh`: TRANSFORMER_* 8개 플래그 defaults/파싱/PY_ARGS 추가
  - `run_factory_ph2.sh`: `USE_CT=1` 환경변수로 CT 모드 활성화 지원

### Human-AI Study Webapp 구현 (2026-03-26)
- **위치**: `webapp/`
- **목적**: 원격 참여자가 Overcooked에서 학습된 AI와 협력 플레이 → 주관 평가 설문 + BC용 trajectory 수집
- **구조**:
  - `app/main.py` — FastAPI 서버 (JAX CPU-only, `JAX_PLATFORMS=cpu`)
  - `app/agent/loader.py` — Orbax checkpoint GPU→CPU 로드 (metadata 기반 sharding 복원)
  - `app/agent/inference.py` — ModelManager: SP/E3T/FCP/PH2/MEP 모두 지원, PH2는 `params_ind` 자동 선택
  - `app/game/engine.py` — GameSession: overcooked-ai 환경 래핑, JaxMARL layout 호환
  - `app/game/obs_adapter.py` — overcooked-ai state → JaxMARL obs (H,W,C) 변환
  - `app/game/layouts/` — JaxMARL layout을 overcooked-ai `.layout` 파일로 변환 (grid 크기 일치)
  - `app/api/routes.py` — WebSocket 게임 루프 + REST (survey, admin export)
  - `app/db/models.py` — SQLite: Participant, Episode(collisions/deliveries), SurveyResponse(7문항)
  - `frontend/` — Vanilla JS SPA: canvas 렌더링, i18n(한/영), layout 선택
- **모델 디렉토리**: `webapp/models/{layout}/{algo}/{run}/ckpt_final/`
  - 5 layout × 5 algo (sp, e3t, fcp, ph2, mep) = 225 checkpoints
  - mep 모델은 `baseline/runs/` 의 `_m2` run에서 복사
- **JaxMARL 환경 호환**:
  - 재료 3개 자동 요리 (interact로 조기 요리 시작 차단 — `_patch_mdp_no_early_cook`)
  - layout grid 크기 일치 (counter_circuit: 5×8 등)
  - obs shape 검증 완료 (cramped_room 4×5×30, counter_circuit 5×8×30 등)
- **BC 데이터**: `data/trajectories/{participant_id}/{episode_id}.pkl`
  - obs_human: JaxMARL-compatible (H,W,C) uint8, action_human: int 0-5
- **설문**: fluency, contribution, trust, human_likeness, obstruction, frustration, play_again (Likert 1-7) + open_text
- **실행**: `cd webapp && JAX_PLATFORMS=cpu uvicorn app.main:app --port 8000 --loop asyncio`

### ToyCoop penalty distance: raw L2 → MLP latent distance (2026-03-31)
- **파일**: `ippo_ph2_core.py`
- ToyCoop penalty 거리 계산을 raw pixel-space L2에서 `encode_blocked()` MLP latent distance로 변경
- `PH1_RAW_DISTANCE` config flag 추가 (기본값 `false` → MLP latent, `true` → 기존 raw L2)
- 기존 forward pass에서 이미 `blocked_emb_slots`를 계산하고 있었으나 ToyCoop에서만 무시하고 있었음 → 해당 분기 조건을 `env_name == "ToyCoop"` → `ph1_raw_distance`로 변경

### Webapp: MEP 알고리즘 호환 + obs_adapter 버그 수정 (2026-04-01)
- **MEP 모델 추가**: `baseline/runs/` 의 `_m2` run에서 5개 레이아웃 × 10 runs의 ckpt_final을 `webapp/models/{layout}/mep/` 로 복사
- **MEP 로드 경로 분리** (`app/agent/inference.py`, `app/agent/loader.py`)
  - `detect_model_source()`: ALG_NAME에 "MEP" 포함 시 `"baseline_native"` 반환
  - `_build_baseline_network_native()`: baseline `ActorCriticRNN`을 importlib로 직접 로드, `actor_only=True`로 forward (value head 없이 추론)
  - SP/E3T/FCP는 기존 PH2 리매핑 경로 유지, PH2는 기존 경로 유지
  - MEP 체크포인트에는 value head (Dense_2/3) params가 저장되지 않으므로, baseline 코드의 `actor_only=True` 옵션으로 value head를 스킵
- **obs_adapter 버그 수정** (`app/game/obs_adapter.py`)
  - 플레이어가 완성된 수프를 들고 있을 때 `is_ready=False` 하드코딩 → `getattr(held, "is_ready", False)` 로 수정
  - 수정 전: held soup의 PLATE=0, COOKED=0 (잘못된 인코딩)
  - 수정 후: held soup의 PLATE=1, COOKED=1 (JaxMARL 일치)
- **config.yaml**: `algo_filter: ["ph2"]` → `["ph2", "sp", "e3t", "fcp", "mep"]` 전 알고리즘 활성화
- **Obs/Action 호환성 검증 완료**: 5개 레이아웃 × 10개 시나리오 (양파 픽업~수프 배달) 전부 PASS
  - `webapp/scripts/test_cooperation_scenarios.py` 신규 생성

### Human-Proxy BC 파이프라인 (2026-04-01 검증)
- **위치**: `human-proxy/`
- **3단계 파이프라인**: extract → train → evaluate
  1. `python code/extract.py --traj-dir ../webapp/data/trajectories --out-dir data/bc`
     - webapp trajectory pkl → 레이아웃/포지션별 `obs.npy + actions.npy` 변환
     - `--recompute-obs` 옵션으로 state dict에서 obs 재계산 가능
  2. `python code/train.py --layout cramped_room --position 0 --num-seeds 5`
     - CNN 기반 BCModel (Conv32→Conv32→FC64→FC6), pickle 저장
  3. `bash sh_scripts/eval_with_bc.sh ALGO_DIR LAYOUT [NUM_EVAL_SEEDS]`
     - runs/ 폴더 지정으로 BC × RL cross-play 평가 (ph2/baseline 자동 감지)
- **현재 상태**: cramped_room만 BC 학습 완료 (5 seeds × 2 positions), eval 완료

### forced_coord PH2 best 갱신 (2026-04-19, 2차)
- 최종: `ph2/runs/20260419-070914_8ixp5kyd_forced_coord_e3t_ph2_e0p3_o3_s8_k1_ct0` (eps=0.3, ω=3, σ=8, k=1, SP=193±12 / XP=158±55)
- 이전 best (pmtjfgzg, ω=4, SP=188 / XP=148) 보다 SP/XP 모두 상승
- `final/ph2/forced_coord/20260419-070914_8ixp5kyd...` 에 top-level run_0..9 + metadata + reward_summary 복사 (91M). 이전 pmtjfgzg 디렉토리도 유지
- `webapp/models/forced_coord/ph2/` 교체 (10 runs, 46M). 이전 pmtjfgzg 기반은 `ph2_old_pmtjfgzg/` 로 백업
- `human-proxy/sh_scripts/eval_final_v2.sh` / `eval_final_v3.sh` 의 forced_coord PH2 경로도 새 path로 갱신

### Webapp 설문·객관지표 재설계 (2026-04-19)
- **Post-episode 설문 8항목** (1-7 Likert, `#8 workload` 만 Very Low~Very High)
  - adaptive / consistent / human_like / enjoyed / coordination → H2
  - in_my_way / frustrating (역채점) → H3
  - workload (NASA-TLX) → H3
- 기존 7항목 (fluency/contribution/trust/human_likeness/obstruction/frustration/play_again) 은 전부 교체
- `frontend/index.html`, `frontend/js/i18n.js`, `frontend/js/main.js`, `app/api/schemas.py`, `app/db/models.py`, `app/api/routes.py`, `config.yaml` 동기 업데이트
- **객관 지표 5종** 자동 로깅 (episode 단위):
  1. `final_score` (기존) — team score
  2. `deliveries` (기존) — successful deliveries
  3. `collisions` (기존) — collision count
  4. `role_specialization` (신규, 0~1) — per-task agent-별 count 에서 `1 - Σ 2·min(h,a) / Σ(h+a)`
  5. `idle_time_ratio` (신규, 0~1) — AI stay + stuck timestep 비율 (interact 는 제외)
  - 추가로 `task_events` (JSON, 5 task × 2 agent) 도 저장하여 사후 재집계 가능
- `app/game/engine.py` GameSession 에 task-event tracker + helpers 추가
- `app/db/session.py` 에 ALTER TABLE ADD COLUMN 마이그레이션 로직 추가 (기존 SQLite DB 호환)
- CSV export (`/admin/export`) 헤더도 새 지표 반영

### forced_coord MEP best 갱신 (2026-04-18)
- `baseline/runs/20260416-162055_lz0c4ozg_forced_coord_mep_pop8` (sp=31, xp=34) 이 기존 plv7yajb (sp=21, xp=25) 를 능가
- `final/baseline/forced_coord/20260416-162055_lz0c4ozg_forced_coord_mep_pop8/` 에 복사 (메타데이터 + mep_population + run_0..9/ckpt_final, 37M)
- `webapp/models/forced_coord/mep/` 교체 (이전 _m2 기반은 `mep_old_m2/` 로 백업)

### PH1 Penalty Linear Mode 추가 (2026-04-21)
- **목적**: 기존 `ω·exp(-σ·dist)` 외에 `α/(ε+dist)` 역수형 penalty 옵션 추가 (Overcooked 실험용)
- `config/experiment/rnn-ph2.yaml`: `PH1_PENALTY_LINEAR_MODE` (default false), `PH1_LINEAR_ALPHA` (default 0.02), `PH1_LINEAR_EPSILON` (default 0.001) 신규
- `ippo_ph2_core.py`: penalty 계산 한 줄(line ~1659)을 if/else 분기로 교체. linear 모드면 `α/(ε+dist)`, 아니면 기존 exp. 분기 외 흐름(slot 합산, warmup, reward 차감, 로깅)은 모두 그대로
- `run_user_wandb.sh`: `--ph1-penalty-linear-mode/--ph1-linear-alpha/--ph1-linear-epsilon` CLI 플래그 추가
- `run_factory_ph2.sh`: 환경변수 default + `run_ph2()` 함수에 3개 플래그 패스스루 추가
- **호환성**: default=false → 기존 모든 실험·체크포인트 동작 100% 동일
- **α 추천값 도출 근거**: forced_coord 12개 wandb run summary 분석 → 학습 종료 시 평균 dist≈2~4, exp penalty 평균≈0.005~0.02. linear 평균 매칭값 α=0.02 기본 (sweep 권장: {0.005, 0.02, 0.05, 0.1})
- **⚠️ dist→0 폭주 주의**: max penalty=α/0.001=1000·α. α=0.02면 cap=20 (reward scale과 동급). α≥0.1 사용 시 warmup 충분히 설정 권장

### 현재 작업 중
- (없음)

### 향후 진행 예정
- `USE_CT=1 bash run_factory_ph2.sh`로 CT 모드 훈련 실행 및 검증
- Webapp: Docker + nginx 배포, 동시 접속 부하 테스트
- Human-Proxy: 나머지 4개 레이아웃 BC 학습 + eval

---

*이 파일은 작업이 진행되면서 지속적으로 업데이트되어야 한다.*
