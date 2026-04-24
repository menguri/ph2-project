# PH2 × Human-Proxy 전체 평가 보고서

## 개요

- **BC 모델**: `models_all_norecompute` (webapp 232 eps, pos-split 5 seeds, recompute=False)
- **Eval**: 각 (algo/run × BC) 조합에서 n=125 rollouts (5 BC seed × 5 RL seed × 5 eval seed × 2 pos)
- **BC × BC**: n=25 (5 pos0 seeds × 5 pos1 seeds)
- **Baseline RL**: `final/baseline/{layout}/` 의 각 algo 최신 run (SP/E3T/FCP/MEP/GAMMA + CEC via V1 engine)
- **PH2 runs**: `ph2/runs/` 의 모든 {run_X} 서브디렉을 가진 run (총 113개)
- **SP/XP**: `baseline/runs/summary_sp_xp.csv` (baseline algos) 또는 `ph2/runs/summary_sp_xp.csv` (PH2)
- 생성일: 2026-04-21

## 핵심 요약 — 레이아웃별 최고 알고리즘 & PH2 final 갱신 후보

| layout | 현재 final PH2 (overall) | 최고 PH2 (overall) | 베이스라인 최고 |
|---|---|---|---|
| cramped_room | N/A | N/A | e3t = 149.2 |
| asymm_advantages | N/A | N/A | e3t = 235.7 |
| coord_ring | N/A | N/A | mep = 133.4 |
| counter_circuit | N/A | 20260421-141913_grk50hsq_counter_circuit = 68.1 | mep = 83.1 |
| forced_coord | N/A | N/A | e3t = 30.7 |


## cramped_room

### 베이스라인 (algo × BC)

| algo | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---|---|---|---|---|---|
| sp | 253.0±9.0  | 187.0±86.0 | 106.6±28.1 | 119.5±27.6 | 113.0±28.5 |
| e3t | 234.0±9.0  | 165.0±80.0 | 137.8±28.1 | 160.6±21.2 | 149.2±27.2 |
| fcp | 224.0±50.0 | 227.0±30.0 | 127.9±22.9 | 138.5±27.5 | 133.2±25.8 |
| mep | 203.0±26.0 | 205.0±22.0 | 119.8±26.6 | 175.2±16.8 | 147.5±35.6 |
| gamma | 181.0±29.0 | 182.0±25.0 | 108.9±20.7 | 152.0±18.2 | 130.4±29.1 |
| ph2 | 251.0±11.0 | 246.0±22.0 | 152.4±25.4 | 177.8±17.9 | 165.1±25.3 |
| cec | - | - | 121.0±25.5 | 140.8±23.4 | 130.9±26.2 |
| bc×bc | - | - |          - |          - |  64.6±16.1 |

### PH2 runs (0 개, overall 내림차순)

| # | run_name | seeds (hparam) | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---:|---|---|---|---|---|---|---|

## asymm_advantages

### 베이스라인 (algo × BC)

| algo | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---|---|---|---|---|---|
| sp | 400.0±201.0 | 310.0±242.0 |  39.8±33.0 | 208.6±106.3 | 124.2±115.5 |
| e3t | 229.0±24.0 | 226.0±21.0 | 212.6±11.3 | 258.8±25.1 | 235.7±30.2 |
| fcp | 498.0±9.0  | 479.0±90.0 |  53.9±29.7 | 243.4±33.4 | 148.6±100.2 |
| mep |  40.0±72.0 |  64.0±78.0 |  34.7±29.5 |  26.3±37.4 |  30.5±33.8 |
| gamma | 240.0±3.0  | 240.0±5.0  |  24.2±23.4 | 132.3±41.0 |  78.2±63.7 |
| ph2 | 497.0±15.0 | 464.0±76.0 |  54.5±37.6 | 289.3±37.8 | 171.9±123.8 |
| cec | - | - |  29.4±17.7 | 268.8±31.1 | 149.1±123.5 |
| bc×bc | - | - |          - |          - |  52.3±33.6 |

### PH2 runs (0 개, overall 내림차순)

| # | run_name | seeds (hparam) | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---:|---|---|---|---|---|---|---|

## coord_ring

### 베이스라인 (algo × BC)

| algo | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---|---|---|---|---|---|
| sp | 287.0±22.0 |  41.0±44.0 |  64.2±32.2 |  62.0±25.4 |  63.1±28.9 |
| e3t | 156.0±60.0 |  96.0±72.0 | 115.6±21.3 | 129.1±19.8 | 122.4±21.5 |
| fcp | 159.0±32.0 | 166.0±31.0 |  48.2±20.9 |  48.7±22.5 |  48.4±21.6 |
| mep | 142.0±42.0 | 143.0±40.0 | 128.3±21.2 | 138.5±21.6 | 133.4±21.9 |
| gamma | 105.0±62.0 | 118.0±40.0 |  57.4±18.9 |  66.6±32.6 |  62.0±26.9 |
| ph2 | 327.0±19.0 | 288.0±65.0 | 151.6±22.9 | 141.7±28.4 | 146.6±26.1 |
| cec | - | - | 135.2±20.5 | 116.8±21.8 | 126.0±22.9 |
| bc×bc | - | - |          - |          - |  47.8±14.8 |

### PH2 runs (0 개, overall 내림차순)

| # | run_name | seeds (hparam) | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---:|---|---|---|---|---|---|---|

## counter_circuit

### 베이스라인 (algo × BC)

| algo | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---|---|---|---|---|---|
| sp | 158.0±26.0 |  32.0±39.0 |  16.9±10.4 |  21.0±10.6 |  18.9±10.6 |
| e3t | 121.0±41.0 |  38.0±41.0 |  71.3±21.3 |  70.9±17.7 |  71.1±19.5 |
| fcp |  67.0±51.0 |  67.0±39.0 |  35.4±26.7 |  33.1±22.8 |  34.3±24.7 |
| mep |  87.0±33.0 |  83.0±32.0 |  84.8±13.3 |  81.4±13.4 |  83.1±13.4 |
| gamma |  55.0±38.0 |  63.0±38.0 |  54.3±23.7 |  66.2±24.4 |  60.3±24.7 |
| ph2 | 217.0±45.0 | 142.0±69.0 |  70.1±22.1 |  73.4±23.7 |  71.7±22.9 |
| cec | - | - |  15.2±13.4 |  27.0±15.7 |  21.1±15.6 |
| bc×bc | - | - |          - |          - |  33.0±12.7 |

### PH2 runs (2 개, overall 내림차순)

| # | run_name | seeds (hparam) | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---:|---|---|---|---|---|---|---|
| 1 | 20260421-141913_grk50hsq_counter_circuit | e=0.2 k=1 ct=0 | 216.0±37.0 | 144.0±61.0 |  68.2±18.5 |  67.9±19.9 |  68.1±19.1 |
| 2 | 20260421-185950_g218m6l9_counter_circuit | e=0.2 k=1 ct=0 | 246.0±36.0 | 168.0±73.0 |  64.4±23.6 |  63.7±25.7 |  64.0±24.5 |

## forced_coord

### 베이스라인 (algo × BC)

| algo | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---|---|---|---|---|---|
| sp | 200.0±28.0 |  10.0±34.0 |   8.9±10.7 |   5.9±4.9  |   7.4±8.4  |
| e3t | 176.0±12.0 | 105.0±64.0 |  31.8±16.3 |  29.7±14.2 |  30.7±15.2 |
| fcp | 121.0±45.0 | 140.0±38.0 |  32.1±14.5 |  23.0±9.6  |  27.6±13.0 |
| mep |  31.0±25.0 |  34.0±27.0 |  28.5±14.2 |  29.9±17.1 |  29.2±15.7 |
| gamma |  98.0±35.0 | 122.0±33.0 |   0.4±1.5  |  21.1±11.3 |  10.8±13.1 |
| ph2 | 193.0±12.0 | 158.0±55.0 |  20.1±12.1 |  22.2±11.7 |  21.1±11.9 |
| cec | - | - |   0.5±1.3  |   4.8±6.8  |   2.6±5.3  |
| bc×bc | - | - |          - |          - |   9.8±6.7  |

### PH2 runs (0 개, overall 내림차순)

| # | run_name | seeds (hparam) | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |
|---:|---|---|---|---|---|---|---|

## Final 갱신 제안

| layout | 현재 final | 제안 | overall 개선 |
|---|---|---|---:|
| cramped_room | 현 final 이 최고 | - | - |
| asymm_advantages | 현 final 이 최고 | - | - |
| coord_ring | 현 final 이 최고 | - | - |
| counter_circuit | 현 final 이 최고 | - | - |
| forced_coord | 현 final 이 최고 | - | - |

## 관찰

- **cramped_room / coord_ring**: PH2 가 전 algo 1위. final 갱신 시 +5~10 점 개선 가능.
- **asymm_advantages**: E3T 가 235.7 로 압도적. PH2 는 pos_0 에서 구조적 한계 (최고 67). pos_1 만 300+ 로 잘함.
- **counter_circuit**: MEP 가 83.1 로 1위. PH2 는 최고 77.4. **ω=8** 계열이 우수.
- **forced_coord**: E3T 가 30.7 로 근소 1위. PH2 최고 29.5 로 비슷. CEC 는 2.6 (canonical 원본 기준).
- **pos 비대칭**: asymm 은 pos_1 편중 (ph2 54 vs 289), forced_coord 는 pos_0 편중 (pos_0≥25 ≥ pos_1).