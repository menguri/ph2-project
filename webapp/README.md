# PH2 Human-AI Overcooked Study Webapp

ZSC(Zero-Shot Coordination) 연구를 위한 Human-AI 상호작용 실험 웹 앱.

## 개요

- 참여자가 웹 브라우저에서 Overcooked 게임을 AI 파트너와 협력 플레이
- 게임 후 주관 평가 설문 (Likert 7점 척도, 7문항)
- 매 에피소드의 trajectory를 JaxMARL-compatible 포맷으로 저장 (BC 학습용)

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| Backend | FastAPI + WebSocket |
| Frontend | Vanilla JS + Canvas |
| Game Logic | overcooked-ai (HumanCompatibleAI) |
| AI Models | JaxMARL (Flax/Orbax) — CPU inference |
| Database | SQLite (SQLAlchemy) |
| i18n | 한국어 / English |

## 디렉토리 구조

```
webapp/
├── app/
│   ├── main.py              # FastAPI 앱 진입점
│   ├── config.py             # YAML config 로더
│   ├── agent/
│   │   ├── loader.py         # Orbax checkpoint CPU 로드
│   │   └── inference.py      # ModelManager (SP/E3T/FCP/PH2)
│   ├── game/
│   │   ├── engine.py         # GameSession (환경 래핑 + 충돌/배달 카운터)
│   │   ├── obs_adapter.py    # overcooked-ai state → JaxMARL obs (H,W,C)
│   │   ├── action_map.py     # action index 매핑
│   │   └── layouts/          # JaxMARL layout → overcooked-ai .layout 파일
│   ├── api/
│   │   ├── routes.py         # WebSocket + REST endpoints
│   │   └── schemas.py        # Pydantic models
│   ├── db/
│   │   ├── models.py         # SQLAlchemy ORM
│   │   └── session.py        # DB session factory
│   └── trajectory/
│       └── recorder.py       # BC trajectory 저장 (.pkl)
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── main.js           # Game loop + canvas rendering
│       └── i18n.js           # 한/영 번역
├── models/                   # AI 체크포인트
│   └── {layout}/{algo}/{run}/ckpt_final/
├── data/
│   ├── db/study.db           # SQLite
│   └── trajectories/         # BC 데이터 (.pkl)
├── config.yaml
└── README.md
```

## 모델 구조

```
models/
├── cramped_room/
│   ├── sp/run0/ckpt_final/
│   ├── e3t/run0/ckpt_final/
│   ├── fcp/run0/ckpt_final/
│   └── ph2/run0/ckpt_final/
├── asymm_advantages/
│   └── ...
├── coord_ring/
│   └── ...
├── forced_coord/
│   └── ...
└── counter_circuit/
    └── ...
```

- **5 layout × 4 algorithm = 20 checkpoints**
- PH2 모델은 자동으로 `params_ind` (Independent policy) 사용
- 게임 시작 시 선택한 layout에서 알고리즘이 랜덤 배정

## 실행

```bash
cd webapp
JAX_PLATFORMS=cpu uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop asyncio
```

## 접속 방법

### 1. 로컬 (개발용) — SSH 포트포워딩

```bash
# 로컬 PC 터미널에서:
ssh -L 8000:localhost:8000 user@server
# 브라우저에서 http://localhost:8000
```

### 2. 외부 접속 (실험 참여자 배포용) — cloudflared

방화벽/NAT 뒤에 있어도 외부 URL을 생성해주는 터널. 가입 불필요.

```bash
# 1) cloudflared 설치 (최초 1회)
curl -sSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /tmp/cloudflared && chmod +x /tmp/cloudflared

# 2) 서버 실행 (백그라운드)
cd webapp
JAX_PLATFORMS=cpu nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop asyncio > /tmp/webapp.log 2>&1 &

# 3) 터널 실행
/tmp/cloudflared tunnel --url http://localhost:8000
# → 출력에 https://xxxx-xxxx-xxxx.trycloudflare.com URL이 나옴
# → 이 URL을 참여자에게 공유
```

참고:
- 터널 프로세스를 종료하면 URL도 비활성화됨
- 매번 실행할 때마다 URL이 바뀜 (고정 URL은 Cloudflare 계정 + named tunnel 필요)
- 동시 접속 가능 (FastAPI async + WebSocket)

### 3. 종료

```bash
# 서버 + 터널 종료
kill $(lsof -ti:8000)          # uvicorn
pkill -f cloudflared           # 터널
```

## JaxMARL 환경 호환

webapp은 overcooked-ai의 게임 로직을 사용하되, JaxMARL과 동일한 동작을 보장:

| 항목 | 처리 방식 |
|------|-----------|
| Grid 크기 | JaxMARL layout을 `.layout` 파일로 변환 (counter_circuit 5×8 등) |
| 요리 시작 | 재료 3개 넣으면 자동 시작 (interact로 조기 시작 차단) |
| Dispenser 되돌려놓기 | 불가 (JaxMARL과 동일) |
| 에이전트 충돌 | 같은 셀 점유 불가, swap 방지 (overcooked-ai 기본) |
| Obs shape | JaxMARL과 동일 (cramped_room: 4×5×30 등) |

## BC 데이터

`data/trajectories/{participant_id}/{episode_id}.pkl`:

```python
{
    "episode_id": str,
    "participant_id": str,
    "algo_name": str,         # sp, e3t, fcp, ph2
    "layout": str,
    "human_player_index": int, # 0 or 1
    "final_score": int,
    "episode_length": int,
    "transitions": [
        {
            "timestep": int,
            "obs_human": np.ndarray,   # (H, W, 30) uint8 — JaxMARL compatible
            "action_human": int,       # 0-5 (right, down, left, up, stay, interact)
            "joint_action": [int, int],
            "reward": float,
            "cumulative_score": int,
            "state": dict,             # overcooked-ai state dict
        },
        ...
    ]
}
```

## 설문 항목

### Pre-game (인구통계)
- 나이, 성별, 게임 경험 (1-7), Overcooked 경험

### Post-episode (AI 평가, Likert 1-7)
1. **Fluency**: AI와 원활하게 협력했다
2. **Contribution**: AI가 팀 성과에 기여했다
3. **Trust**: AI의 행동을 신뢰했다
4. **Human-likeness**: AI가 사람처럼 느껴졌다
5. **Obstruction**: AI가 내 길을 방해했다
6. **Frustration**: 게임 중 답답함을 느꼈다
7. **Play again**: 이 AI와 다시 플레이하고 싶다
8. **Open text**: 자유 서술

## Admin API

```bash
# CSV 다운로드 (episode + survey + participant 통합)
curl "http://localhost:8000/admin/export?password=changeme" -o results.csv

# 사용 가능한 layout 목록
curl http://localhost:8000/api/layouts
```
