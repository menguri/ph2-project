"""
Phase 5-6: FastAPI endpoints — WebSocket game loop + REST APIs.
"""
import csv
import io
import json
import random
import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import StreamingResponse

from .schemas import PreSurveySubmit, PostSurveySubmit
from ..db.session import get_session_sync, init_db
from ..db.models import Participant, Episode, SurveyResponse
from ..agent.inference import ModelManager
from ..agent.loader import scan_models_dir
from ..game.engine import GameSession
from ..game.action_map import keyboard_to_action
from ..trajectory.recorder import TrajectoryRecorder

router = APIRouter()

# 전역 설정 (main.py에서 초기화)
_config = None
_models_by_layout = {}  # {layout: [{algo_name, seed_id, ckpt_path}, ...]}
_model_cache = {}


def init_routes(config: dict):
    """앱 시작 시 설정 로드 + 모델 목록 스캔."""
    global _config, _models_by_layout
    _config = config

    init_db(config["database"]["path"])

    # 모델 스캔: webapp/models/{layout}/{algo}/{run}/ckpt_final/
    from ..agent.loader import scan_models_dir
    model_dir = config["agent"].get("model_dir", "models")
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent.parent.parent / model_dir
    _models_by_layout = scan_models_dir(str(model_path.resolve()))
    # 특정 알고리즘만 필터링 (테스트용, None이면 전체)
    _algo_filter = config.get("agent", {}).get("algo_filter", None)
    if _algo_filter:
        _models_by_layout = {
            layout: [m for m in models if m["algo_name"] in _algo_filter]
            for layout, models in _models_by_layout.items()
        }
        _models_by_layout = {k: v for k, v in _models_by_layout.items() if v}
        print(f"[init] Algo filter: {_algo_filter}")
    total = sum(len(v) for v in _models_by_layout.values())
    print(f"[init] Found {total} model checkpoints in {model_path}")
    for layout, models in _models_by_layout.items():
        for m in models:
            print(f"  - {layout}/{m['algo_name']}/{m['seed_id']}")


def _get_or_load_model(ckpt_path: str, algo_name: str, seed_id: str) -> ModelManager:
    """모델 캐시 (같은 checkpoint 재로드 방지)."""
    cache_key = ckpt_path
    if cache_key not in _model_cache:
        mm = ModelManager()
        mm.load(
            ckpt_path=ckpt_path,
            policy_source=_config["agent"].get("policy_source", "params"),
            stochastic=_config["agent"].get("stochastic", True),
            algo_name=algo_name,
            seed_id=seed_id,
        )
        _model_cache[cache_key] = mm
        print(f"[model] Loaded {algo_name}/{seed_id} from {ckpt_path}")
    return _model_cache[cache_key]


# ========== WebSocket Game Loop ==========

@router.websocket("/ws/{participant_id}")
async def game_websocket(websocket: WebSocket, participant_id: str):
    await websocket.accept()

    # participant 생성/조회
    db = get_session_sync()
    participant = db.query(Participant).filter_by(id=participant_id).first()
    if not participant:
        participant = Participant(id=participant_id)
        db.add(participant)
        db.commit()
    db.close()

    try:
        while True:
            # 클라이언트가 게임 시작 요청 대기
            start_msg = await websocket.receive_json()
            if start_msg.get("type") != "start_game":
                await websocket.send_json({"error": "Expected start_game message"})
                continue

            # layout 결정 (클라이언트가 보내거나 config 기본값)
            layout = start_msg.get("layout", _config["game"].get("default_layout", "cramped_room"))

            # 해당 layout의 모델 중 랜덤 선택
            layout_models = _models_by_layout.get(layout, [])
            if not layout_models:
                await websocket.send_json({"error": f"No models for layout '{layout}'"})
                continue

            model_info = random.choice(layout_models)
            model = _get_or_load_model(
                model_info["ckpt_path"],
                model_info["algo_name"],
                model_info["seed_id"],
            )

            # trajectory recorder
            recorder = TrajectoryRecorder(
                save_dir=_config["trajectory"]["save_dir"],
                save_obs=_config["trajectory"].get("save_obs", True),
                save_state=_config["trajectory"].get("save_state", True),
            )

            # game session
            session = GameSession(
                layout=layout,
                model=model,
                recorder=recorder,
                participant_id=participant_id,
                episode_length=_config["game"].get("episode_length", 400),
            )

            # 초기 상태 전송
            init_info = session.get_init_info()
            init_info["algo_name"] = model_info["algo_name"]
            init_info["seed_id"] = model_info["seed_id"]
            await websocket.send_json({"type": "game_start", **init_info})

            # 게임 루프
            while not session.done:
                try:
                    msg = await websocket.receive_json()
                except WebSocketDisconnect:
                    session.force_end()
                    raise

                action_key = msg.get("action", msg.get("key", ""))
                if isinstance(action_key, int):
                    human_action = action_key
                else:
                    human_action = keyboard_to_action(str(action_key))

                result = session.step(human_action)
                await websocket.send_json({"type": "state_update", **result})

                if result["done"]:
                    # DB에 에피소드 기록
                    db = get_session_sync()
                    ep = Episode(
                        id=session.episode_id,
                        participant_id=participant_id,
                        algo_name=model_info["algo_name"],
                        seed_id=model_info["seed_id"],
                        layout=layout,
                        human_player_index=session.human_idx,
                        final_score=session.score,
                        episode_length=session.timestep,
                        collisions=session.collisions,
                        deliveries=session.deliveries,
                    )
                    db.add(ep)
                    db.commit()
                    db.close()

                    await websocket.send_json({
                        "type": "episode_end",
                        "episode_id": session.episode_id,
                        "final_score": session.score,
                        "algo_name": model_info["algo_name"],
                        "collisions": session.collisions,
                        "deliveries": session.deliveries,
                    })

            # 루프 끝 — 클라이언트가 play_again 보내면 다시 시작

    except WebSocketDisconnect:
        pass


# ========== REST Endpoints ==========

@router.post("/survey/pre")
async def submit_pre_survey(data: PreSurveySubmit):
    db = get_session_sync()
    participant = db.query(Participant).filter_by(id=data.participant_id).first()
    if not participant:
        participant = Participant(id=data.participant_id)
        db.add(participant)

    participant.pre_survey = {
        "age": data.age,
        "gender": data.gender,
        "gaming_exp": data.gaming_exp,
        "overcooked_exp": data.overcooked_exp,
    }
    db.commit()
    db.close()
    return {"status": "ok"}


@router.post("/survey/post")
async def submit_post_survey(data: PostSurveySubmit):
    db = get_session_sync()
    existing = db.query(SurveyResponse).filter_by(episode_id=data.episode_id).first()
    if existing:
        db.close()
        raise HTTPException(400, "Survey already submitted for this episode")

    survey = SurveyResponse(
        id=str(uuid.uuid4()),
        episode_id=data.episode_id,
        fluency=data.fluency,
        contribution=data.contribution,
        trust=data.trust,
        human_likeness=data.human_likeness,
        obstruction=data.obstruction,
        frustration=data.frustration,
        play_again=data.play_again,
        open_text=data.open_text,
    )
    db.add(survey)
    db.commit()
    db.close()
    return {"status": "ok"}


@router.get("/api/layouts")
async def get_available_layouts():
    """사용 가능한 layout 목록 (모델이 있는 것만)."""
    layouts = {}
    for layout, models in _models_by_layout.items():
        algos = sorted(set(m["algo_name"] for m in models))
        layouts[layout] = {"algos": algos, "count": len(models)}
    return {"layouts": layouts}


@router.get("/admin/export")
async def admin_export(password: str = Query(...)):
    if password != _config["admin"]["password"]:
        raise HTTPException(403, "Invalid password")

    db = get_session_sync()
    episodes = db.query(Episode).all()
    participants = {p.id: p for p in db.query(Participant).all()}
    surveys = {s.episode_id: s for s in db.query(SurveyResponse).all()}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "episode_id", "participant_id", "algo_name", "seed_id", "layout",
        "human_player_index", "final_score", "episode_length",
        "collisions", "deliveries", "created_at",
        "age", "gender", "gaming_exp", "overcooked_exp",
        "fluency", "contribution", "trust",
        "human_likeness", "obstruction", "frustration", "play_again",
        "open_text",
    ])

    for ep in episodes:
        p = participants.get(ep.participant_id)
        pre = (p.pre_survey or {}) if p else {}
        s = surveys.get(ep.id)
        writer.writerow([
            ep.id, ep.participant_id, ep.algo_name, ep.seed_id, ep.layout,
            ep.human_player_index, ep.final_score, ep.episode_length,
            getattr(ep, "collisions", ""), getattr(ep, "deliveries", ""),
            ep.created_at,
            pre.get("age"), pre.get("gender"), pre.get("gaming_exp"), pre.get("overcooked_exp"),
            s.fluency if s else "", s.contribution if s else "",
            s.trust if s else "",
            s.human_likeness if s else "", s.obstruction if s else "",
            s.frustration if s else "", s.play_again if s else "",
            s.open_text if s else "",
        ])

    db.close()
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=study_results.csv"},
    )
