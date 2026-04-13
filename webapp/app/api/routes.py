"""
Phase 5-6: FastAPI endpoints — WebSocket game loop + REST APIs.
"""
import asyncio
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
_layout_order = []      # 레이아웃 순서 (모델이 있는 것만)
_algos_per_layout = {}  # {layout: [algo_name, ...]} 고유 알고리즘 목록


def init_routes(config: dict):
    """앱 시작 시 설정 로드 + 모델 목록 스캔."""
    global _config, _models_by_layout, _layout_order, _algos_per_layout
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

    # 레이아웃 순서 및 알고리즘 목록 구축
    _layout_order = sorted(_models_by_layout.keys())
    _algos_per_layout = {}
    for layout, models in _models_by_layout.items():
        _algos_per_layout[layout] = sorted(set(m["algo_name"] for m in models))
    print(f"[init] Layouts: {_layout_order}")
    for layout, algos in _algos_per_layout.items():
        print(f"  {layout}: {len(algos)} algos — {algos}")


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


def _get_study_progress(participant_id: str) -> dict:
    """참가자의 스터디 진행 상황 조회."""
    db = get_session_sync()
    episodes = db.query(Episode).filter_by(participant_id=participant_id).all()
    db.close()

    # 레이아웃별 플레이한 알고리즘 집계
    played = {}  # {layout: [algo_name, ...]}
    for ep in episodes:
        played.setdefault(ep.layout, []).append(ep.algo_name)

    # 레이아웃 순서 (participant_id 기반 시드로 셔플)
    rng = random.Random(participant_id)
    layouts = list(_layout_order)
    rng.shuffle(layouts)

    progress = {}
    for layout in layouts:
        algos_needed = _algos_per_layout.get(layout, [])
        games_per_layout = len(algos_needed)
        played_algos = played.get(layout, [])
        progress[layout] = {
            "played": len(played_algos),
            "total": games_per_layout,
            "played_algos": played_algos,
            "all_algos": algos_needed,
        }

    total_played = sum(p["played"] for p in progress.values())
    total_needed = sum(p["total"] for p in progress.values())

    return {
        "layout_order": layouts,
        "progress": progress,
        "total_played": total_played,
        "total_needed": total_needed,
        "completed": total_played >= total_needed,
    }


def _get_next_game(participant_id: str) -> dict | None:
    """다음 게임의 layout + algo 결정. 모두 완료 시 None 반환."""
    info = _get_study_progress(participant_id)
    if info["completed"]:
        return None

    rng = random.Random()  # 매번 다른 시드

    # 레이아웃 순서대로 미완료된 첫 번째 레이아웃 선택
    for layout in info["layout_order"]:
        p = info["progress"][layout]
        if p["played"] >= p["total"]:
            continue

        # 아직 플레이하지 않은 알고리즘 중 랜덤 선택
        remaining_algos = [a for a in p["all_algos"] if a not in p["played_algos"]]
        if not remaining_algos:
            continue

        chosen_algo = rng.choice(remaining_algos)

        # 해당 algo의 모델 중 랜덤 시드 선택
        algo_models = [m for m in _models_by_layout[layout] if m["algo_name"] == chosen_algo]
        model_info = rng.choice(algo_models)

        return {
            "layout": layout,
            "model_info": model_info,
            "current_game": p["played"] + 1,
            "games_per_layout": p["total"],
            "total_played": info["total_played"],
            "total_needed": info["total_needed"],
        }

    return None


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

            # 다음 게임 자동 배정 (레이아웃 + 알고리즘)
            next_game = _get_next_game(participant_id)
            if next_game is None:
                await websocket.send_json({"type": "study_complete"})
                break

            layout = next_game["layout"]
            model_info = next_game["model_info"]
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

            # 실시간 틱 설정
            tick_interval = _config["game"].get("tick_interval_ms", 250) / 1000.0
            episode_length = _config["game"].get("episode_length", 400)

            # 초기 상태 전송 (진행 상황 포함)
            init_info = session.get_init_info()
            init_info["algo_name"] = model_info["algo_name"]
            init_info["seed_id"] = model_info["seed_id"]
            init_info["tick_interval_ms"] = int(tick_interval * 1000)
            init_info["layout_name"] = layout
            init_info["current_game"] = next_game["current_game"]
            init_info["games_per_layout"] = next_game["games_per_layout"]
            init_info["total_played"] = next_game["total_played"]
            init_info["total_needed"] = next_game["total_needed"]
            await websocket.send_json({"type": "game_start", **init_info})

            # 실시간 게임 루프: 서버가 틱을 주도하고, 인간 액션은 비동기 수신
            latest_action = None
            disconnected = False

            async def action_listener():
                """틱 사이에 WebSocket으로 도착하는 인간 액션을 버퍼링."""
                nonlocal latest_action, disconnected
                try:
                    while not session.done and not disconnected:
                        msg = await websocket.receive_json()
                        action_key = msg.get("action", msg.get("key", ""))
                        if isinstance(action_key, int):
                            latest_action = action_key
                        else:
                            latest_action = keyboard_to_action(str(action_key))
                except WebSocketDisconnect:
                    disconnected = True
                    session.force_end()

            listener_task = asyncio.create_task(action_listener())

            try:
                loop = asyncio.get_event_loop()
                next_tick = loop.time() + tick_interval

                while not session.done and not disconnected:
                    # 다음 틱까지 대기 (drift 방지)
                    now = loop.time()
                    sleep_time = max(0, next_tick - now)
                    await asyncio.sleep(sleep_time)
                    next_tick += tick_interval

                    if disconnected:
                        break

                    # 이번 틱의 액션 결정 (없으면 stay=4)
                    human_action = latest_action if latest_action is not None else 4
                    latest_action = None

                    result = session.step(human_action)

                    # 남은 시간 정보 추가
                    remaining_steps = episode_length - session.timestep
                    result["time_remaining_ms"] = int(remaining_steps * tick_interval * 1000)
                    result["tick_interval_ms"] = int(tick_interval * 1000)

                    try:
                        await websocket.send_json({"type": "state_update", **result})
                    except (WebSocketDisconnect, RuntimeError):
                        session.force_end()
                        break

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

                        try:
                            await websocket.send_json({
                                "type": "episode_end",
                                "episode_id": session.episode_id,
                                "final_score": session.score,
                                "algo_name": model_info["algo_name"],
                                "collisions": session.collisions,
                                "deliveries": session.deliveries,
                            })
                        except (WebSocketDisconnect, RuntimeError):
                            pass
            finally:
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass

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


@router.get("/api/study-progress/{participant_id}")
async def get_study_progress(participant_id: str):
    """참가자의 스터디 진행 상황 반환."""
    return _get_study_progress(participant_id)


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
