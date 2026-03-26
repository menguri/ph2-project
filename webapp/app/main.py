import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # GPU 초기화 자체를 건너뜀

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import load_config
from .api.routes import router, init_routes

# baseline 모델 코드 접근을 위해 sys.path에 추가
# webapp 프로세스에서만 적용 — 기존 훈련 코드에 영향 없음
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = PROJECT_ROOT / "baseline"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))

app = FastAPI(title="PH2 Human-AI Study")

# Static files (frontend)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    config = load_config()
    # 상대경로를 webapp/ 기준 절대경로로 변환
    webapp_dir = Path(__file__).resolve().parent.parent
    for key in ["baseline_runs_dir", "ph2_runs_dir", "model_dir"]:
        val = config["agent"].get(key, "")
        if val and not Path(val).is_absolute():
            config["agent"][key] = str((webapp_dir / val).resolve())
    db_path = config["database"]["path"]
    if not Path(db_path).is_absolute():
        config["database"]["path"] = str((webapp_dir / db_path).resolve())
    traj_dir = config["trajectory"]["save_dir"]
    if not Path(traj_dir).is_absolute():
        config["trajectory"]["save_dir"] = str((webapp_dir / traj_dir).resolve())

    init_routes(config)
    print(f"[startup] PH2 webapp ready on {config['server']['host']}:{config['server']['port']}")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}
