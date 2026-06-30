import os
import uuid
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Video Moderation API",
    description="Upload a video, get a compliance report with violation timestamps.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

jobs = {}
_moderator = None
_lock = threading.Lock()


def get_moderator():
    global _moderator
    with _lock:
        if _moderator is None:
            from moderator import VideoModerator
            _moderator = VideoModerator(model_path="models/best_model.pt")
    return _moderator


def process_video(job_id: str, video_path: str):
    try:
        jobs[job_id]["status"] = "sampling_frames"
        moderator = get_moderator()

        jobs[job_id]["status"] = "classifying"
        report = moderator.moderate(video_path)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["report"] = report

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.get("/health")
def health():
    return {"status": "ok", "service": "AI Video Moderation API"}


@app.post("/moderate")
async def moderate_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    allowed = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())
    video_path = str(UPLOAD_DIR / f"{job_id}{ext}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "report": None, "error": None}
    background_tasks.add_task(process_video, job_id, video_path)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **jobs[job_id]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
