import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

JobStatus = Literal["running", "done", "error"]

# How long a finished job stays visible to a poller before being pruned.
JOB_MAX_AGE_SECONDS = 3600


@dataclass
class UploadJob:
    """
    Tracks progress of one bulk test-image upload. Lives only in memory --
    this is a single-process app with no task queue, so a plain module-level
    dict is enough (all mutation happens on the event loop thread, never
    inside the asyncio.to_thread() workers, so no locking is needed).
    """

    job_id: str
    total_files: int
    processed_files: int = 0
    photos_added: int = 0
    faces_detected: int = 0
    faces_matched: int = 0
    status: JobStatus = "running"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "photos_added": self.photos_added,
            "faces_detected": self.faces_detected,
            "faces_matched": self.faces_matched,
            "error": self.error,
        }


_jobs: dict[str, UploadJob] = {}


def create_job(total_files: int) -> UploadJob:
    _prune_old_jobs()
    job = UploadJob(job_id=uuid.uuid4().hex, total_files=total_files)
    _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> UploadJob | None:
    return _jobs.get(job_id)


def _prune_old_jobs() -> None:
    cutoff = time.time() - JOB_MAX_AGE_SECONDS
    stale_ids = [
        job_id
        for job_id, job in _jobs.items()
        if job.finished_at is not None and job.finished_at < cutoff
    ]
    for job_id in stale_ids:
        del _jobs[job_id]
