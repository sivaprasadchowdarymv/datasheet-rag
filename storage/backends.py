"""
storage/backends.py — replaceable persistence.

    StorageBackend
    ├── LocalStorage     data/ on the local disk (your PC, a VM with a real disk)
    └── HFHubStorage     data/ on local disk  +  mirrored to a PRIVATE Hugging Face
                         dataset repo, so a free cloud host that wipes its disk on
                         restart/sleep (Streamlit Community Cloud) gets the
                         documents, chunks and FAISS indexes back on boot without
                         re-processing anything.

The app always works on the local copy; the backend only syncs. Sync failures
never crash the app — they surface as warnings.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List


class StorageBackend(ABC):
    name = "base"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.last_error = ""

    @abstractmethod
    def pull_all(self) -> str: ...

    @abstractmethod
    def push(self, rel_paths: Iterable[str]) -> str: ...

    @abstractmethod
    def remove(self, rel_paths: Iterable[str]) -> str: ...

    def describe(self) -> str:
        return self.name


class LocalStorage(StorageBackend):
    name = "local"

    def pull_all(self) -> str:
        return ""

    def push(self, rel_paths: Iterable[str]) -> str:
        return ""

    def remove(self, rel_paths: Iterable[str]) -> str:
        return ""

    def describe(self) -> str:
        return f"local disk ({self.data_dir})"


class HFHubStorage(StorageBackend):
    name = "hf"

    def __init__(self, data_dir: Path, repo_id: str, token: str):
        super().__init__(data_dir)
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        self._lock = threading.Lock()
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"Could not create/access dataset repo {repo_id}: {e}"

    def describe(self) -> str:
        return f"local disk mirrored to private HF dataset '{self.repo_id}'"

    def pull_all(self) -> str:
        from huggingface_hub import snapshot_download

        try:
            with self._lock:
                snapshot_download(repo_id=self.repo_id, repo_type="dataset", local_dir=str(self.data_dir),
                                  token=self.token)
            return ""
        except Exception as e:  # noqa: BLE001
            self.last_error = f"Restoring data from Hugging Face failed: {e}"
            return self.last_error

    def push(self, rel_paths: Iterable[str]) -> str:
        errors: List[str] = []
        with self._lock:
            for rel in rel_paths:
                p = self.data_dir / rel
                try:
                    if p.is_dir():
                        self.api.upload_folder(folder_path=str(p), path_in_repo=rel, repo_id=self.repo_id,
                                               repo_type="dataset", commit_message=f"sync {rel}")
                    elif p.exists():
                        self.api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=self.repo_id,
                                             repo_type="dataset", commit_message=f"sync {rel}")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{rel}: {e}")
        self.last_error = "; ".join(errors)
        return ("Cloud backup failed for " + self.last_error) if errors else ""

    def remove(self, rel_paths: Iterable[str]) -> str:
        errors: List[str] = []
        with self._lock:
            for rel in rel_paths:
                try:
                    if (self.data_dir / rel).suffix:
                        self.api.delete_file(path_in_repo=rel, repo_id=self.repo_id, repo_type="dataset")
                    else:
                        self.api.delete_folder(path_in_repo=rel, repo_id=self.repo_id, repo_type="dataset")
                except Exception as e:  # noqa: BLE001  (missing paths are fine)
                    if "404" not in str(e) and "not found" not in str(e).lower():
                        errors.append(f"{rel}: {e}")
        self.last_error = "; ".join(errors)
        return ("Cloud delete failed for " + self.last_error) if errors else ""


def create_storage(settings) -> StorageBackend:
    data_dir = Path(settings.data_dir)
    if settings.storage_backend == "hf" and settings.hf_token and settings.hf_dataset_repo:
        try:
            return HFHubStorage(data_dir, settings.hf_dataset_repo, settings.hf_token)
        except Exception:  # huggingface_hub missing or bad token → local
            pass
    return LocalStorage(data_dir)
