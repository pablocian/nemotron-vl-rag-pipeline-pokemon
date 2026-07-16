from pathlib import Path
import logging

from huggingface_hub import (
    create_repo,
    upload_folder,
    SpaceHardware,
)

REPO_ID = "cian-ps/nemotron-vl-rag-pipeline-pokemon"
FOLDER = Path(__file__).resolve().parents[1]

IGNORE = [
    ".git*",
    ".github/**",
    ".venv/**",
    "__pycache__/**",
    "notebooks/**",
    "uv.lock",
    "pyproject.toml",
    ".python-version",
    "scripts/**",
]

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def upload():
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        private=False, # set to True if you don't want your Space to be accessible to others
        space_sdk="gradio",
        space_hardware=SpaceHardware.ZERO_A10G,
        exist_ok=True, # set to False if you want an error to raise if the repo_id already exists 
    )

    logger.info(f"Uploading files from {FOLDER} to repo: {REPO_ID}")

    folder_upload_url = upload_folder(
        repo_id=REPO_ID,
        folder_path=FOLDER,
        path_in_repo=".", # location in the remote repo
        repo_type="space",
        ignore_patterns=IGNORE,
        delete_patterns=["*"],  # mirror deletions from GitHub
        commit_message="Upload app files",
    )

    logger.info(f"App files successfully uploaded to: {folder_upload_url}")


if __name__ == "__main__":
    upload()
