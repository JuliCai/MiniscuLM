from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def iter_prompt_files(root: Path) -> Iterable[Path]:
    """Yield all prompt files under root, skipping Raw and Parquets folders."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"Raw", "Parquets"}]
        for filename in filenames:
            path = Path(dirpath, filename)
            if path.suffix.lower() != ".txt":
                continue
            yield path


def clean_text(text: str) -> str:
    """Remove real and escaped newlines to keep single-line outputs."""
    return text.replace("\\n", "").replace("\n", "").replace("\r", "").strip()


def load_question_answer(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        question = clean_text(data["question"])
        answer = clean_text(data["answer"])
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Missing keys in {path}") from exc

    return question, answer


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sft_root = repo_root / "Training_Data" / "SFT"
    input_path = repo_root / "ScratchLists" / "input.txt"
    output_path = repo_root / "ScratchLists" / "output.txt"

    prompt_files = sorted(iter_prompt_files(sft_root))

    inputs: list[str] = []
    outputs: list[str] = []

    for path in prompt_files:
        question, answer = load_question_answer(path)
        inputs.append(question)
        outputs.append(answer)

    input_path.write_text("\n".join(inputs) + "\n", encoding="utf-8")
    output_path.write_text("\n".join(outputs) + "\n", encoding="utf-8")

    print(f"Wrote {len(inputs)} items to {input_path} and {output_path}")


if __name__ == "__main__":
    main()
