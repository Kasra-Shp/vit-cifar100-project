"""Synthetic test for task-boundary atomic save and delayed DONE semantics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from imagenet100_continuation_checkpoint import (
    load_torch_payload,
    save_torch_payload_atomic,
    write_done_marker_atomic,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "method__rankext_state.pt"
        marker = root / "method__DONE.marker"
        payload = {
            "method_name": "rank_extension_fullkd_T2_protect30",
            "completed_task_index": 3,
            "previous_rank_state": {"classifier_weight": torch.zeros(100, 100)},
        }
        save_torch_payload_atomic(checkpoint, payload)
        assert checkpoint.is_file()
        assert not marker.exists(), "task checkpoint must not create DONE early"
        reloaded = load_torch_payload(checkpoint)
        assert reloaded["completed_task_index"] == 3
        write_done_marker_atomic(marker)
        assert marker.read_text(encoding="utf-8") == "done\n"
        print("PASS: synthetic continuation checkpoint lifecycle")


if __name__ == "__main__":
    main()
