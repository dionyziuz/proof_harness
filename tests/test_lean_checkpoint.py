"""tests/test_lean_checkpoint.py — Unit test for LeanProver turn-level checkpointing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from core_engine.lean_prover import LeanProver, LeanKernelValidationResult
from core_engine.prover_kernel import SandboxedMathKernel


def test_lean_prover_checkpoint_save_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_file = tmp_path / "test_session.ckpt.json"

        mock_client = MagicMock()
        kernel = SandboxedMathKernel()
        prover = LeanProver(client=mock_client, kernel=kernel, checkpoint_path=ckpt_file)

        # 1. Test direct _save_checkpoint / _load_checkpoint / _unlink_checkpoint
        test_msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
        prover._save_checkpoint(
            ckpt_path=ckpt_file,
            task_id="test_task_123",
            stage="turn_1",
            start_wall_time=0.0,
            messages=test_msgs,
            result=MagicMock(to_dict=lambda: {"status": "RUNNING"}),
            retrieved=[],
        )
        assert ckpt_file.is_file(), "Checkpoint file was not created!"

        loaded = prover._load_checkpoint(ckpt_file, "test_task_123")
        assert loaded is not None, "Failed to load checkpoint!"
        assert loaded["task_id"] == "test_task_123", "Task ID mismatch!"
        assert len(loaded["messages"]) == 2, "Messages length mismatch!"

        prover._unlink_checkpoint(ckpt_file)
        assert not ckpt_file.exists(), "Checkpoint file was not unlinked!"

        # 2. Test turn checkpoint save & resume inside prove_lean()
        turn_counter = [0]

        def mock_stream(messages, **kwargs):
            turn_counter[0] += 1
            return f"Reasoning for turn {turn_counter[0]}", f"```lean\n-- Turn {turn_counter[0]} draft\n```"

        prover._stream_completion_with_retry = mock_stream
        prover.kernel.execute = MagicMock(return_value=MagicMock(stdout="", stderr="", error=None))

        # Mock validation so turn 1 fails and saves checkpoint
        val_res_turn1 = LeanKernelValidationResult(
            passed=False,
            verdict="REJECT",
            status="COMPILE_ERROR",
            lean_code="-- Turn 1",
            compiler_errors=["syntax error"],
            validity_score=0.25,
        )
        prover.validate_lean_candidate = MagicMock(return_value=val_res_turn1)

        # Run only 1 turn
        res1, val1 = prover.prove_lean(
            task_id="task_erdos_test",
            task_prompt="Prove dummy",
            max_iterations=1,
            checkpoint_path=ckpt_file,
        )
        assert ckpt_file.is_file(), "Turn 1 did not save checkpoint file!"
        ckpt_data = prover._load_checkpoint(ckpt_file, "task_erdos_test")
        assert ckpt_data["turn"] == 1, f"Expected turn 1 in ckpt, got {ckpt_data.get('turn')}"
        assert len(ckpt_data["reasoning_traces"]) == 1, "Expected 1 reasoning trace"

        # Now resume with max_iterations=2
        on_step_events = []

        def mock_on_step(event, data):
            on_step_events.append((event, data))

        # For turn 2, mock validation to pass
        val_res_turn2 = LeanKernelValidationResult(
            passed=True,
            verdict="VALID",
            status="KERNEL_PASSED",
            lean_code="-- Turn 2 Complete",
            validity_score=1.0,
        )
        prover.validate_lean_candidate = MagicMock(return_value=val_res_turn2)

        res2, val2 = prover.prove_lean(
            task_id="task_erdos_test",
            task_prompt="Prove dummy",
            max_iterations=2,
            checkpoint_path=ckpt_file,
            on_step=mock_on_step,
        )

        # Verify resume occurred at Turn 2
        resume_events = [e for e in on_step_events if e[0] == "checkpoint_resume"]
        assert len(resume_events) == 1, "Expected checkpoint_resume event!"
        assert resume_events[0][1]["turn"] == 2, f"Expected resume at Turn 2, got {resume_events[0][1]}"

        # Verify that passing unlinked the checkpoint
        assert not ckpt_file.exists(), "Passing proof should unlink checkpoint!"


def test_lean_prover_checkpoint_multi_turn_failure_and_resume():
    """Verify that multiple failing turns update the checkpoint monotonically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_file = tmp_path / "multi_turn.ckpt.json"

        mock_client = MagicMock()
        kernel = SandboxedMathKernel()
        prover = LeanProver(client=mock_client, kernel=kernel)

        turn_counter = [0]

        def mock_stream(messages, **kwargs):
            turn_counter[0] += 1
            return f"Reasoning {turn_counter[0]}", f"```lean\n-- Code {turn_counter[0]}\n```"

        prover._stream_completion_with_retry = mock_stream
        prover.kernel.execute = MagicMock(return_value=MagicMock(stdout="", stderr="", error=None))
        prover.validate_lean_candidate = MagicMock(
            return_value=LeanKernelValidationResult(
                passed=False,
                verdict="REJECT",
                status="COMPILE_ERROR",
                lean_code="-- Draft",
                compiler_errors=["err"],
                validity_score=0.1,
            )
        )

        # Run 2 turns
        prover.prove_lean(
            task_id="multi_turn_task",
            task_prompt="Prove something",
            max_iterations=2,
            checkpoint_path=ckpt_file,
        )

        assert ckpt_file.is_file()
        data = json.loads(ckpt_file.read_text(encoding="utf-8"))
        assert data["turn"] == 2
        assert len(data["messages"]) > 2

        # Resume from turn 3 up to turn 4
        on_step_events = []

        def mock_on_step(event, data):
            on_step_events.append((event, data))

        prover.prove_lean(
            task_id="multi_turn_task",
            task_prompt="Prove something",
            max_iterations=4,
            checkpoint_path=ckpt_file,
            on_step=mock_on_step,
        )

        resume_events = [e for e in on_step_events if e[0] == "checkpoint_resume"]
        assert len(resume_events) == 1
        assert resume_events[0][1]["turn"] == 3

        data2 = json.loads(ckpt_file.read_text(encoding="utf-8"))
        assert data2["turn"] == 4


if __name__ == "__main__":
    test_lean_prover_checkpoint_save_and_resume()
    print("ALL TESTS PASSED!")
