"""Interactive Step-by-Step Session Solver for Prime Math.

Allows running one CoALA reasoning-and-execution turn at a time.
Each step executes synchronously in 3-6 seconds, enabling live milestone
visibility inside AGY without triggering async background task thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .compactor import MathContextCompactor
from .kernel import ExecutionResult, PersistentMathKernel
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager
from .solver import MATH_SYSTEM_PROMPT, MathSolver, SolveResult, DEFAULT_LOG_PATH
from .token_shield import extract_high_level_steps

SESSION_STATE_FILE = Path("/tmp/prime_math_active_session.json")


@dataclass
class StepReport:
    turn: int
    reasoning_steps: List[str]
    code_blocks: List[str]
    exec_results: List[ExecutionResult]
    boxed_answer: Optional[str]
    finished: bool
    status_message: str


class SolverSession:
    """Stateful step-by-step solver maintaining context between individual turn calls."""

    def __init__(
        self,
        problem: str,
        client: Optional[TunnelThinkingClient] = None,
        kernel: Optional[PersistentMathKernel] = None,
        memory: Optional[CoALAMemoryManager] = None,
        compactor: Optional[MathContextCompactor] = None,
        max_iterations: int = 6,
    ):
        self.problem = problem
        self.client = client or TunnelThinkingClient()
        self.kernel = kernel or PersistentMathKernel()
        self.memory = memory or CoALAMemoryManager()
        self.compactor = compactor or MathContextCompactor(
            client=self.client,
            max_context_tokens=self.client.config.max_context_window,
            reserve_tokens=self.client.config.max_tokens,
        )
        self.max_iterations = max_iterations
        self.turn = 0
        self.finished = False

        # 1. Health check
        self.memory.check_health()

        # 2. Retrieve procedural heuristics
        self.retrieved = self.memory.query_procedural_heuristics(problem, limit=5)

        system_content = MATH_SYSTEM_PROMPT
        if self.retrieved:
            heuristics_block = "\n\nRELEVANT PROCEDURAL RULES & PROVEN HEURISTICS (FROM COALA MEMORY):\n"
            for h in self.retrieved:
                heuristics_block += f"- {h}\n"
            system_content += heuristics_block

        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Solve the following mathematical problem:\n\n{problem}"},
        ]

        self.result = SolveResult(
            problem=problem,
            answer=None,
            boxed_answer=None,
            verified=False,
            iterations=0,
            retrieved_heuristics=self.retrieved,
            log_file=str(DEFAULT_LOG_PATH),
        )

    def step(self) -> StepReport:
        """Executes a single reasoning and calculation turn."""
        if self.finished:
            return StepReport(
                turn=self.turn,
                reasoning_steps=[],
                code_blocks=[],
                exec_results=[],
                boxed_answer=self.result.boxed_answer,
                finished=True,
                status_message="Session already finished.",
            )

        self.turn += 1
        self.result.iterations = self.turn

        solver_helper = MathSolver(
            client=self.client,
            kernel=self.kernel,
            memory=self.memory,
            compactor=self.compactor,
        )

        if self.compactor.should_compact(self.messages):
            comp_res = self.compactor.compact(self.messages)
            if comp_res.compacted:
                self.messages = comp_res.compacted_messages
                solver_helper._log_write(
                    f"\n[Context Compaction] Context window compacted: "
                    f"{comp_res.original_tokens} -> {comp_res.new_tokens} tokens\n"
                )

        # Stream thinking model response to live log
        solver_helper._log_write(f"\n--- [TURN {self.turn}] INNER THINKING (STREAMING FROM QWEN ENGINE) ---\n")
        reasoning_chunks: List[str] = []
        content_chunks: List[str] = []
        switched = False

        for typ, delta in self.client.stream_complete(self.messages, max_tokens=self.client.config.max_tokens):
            if typ == "reasoning":
                reasoning_chunks.append(delta)
                solver_helper._log_write(delta)
            elif typ == "content":
                if not switched:
                    solver_helper._log_write(f"\n\n--- [TURN {self.turn}] MODEL GENERATED PROPOSAL & CODE ---\n")
                    switched = True
                content_chunks.append(delta)
                solver_helper._log_write(delta)

        reasoning_text = "".join(reasoning_chunks).strip()
        model_text = "".join(content_chunks).strip()

        if reasoning_text:
            self.result.reasoning_traces.append(reasoning_text)
        self.result.final_response = model_text

        high_level_steps = extract_high_level_steps(reasoning_text or model_text, max_points=3)
        code_blocks = solver_helper.extract_code_blocks(model_text)

        turn_exec_results: List[ExecutionResult] = []

        if code_blocks:
            self.messages.append({"role": "assistant", "content": model_text})
            feedbacks = []

            for code in code_blocks:
                self.result.code_history.append(code)
                exec_res = self.kernel.execute(code)
                self.result.execution_history.append(exec_res)
                turn_exec_results.append(exec_res)

                if exec_res.success:
                    feedback = f"[Execution Output]:\n{exec_res.stdout}"
                    if exec_res.result_repr:
                        feedback += f"\n[Result Value]: {exec_res.result_repr}"
                    if not exec_res.stdout and not exec_res.result_repr:
                        feedback += "[Executed successfully with no output]"
                else:
                    feedback = (
                        f"[Execution Error]:\n{exec_res.error}\n"
                        "Please analyze why this error occurred, adjust your code or approach, and continue."
                    )
                feedbacks.append(feedback)

            self.messages.append({"role": "user", "content": "\n\n".join(feedbacks)})

        boxed = solver_helper.extract_boxed_answer(model_text)
        if boxed:
            self.result.boxed_answer = boxed
            self.result.answer = boxed
            self.result.verified = len(self.result.code_history) > 0
            self.finished = True
        elif self.turn >= self.max_iterations:
            self.finished = True
        elif not code_blocks:
            if model_text.strip():
                self.messages.append({"role": "assistant", "content": model_text})
            self.messages.append({
                "role": "user",
                "content": "Please write Python code to compute or verify your steps, or provide the final answer in \\boxed{}.",
            })

        if self.finished:
            # Consolidate learning
            solver_helper._consolidate_learning(self.problem, self.result)

        return StepReport(
            turn=self.turn,
            reasoning_steps=high_level_steps,
            code_blocks=code_blocks,
            exec_results=turn_exec_results,
            boxed_answer=self.result.boxed_answer,
            finished=self.finished,
            status_message="Turn completed.",
        )

    def save(self, path: Path = SESSION_STATE_FILE) -> None:
        """Saves session state to disk as JSON."""
        state = {
            "problem": self.problem,
            "max_iterations": self.max_iterations,
            "turn": self.turn,
            "finished": self.finished,
            "retrieved": self.retrieved,
            "messages": self.messages,
            "code_history": self.result.code_history,
            "boxed_answer": self.result.boxed_answer,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: Path = SESSION_STATE_FILE) -> "SolverSession":
        """Loads active session from disk and replays execution history."""
        if not path.exists():
            raise FileNotFoundError(f"No active session found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        session = cls(
            problem=state["problem"],
            max_iterations=state.get("max_iterations", 6),
        )
        session.turn = state["turn"]
        session.finished = state["finished"]
        session.retrieved = state.get("retrieved", [])
        session.messages = state["messages"]
        session.result.boxed_answer = state.get("boxed_answer")
        session.result.answer = state.get("boxed_answer")
        session.result.code_history = state.get("code_history", [])

        # Replay code history in kernel so state is restored
        for code in session.result.code_history:
            session.kernel.execute(code)

        return session
