"""Competition entrypoint backed by the migrated MathSolve-Agent design."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from math_prove.agent import MathSolverAgent


class ReasoningAgent:
    """Adapter from the Challenge Cup interface to MathSolve-Agent."""

    def __init__(self, client: Any, *args: Any, **kwargs: Any) -> None:
        del args
        self.client = client
        self.ablation = str(
            kwargs.pop("ablation", os.environ.get("MATH_PROVE_ABLATION", "official_stable"))
        )
        self.model_type = str(
            kwargs.pop("model_type", os.environ.get("INTERN_MODEL", "intern-s2-preview"))
        )
        self.temperature = float(kwargs.pop("temperature", 0.0))
        self.max_tokens = int(kwargs.pop("max_tokens", 4096))
        self.solver = MathSolverAgent(
            client=client,
            model_type=self.model_type,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            ablation=self.ablation,
        )

    def solve(self, problem: str, metadata: Dict) -> Dict:
        problem_id = self._problem_id(metadata)
        safe_metadata = self._safe_metadata(metadata)

        solution = self.solver.solve(
            problem=problem,
            problem_id=problem_id,
            raw_metadata=safe_metadata,
        )
        final_response = str(solution.answer or "").strip()
        trace = self._build_trace(solution, self.solver.last_run_log)

        if not final_response or final_response == "unable_to_determine":
            fallback_answer, fallback_trace = self._direct_fallback(problem)
            if fallback_answer:
                final_response = fallback_answer
                trace.extend(fallback_trace)

        return {
            "final_response": final_response or "unable_to_determine",
            "trace": trace,
        }

    @staticmethod
    def _problem_id(metadata: Dict) -> str:
        metadata = metadata or {}
        for key in ("idx", "problem_id", "id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return "0"

    @staticmethod
    def _safe_metadata(metadata: Dict) -> Dict:
        blocked = {
            "answer",
            "expected",
            "expected_answer",
            "reference",
            "reference_answer",
            "gold",
            "label",
        }
        return {
            str(key): value
            for key, value in (metadata or {}).items()
            if str(key).lower() not in blocked
        }

    @staticmethod
    def _build_trace(solution: Any, run_log: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = [
            {
                "step": "diagnosis",
                "content": {
                    "domain": solution.domain,
                    "answer_type": solution.answer_type,
                },
            },
            {
                "step": "reasoning_summary",
                "content": solution.reasoning_summary,
            },
            {
                "step": "key_steps",
                "content": solution.key_steps,
            },
            {
                "step": "verification",
                "content": {
                    "passed": solution.verification.passed,
                    "confidence": solution.verification.confidence,
                    "issues": solution.verification.issues,
                },
            },
        ]
        for stage in (run_log or {}).get("stages", [])[:8]:
            trace.append(
                {
                    "step": str(stage.get("stage", "model_stage")),
                    "content": {
                        "response_preview": stage.get("cleaned_response_preview", ""),
                        "error": stage.get("error", ""),
                    },
                }
            )
        return trace

    def _direct_fallback(self, problem: str) -> tuple[str, List[Dict[str, Any]]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful mathematics solver. Solve the problem and put "
                    "the shortest judgeable result on the last line as "
                    "Final Answer: <answer>."
                ),
            },
            {"role": "user", "content": f"Problem:\n{problem}"},
        ]
        try:
            raw = self.client.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=min(self.max_tokens, 4096),
            )
        except TypeError:
            raw = self.client.chat(
                messages,
                temperature=0.0,
                max_tokens=min(self.max_tokens, 4096),
            )
        except Exception as exc:  # noqa: BLE001 - single item fallback should not crash.
            return "", [{"step": "direct_fallback_error", "content": str(exc)}]

        text = str(raw or "").strip()
        answer = self._extract_final_answer(text)
        return answer, [{"step": "direct_fallback", "content": text[:1200]}]

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        matches = re.findall(r"Final Answer\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
