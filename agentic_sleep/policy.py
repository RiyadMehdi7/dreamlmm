from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .models import EpisodeStep, TurnTask


@dataclass
class PolicyDecision:
    action: str
    action_probs: Dict[str, float]
    rationale_summary: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_text: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PolicyContext:
    mode: str
    task: "TurnTask"
    recent_steps: List["EpisodeStep"]
    top_rule_descriptions: List[str]
    community_summaries: Dict[str, str]
    effective_context_tokens: int


class BasePolicy:
    kind: str = "base"

    def select_action(self, context: PolicyContext) -> PolicyDecision:
        raise NotImplementedError


@dataclass
class PolicySpec:
    kind: str = "sim"  # sim | openai
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_output_tokens: int = 300
    base_url: str = "https://api.openai.com/v1/responses"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 60.0


class OpenAIResponsesPolicy(BasePolicy):
    kind = "openai"

    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self.api_key = os.getenv(spec.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {spec.api_key_env}. "
                "Set it before running with --policy openai."
            )

    def select_action(self, context: PolicyContext) -> PolicyDecision:
        prompt = self._build_prompt(context)
        payload = {
            "model": self.spec.model,
            "input": prompt,
            "temperature": self.spec.temperature,
            "max_output_tokens": self.spec.max_output_tokens,
        }
        raw = self._post_with_compat_fallback(payload)

        parsed = json.loads(raw)
        output_text = _extract_output_text(parsed)
        decision_obj = _extract_json_object(output_text)
        candidates = context.task.candidates
        action = str(decision_obj.get("action", "")).strip()
        if action not in candidates:
            action = candidates[0]
        probs = _normalize_action_probs(decision_obj.get("action_probs"), candidates, chosen=action)
        rationale = str(decision_obj.get("rationale_summary", "")).strip() or "LLM-selected action."
        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        prompt_tokens = int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0)
        return PolicyDecision(
            action=action,
            action_probs=probs,
            rationale_summary=rationale,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_text=output_text,
            metadata={"model": self.spec.model},
        )

    def _build_prompt(self, context: PolicyContext) -> str:
        task = context.task
        recent = []
        for step in context.recent_steps[-8:]:
            recent.append(
                {
                    "turn": step.turn,
                    "action": step.action,
                    "success": step.success,
                    "entropy": round(step.entropy, 3),
                    "tags": sorted(step.tags),
                }
            )
        rules = context.top_rule_descriptions[:5]
        communities = context.community_summaries
        schema = {
            "action": "one of the provided candidates",
            "action_probs": {"candidate_name": "float probability, probabilities sum to 1.0"},
            "rationale_summary": "one short sentence"
        }
        return (
            "You are the waking policy for an autonomous agent.\n"
            "Choose exactly one action from the candidate list.\n"
            "Return ONLY valid JSON (no markdown).\n"
            "You do not have access to hidden labels; infer from tags and memory.\n\n"
            f"Mode: {context.mode}\n"
            f"Domain: {task.domain}\n"
            f"Turn: {task.turn}\n"
            f"Tags: {sorted(task.tags)}\n"
            f"Candidates: {task.candidates}\n"
            f"EffectiveContextTokensApprox: {context.effective_context_tokens}\n"
            f"RecentSteps: {json.dumps(recent, ensure_ascii=True)}\n"
            f"TopHeuristics: {json.dumps(rules, ensure_ascii=True)}\n"
            f"CommunitySummaries: {json.dumps(communities, ensure_ascii=True)}\n"
            f"OutputSchema: {json.dumps(schema, ensure_ascii=True)}\n"
        )

    def _post_with_compat_fallback(self, payload: dict) -> str:
        variants = [payload]
        if "max_output_tokens" in payload and "max_completion_tokens" not in payload:
            alt = dict(payload)
            alt["max_completion_tokens"] = alt.pop("max_output_tokens")
            variants.append(alt)

        last_error: Exception | None = None
        for body in variants:
            try:
                return self._post_json(body)
            except RuntimeError as exc:
                last_error = exc
                # Retry only if it looks like a 400-style schema issue.
                if "HTTP 400" not in str(exc):
                    break
                continue
        raise last_error or RuntimeError("Unknown OpenAI API error")

    def _post_json(self, payload: dict) -> str:
        req = urllib.request.Request(
            self.spec.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.spec.timeout_seconds) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API connection error: {exc}") from exc


def build_policy(spec: PolicySpec | None) -> Optional[BasePolicy]:
    if spec is None or spec.kind == "sim":
        return None
    if spec.kind == "openai":
        return OpenAIResponsesPolicy(spec)
    raise ValueError(f"Unsupported policy kind: {spec.kind}")


def _extract_output_text(resp_json: object) -> str:
    if isinstance(resp_json, dict):
        if isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]
        out = resp_json.get("output")
        if isinstance(out, list):
            texts: List[str] = []
            for item in out:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    txt = c.get("text")
                    if isinstance(txt, str):
                        texts.append(txt)
            if texts:
                return "\n".join(texts)
    return json.dumps(resp_json)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    # Fast path: exact JSON object.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_action_probs(
    raw_probs: object,
    candidates: List[str],
    chosen: str,
) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            if k in candidates:
                try:
                    probs[k] = max(0.0, float(v))
                except (TypeError, ValueError):
                    continue

    # Fill missing actions with epsilon.
    for c in candidates:
        probs.setdefault(c, 0.0)
    total = sum(probs.values())
    if total <= 0:
        # one-hot fallback if model didn't provide usable probabilities
        return {c: (1.0 if c == chosen else 0.0) for c in candidates}
    return {c: probs[c] / total for c in candidates}
