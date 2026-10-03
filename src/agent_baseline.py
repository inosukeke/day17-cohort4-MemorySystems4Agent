from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model

_QUESTION_MARKERS = (
    "nhắc lại",
    "là gì",
    "là ai",
    "ở đâu",
    "làm nghề gì",
    "không?",
    "nhớ không",
    "bạn biết",
    "nêu lại",
    "tóm tắt",
)


def _looks_like_question(message: str) -> bool:
    if "?" in message:
        return True
    lower = message.lower()
    return any(marker in lower for marker in _QUESTION_MARKERS)


def _format_facts_answer(facts: dict[str, str]) -> str:
    parts: list[str] = []
    if "name" in facts:
        parts.append(f"Tên của bạn là {facts['name']}")
    if "profession" in facts:
        parts.append(f"hiện đang làm {facts['profession']}")
    if "location" in facts:
        parts.append(f"đang ở {facts['location']}")
    if "favorite_drink" in facts:
        parts.append(f"đồ uống yêu thích là {facts['favorite_drink']}")
    if "favorite_food" in facts:
        parts.append(f"món ăn yêu thích là {facts['favorite_food']}")
    if "pet" in facts:
        parts.append(f"bạn đang nuôi {facts['pet']}")
    if "interests" in facts:
        parts.append(f"quan tâm chính tới {facts['interests']}")
    if "style" in facts:
        parts.append(f"bạn thích phong cách trả lời {facts['style']}")
    if not parts:
        return "Mình chưa có đủ thông tin để nhớ lại điều đó trong phiên này."
    return "; ".join(parts) + "."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    facts: dict[str, str] = field(default_factory=dict)


class BaselineAgent:
    """Agent A: within-session memory only, no persistent `User.md`.

    Each `thread_id` gets a fresh, isolated `SessionState`. Facts learned in
    one thread never leak into another thread, which is the point of the
    baseline: it is a fair, naive comparison point for the advanced agent.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                # Fall back to the deterministic offline path if the live call fails.
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        updates = extract_profile_updates(message)
        session.facts.update(updates)

        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        if _looks_like_question(message) and session.facts:
            answer = _format_facts_answer(session.facts)
        elif session.facts:
            answer = "Mình đã ghi nhận thông tin này trong phiên hiện tại."
        else:
            answer = "Cảm ơn bạn đã chia sẻ."

        session.messages.append({"role": "assistant", "content": answer})
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens

        return {
            "answer": answer,
            "tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        history: list[Any] = [
            SystemMessage(
                content=(
                    "Bạn là trợ lý chỉ nhớ trong phạm vi cuộc trò chuyện hiện tại. "
                    "Đừng giả vờ nhớ thông tin từ các phiên trước."
                )
            )
        ]
        for item in session.messages:
            if item["role"] == "user":
                history.append(HumanMessage(content=item["content"]))
            else:
                history.append(AIMessage(content=item["content"]))

        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        result = self.langchain_agent.invoke(history)
        answer = getattr(result, "content", str(result))

        session.messages.append({"role": "assistant", "content": answer})
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens

        return {
            "answer": answer,
            "tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Build a real chat model when possible; stay `None` (offline) otherwise."""

        if self.force_offline:
            return None
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
