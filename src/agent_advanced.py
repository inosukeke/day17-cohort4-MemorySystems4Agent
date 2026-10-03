from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
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

# Facts whose new mentions should be merged with what is already on file
# instead of overwriting it (e.g. "Python" learned in one turn, "AI" in
# another should both survive in `interests`).
_MERGEABLE_FACTS = {"interests"}


def _looks_like_question(message: str) -> bool:
    if "?" in message:
        return True
    lower = message.lower()
    return any(marker in lower for marker in _QUESTION_MARKERS)


def _merge_list_fact(existing: str, new_value: str) -> str:
    existing_items = [item.strip() for item in existing.split(",") if item.strip()]
    new_items = [item.strip() for item in new_value.split(",") if item.strip()]
    for item in new_items:
        if item not in existing_items:
            existing_items.append(item)
    return ", ".join(existing_items)


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
        return "Mình chưa có đủ thông tin để nhớ lại điều đó."
    return "; ".join(parts) + "."


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_users: dict[str, str] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self.thread_users[thread_id] = user_id
        if self.langchain_agent is not None:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _apply_profile_updates(self, user_id: str, message: str) -> None:
        updates = extract_profile_updates(message)
        if not updates:
            return
        current = self.profile_store.facts(user_id)
        for key, value in updates.items():
            if key in _MERGEABLE_FACTS and key in current:
                value = _merge_list_fact(current[key], value)
            self.profile_store.upsert_fact(user_id, key, value)
            current[key] = value

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1-2. Extract stable profile facts and persist them into User.md.
        self._apply_profile_updates(user_id, message)

        # 3. Append the message into compact memory (short-term layer).
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load from User.md + summary + recent messages.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        # 5. Generate a response that can answer long-term recall questions.
        answer = self._offline_response(user_id, thread_id, message)

        # 6. Append the assistant reply and update token counters.
        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens

        return {
            "answer": answer,
            "tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary = str(context.get("summary", ""))
        messages: list[dict[str, str]] = context.get("messages", [])  # type: ignore[assignment]

        total = estimate_tokens(profile_text) + estimate_tokens(summary)
        total += sum(estimate_tokens(m["content"]) for m in messages)
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)

        if _looks_like_question(message) and facts:
            return _format_facts_answer(facts)
        if facts:
            return "Mình đã ghi nhận thông tin này và sẽ nhớ lâu dài trong User.md."
        return "Cảm ơn bạn, mình sẽ bắt đầu ghi nhớ các thông tin quan trọng."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Keep the persistent/compact memory bookkeeping identical to the
        # offline path so benchmark + tests behave the same regardless of
        # whether a live model answers the message.
        self._apply_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        profile_text = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary = str(context.get("summary", ""))
        messages: list[dict[str, str]] = context.get("messages", [])  # type: ignore[assignment]

        system_prompt = (
            "Bạn là trợ lý có trí nhớ dài hạn cho người dùng.\n\n"
            f"Hồ sơ người dùng (User.md):\n{profile_text}\n\n"
            f"Tóm tắt hội thoại cũ đã được nén:\n{summary or '(chưa có)'}"
        )
        history: list[Any] = [SystemMessage(content=system_prompt)]
        for item in messages:
            if item["role"] == "user":
                history.append(HumanMessage(content=item["content"]))
            else:
                history.append(AIMessage(content=item["content"]))

        result = self.langchain_agent.invoke(history)
        answer = getattr(result, "content", str(result))

        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens

        return {
            "answer": answer,
            "tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Build a real chat model when possible; stay `None` (offline) otherwise."""

        if self.force_offline:
            return None
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
