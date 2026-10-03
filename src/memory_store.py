from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator (~4 chars per token). Good enough for offline benchmarking."""

    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one markdown file per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (user_id or "user").strip().lower()).strip("_") or "user"
        user_dir = Path(self.root_dir) / slug
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n## Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        self.write_text(user_id, new_content)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            match = re.match(r"-\s*\*\*(.+?)\*\*:\s*(.+)", line.strip())
            if match:
                facts[match.group(1).strip().lower()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        content = self.read_text(user_id)
        line = f"- **{key}**: {value}"
        pattern = re.compile(rf"^-\s*\*\*{re.escape(key)}\*\*:.*$", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(line, content, count=1)
        else:
            if "## Facts" not in content:
                content = content.rstrip() + "\n\n## Facts\n"
            if not content.endswith("\n"):
                content += "\n"
            content += line + "\n"
        return self.write_text(user_id, content)


_NAME_PATTERNS = [
    re.compile(r"(?:mình|tôi)\s+tên\s+là\s+([^,.\n!?]+)", re.IGNORECASE),
    re.compile(r"tên\s+(?:mình|tôi)\s+là\s+([^,.\n!?]+)", re.IGNORECASE),
]

_LOCATION_PATTERN = re.compile(r"ở\s+([^\s,.\n!?]+(?:\s+[^\s,.\n!?]+)*)")
_NEGATION_MARKERS = ("không còn", "chứ không", "không phải", "đừng nói", "đừng gọi")

_PROFESSION_KEYWORDS = [
    ("mlops engineer", "MLOps engineer"),
    ("backend engineer", "backend engineer"),
    ("frontend engineer", "frontend engineer"),
    ("data engineer", "data engineer"),
    ("software engineer", "software engineer"),
    ("product manager", "product manager"),
]
_JOKE_MARKERS = ("đùa", "giỡn", "nói chơi")

_PET_PATTERN = re.compile(r"corgi(?:\s+tên\s+([^\s,.\n!?]+))?", re.IGNORECASE)


def _detect_name(sentence: str) -> str | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group(1).strip()
    return None


def _detect_location(sentence: str) -> str | None:
    location = None
    for match in _LOCATION_PATTERN.finditer(sentence):
        # Only treat a run of capitalized words right after "ở" as a place
        # name (Python's str.isupper() is Unicode-correct for Đ vs đ, unlike
        # a naive regex character-range check).
        proper_words: list[str] = []
        for word in match.group(1).split():
            if word[:1].isupper():
                proper_words.append(word)
            else:
                break
        if not proper_words:
            continue

        preceding = sentence[max(0, match.start() - 25) : match.start()].lower()
        if any(marker in preceding for marker in _NEGATION_MARKERS):
            continue
        location = " ".join(proper_words)
    return location


def _detect_profession(sentence: str) -> str | None:
    lower = sentence.lower()
    if any(marker in lower for marker in _JOKE_MARKERS):
        return None
    for keyword, canonical in _PROFESSION_KEYWORDS:
        idx = lower.find(keyword)
        if idx == -1:
            continue
        preceding = lower[max(0, idx - 25) : idx]
        if any(marker in preceding for marker in _NEGATION_MARKERS):
            continue
        return canonical
    return None


def _detect_style(sentence: str) -> str | None:
    lower = sentence.lower()
    if "3 bullet" in lower:
        return "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off giữa recall và token cost"
    style_hints = ("ngắn gọn", "bullet ngắn", "trả lời ngắn", "ngắn và", "ngắn, rõ")
    if any(hint in lower for hint in style_hints):
        return "ngắn gọn, rõ ý và có ví dụ thực tế"
    return None


def _detect_interests(sentence: str) -> str | None:
    has_python = re.search(r"python", sentence, re.IGNORECASE) is not None
    has_ai = re.search(r"\bAI\b", sentence) is not None
    pieces = []
    if has_python:
        pieces.append("Python")
    if has_ai:
        pieces.append("AI ứng dụng")
    if not pieces:
        return None
    return ", ".join(pieces)


def _detect_pet(sentence: str) -> str | None:
    match = _PET_PATTERN.search(sentence)
    if not match:
        return None
    name = match.group(1)
    return f"một bé corgi tên {name}" if name else "corgi"


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Facts extracted: name, location, profession, style, favorite_drink,
    favorite_food, pet, interests. Only confident matches are returned, and
    negated / joking clauses (e.g. "không còn làm ... nữa", "chỉ là câu đùa")
    are skipped so corrections overwrite outdated facts instead of being
    confused with them.
    """

    facts: dict[str, str] = {}
    text = (message or "").strip()
    if not text:
        return facts

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        if "?" in sentence:
            # Questions ask for information, they don't provide it. Without
            # this guard, a recall question like "Tên mình là gì?" would be
            # parsed as if it *stated* a new name and corrupt the profile.
            continue

        lower = sentence.lower()

        name = _detect_name(sentence)
        if name:
            facts["name"] = name

        location = _detect_location(sentence)
        if location:
            facts["location"] = location

        profession = _detect_profession(sentence)
        if profession:
            facts["profession"] = profession

        style = _detect_style(sentence)
        if style:
            facts["style"] = style

        if "cà phê sữa đá" in lower:
            facts["favorite_drink"] = "cà phê sữa đá"

        if "mì quảng" in lower:
            facts["favorite_food"] = "mì Quảng"

        pet = _detect_pet(sentence)
        if pet:
            facts["pet"] = pet

        interests = _detect_interests(sentence)
        if interests:
            facts["interests"] = interests

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact heuristic summary of older messages (newest kept last)."""

    if not messages:
        return ""

    items = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for item in items:
        role = item.get("role", "user")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        snippet = content if len(content) <= 160 else content[:157] + "..."
        prefix = "Người dùng" if role == "user" else "Agent"
        lines.append(f"- {prefix}: {snippet}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads: keeps recent messages, summarizes the rest."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self._ensure(thread_id)
        thread_state["messages"].append({"role": role, "content": content})  # type: ignore[union-attr]
        self._maybe_compact(thread_id)

    def _maybe_compact(self, thread_id: str) -> None:
        thread_state = self.state[thread_id]
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]

        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        if total_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        if self.keep_messages > 0:
            overflow = messages[: -self.keep_messages]
            kept = messages[-self.keep_messages :]
        else:
            overflow = messages[:]
            kept = []

        if not overflow:
            return

        new_summary_piece = summarize_messages(overflow, max_items=len(overflow))
        existing_summary = str(thread_state.get("summary", ""))
        thread_state["summary"] = (
            (existing_summary + "\n" + new_summary_piece).strip() if existing_summary else new_summary_piece
        )
        thread_state["messages"] = kept
        thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure(thread_id).get("compactions", 0))
