from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests: tmp state dir + a low compact threshold."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0, api_key=None)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=_DATA_DIR,
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "tester"

    default_content = store.read_text(user_id)
    assert default_content  # a default template is returned even before any write

    store.upsert_fact(user_id, "name", "Tester")
    content = store.read_text(user_id)
    assert "Tester" in content
    assert store.file_size(user_id) > 0

    changed = store.edit_text(user_id, "Tester", "Tester Updated")
    assert changed is True
    assert "Tester Updated" in store.read_text(user_id)

    not_found = store.edit_text(user_id, "Does not exist", "Something else")
    assert not_found is False

    store.upsert_fact(user_id, "name", "Tester Updated Again")
    facts = store.facts(user_id)
    assert facts["name"] == "Tester Updated Again"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages,
    )
    thread_id = "thread-1"

    long_text = "Đây là một đoạn hội thoại khá dài để vượt ngưỡng token nén bộ nhớ. " * 5
    for i in range(10):
        manager.append(thread_id, "user", f"{long_text} lượt {i}")

    assert manager.compaction_count(thread_id) > 0

    context = manager.context(thread_id)
    assert context["summary"]
    assert len(context["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct_test"

    baseline.reply(user_id, "thread-a", "Mình tên là DũngCT.")
    baseline.reply(user_id, "thread-a", "Mình ở Đà Nẵng.")
    baseline_answer = baseline.reply(user_id, "thread-b", "Mình tên gì?")["answer"]
    assert "DũngCT" not in baseline_answer

    advanced.reply(user_id, "thread-a", "Mình tên là DũngCT.")
    advanced.reply(user_id, "thread-a", "Mình ở Đà Nẵng.")
    advanced_answer = advanced.reply(user_id, "thread-b", "Mình tên gì?")["answer"]
    assert "DũngCT" in advanced_answer
    assert advanced.memory_file_size(user_id) > 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct_stress_test"
    thread_id = "stress-thread"

    long_text = (
        "Đây là một đoạn tin tức khá dài được lặp lại nhiều lần để ép ngữ cảnh "
        "phình to và kiểm tra khả năng nén bộ nhớ của advanced agent. "
    ) * 4

    for i in range(12):
        baseline.reply(user_id, thread_id, f"{long_text} lượt {i}")
        advanced.reply(user_id, thread_id, f"{long_text} lượt {i}")

    assert advanced.compaction_count(thread_id) > 0
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
