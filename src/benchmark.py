from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def recall_points(answer: str, expected: list[str]) -> float:
    """0 if nothing matched, 0.5 if partial, 1 if every expected fact is present."""

    if not expected:
        return 1.0
    answer = answer or ""
    hits = sum(1 for item in expected if item in answer)
    ratio = hits / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio > 0.0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality score: fact coverage + conciseness."""

    answer = (answer or "").strip()
    if not answer:
        return 0.0

    coverage = recall_points(answer, expected)
    length = len(answer)
    conciseness = 1.0 if length <= 300 else max(0.3, 300 / length)

    score = 0.7 * coverage + 0.3 * conciseness
    return round(min(1.0, score), 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    agent_tokens_total = 0
    prompt_tokens_total = 0
    compactions_total = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    seen_users: set[str] = set()

    for conv in conversations:
        user_id = conv["user_id"]
        seen_users.add(user_id)
        thread_id = f"{agent_name}:{conv['id']}"

        for turn in conv["turns"]:
            result = agent.reply(user_id, thread_id, turn)
            agent_tokens_total += result.get("tokens", 0)
            prompt_tokens_total += result.get("prompt_tokens", 0)

        compactions_total += agent.compaction_count(thread_id)

        for q_index, question in enumerate(conv.get("recall_questions", [])):
            recall_thread_id = f"{agent_name}:{conv['id']}:recall:{q_index}"
            result = agent.reply(user_id, recall_thread_id, question["question"])
            agent_tokens_total += result.get("tokens", 0)
            prompt_tokens_total += result.get("prompt_tokens", 0)

            answer = result.get("answer", "")
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    memory_growth_bytes = 0
    size_fn = getattr(agent, "memory_file_size", None)
    if callable(size_fn):
        memory_growth_bytes = sum(size_fn(user_id) for user_id in seen_users)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens_total,
        prompt_tokens_processed=prompt_tokens_total,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth_bytes,
        compactions=compactions_total,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    return tabulate(table, headers=headers, tablefmt="github")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_conversations = load_conversations(config.data_dir / "conversations.json")
    stress_conversations = load_conversations(config.data_dir / "advanced_long_context.json")

    print("## Standard Benchmark\n")
    standard_rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config=config, force_offline=True), standard_conversations, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config=config, force_offline=True), standard_conversations, config),
    ]
    print(format_rows(standard_rows))

    print("\n## Long-Context Stress Benchmark\n")
    stress_rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config=config, force_offline=True), stress_conversations, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config=config, force_offline=True), stress_conversations, config),
    ]
    print(format_rows(stress_rows))


if __name__ == "__main__":
    main()
