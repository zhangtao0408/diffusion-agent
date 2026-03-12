"""Tests for state management modules."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.state_mgmt.daily_log import append_log, read_log
from diffusion_agent.state_mgmt.feature_list import (
    Feature,
    get_next_pending,
    read_features,
    update_feature_status,
    write_features,
)
from diffusion_agent.state_mgmt.rules import read_rules, write_rules
from diffusion_agent.state_mgmt.task_file import (
    clear_current_task,
    read_current_task,
    write_current_task,
)


class TestTaskFile:
    def test_write_and_read(self, tmp_repo: Path) -> None:
        task = {"id": "feat-001", "status": "pending", "feature_name": "test feature"}
        write_current_task(tmp_repo, task)
        result = read_current_task(tmp_repo)
        assert result is not None
        assert result["id"] == "feat-001"
        assert result["status"] == "pending"

    def test_read_nonexistent(self, tmp_repo: Path) -> None:
        result = read_current_task(tmp_repo)
        assert result is None

    def test_clear(self, tmp_repo: Path) -> None:
        write_current_task(tmp_repo, {"id": "x"})
        clear_current_task(tmp_repo)
        assert read_current_task(tmp_repo) is None


class TestDailyLog:
    def test_append_and_read(self, tmp_repo: Path) -> None:
        append_log(tmp_repo, "First entry")
        append_log(tmp_repo, "Second entry")
        content = read_log(tmp_repo)
        assert content is not None
        assert "First entry" in content
        assert "Second entry" in content
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_read_nonexistent(self, tmp_repo: Path) -> None:
        assert read_log(tmp_repo) is None


class TestRules:
    def test_write_and_read(self, tmp_repo: Path) -> None:
        write_rules(tmp_repo, "# Rules\n- Rule 1\n- Rule 2")
        content = read_rules(tmp_repo)
        assert content is not None
        assert "Rule 1" in content

    def test_read_nonexistent(self, tmp_repo: Path) -> None:
        assert read_rules(tmp_repo) is None


class TestFeatureList:
    def test_write_and_read(self, tmp_repo: Path) -> None:
        features = [
            Feature(id="f1", name="Feature 1", description="Desc 1"),
            Feature(id="f2", name="Feature 2", description="Desc 2", status="completed"),
        ]
        write_features(tmp_repo, features)
        result = read_features(tmp_repo)
        assert len(result) == 2
        assert result[0].id == "f1"
        assert result[0].status == "pending"
        assert result[1].status == "completed"

    def test_read_empty(self, tmp_repo: Path) -> None:
        result = read_features(tmp_repo)
        assert result == []

    def test_update_status(self, tmp_repo: Path) -> None:
        features = [
            Feature(id="f1", name="F1", description="D1"),
            Feature(id="f2", name="F2", description="D2"),
        ]
        write_features(tmp_repo, features)
        update_feature_status(tmp_repo, "f1", "completed")
        result = read_features(tmp_repo)
        assert result[0].status == "completed"
        assert result[1].status == "pending"

    def test_update_status_with_error(self, tmp_repo: Path) -> None:
        features = [Feature(id="f1", name="F1", description="D1")]
        write_features(tmp_repo, features)
        update_feature_status(tmp_repo, "f1", "failed", error="Something went wrong")
        result = read_features(tmp_repo)
        assert result[0].status == "failed"
        assert result[0].error == "Something went wrong"

    def test_get_next_pending(self, tmp_repo: Path) -> None:
        features = [
            Feature(id="f1", name="F1", description="D1", status="completed"),
            Feature(id="f2", name="F2", description="D2", status="pending"),
            Feature(id="f3", name="F3", description="D3", status="pending"),
        ]
        write_features(tmp_repo, features)
        nxt = get_next_pending(tmp_repo)
        assert nxt is not None
        assert nxt.id == "f2"

    def test_get_next_pending_none(self, tmp_repo: Path) -> None:
        features = [Feature(id="f1", name="F1", description="D1", status="completed")]
        write_features(tmp_repo, features)
        assert get_next_pending(tmp_repo) is None
