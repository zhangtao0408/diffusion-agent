"""Parse Ascend pytorch HTML API docs into an op compatibility matrix."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class OpEntry:
    """A single API entry parsed from the Ascend pytorch API doc."""

    api_name: str
    compatible: bool
    limitations: str
    section: str


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode common entities, return clean text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


def _normalize_newlines(content: str) -> str:
    """Replace escaped literal \\n sequences with actual newlines."""
    return content.replace("\\n", "\n")


def _detect_column_indices(section_block: str) -> tuple[int, int, int]:
    """Detect which column indices hold API name, compatibility, and limitations.

    Most sections have 3 columns (API, Compat, Limits).
    ``torch.cuda`` and ``torch.cuda.amp`` have 4 (API, NPU API, Compat, Limits).
    We detect this by inspecting the ``<th>`` headers.
    """
    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", section_block, re.DOTALL)
    headers = [_clean_html(h).lower() for h in th_matches]

    api_idx = 0  # always first column
    compat_idx = 1
    limit_idx = 2

    for i, h in enumerate(headers):
        if "compatibility" in h:
            compat_idx = i
        elif "limitation" in h:
            limit_idx = i

    return api_idx, compat_idx, limit_idx


def parse_api_doc(content: str) -> list[OpEntry]:
    """Parse HTML API doc content into a list of OpEntry objects.

    The doc uses ``## section_name`` headers followed by HTML ``<table>`` blocks.
    Each data row ``<tr id="row...">`` contains cells for API name, compatibility
    (``Y`` or empty), and limitations text.  Column layout is auto-detected from
    the ``<th>`` headers to handle both 3-column and 4-column sections.
    """
    if not content:
        return []

    content = _normalize_newlines(content)

    entries: list[OpEntry] = []
    sections = re.split(r"(?:^|\n)## ", content)

    for section_block in sections:
        if not section_block.strip():
            continue

        header_line = section_block.split("\n", 1)[0].strip()
        section_name = header_line.strip()

        api_idx, compat_idx, limit_idx = _detect_column_indices(section_block)
        min_cells = max(api_idx, compat_idx, limit_idx) + 1

        # Find all data rows: <tr id="row...">...</tr>
        rows = re.findall(r'<tr\s+id="row[^"]*">(.*?)</tr>', section_block, re.DOTALL)

        for row_html in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
            if len(cells) < min_cells:
                continue

            api_name = _clean_html(cells[api_idx])
            if not api_name or api_name in ("PyTorch API",):
                continue

            compat_text = _clean_html(cells[compat_idx])
            compatible = compat_text.upper() == "Y"

            limitations = _clean_html(cells[limit_idx]) if limit_idx < len(cells) else ""

            entries.append(
                OpEntry(
                    api_name=api_name,
                    compatible=compatible,
                    limitations=limitations,
                    section=section_name,
                )
            )

    return entries


def build_op_matrix(entries: list[OpEntry]) -> dict:
    """Convert parsed OpEntry list into the format used by torch_npu_checker.

    Returns a dict with "ops" and "patterns" keys, compatible with op_support.json.
    """
    ops: dict[str, dict[str, str]] = {}
    for entry in entries:
        if entry.compatible:
            ops[entry.api_name] = {
                "status": "supported",
                "note": entry.limitations,
            }
        else:
            ops[entry.api_name] = {
                "status": "unknown",
                "note": "Not tested on NPU",
            }

    return {"ops": ops, "patterns": {}}
