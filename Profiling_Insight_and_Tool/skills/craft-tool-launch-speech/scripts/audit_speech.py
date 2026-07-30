#!/usr/bin/env python3
"""Audit the structural promises of a tool-launch speech."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("speech", type=Path, help="Markdown speech to audit")
    parser.add_argument(
        "--required-rows",
        default="",
        help="Comma-separated CheckList row IDs that must appear",
    )
    parser.add_argument("--top-count", type=int, choices=(3, 4), default=4)
    parser.add_argument("--min-demo", type=int, default=3)
    parser.add_argument("--min-slogans", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.speech.is_file():
        print(f"ERROR: speech not found: {args.speech}")
        return 2

    text = args.speech.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    name_match = re.fullmatch(r".+-体验改进总结-\d{8}\.md", args.speech.name)
    if not name_match:
        errors.append(
            "filename must match 业务主题-体验改进总结-YYYYMMDD.md"
        )
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if not h1_match or h1_match.group(1).strip() != args.speech.stem:
        errors.append("H1 must exactly match the filename stem")

    required_rows = {
        int(value.strip())
        for value in args.required_rows.split(",")
        if value.strip()
    }
    found_rows = {int(value) for value in re.findall(r"\bRow\s+(\d+)\b", text)}
    missing_rows = sorted(required_rows - found_rows)
    if missing_rows:
        errors.append("missing required voice rows: " + ", ".join(map(str, missing_rows)))

    demo_count = len(re.findall(r"【Demo｜工具：[^】]+】", text))
    if demo_count < args.min_demo:
        errors.append(f"only {demo_count} Demo blocks; require at least {args.min_demo}")

    voice_block_count = text.count("【覆盖用户原声】")
    if voice_block_count < demo_count:
        warnings.append(
            f"{demo_count} Demo blocks but only {voice_block_count} voice-coverage blocks"
        )

    slogan_count = len(re.findall(r"(?:体验亮点\s+Slogan|>\s*\*\*Slogan：)", text))
    if slogan_count < args.min_slogans:
        errors.append(
            f"only {slogan_count} Slogan markers; require at least {args.min_slogans}"
        )

    top_headings = {
        int(value) for value in re.findall(r"^###\s+Top\s+(\d+)\b", text, re.MULTILINE)
    }
    expected_top = set(range(1, args.top_count + 1))
    if not expected_top.issubset(top_headings):
        missing = sorted(expected_top - top_headings)
        errors.append("missing Top headings: " + ", ".join(map(str, missing)))

    required_sections = (
        "用户原声覆盖矩阵",
    )
    for section in required_sections:
        if section not in text:
            errors.append(f"missing required section: {section}")

    demo_blocks = re.findall(
        r"【Demo｜工具：[^】]+】(?P<body>.*?)(?=【覆盖用户原声】|\Z)",
        text,
        re.DOTALL,
    )
    operation_terms = re.compile(r"打开|点击|切换|选择|勾选|输入|展开|停留|指向|返回")
    for index, block in enumerate(demo_blocks, start=1):
        if not operation_terms.search(block):
            warnings.append(f"Demo block {index} lacks a concrete operation verb")

    demo_beats = re.findall(
        r"【Demo｜工具：[^】]+】.*?【覆盖用户原声】(?P<voices>.*?)(?=【体验亮点\s+Slogan】|\Z)",
        text,
        re.DOTALL,
    )
    for index, voices in enumerate(demo_beats, start=1):
        if not re.search(r"\bRow\s+\d+\b", voices):
            errors.append(f"Demo block {index} has no valid Row ID in its voice coverage")

    voice_sections = re.findall(
        r"【覆盖用户原声】(?P<voices>.*?)(?=【体验亮点\s+Slogan】|^---$|^###\s|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    for index, voices in enumerate(voice_sections, start=1):
        for line in voices.splitlines():
            if re.search(r"\b(?:Row\s+\d+|补充原声-[\w-]+)\b", line):
                if not re.match(r"^\s*>\s+", line):
                    errors.append(
                        f"voice-coverage block {index} contains a citation outside Markdown blockquote syntax"
                    )

    if re.search(r"^##\s+五、现场 Demo 准备与口径检查", text, re.MULTILINE):
        errors.append("standalone 现场 Demo 准备与口径检查 chapter must be removed")

    chapter_headings = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    if not chapter_headings or not chapter_headings[0].startswith("一、用户原声覆盖矩阵"):
        errors.append("用户原声覆盖矩阵 must be Chapter 1")

    matrix = re.search(
        r"^##\s+一、用户原声覆盖矩阵(?P<body>.*?)(?=^##\s+二、|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if matrix:
        matrix_text = matrix.group("body")
        matrix_rows = {int(value) for value in re.findall(r"\bRow\s+(\d+)\b", matrix_text)}
        missing_matrix_rows = sorted(required_rows - matrix_rows)
        if missing_matrix_rows:
            errors.append(
                "required voice rows missing from Chapter 1 matrix: "
                + ", ".join(map(str, missing_matrix_rows))
            )
        header = next(
            (
                line
                for line in matrix_text.splitlines()
                if line.startswith("|") and "业务流程阶段" in line
            ),
            "",
        )
        for column in (
            "用户痛点主题",
            "产品机会点",
            "对应 Top 体验亮点",
            "关键设计点",
            "工具 / Demo 承载",
        ):
            if column not in header:
                errors.append(f"Chapter 1 matrix is missing column: {column}")
        if "原声输入" in header:
            errors.append("Chapter 1 matrix must not have a separate 原声输入 column")
        data_rows = [
            line
            for line in matrix_text.splitlines()
            if line.startswith("|") and "**摘要：**" in line
        ]
        for index, row in enumerate(data_rows, start=1):
            if "原声：" not in row:
                errors.append(
                    f"Chapter 1 matrix row {index} lacks numbered verbatim under pain summary"
                )

    if re.search(r"###\s+(?:第一层映射|第二层映射)", text):
        errors.append("mapping must use one table, not separate first/second-layer tables")

    if re.search(r"\{\{[^}]+\}\}", text):
        errors.append("unresolved template placeholders remain")

    for message in errors:
        print(f"ERROR: {message}")
    for message in warnings:
        print(f"WARN: {message}")

    print(
        "SUMMARY: "
        f"demos={demo_count}, slogans={slogan_count}, "
        f"voice_rows={len(found_rows)}, errors={len(errors)}, warnings={len(warnings)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
