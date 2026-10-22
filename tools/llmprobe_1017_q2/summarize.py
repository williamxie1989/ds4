#!/usr/bin/env python3
"""Summarize an llmprobe --eval-only report.

Reads the JSON written by `llmprobe --save`. The reasoning eval is in
`report["reasoning"]`; each case has a status:

  passed  = correct
  failed  = wrong answer (this is the "7 wrong" in the comment)
  stopped = ran out of tokens / no Answer line (a budget fact, not wrong)
  error   = request/engine error (infrastructure, not wrong)

Usage:
  python3 summarize.py report.json
  python3 summarize.py report.json --baseline baseline.json

The 10-question run is a floor check; scopeNote is set and it is not
comparable to the full 92-question eval.
"""

import argparse
import json
import sys
from collections import Counter


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def trunc(value, limit: int = 180) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def status_counts(cases):
    counts = Counter()
    for case in cases:
        counts[case.get("status", "unknown")] += 1
    return counts


def print_header(data, report_path):
    target = data.get("target") or {}
    run = data.get("run") or {}
    machine = data.get("machine") or {}

    print("=== llmprobe reasoning eval ===")
    print(f"report: {report_path}")
    if isinstance(target, dict):
        bits = []
        for key in ("model", "baseUrl", "base_url", "endpoint"):
            if key in target:
                bits.append(f"{key}={target[key]}")
        if bits:
            print("target: " + " ".join(bits))
    if isinstance(run, dict):
        print(
            "run: mode=%s depth=%s label=%s startedAt=%s"
            % (
                run.get("mode", "?"),
                run.get("depth", "?"),
                run.get("label", ""),
                run.get("startedAt", ""),
            )
        )
    if isinstance(machine, dict):
        chip = machine.get("chip") or machine.get("model") or ""
        ram = machine.get("ram") or machine.get("memory") or ""
        if chip or ram:
            print(f"machine: chip={chip} ram={ram}")


def print_reasoning_summary(reasoning, cases):
    counts = status_counts(cases)
    total = len(cases)
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    stopped = counts.get("stopped", 0)
    error = counts.get("error", 0)

    print(
        "suite=%s total=%d passed=%d failed=%d stopped=%d error=%d"
        % (
            reasoning.get("suite", "?"),
            total,
            passed,
            failed,
            stopped,
            error,
        )
    )
    print(
        "maxTokens=%s temperature=%s reasoningEffort=%s rejected=%s thinkingOff=%s"
        % (
            reasoning.get("maxTokens", "?"),
            reasoning.get("temperature", "?"),
            reasoning.get("reasoningEffort", "?"),
            reasoning.get("reasoningEffortRejected", "?"),
            reasoning.get("thinkingOff", "?"),
        )
    )
    print(f"scopeNote: {reasoning.get('scopeNote')}")

    aborted = reasoning.get("aborted")
    if aborted:
        print(
            "ABORTED: %s — %s"
            % (aborted.get("reason", "?"), aborted.get("message", ""))
        )


def print_by_source(reasoning):
    rows = reasoning.get("bySource") or []
    if not rows:
        return
    print("\nbySource:")
    print("  source                    passed failed stopped error total")
    for row in rows:
        print(
            "  %-24s %6s %6s %7s %5s %5s"
            % (
                trunc(row.get("source", "?"), 24),
                row.get("passed", 0),
                row.get("failed", 0),
                row.get("stopped", 0),
                row.get("error", 0),
                row.get("total", 0),
            )
        )


def print_cases(cases):
    if not cases:
        return
    print("\ncases:")
    print("  id                         source             status   expected -> got")
    for case in cases:
        print(
            "  %-26s %-18s %-8s %s -> %s"
            % (
                trunc(case.get("id", "?"), 26),
                trunc(case.get("source", "?"), 18),
                case.get("status", "?"),
                trunc(case.get("expected", ""), 40),
                trunc(case.get("got", ""), 40),
            )
        )


def print_failures(cases):
    failed = [case for case in cases if case.get("status") == "failed"]
    stopped = [case for case in cases if case.get("status") == "stopped"]
    errors = [case for case in cases if case.get("status") == "error"]

    if failed:
        print("\nfailed cases (wrong answers):")
        for case in failed:
            print(
                "  %s | %s | expected=%s got=%s | finish=%s out=%s ms=%s"
                % (
                    case.get("id", "?"),
                    case.get("source", "?"),
                    trunc(case.get("expected", ""), 80),
                    trunc(case.get("got", ""), 80),
                    case.get("finishReason"),
                    case.get("outputTokens"),
                    case.get("durationMs"),
                )
            )
            if case.get("text"):
                print(f"    text: {trunc(case.get('text'), 220)}")

    if stopped:
        print("\nstopped cases (no answer / token cap; not wrong):")
        for case in stopped:
            print(
                "  %s | %s | finish=%s out=%s"
                % (
                    case.get("id", "?"),
                    case.get("source", "?"),
                    case.get("finishReason"),
                    case.get("outputTokens"),
                )
            )

    if errors:
        print("\nerror cases (request/engine errors; not wrong):")
        for case in errors:
            print(
                "  %s | %s | %s"
                % (
                    case.get("id", "?"),
                    case.get("source", "?"),
                    trunc(case.get("error", ""), 220),
                )
            )


def compare_baseline(current_data, current_reasoning, current_cases, baseline_path):
    baseline = load_json(baseline_path)
    baseline_reasoning = baseline.get("reasoning")
    if not isinstance(baseline_reasoning, dict):
        print("\nWARN: baseline has no reasoning section; not comparable.")
        return

    current_target = current_data.get("target") or {}
    baseline_target = baseline.get("target") or {}
    current_model = current_target.get("model") if isinstance(current_target, dict) else None
    baseline_model = baseline_target.get("model") if isinstance(baseline_target, dict) else None
    if current_model and baseline_model and current_model != baseline_model:
        print(
            "\nWARN: model id differs (current=%s baseline=%s); confirm it is the same "
            "base model/checkpoint, otherwise the comparison is not apples-to-apples."
            % (current_model, baseline_model)
        )

    baseline_cases = baseline_reasoning.get("cases") or []
    if not baseline_cases:
        print("\nWARN: baseline has no reasoning cases; not comparable.")
        return

    baseline_by_id = {case.get("id"): case for case in baseline_cases}
    current_by_id = {case.get("id"): case for case in current_cases}

    if current_reasoning.get("suite") != baseline_reasoning.get("suite"):
        print(
            "\nWARN: suite differs (current=%s baseline=%s); not comparable."
            % (
                current_reasoning.get("suite", "?"),
                baseline_reasoning.get("suite", "?"),
            )
        )
    if (
        current_reasoning.get("scopeNote")
        and baseline_reasoning.get("scopeNote")
        and current_reasoning.get("scopeNote") != baseline_reasoning.get("scopeNote")
    ):
        print(
            "\nWARN: scopeNote differs (current=%r baseline=%r); check the question set."
            % (
                current_reasoning.get("scopeNote"),
                baseline_reasoning.get("scopeNote"),
            )
        )
    only_current = sorted(set(current_by_id) - set(baseline_by_id))
    only_baseline = sorted(set(baseline_by_id) - set(current_by_id))
    if only_current or only_baseline:
        print(
            "\nWARN: case ids differ; only_current=%d only_baseline=%d"
            % (len(only_current), len(only_baseline))
        )

    baseline_counts = status_counts(baseline_cases)
    print("\n=== baseline comparison ===")
    print(
        "baseline: suite=%s total=%d passed=%d failed=%d stopped=%d error=%d"
        % (
            baseline_reasoning.get("suite", "?"),
            len(baseline_cases),
            baseline_counts.get("passed", 0),
            baseline_counts.get("failed", 0),
            baseline_counts.get("stopped", 0),
            baseline_counts.get("error", 0),
        )
    )
    print(f"baseline scopeNote: {baseline_reasoning.get('scopeNote')}")

    newly_failed = []
    newly_passed = []
    same_failed = []
    both_passed = []
    other_changes = []

    for case_id, current in current_by_id.items():
        base = baseline_by_id.get(case_id)
        if base is None:
            continue
        current_status = current.get("status")
        baseline_status = base.get("status")
        if baseline_status == "passed" and current_status == "failed":
            newly_failed.append((case_id, current))
        elif baseline_status == "failed" and current_status == "passed":
            newly_passed.append((case_id, current))
        elif baseline_status == "failed" and current_status == "failed":
            same_failed.append((case_id, current))
        elif baseline_status == "passed" and current_status == "passed":
            both_passed.append(case_id)
        elif baseline_status != current_status:
            other_changes.append((case_id, baseline_status, current_status))

    print(
        "both_passed=%d both_failed=%d newly_failed=%d newly_passed=%d other_changes=%d"
        % (
            len(both_passed),
            len(same_failed),
            len(newly_failed),
            len(newly_passed),
            len(other_changes),
        )
    )

    if newly_failed:
        print("\nnewly failed (baseline passed -> current failed):")
        for case_id, current in newly_failed:
            print(
                "  %s | %s | expected=%s got=%s"
                % (
                    case_id,
                    current.get("source", "?"),
                    trunc(current.get("expected", ""), 80),
                    trunc(current.get("got", ""), 80),
                )
            )
            if current.get("text"):
                print(f"    text: {trunc(current.get('text'), 220)}")

    if newly_passed:
        print("\nnewly passed (baseline failed -> current passed):")
        for case_id, current in newly_passed:
            print(
                "  %s | %s | got=%s"
                % (
                    case_id,
                    current.get("source", "?"),
                    trunc(current.get("got", ""), 80),
                )
            )

    if other_changes:
        print("\nother status changes:")
        for case_id, baseline_status, current_status in other_changes:
            print(f"  {case_id}: {baseline_status} -> {current_status}")


def print_interpretation(reasoning):
    print("\n判读:")
    print("- failed = 错题；stopped = 没在 token cap 内给出 Answer；error = 请求/引擎错误。")
    print("  不要把 stopped/error 算进“错题”；先看它们是否多，多则先修预算/超时/服务。")
    print("- scopeNote 非空表示只跑了前 N 题；10 题是 floor check，不能和完整 92 题比。")
    if reasoning.get("reasoningEffortRejected"):
        print("- reasoningEffortRejected=true：引擎拒绝了 reasoning_effort，按默认跑；与显式 medium 不完全可比。")
    if reasoning.get("thinkingOff"):
        print("- thinkingOff 非空：off 的强制方式有回退/异常；看报告细节。")
    print("- 要判断“这份 Q2 量化本身有缺陷”，必须用同样 10 题、同样 llmprobe 版本/设置，")
    print("  对比官方 1017 全精度（Transformers/vLLM）或更高精度 quant/重新量化文件；")
    print("  GLM-5.3-Flash-Sushi-2.4bpw 是不同模型/EXL3/Sushi 工具链，不是 apples-to-apples。")
    print("- 若 failed=7/10：复现了那条评论的 anecdote，但仍需基线才能归因量化；")
    print("  若 failed!=7/10：没有复现，可能是版本/题集/sampling/文件/可执行程序差异。")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize an llmprobe --eval-only JSON report."
    )
    parser.add_argument("report", help="Path to the llmprobe --save JSON report")
    parser.add_argument("--baseline", help="Optional baseline llmprobe --save report")
    args = parser.parse_args()

    data = load_json(args.report)
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, dict):
        print(
            "ERROR: no reasoning section. Run llmprobe with --eval-only.",
            file=sys.stderr,
        )
        return 2

    cases = reasoning.get("cases") or []
    print_header(data, args.report)
    print_reasoning_summary(reasoning, cases)
    print_by_source(reasoning)
    print_cases(cases)
    print_failures(cases)
    if args.baseline:
        compare_baseline(data, reasoning, cases, args.baseline)
    print_interpretation(reasoning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
