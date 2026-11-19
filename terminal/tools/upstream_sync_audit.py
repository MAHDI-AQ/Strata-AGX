#!/usr/bin/env python3
"""Audit divergence and dry-run merge compatibility between Strata-AGX main and upstream."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ANCHOR = "d2f096475"


def run_git(repo: Path, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def classify_subsystem(path: str) -> str:
    if path.startswith(("src/core/", "include/strata/core/")):
        return "core"
    if path.startswith(("src/kernels/", "include/strata/kernels/")):
        return "kernels"
    if path.startswith(("src/prefill/", "include/strata/prefill/")):
        return "prefill"
    if path.startswith("src/program/"):
        return "program"
    if path.startswith("serve/"):
        return "serve"
    if path.startswith("tools/"):
        return "tools"
    return "other"


def is_artifact_or_backup(path: str) -> bool:
    return path.endswith((".orig", ".bak", ".json"))


def resolve_ref(repo: Path, ref: str) -> str | None:
    res = run_git(repo, ["rev-parse", "--verify", "--quiet", ref])
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def resolve_merge_base(repo: Path, main_ref: str, upstream_ref: str, explicit_base: str | None) -> tuple[str, str]:
    if explicit_base:
        sha = resolve_ref(repo, explicit_base)
        if not sha:
            raise RuntimeError(f"Explicit --merge-base ref not found: {explicit_base}")
        return sha, "explicit"
    mb = run_git(repo, ["merge-base", main_ref, upstream_ref])
    if mb.returncode == 0 and mb.stdout.strip():
        return mb.stdout.strip().splitlines()[0], "git-merge-base"
    for candidate in (DEFAULT_ANCHOR, "v0.2.13"):
        sha = resolve_ref(repo, candidate)
        if sha:
            return sha, f"fallback:{candidate}"
    raise RuntimeError("Could not determine merge-base (orphan root and no fallback v0.2.13/d2f096475)")


def parse_diff_numstat(repo: Path, base_ref: str, target_ref: str) -> dict:
    res = run_git(repo, ["diff", "--numstat", f"{base_ref}..{target_ref}"])
    if res.returncode != 0:
        raise RuntimeError(f"git diff --numstat failed: {res.stderr.strip()}")
    by_sub: dict[str, dict] = defaultdict(lambda: {"files": 0, "insertions": 0, "deletions": 0, "paths": []})
    flagged: list[str] = []
    total_files = 0
    total_ins = 0
    total_del = 0
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        ins_s, del_s, path = parts
        ins = int(ins_s) if ins_s.isdigit() else 0
        dels = int(del_s) if del_s.isdigit() else 0
        sub = classify_subsystem(path)
        by_sub[sub]["files"] += 1
        by_sub[sub]["insertions"] += ins
        by_sub[sub]["deletions"] += dels
        by_sub[sub]["paths"].append(path)
        if is_artifact_or_backup(path):
            flagged.append(path)
        total_files += 1
        total_ins += ins
        total_del += dels
    return {
        "total_files": total_files,
        "insertions": total_ins,
        "deletions": total_del,
        "subsystems": dict(sorted(by_sub.items())),
        "flagged_artifacts": sorted(flagged),
    }


def dry_run_merge_tree(repo: Path, anchor: str, main_ref: str, upstream_ref: str) -> dict:
    res = run_git(
        repo,
        ["merge-tree", "--write-tree", f"--merge-base={anchor}", "--no-messages", main_ref, upstream_ref],
        timeout=60,
    )
    if res.returncode == 0:
        return {
            "clean": True,
            "tree": res.stdout.strip().splitlines()[0] if res.stdout.strip() else "",
            "conflicts_by_subsystem": {},
            "conflict_count": 0,
        }
    if res.returncode == 1:
        # git merge-tree --write-tree prints tree OID on line 1, then conflicted paths/info
        conflicts: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        for line in res.stdout.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            # Format may be "<mode> <oid> <stage>\t<path>" or "CONFLICT ...: Merge conflict in <path>"
            path = None
            if "\t" in line:
                path = line.split("\t", 1)[1].strip()
            elif "Merge conflict in " in line:
                path = line.split("Merge conflict in ", 1)[1].strip()
            if path and path not in seen:
                seen.add(path)
                conflicts[classify_subsystem(path)].append(path)
        return {
            "clean": False,
            "tree": res.stdout.splitlines()[0].strip() if res.stdout.splitlines() else "",
            "conflicts_by_subsystem": {k: sorted(v) for k, v in sorted(conflicts.items())},
            "conflict_count": len(seen),
        }
    raise RuntimeError(f"git merge-tree failed ({res.returncode}): {res.stderr.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Strata-AGX main vs upstream compatibility.")
    ap.add_argument("--repo", default=".", help="Path to git repository (default: .)")
    ap.add_argument("--main-ref", default="main", help="Main/local ref (default: main)")
    ap.add_argument("--upstream-ref", default="upstream/main", help="Upstream ref (default: upstream/main)")
    ap.add_argument("--merge-base", default=None, help="Explicit merge-base anchor (fallback: d2f096475 / v0.2.13)")
    ap.add_argument("--fetch", action="store_true", help="Fetch upstream remote before auditing (15s timeout)")
    ap.add_argument("--bundle", default=None, help="Optional path to create a git bundle of main-ref")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    ap.add_argument("--fail-on-conflict", action="store_true", help="Exit non-zero if dry-run merge has conflicts")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"ERROR: not a git repository: {repo}", file=sys.stderr)
        return 2

    if args.fetch:
        remote = args.upstream_ref.split("/", 1)[0] if "/" in args.upstream_ref else "upstream"
        f_res = run_git(repo, ["fetch", remote], timeout=15)
        if f_res.returncode != 0:
            print(f"ERROR: git fetch {remote} failed: {f_res.stderr.strip()}", file=sys.stderr)
            return 2

    main_sha = resolve_ref(repo, args.main_ref)
    if not main_sha:
        print(f"ERROR: main-ref not found: {args.main_ref}", file=sys.stderr)
        return 2

    upstream_sha = resolve_ref(repo, args.upstream_ref)
    if not upstream_sha:
        print(f"ERROR: upstream-ref not found: {args.upstream_ref}", file=sys.stderr)
        return 2

    anchor_sha, anchor_source = resolve_merge_base(repo, args.main_ref, args.upstream_ref, args.merge_base)
    divergence = parse_diff_numstat(repo, args.upstream_ref, args.main_ref)
    merge_check = dry_run_merge_tree(repo, anchor_sha, args.main_ref, args.upstream_ref)

    bundle_ok = None
    if args.bundle:
        b_res = run_git(repo, ["bundle", "create", args.bundle, args.main_ref], timeout=60)
        if b_res.returncode != 0:
            print(f"ERROR: git bundle create failed: {b_res.stderr.strip()}", file=sys.stderr)
            return 2
        v_res = run_git(repo, ["bundle", "verify", args.bundle], timeout=60)
        bundle_ok = v_res.returncode == 0

    report = {
        "repo": str(repo),
        "main_ref": args.main_ref,
        "main_sha": main_sha,
        "upstream_ref": args.upstream_ref,
        "upstream_sha": upstream_sha,
        "merge_base_sha": anchor_sha,
        "merge_base_source": anchor_source,
        "divergence": divergence,
        "merge_tree": merge_check,
        "bundle_verified": bundle_ok,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Repo:        {repo}")
        print(f"Main:        {args.main_ref} -> {main_sha[:12]}")
        print(f"Upstream:    {args.upstream_ref} -> {upstream_sha[:12]}")
        print(f"Merge-base:  {anchor_sha[:12]} ({anchor_source})")
        print(
            f"Divergence:  {divergence['total_files']} files, "
            f"+{divergence['insertions']}/-{divergence['deletions']}"
        )
        for sub, info in divergence["subsystems"].items():
            print(f"  - {sub:8s}: {info['files']:3d} files (+{info['insertions']}/-{info['deletions']})")
        if divergence["flagged_artifacts"]:
            print(f"Flagged (.orig/.bak/.json): {len(divergence['flagged_artifacts'])} files")
        print(
            f"Merge-tree:  {'CLEAN' if merge_check['clean'] else 'CONFLICTS'} "
            f"(tree={merge_check['tree'][:12]}, conflicts={merge_check['conflict_count']})"
        )
        for sub, paths in merge_check["conflicts_by_subsystem"].items():
            print(f"  * {sub}: {', '.join(paths)}")
        if bundle_ok is not None:
            print(f"Bundle:      {'OK' if bundle_ok else 'FAILED'} ({args.bundle})")

    if args.fail_on_conflict and not merge_check["clean"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
