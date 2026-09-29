"""운영자가 승인한 현재 Pod에서만 실행하는 제한적 read-only worker inventory.

특정 PID/소스/status/log 파일만 읽는다. dataset/cache 재귀 scan, 전체 environ 출력은 없다.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import stat
import re
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entrypoint(argv):
    """Python options 뒤의 script/module만 표시하며 inline code는 공개하지 않는다."""
    result = [argv[0]]
    for i, arg in enumerate(argv[1:], 1):
        if arg == "-c":
            return result + ["-c [redacted]"]
        if arg == "-m":
            return result + ["-m", argv[i+1] if i+1 < len(argv) else "UNKNOWN"]
        if not arg.startswith("-"):
            return result + [arg]
        result.append(arg)
    return result


def main():
    p = argparse.ArgumentParser(description="승인된 현재 endpoint에서 worker PID별 read-only 소수 파일 inventory")
    p.add_argument("--pid", type=int, action="append", required=True)
    p.add_argument("--source-file", type=Path, action="append", default=[])
    p.add_argument("--status-file", type=Path, action="append", default=[])
    p.add_argument("--log-file", type=Path, action="append", default=[])
    args = p.parse_args()
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "inspection_interpreter": sys.executable,
              "workers": [], "source_files": [], "status_files": [], "log_files": []}
    spec = importlib.util.find_spec("vos_memory_inspector")
    result["inspection_module_path"] = spec.origin if spec else None
    allow = {"--shard-count", "--shard-index", "--paired-split", "--device", "--dataset-root",
             "--network-volume-root", "--run-directory", "--sam2-repo", "--source-manifest"}
    for pid in args.pid:
        proc = Path("/proc") / str(pid)
        row = {"pid": pid}
        try:
            row["cwd"], row["executable"] = str((proc/"cwd").resolve()), str((proc/"exe").resolve())
            argv = (proc/"cmdline").read_bytes().decode(errors="replace").split("\0")
            row["entrypoint"] = entrypoint(argv)
            row["allowed_arguments"] = {arg: argv[i+1] for i,arg in enumerate(argv[:-1]) if arg in allow}
            row["interpreter_command"] = argv[0]
            env = (proc/"environ").read_bytes().split(b"\0")
            row["CUDA_VISIBLE_DEVICES"] = next((v.split(b"=",1)[1].decode() for v in env
                                               if v.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
            row["stat"] = (proc/"stat").read_text().strip()  # PID/parent/session/CPU ticks; environment 아님
            row["open_regular_file_writes"] = []
            for fd in (proc/"fd").iterdir():
                try:
                    info = (proc/"fdinfo"/fd.name).read_text()
                    flags = int(next(line.split()[1] for line in info.splitlines() if line.startswith("flags:")), 8)
                    if flags & 3 and stat.S_ISREG(fd.stat().st_mode):
                        row["open_regular_file_writes"].append({"fd": fd.name, "path": str(fd.resolve())})
                except (OSError, StopIteration):
                    pass
            row["cwd_revision"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=row["cwd"],
                check=False, capture_output=True, text=True, timeout=10).stdout.strip()
            probe = "import importlib.util,json,sys; s=importlib.util.find_spec('vos_memory_inspector'); print(json.dumps({'python':sys.executable,'module':s.origin if s else None}))"
            # shell/launcher PID에는 -c를 실행하지 않는다. 이 probe는 현재 새 process의
            # 설치 경로이며, 실행 중 worker에 이미 import된 모듈의 증거는 아니다.
            if (Path(argv[0]).is_absolute() and Path(argv[0]).is_file() and
                    re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(row["executable"]).name)):
                row["interpreter_module_probe"] = subprocess.run([argv[0], "-B", "-c", probe],
                    cwd=row["cwd"], check=False, capture_output=True, text=True, timeout=10).stdout.strip()
                row["module_probe_scope"] = "fresh_process_not_loaded_worker_modules"
            else:
                row["interpreter_module_probe"] = "UNKNOWN: explicit Python executable unavailable"
        except (OSError, subprocess.TimeoutExpired) as exc:
            row["error"] = str(exc)
        result["workers"].append(row)
    for path in args.source_file:
        result["source_files"].append({"path": str(path), "sha256": digest(path)})
    for path in args.status_file:
        value = json.loads(path.read_text())
        cases = value.get("cases", {})
        counts = {}
        for row in cases.values():
            label = row.get("state", "unknown")
            counts[label] = counts.get(label, 0)+1
        result["status_files"].append({"path": str(path), "mtime_ns": path.stat().st_mtime_ns,
            "state": value.get("state"), "case_state_counts": counts,
            "failed_case_ids": [k for k,v in cases.items() if v.get("state") in {"failed", "rejected"}]})
    for path in args.log_file:
        log_stat = path.stat()
        result["log_files"].append({"path": str(path), "bytes": log_stat.st_size, "mtime_ns": log_stat.st_mtime_ns})
    commands = {"devices": ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                "gpu_processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader"]}
    for name, cmd in commands.items():
        try:
            result[name] = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"unavailable": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
