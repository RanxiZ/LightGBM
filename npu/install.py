"""Build the validated NPU backend and provision the ordinary Qlib qrun entry."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time

REPO = Path(__file__).resolve().parents[1]
QLIB_COMMIT = "be338d3c2ee8f0aa462015e4ba888a5a7682a10e"


def run(args, **kwargs):
    print("+", shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_root(name):
    # A pip-installed editable finder is registered in a fresh interpreter.
    code = "import importlib.util,sys;s=importlib.util.find_spec(sys.argv[1]);print(s.origin if s else '')"
    origin = subprocess.check_output([sys.executable, "-c", code, name], text=True).strip()
    if not origin:
        raise RuntimeError(f"Python package is missing: {name}")
    return Path(origin).parent


class Installation:
    def __init__(self, state):
        self.state = state
        self.files = {}
        self.state.mkdir(parents=True, exist_ok=False)
        self.save()

    def save(self):
        (self.state / "backup.json").write_text(json.dumps(self.files, indent=2) + "\n")

    def backup(self, path):
        path = Path(path).absolute()
        if str(path) in self.files:
            return
        record = {"existed": path.exists() or path.is_symlink()}
        if path.is_symlink():
            record["symlink"] = os.readlink(path)
        elif path.is_file():
            backup = self.state / "files" / str(path).lstrip("/")
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            record.update(backup=str(backup), sha256=digest(backup))
        self.files[str(path)] = record
        self.save()

    def copy(self, source, destination):
        destination = Path(destination)
        self.backup(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".npu-install-tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    def write(self, destination, text):
        destination = Path(destination)
        self.backup(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".npu-install-tmp")
        temporary.write_text(text)
        os.replace(temporary, destination)

    def rollback(self):
        for name, record in reversed(list(self.files.items())):
            path = Path(name)
            if "symlink" in record:
                if path.exists() or path.is_symlink():
                    path.unlink()
                path.symlink_to(record["symlink"])
            elif record["existed"]:
                assert digest(record["backup"]) == record["sha256"]
                temporary = path.with_name(path.name + ".npu-rollback-tmp")
                shutil.copy2(record["backup"], temporary)
                os.replace(temporary, path)
            elif path.exists() or path.is_symlink():
                path.unlink()


def ensure_qlib(directory):
    directory = directory.resolve()
    metadata = json.loads((REPO / "npu/reference/qlib_source.json").read_text())
    marker = directory / ".npu-qlib-source.json"
    if not directory.exists() or not any(directory.iterdir()):
        archive = REPO / "npu/qlib_source.tar.gz"
        if digest(archive) != metadata["archive_sha256"]:
            raise RuntimeError("Qlib source archive SHA256 mismatch")
        directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                destination = (directory / member.name).resolve()
                if not destination.is_relative_to(directory):
                    raise RuntimeError("Unsafe Qlib archive member")
                if member.issym() or member.islnk():
                    base = destination.parent if member.issym() else directory
                    if not (base / member.linkname).resolve().is_relative_to(directory):
                        raise RuntimeError("Unsafe Qlib archive link")
            source.extractall(directory)
        marker.write_text(json.dumps(metadata, indent=2) + "\n")
        return
    if marker.is_file():
        if json.loads(marker.read_text()) != metadata:
            raise RuntimeError("Existing Qlib source snapshot differs; use a separate --qlib-dir")
        return
    head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if head != QLIB_COMMIT:
        raise RuntimeError(f"Qlib HEAD is {head}; expected {QLIB_COMMIT}. Use a separate --qlib-dir.")


def qrun_script(python, cann):
    return f'''#!/usr/bin/env bash
# lightgbm_npu_npa1 runner: prepare CANN, then use the original Qlib CLI.
set -eo pipefail
source {shlex.quote(str(cann / 'set_env.sh'))}
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export PATH={shlex.quote(str(Path(python).parent))}:"$PATH"
export PYTHONHASHSEED="${{PYTHONHASHSEED:-0}}"
export PYTHONDONTWRITEBYTECODE=1
exec {shlex.quote(python)} -c 'import sys; from qlib.cli.run import run; sys.argv[0]="qrun"; sys.exit(run())' "$@"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlib-dir", type=Path, default=Path("/workspace/qlib"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--download-data", action="store_true", help="Download standard CN Qlib data if missing")
    parser.add_argument("--skip-python-install", action="store_true", help="Reuse an already installed compatible Python environment")
    parser.add_argument("--skip-build", action="store_true", help="Reuse this checkout's existing NPU lib_lightgbm.so")
    parser.add_argument("--qrun-bin-dir", type=Path, help="Additional command directory; defaults to writable /usr/local/bin")
    parser.add_argument("--rollback", type=Path, metavar="STATE_DIRECTORY")
    args = parser.parse_args()
    if args.rollback:
        installation = Installation.__new__(Installation)
        installation.state = args.rollback.resolve()
        installation.files = json.loads((installation.state / "backup.json").read_text())
        installation.rollback()
        print("Restored native library, Qlib overlays, workflow and qrun wrappers.")
        return
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12: NPU_PYTHON=/path/to/python3.12 bash npu/setup.sh ...")
    if sys.platform != "linux":
        raise RuntimeError("This NPU target requires Linux with Ascend drivers and CANN.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cann = Path(os.environ["CANN_ROOT"]).resolve()
    qlib_dir = args.qlib_dir.resolve()
    ensure_qlib(qlib_dir)
    manifest = json.loads((REPO / "npu/reference/qlib_overlay_manifest.json").read_text())
    for relative, expected in manifest.items():
        target = qlib_dir / relative
        if digest(target) not in {expected["original_sha256"], expected["patched_sha256"]}:
            raise RuntimeError(f"Refusing to overwrite a different Qlib modification: {target}")
    if not args.skip_python_install:
        run([sys.executable, "-m", "pip", "install", "-r", REPO / "npu/requirements-reproduce.txt"])
        run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-c", REPO / "npu/requirements-reproduce.txt", qlib_dir],
            env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB": "0.9.8"})
    library = REPO / "lib_lightgbm.so"
    if not args.skip_build:
        if any(not (REPO / "external_libs" / name / "CMakeLists.txt").exists() for name in ["eigen", "fmt", "nanoarrow"]):
            run(["git", "submodule", "update", "--init", "--recursive", "external_libs/eigen", "external_libs/fmt", "external_libs/fast_double_parser", "external_libs/nanoarrow"], cwd=REPO)
        build = REPO / "build-npu-npa1"
        cmake = package_root("cmake") / "data/bin/cmake"
        run([cmake, "-S", REPO, "-B", build, "-DUSE_NPU=ON", "-DUSE_CUDA=OFF", "-DUSE_GPU=OFF", "-DBUILD_CLI=OFF", "-DCMAKE_BUILD_TYPE=Release", f"-DNPU_TOOLKIT_ROOT={cann}", f"-DNPU_SIMT_ARCH={args.arch}"])
        run([cmake, "--build", build, "-j", args.jobs])
    if not library.is_file():
        raise RuntimeError("NPU build did not produce lib_lightgbm.so")
    if not args.skip_python_install:
        run(["bash", "build-python.sh", "install", "--precompile", "--no-isolation"], cwd=REPO)
    package = package_root("lightgbm")
    native = package / "lib/lib_lightgbm.so"
    state = REPO / "npu/_state" / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    installation = Installation(state)
    try:
        installation.copy(library, native)
        installed_qlib = package_root("qlib")
        for relative, expected in manifest.items():
            source = REPO / "npu/qlib_overlay" / relative
            assert digest(source) == expected["patched_sha256"]
            for target in {qlib_dir / relative, installed_qlib / Path(relative).relative_to("qlib")}:
                if digest(target) not in {expected["original_sha256"], expected["patched_sha256"]}:
                    raise RuntimeError(f"Refusing to replace a different installed Qlib module: {target}")
                installation.copy(source, target)
        workflow = qlib_dir / "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"
        installation.copy(REPO / "npu/workflow_config_lightgbm_Alpha158.yaml", workflow)
        commands = {Path(sysconfig.get_path("scripts")) / "qrun"}
        global_bin = args.qrun_bin_dir
        if global_bin is None and os.access("/usr/local/bin", os.W_OK):
            global_bin = Path("/usr/local/bin")
        if global_bin is not None:
            commands.add(global_bin.resolve() / "qrun")
        for command in commands:
            installation.write(command, qrun_script(sys.executable, cann))
            command.chmod(0o755)
        data = Path.home() / ".qlib/qlib_data/cn_data"
        if not (data / "calendars/day.txt").is_file():
            if not args.download_data:
                raise RuntimeError("CN Qlib data missing. Re-run with --download-data or provision the reference dataset.")
            run([sys.executable, qlib_dir / "scripts/get_data.py", "qlib_data", "--target_dir", data, "--region", "cn"])
        report = {"native_library": str(native), "native_sha256": digest(native), "qlib": str(installed_qlib), "qlib_commit": QLIB_COMMIT, "qrun": sorted(map(str, commands)), "workflow": str(workflow), "data": str(data), "state": str(state)}
        (state / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
        run([sys.executable, REPO / "npu/verify_install.py", "--installation", state / "installation.json"])
    except BaseException:
        installation.rollback()
        raise
    print("Installation complete. Run:\ncd " + str(qlib_dir / "examples") + "\nqrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml")
    print("Backup state:", state)


if __name__ == "__main__":
    main()
