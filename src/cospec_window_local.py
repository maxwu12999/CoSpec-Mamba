#!/usr/bin/env python3
"""Desktop/WSL launcher for the unchanged cospec_window_lr001_v1 package.

Put this file in src beside cospec_window_lr001_v1, then run it using the
existing convmamba environment. Default concurrency is two. The original
controller is stopped through its own STOP flag, never a global kill command.
Only the initial idle-GPU memory threshold changes in memory: 90% free becomes
at least 3072 MiB (and the original headroom). Measured peak reservations,
OOM handling, training, masks, checkpoints and protocol IDs are unchanged.
"""
import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
EXPECTED_CONTROLLER_SHA256 = 'e47cb506348019677c06e57c09e0bdc9bcd731f4089821f68c83b3a2cf7cb340'
MIN_FREE_MIB = 3072.
OLD_GATE = "if not gpu_processes(args.gpu) and gpu['free_mib']>=max(gpu['total_mib']*.9,args.headroom_mib+1024):"
NEW_GATE = "if not gpu_processes(args.gpu) and gpu['free_mib']>=max(3072.,args.headroom_mib+1024):"
OLD_WAIT = "snapshot(state='preflight',phase='等待空闲显卡进行显存预检')"
NEW_WAIT = '''snapshot(state='preflight',phase=(f"等待显存预检：空闲{gpu['free_mib']:.0f}MiB，"
                         f"需要至少{max(3072.,args.headroom_mib+1024):.0f}MiB且无计算进程"))'''


def patched_source(core):
    raw = (core / 'scheduler.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_CONTROLLER_SHA256:
        raise ValueError('原调度器版本不同；未修改源码、配置或检查点。请使用本对话提供的补跑代码包。')
    source = raw.decode('utf-8')
    if source.count(OLD_GATE) != 1 or source.count(OLD_WAIT) != 1:
        raise ValueError('找不到唯一预检门槛，未执行替换')
    return source.replace(OLD_GATE, NEW_GATE).replace(OLD_WAIT, NEW_WAIT)


def controller(core):
    runtime = types.ModuleType('cospec_window_desktop_controller')
    runtime.__file__ = str(core / 'scheduler.py')
    exec(compile(patched_source(core), runtime.__file__, 'exec'), runtime.__dict__)
    return runtime


def is_our_controller(output):
    from common import read_json
    marker = read_json(output / 'runtime_extensions/local_preflight/source.json', {})
    if marker.get('launcher_sha256') != hashlib.sha256(SELF.read_bytes()).hexdigest():
        return False
    owner = marker.get('controller_identity', {})
    try:
        fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        command = Path(f"/proc/{owner['pid']}/cmdline").read_bytes().split(b'\0')
        return (fields[19] == owner['start_ticks'] and b'--controller' in command
                and str(SELF).encode() in command)
    except (KeyError, OSError, IndexError):
        return False


def stop_previous(output, timeout=60.):
    from common import locked, read_json
    config = read_json(output / 'config.json', {})
    if config and config.get('version') != 'cospec_window_lr001_v1':
        raise ValueError('该输出属于其他实验，未写入停止标记')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'launch.lock').open('a+') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        if not locked(output / 'queue.lock') or is_our_controller(output):
            return
        (output / 'STOP').write_text('desktop launcher: save own queue and restart unchanged protocol\n')
        print('正在让原预检退出，随后自动接续同一补跑队列……', flush=True)
        deadline = time.monotonic() + timeout
        while locked(output / 'queue.lock'):
            if time.monotonic() >= deadline:
                raise RuntimeError('原队列仍在保存。稍后重跑本命令即可；没有强制结束进程。')
            time.sleep(.2)


def run_controller(core, argv):
    runtime = controller(core)
    original_prepare = runtime.prepare

    def prepare(args):
        # main() owns queue.lock when calling prepare; provenance cannot be
        # written by a second controller that failed to acquire that lock.
        from common import identity, source_hash, write_json
        metadata = dict(kind='cospec_window_desktop_preflight_v1',
            controller_identity=identity(), launcher_sha256=hashlib.sha256(SELF.read_bytes()).hexdigest(),
            original_controller_sha256=EXPECTED_CONTROLLER_SHA256,
            scientific_source_sha256=source_hash(),
            preflight_minimum_free_mib=MIN_FREE_MIB,
            require_no_compute_processes=True, measured_peak_admission_unchanged=True,
            training_and_checkpoint_protocol_unchanged=True,
            launcher_source=SELF.read_text(encoding='utf-8'),
            runtime_controller_source=patched_source(core))
        write_json(args.output.expanduser().resolve() / 'runtime_extensions/local_preflight/source.json', metadata)
        return original_prepare(args)

    runtime.prepare = prepare
    previous = sys.argv
    sys.argv = [str(SELF)] + argv
    try:
        return runtime.main()
    finally:
        sys.argv = previous


def start(core, argv):
    import control
    import scheduler
    from window_protocol import protect_output, verify_science
    verify_science()
    patched_source(core)
    if '-h' in argv or '--help' in argv:
        print('本地启动：python cospec_window_local.py [--max-parallel 2] [--headless]\n'
              '默认使用src内的原代码和原输出，自动接续同一12项队列。')
        return 0
    # Headless is a viewer option handled by the existing control entry point.
    options = scheduler.parser().parse_args(['--max-parallel', '2'] + [v for v in argv if v != '--headless'])
    if not 1 <= options.max_parallel <= 10:
        raise ValueError('并行上限必须1至10')
    protect_output(options.output, options.reference, options.parent_output)
    stop_previous(options.output.expanduser().resolve())
    original_popen = subprocess.Popen

    def popen(command, *args, **kwargs):
        # Only redirect this package's controller; worker launch arguments,
        # detached-session handling, environment and logs remain original.
        command = list(command)
        target = str(core / 'scheduler.py')
        if target in command:
            index = command.index(target)
            command[index:index + 1] = [str(SELF), '--controller', '--core', str(core)]
        return original_popen(command, *args, **kwargs)

    previous_argv = sys.argv
    subprocess.Popen = popen
    sys.argv = [str(SELF), 'start', '--max-parallel', '2'] + argv
    try:
        return control.main()
    finally:
        subprocess.Popen = original_popen
        sys.argv = previous_argv


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--core', type=Path, default=HERE / 'cospec_window_lr001_v1')
    local, forwarded = parser.parse_known_args(argv)
    core = local.core.expanduser().resolve()
    if not (core / 'scheduler.py').is_file():
        raise FileNotFoundError('把cospec_window_local.py直接放在src，与cospec_window_lr001_v1文件夹并列。')
    sys.path.insert(0, str(core))
    return run_controller(core, forwarded) if local.controller else start(core, forwarded)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print('本地启动未完成：' + str(error), file=sys.stderr)
        raise SystemExit(1)
