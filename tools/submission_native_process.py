"""Native inference in a persistent Python process isolated from YOLO globals.

The constructor snapshots the environment before YOLO is imported/constructed.
Only stdlib is imported here; numerical libraries are imported in ``native`` or
in the child after its captured environment is in effect. IPC uses ordinary
stdin/stdout pipes plus files in this client's private temporary directory.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROTOCOL = 'solar-native-v1'


class NativeProcessClient:
    def __init__(self, checkpoint, device='cuda:0', batch=4, *, _command=None):
        self._environment = dict(os.environ)
        self.checkpoint = str(checkpoint)
        self.device = str(device)
        self.batch = int(batch)
        if self.batch <= 0:
            raise ValueError('Native batch must be positive')
        self._command = list(_command) if _command is not None else [
            sys.executable, '-u', str(Path(__file__).with_name('submission_inference.py')),
            '--native-server', '--checkpoint', self.checkpoint,
            '--device', self.device, '--batch', str(self.batch)]
        self._process = None
        self._temporary = None
        self._counter = 0
        self._closed = False

    def _start(self):
        if self._closed:
            raise RuntimeError('Native inference client is closed')
        if self._process is None:
            self._temporary = tempfile.TemporaryDirectory(prefix='solar_native_process_')
            try:
                self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
                    env=self._environment)
            except BaseException:
                self._temporary.cleanup()
                self._temporary = None
                raise

    def _response(self, request_id):
        diagnostics = []
        while True:
            line = self._process.stdout.readline()
            if not line:
                code = self._process.poll()
                detail = '; '.join(diagnostics[-3:])
                raise RuntimeError(f'Native inference worker ended without a response '
                    f'(exit code {code})'+(f': {detail}' if detail else ''))
            try:
                response = json.loads(line)
            except ValueError:
                diagnostics.append(line.rstrip()[:300])
                continue
            if not isinstance(response, dict) or response.get('protocol') != PROTOCOL:
                diagnostics.append(line.rstrip()[:300])
                continue
            if response.get('id') != request_id:
                raise RuntimeError('Native inference worker response ID differs from the request')
            if response.get('status') != 'ok':
                raise RuntimeError('Native inference worker failed: '
                    f"{response.get('error_type', 'Error')}: {response.get('message', '')}")
            return response

    def native(self, image_id, image, *, file_name=None, image_sha256='', identity='submission'):
        import numpy as np
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (2048, 2048):
            raise ValueError('Original grayscale uint8 2048-square image required')
        self._start()
        self._counter += 1
        request_id = self._counter
        root = Path(self._temporary.name)
        pixels = root/f'{request_id}.npy'
        output = root/f'{request_id}.json'
        request = dict(protocol=PROTOCOL, command='native', id=request_id,
            input_path=str(pixels), output_path=str(output), image_id=str(image_id),
            file_name=file_name or f'{image_id}.jpeg', image_sha256=str(image_sha256), identity=str(identity))
        try:
            np.save(pixels, image, allow_pickle=False)
            self._process.stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
            self._process.stdin.flush()
            response = self._response(request_id)
            if response.get('output_path') != str(output):
                raise RuntimeError('Native inference worker returned an unexpected output file')
            record = json.loads(output.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise RuntimeError('Native inference worker did not return a JSON object')
            return record
        except BaseException:
            self.close()
            raise
        finally:
            for path in (pixels, output, output.with_name(output.name+'.pending')):
                if path.is_file():
                    path.unlink()

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write(json.dumps(dict(protocol=PROTOCOL, command='close'))+'\n')
                        process.stdin.flush()
                    except (OSError, ValueError):
                        pass
                    try:
                        process.stdin.close()
                    except (OSError, ValueError):
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                else:
                    process.wait()
                for handle in (process.stdin, process.stdout):
                    if handle is not None:
                        try:
                            handle.close()
                        except (OSError, ValueError):
                            pass
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
