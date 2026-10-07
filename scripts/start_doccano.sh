#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DOCCANO_HOME="$PWD/.doccano"
export DEBUG=False
mkdir -p "$DOCCANO_HOME"
exec .venv-doccano/bin/python - <<'PY'
import os
import subprocess
from pathlib import Path
import sys
import time
from urllib.request import urlopen

home = Path(os.environ['DOCCANO_HOME'])
commands = {
    'web': [sys.executable, '-c',
            'import backend.cli; import django; django.setup(); '
            'from config.wsgi import application; from waitress import serve; '
            'serve(application, host="127.0.0.1", port=8000, threads=4)'],
    'task': [sys.executable, '-c',
             'import backend.cli as cli; import django; django.setup(); '
             'cli.app.worker_main(["--app=config", "--workdir=" + cli.base, '
             '"worker", "--loglevel=info", "--pool=solo", "--concurrency=1"])'],
}
for name, command in commands.items():
    pid_file = home / f'{name}.pid'
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            continue
        except (ProcessLookupError, ValueError):
            pass
    with (home / f'{name}.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    pid_file.write_text(str(process.pid))
for _ in range(30):
    try:
        with urlopen('http://127.0.0.1:8000', timeout=2) as response:
            if response.status == 200:
                print('Doccano is running: http://127.0.0.1:8000')
                print(f'Logs and database: {home}')
                break
    except OSError:
        time.sleep(1)
else:
    sys.exit(f'Server did not start. Check {home / "web.log"}')
PY
