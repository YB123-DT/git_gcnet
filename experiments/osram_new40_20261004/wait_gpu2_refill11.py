"""Persistently wait for safe GPU2 capacity; never duplicate a created run."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    record = json.loads(Path(sys.argv[1]).read_text())
    while True:
        result = subprocess.run(record['command'], cwd=record['snapshot'], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with Path(record['log']).open('a') as stream:
            stream.write(result.stdout)
        if result.returncode == 0:
            return
        if 'Occupied-lane admission rejected before job creation:' not in result.stdout:
            raise SystemExit(result.returncode)
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'capacity waiting; no PointCNN training process created', flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
