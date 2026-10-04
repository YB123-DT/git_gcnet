"""Retry only pre-creation resource admission for the twelfth authorized job."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    record = json.loads(Path(sys.argv[1]).read_text())
    job = record['new_jobs'][-1]
    command = [value.replace('occupied_lane_fill12.py', 'occupied_lane_fill12_v2.py')
               for value in job['command']] + ['--global-reserve-mib', '1792']
    log = Path(job['log'])
    while True:
        result = subprocess.run(command, cwd=record['model_snapshot'], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with log.open('a') as stream:
            stream.write(result.stdout)
        if result.returncode == 0:
            return
        if 'Occupied-lane admission rejected before job creation:' not in result.stdout:
            raise SystemExit(result.returncode)
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'twelfth job waiting for capacity; no training process created', flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
