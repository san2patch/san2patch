# !/bin/bash
pgrep -f 'run_test.py' | xargs -n 1 --no-run-if-empty kill -9