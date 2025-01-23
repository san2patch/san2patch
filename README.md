# San2Patch
Logs In, Patches Out: Automated Vulnerability Repair via Tree-of-Thought LLM Analysis


# Install
```bash
poetry install
```

# Setup Experiment Framework
- Recommended to use Aim (https://github.com/aimhubio/aim)
- Run Aim either by using the `docker compose up` command, or
- Execute `aim up` and `aim server` commands separately to set up the server


# Setup dataset
- Setup dataset directories in .env file. (see .env_example)

```bash
poetry shell
python ./scripts/run_dataset.py VulnlLoc all
python ./scripts/run_dataset.py San2Patch all
python ./scripts/run_dataset.py FinalTest aggregate
```

# Usage
- Setup API keys in .env file (see .env_example)

```bash
poetry shell
# Print help
python ./scripts/run_test.py Final run-patch --help
Usage: run_test.py [[Final]] run-patch [OPTIONS] [NUM_WORKERS]

Options:
  --experiment-name TEXT      Experiment name
  --retry-cnt INTEGER         Number of retries
  --max-retry-cnt INTEGER     Maximum number of retries for accumulative
                              experiments. If 0, it will retry indefinitely
                              until retry-cnt is reached
  --model TEXT                Model name to run
  --version TEXT              Prompting version
  --docker-id TEXT            Docker ID to run
  --vuln-ids TEXT             Comma separated list of vuln_ids to run
  --select-method TEXT        Select method for ToT (sample, greedy)
  --temperature-setting TEXT  Temperature setting for ToT (zero, low, medium,
                              high)
  --raise-exception           Raise exception
  --help                      Show this message and exit.

# Run example
python ./scripts/run_test.py Final run-patch 1 --experiment-name tot_sample_medium_high_vulnloc_experiment --retry-cnt 5 --max-retry-cnt 5 --model gpt-4o --version tot --select-method sample --vuln-ids vulnloc  --temperature-setting medium_high
```

# Note
- The appropriate LLM API keys must be set in the .env. Please check ".env_example" file.