# Resampling Implementation

This repository includes a standalone Python script that illustrates a resampling approach for evaluating large‑language‑model (LLM) agents on a list of tasks. The script does not rely on any other pieces of AgentBench—it is self‑contained and can be used with minimal setup.

# 🎯 What It Does

The script reads a JSON file containing tasks, runs a baseline and a resampling pass over those tasks, and reports basic metrics. In the baseline pass the model samples a fixed number of actions and picks the majority vote. In the resampling pass it does the same, but if the initial set of actions is too uncertain (based on simple disagreement), it draws a fresh set of actions and uses those instead.

# 🚀 Quick Start

1. Prepare your data. Create a JSON file with a list of tasks. The script doesn’t inspect the task contents—only the number of tasks matters—so each entry can be any JSON value.

2. Run the script. At minimum you need to specify the path to your data file and an API key (which is stored but not used when using the stub models):

```bash
python run_resampling_on_dataset_modified.py \
  --input path/to/your_dataset.json \
  --api-key dummykey
```

The script runs one pass without resampling and one with resampling. When finished it prints a summary to the terminal and writes a resampling_modified_summary.json file in the working directory.

Check the results. Look at the console output for pass rate, average uncertainty, resample rate, and latency statistics. The JSON summary file contains the same information for programmatic consumption.

# ⚙️ Key Parameters

You can customise the behaviour of the resampling experiment via the following command‑line options:

**`--threshold`** – uncertainty threshold between 0 and 1 (default: 0.40). Lower values trigger resampling more often.

**`--k`** – number of extra samples to generate in each phase (k + 1 total candidates). Default: 7.

**`--p-correct-base`** – probability that the base stub returns the correct action. Default: 0.55.

**`--p-correct-resample`** – probability that the resampling stub returns the correct action. Default: 0.90.

**`--temperature`** – sampling temperature (unused in the stubs). Default: 0.7.

**`--limit`** – limit the number of tasks processed. Use this to run a quick test on a subset of your dataset.

**`--seed`** – random seed for reproducible runs. Default: 42.

All parameters are optional. The defaults should give reasonable behaviour, but feel free to experiment.

# 📋 Example Usage

Run the script on a dataset called test_dataset.json with a lower uncertainty threshold and fewer samples:

```bash
python run_resampling_on_dataset_modified.py \
  --input test_dataset.json \
  --api-key anystring \
  --threshold 0.30 \
  --k 5
```

This command processes all tasks in test_dataset.json, triggers resampling whenever the initial set of actions is more than 30 % uncertain, and reports the results. You will see output similar to:

```vbnet
Processed 100 tasks from test_dataset.json
=== Resampling OFF ===
pass_rate: 0.530000
avg_uncertainty: 0.290000
resample_rate: 0.000000
latency_mean: 0.003200
latency_p95: 0.005800
=== Resampling ON ===
pass_rate: 0.780000
avg_uncertainty: 0.290000
resample_rate: 0.150000
latency_mean: 0.004500
latency_p95: 0.009000
=== Delta (ON - OFF) ===
pass_rate_delta: +0.250000
latency_delta: +0.001300
Summary written to resampling_modified_summary.json
```
# 🧠 How It Works

1. Load tasks. The script reads the provided JSON file and counts the number of tasks. It does not use the task contents when using the stub models.

2. Create stub models. Two simple functions simulate an LLM: the “base” model returns the correct action with probability p_correct_base, and the “resample” model returns the correct action with probability p_correct_resample.

3. Sample actions. For each task, the baseline pass samples k + 1 actions from the base model and picks the most common. The resampling pass does the same initial sampling, computes disagreement among the samples (uncertainty), and if the uncertainty exceeds threshold, it samples k + 1 actions from the resample model instead.

4. Compute metrics. After processing all tasks, the script calculates statistics for both passes: pass rate, average uncertainty, resample rate, mean latency, and 95th percentile latency.

5. Save output. Results are printed and saved in resampling_modified_summary.json for easy reference.

# 🔧 Using a Real LLM

The script uses stub models by default. To evaluate a real LLM (e.g. GPT‑4, Claude, or Gemini), provide your own functions that call the model’s API and return the <action>...</action> text from the response. Make sure they accept a prompt and a temperature argument and return a string. Replace calls to make_stub_llm with your own functions and pass them into run_trial_modified.

# 📄 License

This script is provided under the same license as the AgentBench repository. Use it for research and experimentation, and test thoroughly before integrating into production.
