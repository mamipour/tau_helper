# Quick Start Guide

**Tau-Helper** accelerates domain development for **[Tau2-Bench](https://github.com/sierra-research/tau2-bench)** by helping you create high-quality, deterministic ground truth tasks.

With perfect tools, rules, and instructions → there's only ONE path to ground truth. Tau-Helper gets you there faster.

> **📌 Cursor IDE Users**: This directory includes `.cursorrules` for AI assistant integration.

## Recommended Setup

**Place `tau_helper` as a sibling directory to `warrior-tau-bench`:**

```
your-workspace/
├── warrior-tau-bench/    ← Your main project
│   ├── domains/
│   ├── ...
└── tau_helper/           ← Helper tools (this folder)
    ├── run.py
    ├── .env
    ├── requirements.txt
    └── ...
```

## Installation (One-Time)

### Move to Sibling Directory

If `tau_helper` is currently inside `warrior-tau-bench`, move it out:

```bash
# From the parent directory
cd /path/to/your-workspace
mv warrior-tau-bench/tau_helper .

# Your structure should now be:
# your-workspace/
#   ├── warrior-tau-bench/
#   └── tau_helper/
```

### Install Dependencies

```bash
pip install -r tau_helper/requirements.txt
```

## Configuration

Create `.env` file in `tau_helper/` directory:

```bash
cd tau_helper
cat > .env << 'EOF'
# Default model (for instruction evaluation)
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_API_KEY=your-openai-api-key
DEFAULT_BASE_URL=https://api.openai.com/v1

# Reasoning model (for SOP mapping and scaffolding)
DEFAULT_MODEL_R=gpt-4o-mini
DEFAULT_API_KEY_R=your-openai-api-key
DEFAULT_BASE_URL_R=https://api.openai.com/v1

# Optional: Multi-Agent Architecture (for better scaffolding)
# R2 Model: Can use stronger OpenAI models
DEFAULT_MODEL_R2=gpt-5-mini
DEFAULT_API_KEY_R2=your-openai-api-key
DEFAULT_BASE_URL_R2=https://api.openai.com/v1

# Judge Model: Reasoning models work well for evaluation
DEFAULT_MODEL_R_JUDGE=deepseek-ai/DeepSeek-R1-0528
DEFAULT_API_KEY_R_JUDGE=your-friendli-api-key
DEFAULT_BASE_URL_R_JUDGE=https://api.friendli.ai/serverless/v1
EOF
```

## Usage Pattern

**Always run from `warrior-tau-bench/` root:**

```bash
cd warrior-tau-bench
python ../tau_helper/run.py <command>
```

## Quick Examples

### 1. Check Installation

```bash
cd warrior-tau-bench
python ../tau_helper/run.py info
```

### 2. List Available Domains

```bash
cd warrior-tau-bench
python ../tau_helper/run.py list-domains
python ../tau_helper/run.py list-domains --domain sec
```

### 3. Evaluate Instruction Quality

```bash
cd warrior-tau-bench
python ../tau_helper/run.py evaluate "You are analyzing Tesla's financial data"
```

**Score:** 0-100 (higher = more user-facing and non-procedural)

### 4. Map Instruction to SOPs

```bash
cd warrior-tau-bench
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001 --compare
```

**Output:**
- SOP chain with confidence
- Ambiguity detection
- Missing information
- Suggested instruction fix

### 5. Generate Complete Task (Scaffolding) ⭐

```bash
cd warrior-tau-bench

# Generate from existing task
python ../tau_helper/run.py scaffold sec --variation variation_2 --task task_001

# Generate from custom instruction
python ../tau_helper/run.py scaffold sec --variation variation_2 \
  --instruction "Extract Apple's last 3 fiscal years of financial statements"

# With verbose output (shows execution details)
python ../tau_helper/run.py scaffold sec --variation variation_2 --task task_001 --verbose
```

**Features:**
- Maps instruction → SOP chain
- **Iterative execution**: Generates ONE action at a time, executes immediately
- **Real values only**: No placeholders! Uses actual execution results
- Adapts based on execution feedback (errors, results)
- Outputs ready-to-use task code with real values

### 6. Execute Task Actions (Testing)

```bash
cd warrior-tau-bench

# List all tasks
python ../tau_helper/run.py execute sec --variation variation_2 --list-tasks

# Show task details
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001 --show

# Execute specific action
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001 --action 0

# Execute all actions
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001
```

### 7. Analyze Agent Logs

```bash
cd warrior-tau-bench

# Show statistics
python ../tau_helper/run.py agent-logs sec --variation variation_2 --stats

# Analyze specific task
python ../tau_helper/run.py agent-logs sec --variation variation_2 --task task_001

# Compare ground truth vs agent
python ../tau_helper/run.py agent-logs sec --variation variation_2 --task task_001 --compare
```

## Common Workflows

### Creating a New Task

```bash
cd warrior-tau-bench

# 1. Write your instruction
INSTRUCTION="You are a financial analyst. Extract Tesla's income statements for the last 3 years."

# 2. Generate the scaffold (executes iteratively with real values)
python ../tau_helper/run.py scaffold sec --variation variation_2 \
  --instruction "$INSTRUCTION" \
  --task-id task_new_042 \
  --verbose

# 3. Copy the output to tasks.py
# The code already contains real values - ready to use!

# 4. Validate it
uv run alignerr validate --domain sec --variation variation_2 --task-id task_new_042
```

### Improving Instruction Quality

```bash
cd warrior-tau-bench

# 1. Evaluate current instruction
python ../tau_helper/run.py evaluate "Call get_financials() for AAPL"
# → Low score (procedural)

# 2. Get suggestion
python ../tau_helper/run.py evaluate "Call get_financials() for AAPL" --json

# 3. Use improved version
python ../tau_helper/run.py evaluate "You are analyzing Apple's quarterly earnings"
# → High score (user-facing)
```

### Debugging SOP Mapping

```bash
cd warrior-tau-bench

# 1. Check what SOPs are predicted
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001

# 2. Compare with actual task actions
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001 --compare

# 3. If ambiguous, use suggested fix
```

## Troubleshooting

### "No domains found"

Make sure you're running from `warrior-tau-bench/` root:
```bash
pwd  # Should show: .../warrior-tau-bench
ls domains/  # Should list: sec, airline, etc.
```

### "API key not provided"

Check `.env` file exists in `tau_helper/`:
```bash
ls -la tau_helper/.env
cat tau_helper/.env | grep API_KEY
```

### "Module not found"

Install dependencies:
```bash
pip install -r tau_helper/requirements.txt
```

### Scaffold generation fails

1. Check if multi-agent models are misconfigured in `.env`
2. Try with `--verbose` to see detailed execution logs
3. Use `--max-actions 150` to allow more actions (default: 100)
4. Check database state if execution gets stuck

## Key Features

| Feature | Command | Use Case |
|---------|---------|----------|
| **Instruction Evaluation** | `evaluate` | Check if instruction is user-facing |
| **SOP Mapping** | `map-sop` | Map instruction to SOP chain |
| **Task Scaffolding** | `scaffold` | Generate complete task from instruction |
| **Action Execution** | `execute` | Test task actions incrementally |
| **Agent Log Analysis** | `agent-logs` | Analyze agent evaluation results |

## Next Steps

- **Full Documentation**: See `README.md`
- **All Commands**: Run `python ../tau_helper/run.py --help`
- **Domain-Specific**: Run commands with `--help` (e.g., `scaffold --help`)
- **Cursor IDE Users**: See `.cursorrules` for AI assistant integration

## Pro Tips

1. **Use `--verbose`** with `scaffold` to see the full generation process
2. **Multi-agent mode** (R2 + Judge) significantly improves scaffold quality
3. **Always validate** generated tasks with `execute` before committing
4. **Use `map-sop --compare`** to understand why scaffolds differ from ground truth
5. **Batch evaluate** instructions to ensure consistency across tasks
