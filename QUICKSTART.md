# Quick Start Guide

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

### Option A: Move to Sibling Directory (Recommended)

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

### Option B: Keep Inside Repository (Also Works)

You can keep `tau_helper` inside `warrior-tau-bench`. Both setups are fully supported.

### Install Dependencies

```bash
# If you moved it (Option A):
pip install -r tau_helper/requirements.txt

# If it's still inside the repo (Option B):
pip install -r warrior-tau-bench/tau_helper/requirements.txt
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
DEFAULT_MODEL_R=gpt-5-mini
DEFAULT_API_KEY_R=your-openai-api-key
DEFAULT_BASE_URL_R=https://api.openai.com/v1

# Optional: Multi-Agent Architecture (for better scaffolding)
DEFAULT_MODEL_R2=meta-llama-3.3-70b-instruct
DEFAULT_API_KEY_R2=your-friendli-api-key
DEFAULT_BASE_URL_R2=https://api.friendli.ai/serverless/v1

DEFAULT_MODEL_R_JUDGE=deepseek-ai/DeepSeek-R1-0528
DEFAULT_API_KEY_R_JUDGE=your-friendli-api-key
DEFAULT_BASE_URL_R_JUDGE=https://api.friendli.ai/serverless/v1
EOF
```

## Usage Pattern

**Always run from `warrior-tau-bench/` root:**

```bash
# If tau_helper is a sibling directory (recommended):
cd warrior-tau-bench
python ../tau_helper/run.py <command>

# If tau_helper is inside the repository (also works):
cd warrior-tau-bench
python tau_helper/run.py <command>
```

**Note:** All examples below use `../tau_helper/run.py` (sibling setup). If you haven't moved `tau_helper` yet, just replace `../tau_helper/run.py` with `tau_helper/run.py`.

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

# With verbose output
python ../tau_helper/run.py scaffold sec --variation variation_2 --task task_001 --verbose
```

**Features:**
- Maps instruction → SOP chain
- Generates action sequence with placeholders
- Executes actions incrementally
- Fills placeholders from tool outputs
- Outputs ready-to-use task code

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

# 2. Generate the scaffold
python ../tau_helper/run.py scaffold sec --variation variation_2 \
  --instruction "$INSTRUCTION" \
  --task-id task_new_042

# 3. Copy the output to tasks.py

# 4. Test it
python ../tau_helper/run.py execute sec --variation variation_2 --task task_new_042
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
2. Try with `--verbose` to see detailed logs
3. Use `--max-retries 5` for more retry attempts

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

## Pro Tips

1. **Use `--verbose`** with `scaffold` to see the full generation process
2. **Multi-agent mode** (R2 + Judge) significantly improves scaffold quality
3. **Always validate** generated tasks with `execute` before committing
4. **Use `map-sop --compare`** to understand why scaffolds differ from ground truth
5. **Batch evaluate** instructions to ensure consistency across tasks
