# Tau-Helper: Warrior Tau-Bench Helper Library

A comprehensive toolkit for creating, validating, and managing tasks, tools, and SOPs across all Warrior Tau-Bench domains.

> **📌 Cursor IDE Users**: This directory includes `.cursorrules` for AI assistant integration. The file is automatically loaded by Cursor to help you work with this library.

## Installation

**Setup:** Place `tau_helper` as a sibling directory to `warrior-tau-bench`:

```
your-workspace/
├── warrior-tau-bench/    ← Main project
└── tau_helper/           ← Helper tools
```

**Install:**

```bash
# Move tau_helper out of the repository if needed
cd /path/to/your-workspace
mv warrior-tau-bench/tau_helper .

# Install dependencies
pip install -r tau_helper/requirements.txt
```

**Important**: Always run `tau_helper` commands from the `warrior-tau-bench/` root directory. The tool uses the current working directory to find domains.

## Configuration

Create `.env` in `tau_helper/` directory:

```env
# Default model (used for instruction evaluation)
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_API_KEY=your-api-key
DEFAULT_BASE_URL=https://api.openai.com/v1

# Reasoning model (used for SOP mapping, scaffolding)
DEFAULT_MODEL_R=gpt-5-mini
DEFAULT_API_KEY_R=your-api-key
DEFAULT_BASE_URL_R=https://api.openai.com/v1

# Multi-Agent Architecture (OPTIONAL)
# If both R2 and JUDGE are configured, scaffolding uses multi-agent mode:
# - R and R2 generate scaffolds independently
# - If they differ, JUDGE picks the best one
# - Improves quality through model consensus

DEFAULT_MODEL_R2=meta-llama-3.3-70b-instruct
DEFAULT_API_KEY_R2=your-r2-api-key
DEFAULT_BASE_URL_R2=https://api.friendli.ai/serverless/v1

DEFAULT_MODEL_R_JUDGE=deepseek-ai/DeepSeek-R1-0528
DEFAULT_API_KEY_R_JUDGE=your-judge-api-key
DEFAULT_BASE_URL_R_JUDGE=https://api.friendli.ai/serverless/v1
```

## Features

1. **Instruction Evaluation** - Score instructions 0-100 for user-facing, non-procedural quality
2. **SOP Chain Mapper** - Map instructions to SOPs, detect ambiguity, suggest fixes
3. **Task Scaffolder** ⭐ NEW - Generate complete, executable tasks from instructions
4. **Action Executor** - Execute and debug domain task actions
5. **Agent Log Reader** - Analyze agent evaluation results

## Usage

**Always run from `warrior-tau-bench/` root:**

```bash
cd /path/to/warrior-tau-bench
python ../tau_helper/run.py <command>
```

## Quick Command Reference

### General

```bash
# Show info and list domains
python ../tau_helper/run.py info
python ../tau_helper/run.py list-domains
python ../tau_helper/run.py list-domains --domain sec
```

### Instruction Evaluation

```bash
# Evaluate single instruction
python ../tau_helper/run.py evaluate "You are analyzing financial data"

# Batch evaluation from file
python ../tau_helper/run.py evaluate -f instructions.txt --batch
```

### SOP Chain Mapping

```bash
# Map instruction from task
python ../tau_helper/run.py map-sop <domain> --variation <variation> --task <task_id>
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001

# Compare predicted vs actual actions
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001 --compare

# Map custom instruction
python ../tau_helper/run.py map-sop sec --variation variation_2 --instruction "Calculate WACC for Tesla"

# Use specific model (override DEFAULT_MODEL_R)
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001 --model gpt-4o
```

**SOP Mapper Output:**
- Primary SOP chain with confidence score
- Alternative interpretations (if ambiguous)
- Ambiguity explanation
- Missing information needed to disambiguate
- **Suggested instruction fix** (non-procedural, minimal changes)

### Task Scaffolder

```bash
# Generate complete task from instruction
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "Your instruction"

# Generate task from existing task (re-scaffold)
python ../tau_helper/run.py scaffold <domain> --variation <variation> --task <task_id>

# Generate without executing (placeholders remain)
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "..." --no-execute

# Specify custom task ID
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "..." --task-id task_new_001

# Configure max retries (default: 3)
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "..." --max-retries 5

# Show detailed progress (or auto-show on error)
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "..." --verbose

# Examples
python ../tau_helper/run.py scaffold sec --variation variation_2 --instruction "Extract Apple's financial data"
python ../tau_helper/run.py scaffold sec --variation variation_2 --task task_001
python ../tau_helper/run.py scaffold sec --variation variation_2 --task task_001 --verbose
```

**Task Scaffolder Features:**
- Maps instruction to SOP chain (reuses `map-sop` logic)
- Generates action sequence with placeholders (`{{company_ticker}}`, `{{spreadsheet_id}}`)
- Executes actions incrementally to fill placeholders from:
  - LLM-provided defaults
  - Previous tool outputs
  - Deterministic calculations
- **Automatic retry logic**: Retries up to 3 times (configurable) if generation fails
- **Smart transformation detection**: LLM marks complex data transformations with `{{TRANSFORM: ...}}`
- Shows progress and identifies missing values
- Provides helpful guidance when manual intervention is needed
- Outputs complete task in `tasks.py` format (ready to copy-paste)
- **Model transparency**: Shows which model(s) generated the output:
  - `Model: Single Model` - Default single model mode
  - `Model: Consensus (R + R2)` - Both models agreed
  - `Model: Judge → Model R` or `Model: Judge → Model R2` - Judge resolved disagreement
  - `Model: Model R (R2 failed)` - Fallback to R after R2 failure

**🚀 Advanced Feature: LLM-Powered Data Transformations**

When a task requires complex data manipulation (e.g., merging multiple data sources, extracting specific fields, date parsing), the scaffolder can **automatically generate and execute transformation code**.

**How it works:**
1. The scaffolding LLM detects when a transformation is needed
2. It marks the placeholder with transformation syntax: `{{TRANSFORM: merge income_statements, balance_sheets by period}}`
3. A second LLM call generates Python code for the transformation
4. The code is executed in a safe environment with available context
5. The result is used to fill the placeholder

**Example Transformation:**
```
Instruction: "Extract the last 3 fiscal years"
→ LLM generates code to parse dates, sort by year, and take top 3
→ Code executes using actual data from context
→ Result: ["2023-FY", "2022-FY", "2021-FY"]
```

**🤖 Advanced Feature: Multi-Agent Architecture**

When `DEFAULT_MODEL_R2` and `DEFAULT_MODEL_R_JUDGE` are configured in `.env`, scaffolding automatically uses **multi-agent mode** for improved quality:

**How it works:**
1. **Parallel Generation**: Both R and R2 models generate scaffolds independently
2. **Consensus Check**: If both models produce the same scaffold → use it (high confidence)
3. **Judge Resolution**: If they differ → JUDGE model evaluates both and picks the best one
4. **Fault Tolerance**: If R2 fails → automatically fall back to R model

**Benefits:**
- **Higher quality**: Multiple perspectives catch edge cases
- **Reduced errors**: Model consensus validates correctness
- **Automatic fallback**: System degrades gracefully if one model fails

**Example Output (with `--verbose`):**
```
🤖 Multi-Agent Mode: Generating with both R and R2 models...
  → Generating with Model R...
    ✓ Model R: 12 actions
  → Generating with Model R2...
    ✓ Model R2: 13 actions
  ⚠️  Models disagree - invoking judge...
🔶 Judge selected: Model R2 scaffold

Step 3: Executing actions to fill placeholders...
  [... detailed execution logs ...]

✅ Scaffold generation complete!
Copy the above code to add this task to tasks.py

Model: Judge → Model R2
```

**Output Control:**
- **Default mode**: Shows only the generated task code and model summary
- **With `--verbose`**: Shows full workflow including multi-agent process and execution logs
- **On error**: Automatically shows verbose logs to help debug

**Recommended Models:**
- **R2**: `meta-llama-3.3-70b-instruct`, `deepseek-ai/DeepSeek-V3`, or other instruction-tuned models
- **Judge**: `deepseek-ai/DeepSeek-R1-0528` (reasoning model works well for evaluation)

**Note**: Models must support structured JSON output. If R2 consistently fails with parsing errors, consider using a different model or disabling multi-agent mode by removing the R2/JUDGE configuration.

**This means the scaffolder can handle agent-level complexity!** It leverages LLMs to write and execute transformation logic just like agents do.

### Action Executor

```bash
# List tasks
python ../tau_helper/run.py execute <domain> --variation <variation> --list-tasks

# Show task details
python ../tau_helper/run.py execute <domain> --variation <variation> --task <task_id> --show

# Execute specific action
python ../tau_helper/run.py execute <domain> --variation <variation> --task <task_id> --action <action_index>

# Execute all actions in sequence
python ../tau_helper/run.py execute <domain> --variation <variation> --task <task_id>

# Show database state
python ../tau_helper/run.py execute <domain> --variation <variation> --task <task_id> --db-state

# Reset database
python ../tau_helper/run.py execute <domain> --variation <variation> --task <task_id> --reset-db

# Examples
python ../tau_helper/run.py execute sec --variation variation_2 --list-tasks
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001 --action 0
python ../tau_helper/run.py execute airline --variation variation_1 --task task_003
```

### Agent Log Reader

```bash
# Show statistics
python ../tau_helper/run.py agent-logs <domain> --variation <variation> --stats

# Analyze specific task
python ../tau_helper/run.py agent-logs <domain> --variation <variation> --task <task_id>

# Compare ground truth vs agent
python ../tau_helper/run.py agent-logs <domain> --variation <variation> --task <task_id> --compare

# Find problematic tasks
python ../tau_helper/run.py agent-logs <domain> --variation <variation> --task <task_id> --errors-only

# Custom agent.json location
python ../tau_helper/run.py agent-logs <domain> --variation <variation> --agent-json path/to/agent.json --stats

# Examples
python ../tau_helper/run.py agent-logs sec --variation variation_2 --stats
python ../tau_helper/run.py agent-logs sec --variation variation_2 --task task_072 --compare
```

## Understanding Instruction Quality

### High Quality (80-100) ✅
User-facing, non-procedural:
```
"You are Sarah Chen, a financial analyst. You need to analyze 
Tesla's Q4 2023 financial performance for your investment report."
```

**Characteristics:**
- Describes WHAT the user wants, not HOW to do it
- Natural, conversational language
- Role and context provided
- No function names or technical details

### Low Quality (0-40) ❌
Procedural:
```
"First, call get_balance_sheet() for Tesla, then execute 
calculate_wacc() with the parameters, and save the results."
```

**Problems:**
- Step-by-step instructions
- Function names and API calls
- Implementation details

## Python API

```python
from tau_helper.llm import get_llm_client, get_reasoning_llm
from tau_helper.evaluator import InstructionEvaluator

# Instruction evaluation
llm = get_llm_client()
evaluator = InstructionEvaluator(llm)
result = evaluator.evaluate("You are analyzing financial data")
print(f"Score: {result.score}/100")

# SOP mapping
from tau_helper.sop_mapper import SOPMapper
llm_r = get_reasoning_llm()
mapper = SOPMapper(llm_r, "sec", "variation_2")
mapping = mapper.map_instruction("Extract AAPL financials")

# Action execution
from tau_helper.action_executor import ActionExecutor
executor = ActionExecutor("sec", "variation_2")
result = executor.execute_action("task_001", 0)

# Agent log reading
from tau_helper.agent_log_reader import AgentLogReader
reader = AgentLogReader("sec", "variation_2")
stats = reader.analyze_stats()
```

## Common Workflows

### 1. Validate Task Instructions
```bash
# Check if task_001 instruction is clear
python ../tau_helper/run.py map-sop sec --variation variation_2 --task task_001

# If ambiguous, you'll see:
# - Ambiguity explanation
# - Missing information
# - Suggested instruction fix (copy-paste ready)
```

### 2. Debug Failed Task
```bash
# See what agent did wrong
python ../tau_helper/run.py agent-logs sec --variation variation_2 --task task_072 --compare

# Execute actions manually to reproduce
python ../tau_helper/run.py execute sec --variation variation_2 --task task_072 --show
python ../tau_helper/run.py execute sec --variation variation_2 --task task_072 --action 0
python ../tau_helper/run.py execute sec --variation variation_2 --task task_072 --db-state
```

### 3. Test New Task
```bash
# Execute all actions
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001

# Check final database state
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001 --db-state

# Reset and retry
python ../tau_helper/run.py execute sec --variation variation_2 --task task_001 --reset-db
```

## Architecture

```
tau_helper/
├── cli.py                   # Main CLI entry point
├── llm.py                   # LLM client (DEFAULT_MODEL, DEFAULT_MODEL_R)
├── evaluator.py             # Instruction evaluation
├── sop_mapper.py            # SOP chain mapping + ambiguity detection
├── action_executor.py       # Domain-agnostic action executor
├── agent_log_reader.py      # Agent log analysis
├── run.py                   # Wrapper for easy execution
├── .env                     # Configuration
└── requirements.txt         # Dependencies
```

## Documentation

- **README.md** - This file (command reference)
- **QUICKSTART.md** - 5-minute quick start guide
- **.cursorrules** - Cursor IDE integration (auto-loaded for AI assistance)

## Support

```bash
# Get help
python ../tau_helper/run.py --help
python ../tau_helper/run.py <command> --help
```

## Version

Current version: **0.2.0**
