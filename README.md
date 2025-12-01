# Tau-Helper: Tau-Bench Helper Library

A comprehensive toolkit for creating, validating, and managing tasks, tools, and SOPs for **[Tau2-Bench](https://github.com/sierra-research/tau2-bench)** domains.

## Philosophy

When you have the **perfect combination** of:
- ✅ A complete, well-designed **tool set**
- ✅ Clear, comprehensive **domain rules**
- ✅ High-quality, unambiguous **instructions**

...there should be **only ONE path** to achieve the ground truth.

**Tau-Helper accelerates domain development** by helping you:
1. **Validate instructions** for quality and clarity
2. **Map instructions to SOPs** and detect ambiguity
3. **Generate ground truth tasks** automatically through iterative execution
4. **Debug and verify** task execution

This ensures your domains provide deterministic, high-quality benchmarks for agent evaluation.

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
# Install dependencies
pip install -r tau_helper/requirements.txt
```

**Important**: Always run `tau_helper` commands from the `warrior-tau-bench/` root directory. The tool uses the current working directory to find domains.

## Configuration

Create `.env` in `tau_helper/` directory:

```env
# Default model (used for instruction evaluation)
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_API_KEY=your-openai-api-key
DEFAULT_BASE_URL=https://api.openai.com/v1

# Reasoning model (used for SOP mapping, scaffolding)
DEFAULT_MODEL_R=gpt-4o-mini
DEFAULT_API_KEY_R=your-openai-api-key
DEFAULT_BASE_URL_R=https://api.openai.com/v1

# Multi-Agent Architecture (OPTIONAL)
# If both R2 and JUDGE are configured, scaffolding uses multi-agent mode:
# - R and R2 generate scaffolds independently
# - If they differ, JUDGE picks the best one
# - Improves quality through model consensus

# R2 Model: Can use stronger OpenAI models like gpt-5-mini, gpt-4o, etc.
DEFAULT_MODEL_R2=gpt-5-mini
DEFAULT_API_KEY_R2=your-openai-api-key
DEFAULT_BASE_URL_R2=https://api.openai.com/v1

# Judge Model: Reasoning models work well (DeepSeek R1, etc.)
DEFAULT_MODEL_R_JUDGE=deepseek-ai/DeepSeek-R1-0528
DEFAULT_API_KEY_R_JUDGE=your-friendli-api-key
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
# Generate task from custom instruction
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "Your instruction"

# Generate task from existing task instruction
python ../tau_helper/run.py scaffold <domain> --variation <variation> --task <task_id>

# Specify custom task ID
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "..." --task-id task_new_001

# Show detailed progress including execution results
python ../tau_helper/run.py scaffold <domain> --variation <variation> --instruction "..." --verbose

# Examples
python ../tau_helper/run.py scaffold salesforce_management --variation variation_2 --instruction "Transfer accounts to Chris Sullivan"
python ../tau_helper/run.py scaffold salesforce_management --variation variation_2 --task task_001 --verbose
```

**Task Scaffolder Features:**
- **Code-based generation**: Agents write Python code that calls tools
- **R/R2 Roundtable**: R generates code, R2 reviews for correctness, up to 5 refinement rounds
- **Live editing**: Execution failures trigger automatic diagnosis (R2) and fixes (R)
- **Real values only**: No placeholders! Uses actual values from execution results
- **Domain-agnostic**: Works with any domain automatically
- Outputs complete task in `tasks.py` format with real, executable values

**🤖 Multi-Agent Architecture**

Configure `DEFAULT_MODEL_R2` and optionally `DEFAULT_MODEL_R_JUDGE` in `.env`:

**How it works:**
1. **Model R** generates Python code for the task
2. **Model R2** reviews for correctness (tool calls, data flow, SOP compliance)
3. Up to 5 rounds of refinement until R2 approves
4. **Judge** (optional) mediates if R2 has critical concerns after all rounds
5. **Live editing** on execution: failures trigger R2 diagnosis → R fix → re-execute

**Recommended Models:**
- **R/R2**: OpenAI models like `gpt-4o`, `gpt-5-mini`
- **Judge**: Reasoning models like `deepseek-ai/DeepSeek-R1-0528`

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

### 1. Generate New Task (Complete Workflow)

The correct workflow for generating tasks:

```bash
# Step 1: Evaluate instruction quality
python ../tau_helper/run.py evaluate "You are Pat Manager. You want Chris Sullivan to succeed in Q1 2026..."

# Step 2: Map instruction to SOPs (check for ambiguity)
python ../tau_helper/run.py map-sop salesforce_management --variation variation_2 --instruction "You are Pat Manager..."

# If ambiguous, use the suggested fix from Step 2, then:

# Step 3: Evaluate the fixed instruction again
python ../tau_helper/run.py evaluate "FIXED_INSTRUCTION_HERE"

# Step 4: Scaffold with iterative execution (generates task with REAL values)
python ../tau_helper/run.py scaffold salesforce_management --variation variation_2 --instruction "..." --task-id task_006 --verbose

# Step 5: Copy the scaffolded code to tasks.py
# The code already contains real values from execution - no manual filling needed!

# Step 6: Validate the completed task
uv run alignerr validate --domain salesforce_management --variation variation_2 --task-id task_006
```

**Why this workflow?**
- **Step 1-3**: Ensures instruction is high quality, user-facing, and non-procedural
- **Step 2**: Detects missing information before scaffolding
- **Step 4**: Generates actions iteratively with real execution, producing ready-to-use code
- **Step 5**: Copy-paste the output - it's already complete with real values!
- **Step 6**: Validates the final task works correctly

### 2. Validate Task Instructions
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
