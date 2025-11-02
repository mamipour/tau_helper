"""
Task scaffolding module for generating complete, executable tasks from instructions.

Uses SOP mapping to generate action sequences, then executes actions incrementally
to fill placeholders with real values.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tau_helper.sop_mapper import SOPMapper
from tau_helper.action_executor import ActionExecutor


class TaskAction(BaseModel):
    """Represents a single action in a task."""
    name: str = Field(description="Tool/function name")
    kwargs: Dict[str, Any] = Field(description="Arguments for the action")
    comment: Optional[str] = Field(default=None, description="Explanatory comment")


class TaskScaffold(BaseModel):
    """Complete task scaffold with actions and metadata."""
    user_id: str = Field(description="Task ID (e.g., task_001)")
    instruction: str = Field(description="User-facing task instruction")
    actions: List[TaskAction] = Field(description="Sequence of actions to execute")
    outputs: List[str] = Field(
        default_factory=list,
        description="Expected outputs (list of strings)"
    )
    sop_chain: List[str] = Field(
        default_factory=list,
        description="SOP chain that this task implements"
    )


class TaskScaffolder:
    """
    Generates complete, executable tasks from instructions.
    
    Workflow:
    1. Map instruction to SOP chain using SOPMapper
    2. Generate action sequence with placeholders
    3. Execute actions incrementally, filling placeholders
    4. Return complete task or error with progress
    """
    
    SCAFFOLD_PROMPT = """You are a task scaffolding expert for Warrior Tau-Bench.

Given an instruction and the SOP chain it maps to, generate a COMPLETE action sequence.

## Instruction

{instruction}

## SOP Chain

{sop_chain}

## SOP Definitions (CRITICAL - Follow These Step-by-Step)

{sop_definitions}

## Available Tools

{tools_description}

## Rules

{rules_summary}

## Your Task

Generate a complete action sequence that implements the SOP chain.

### Action Generation Rules (CRITICAL - READ CAREFULLY)

1. **Follow SOP Steps EXACTLY**: Each SOP has numbered steps. Generate actions for ALL steps in EXACT order.
   - ❌ DO NOT skip steps
   - ❌ DO NOT reorder steps for "optimization"
   - ❌ DO NOT batch operations unless SOP explicitly does so
   - ✅ DO follow step-by-step: "Step 1: do X, then do Y" means action_X, then action_Y

2. **Example - SOP Pull Core Statements**:
   ```
   Step 1: Extract income → store income     ← TWO actions in sequence
   Step 2: Extract balance → store balance   ← TWO actions in sequence
   Step 3: Extract cashflow → store cashflow ← TWO actions in sequence
   Step 4: Validate missing data             ← ONE action
   ```
   **CORRECT sequence**: extract_income → store_income → extract_balance → store_balance → extract_cashflow → store_cashflow → validate_missing_data (7 actions)
   
   **WRONG (DO NOT DO THIS)**: extract_income → extract_balance → extract_cashflow → store_income → store_balance → store_cashflow (6 actions, missing validate, wrong order)

3. **Include Prerequisites**: All prerequisite actions must come first

4. **Use Exact Parameter Names**: CRITICAL - Use ONLY parameter names from the tool signatures above

5. **No Extra Parameters**: DO NOT add parameters that are not in the tool signature

6. **Placeholders**: Use {{placeholder_name}} for values that depend on previous actions

7. **Deterministic Defaults**: Use defaults from rules for known values

8. **Comments**: Add comments to explain what each action does (mention SOP step number)

### When to Use Placeholders vs Concrete Values (CRITICAL)

**Use CONCRETE VALUES for:**
- ✅ Values explicitly stated in the instruction (e.g., "AAPL", "3 fiscal years")
- ✅ Simple string parameters that are context-specific (e.g., application_name: "DCF Analysis Tool")
- ✅ Configuration keys (e.g., param_name: "UserEmailAddress")
- ✅ Enum values (e.g., env_type: "venv")
- ✅ Numeric constants (e.g., num_periods: 3)

**Use PLACEHOLDERS for:**
- ❌ Values that come from PREVIOUS action outputs (e.g., {{{{spreadsheet_id}}}}, {{{{income_statements}}}})
- ❌ Values that need TRANSFORMATION (e.g., {{{{TRANSFORM: merge data}}}})
- ❌ Dynamic IDs or generated values

**Example (CORRECT)**:
```json
{{
  "name": "configure_sec_api_identity",
  "kwargs": {{"application_name": "DCF Analysis Tool"}},  // ✅ CONCRETE - descriptive name
  "comment": "Configure SEC identity"
}}
```

**Example (WRONG)**:
```json
{{
  "name": "configure_sec_api_identity",
  "kwargs": {{"application_name": "{{{{application_name}}}}"}},  // ❌ WRONG - use concrete value
  "comment": "Configure SEC identity"
}}
```

### Placeholder Format (For Previous Action Outputs Only)

**Direct field names** (when action output contains the field):
- {{{{spreadsheet_id}}}} - if previous action returned {{"spreadsheet_id": "..."}}
- {{{{income_statements}}}} - if previous action returned {{"income_statements": [...]}}

**Nested access** (when you need a specific field from an action):
- {{{{extract_income_statements_output.income_statements}}}} - get income_statements from extract_income_statements
- {{{{action_0_output.environment}}}} - get environment field from first action

**Critical Rules**:
- Use EXACT field names from tool outputs (check tool return schemas!)
- If a tool returns {{"success": true, "income_statements": [...]}}, use {{{{income_statements}}}}
- DON'T invent field names like {{{{extracted_data}}}} - use the actual field name
- DON'T prefix with "extracted_" or "data_" unless that's the actual field name

**Data Transformation Detection** (CRITICAL):
When you need to COMBINE/MERGE/TRANSFORM data from multiple sources, use special syntax:

- {{{{TRANSFORM: merge income_statements, balance_sheets, cash_flow_statements by period_label}}}}
- {{{{CALCULATE: combine revenue from income_statements with total_assets from balance_sheets}}}}
- {{{{MANUAL: create normalized_records from income/balance/cashflow statements}}}}

Use TRANSFORM/CALCULATE/MANUAL when:
1. Combining arrays from multiple sources into one
2. Extracting fields from different sources into new structure
3. Matching/joining records across multiple arrays
4. Any logic beyond simple field lookup

Example needing transformation:
```
normalize_and_persist_financials(
  company_ticker: "AAPL",
  normalized_records: {{{{TRANSFORM: merge income_statements, balance_sheets, cash_flow_statements by period_label}}}}
)
```

This tells the system that manual intervention is needed.

### JSON Formatting Rules (CRITICAL - MUST FOLLOW)

**ALL placeholder values MUST be quoted strings in valid JSON:**

✅ CORRECT:
```json
{{
  "kwargs": {{
    "income_statements": "{{{{income_statements}}}}",
    "available_periods": "{{{{TRANSFORM: get periods from statements}}}}",
    "spreadsheet_id": "{{{{spreadsheet_id}}}}"
  }}
}}
```

❌ WRONG (will fail parsing):
```json
{{
  "kwargs": {{
    "income_statements": {{{{income_statements}}}},  // ❌ Missing quotes
    "available_periods": {{{{TRANSFORM: get periods}}}},  // ❌ Missing quotes
    "spreadsheet_id": {{{{spreadsheet_id}}}}  // ❌ Missing quotes
  }}
}}
```

**Remember**: In JSON, ALL string values including placeholders MUST be wrapped in double quotes.
Even if the placeholder looks like {{{{variable}}}}, it must be written as "{{{{variable}}}}" in JSON.

### Action Schema

Each action must have:
- name: str (tool name)
- kwargs: dict (parameters)
- comment: str (optional explanation)

### Example

For instruction "Extract Apple's financials":
```json
{{
  "actions": [
    {{
      "name": "setup_python_environment",
      "kwargs": {{"env_name": "dcf_modeling_env", "env_type": "venv"}},
      "comment": "SOP: Setup Python Environment"
    }},
    {{
      "name": "get_configuration",
      "kwargs": {{"param_name": "UserEmailAddress"}},
      "comment": "SOP: Set SEC Identity - Step 1"
    }},
    {{
      "name": "extract_income_statements",
      "kwargs": {{"company_ticker": "AAPL", "num_periods": 3}},
      "comment": "SOP: Pull Core Statements - Extract income"
    }},
    {{
      "name": "store_income_statements",
      "kwargs": {{"company_ticker": "AAPL", "income_statements": "{{{{income_statements}}}}"}},
      "comment": "SOP: Pull Core Statements - Store income (NOTE: placeholder is QUOTED)"
    }}
  ]
}}
```

## Output Format

{format_instructions}

Generate the complete action sequence now.
"""
    
    def __init__(
        self, 
        llm: ChatOpenAI, 
        domain: str, 
        variation: str,
        llm_r2: Optional[ChatOpenAI] = None,
        llm_judge: Optional[ChatOpenAI] = None
    ):
        """
        Initialize task scaffolder.
        
        Args:
            llm: LangChain ChatOpenAI instance (primary/R model)
            domain: Domain name (e.g., 'sec')
            variation: Variation name (e.g., 'variation_2')
            llm_r2: Optional second reasoning model for multi-agent architecture
            llm_judge: Optional judge model to pick best scaffold when R and R2 differ
        """
        self.llm = llm
        self.llm_r2 = llm_r2
        self.llm_judge = llm_judge
        self.domain = domain
        self.variation = variation
        
        # Multi-agent is enabled if both R2 and judge are provided
        self.multi_agent_enabled = llm_r2 is not None and llm_judge is not None
        
        # Initialize SOP mapper (reuse logic)
        self.sop_mapper = SOPMapper(llm, domain, variation)
        
        # Initialize action executor
        self.executor = ActionExecutor(domain, variation)
        
        # Setup parser and prompt
        self.parser = PydanticOutputParser(pydantic_object=TaskScaffold)
        self.prompt = ChatPromptTemplate.from_template(self.SCAFFOLD_PROMPT)
    
    def _get_tools_description(self) -> str:
        """Get description of available tools with their parameter schemas."""
        try:
            tools = self.executor.get_available_tools()
            lines = []
            for tool_name in sorted(tools.keys()):
                tool = tools[tool_name]
                info = tool.get_info()
                params = info.get('function', {}).get('parameters', {})
                
                # Format tool signature
                required_params = params.get('required', [])
                properties = params.get('properties', {})
                
                # Build parameter list
                param_strs = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'any')
                    is_required = param_name in required_params
                    param_str = f"{param_name}: {param_type}"
                    if not is_required:
                        param_str = f"[{param_str}]"  # Optional
                    param_strs.append(param_str)
                
                signature = f"{tool_name}({', '.join(param_strs)})"
                lines.append(f"- {signature}")
                
                # Add description if available
                if hasattr(tool, '__doc__') and tool.__doc__:
                    doc = tool.__doc__.strip().split('\n')[0]
                    lines.append(f"  {doc}")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Could not load tools: {e}"
    
    def _get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the full schema for a specific tool."""
        try:
            tools = self.executor.get_available_tools()
            if tool_name in tools:
                tool = tools[tool_name]
                info = tool.get_info()
                return info.get('function', {})
            return None
        except Exception:
            return None
    
    def _get_rules_summary(self) -> str:
        """Get summary of key rules from rules.py."""
        try:
            sops = self.sop_mapper.sops
            if isinstance(sops, list) and sops:
                # Take first 10 rules as summary
                return "\n".join([f"- {rule[:200]}..." if len(rule) > 200 else f"- {rule}" 
                                 for rule in sops[:10]])
            return "See rules.py for details"
        except Exception:
            return "See rules.py for details"
    
    def _get_sop_definitions(self, sop_names: List[str]) -> str:
        """
        Get full SOP definitions for the given SOP names.
        
        This extracts the detailed step-by-step instructions for each SOP
        from rules.py to help the LLM generate the correct action sequence.
        """
        try:
            sops = self.sop_mapper.sops
            if not isinstance(sops, list):
                return "SOP definitions not available"
            
            # Find the SOPs section in rules
            sop_section = None
            for rule in sops:
                if "SOP Pull Core Statements" in rule or "SOP Set SEC Identity" in rule:
                    sop_section = rule
                    break
            
            if not sop_section:
                return "SOP definitions not found"
            
            # Extract definitions for requested SOPs
            lines = []
            for sop_name in sop_names:
                # Find the SOP in the section
                if sop_name in sop_section:
                    # Extract the SOP definition
                    start = sop_section.find(sop_name)
                    # Find the next SOP or end
                    next_sop = sop_section.find("\n    =========================", start + 1)
                    if next_sop == -1:
                        next_sop = len(sop_section)
                    
                    sop_def = sop_section[start:next_sop].strip()
                    lines.append(sop_def)
                    lines.append("")
            
            if lines:
                return "\n".join(lines)
            else:
                return f"Requested SOPs: {', '.join(sop_names)}"
                
        except Exception as e:
            return f"Could not extract SOP definitions: {e}"
    
    def _scaffolds_are_equal(self, scaffold1: TaskScaffold, scaffold2: TaskScaffold) -> bool:
        """
        Check if two scaffolds are functionally equivalent.
        
        Compares action names and critical parameters.
        """
        if len(scaffold1.actions) != len(scaffold2.actions):
            return False
        
        for a1, a2 in zip(scaffold1.actions, scaffold2.actions):
            if a1.name != a2.name:
                return False
            # Could add more sophisticated kwargs comparison here if needed
        
        return True
    
    def _judge_scaffolds(
        self, 
        instruction: str,
        scaffold_r: TaskScaffold,
        scaffold_r2: TaskScaffold,
        progress: List[str]
    ) -> TaskScaffold:
        """
        Use judge LLM to pick the better scaffold.
        
        Args:
            instruction: Original instruction
            scaffold_r: Scaffold from R model
            scaffold_r2: Scaffold from R2 model
            progress: Progress log to append messages
        
        Returns:
            The chosen scaffold (either scaffold_r or scaffold_r2)
        """
        JUDGE_PROMPT = """You are an expert judge evaluating two task scaffolds.

Given the original instruction and two independently generated scaffolds, pick the BETTER one.

## Instruction

{instruction}

## Scaffold A (from Model R)

Actions ({count_a}):
{scaffold_a}

## Scaffold B (from Model R2)

Actions ({count_b}):
{scaffold_b}

## Evaluation Criteria

1. **Correctness**: Does it follow the SOPs correctly?
2. **Completeness**: Are all required steps included (setup, extraction, validation)?
3. **No extra steps**: Doesn't add unrequested functionality
4. **Action ordering**: Follows Extract→Store pattern correctly
5. **Parameter accuracy**: Uses correct field names and placeholders

## Your Task

Analyze both scaffolds and pick the better one.

Respond with ONLY: "A" or "B"

Your choice:"""

        try:
            # Format scaffolds for comparison
            scaffold_a_str = "\n".join([
                f"{i+1}. {action.name}({', '.join(f'{k}={v}' for k, v in action.kwargs.items())})"
                for i, action in enumerate(scaffold_r.actions)
            ])
            
            scaffold_b_str = "\n".join([
                f"{i+1}. {action.name}({', '.join(f'{k}={v}' for k, v in action.kwargs.items())})"
                for i, action in enumerate(scaffold_r2.actions)
            ])
            
            prompt = JUDGE_PROMPT.format(
                instruction=instruction,
                count_a=len(scaffold_r.actions),
                scaffold_a=scaffold_a_str,
                count_b=len(scaffold_r2.actions),
                scaffold_b=scaffold_b_str
            )
            
            response = self.llm_judge.invoke(prompt)
            decision = response.content.strip().upper()
            
            if "A" in decision[:10]:
                progress.append("🔷 Judge selected: Model R scaffold")
                return scaffold_r
            else:
                progress.append("🔶 Judge selected: Model R2 scaffold")
                return scaffold_r2
                
        except Exception as e:
            progress.append(f"⚠️  Judge failed ({e}), defaulting to Model R")
            return scaffold_r
    
    def scaffold(
        self,
        instruction: str,
        task_id: str = "task_new",
        execute: bool = True,
        max_retries: int = 3
    ) -> Tuple[Optional[TaskScaffold], Optional[str], List[str], Optional[str]]:
        """
        Generate a complete task scaffold from instruction.
        
        Automatically retries up to max_retries times if generation fails.
        
        Args:
            instruction: User-facing task instruction
            task_id: Task ID (default: task_new)
            execute: Whether to execute actions to fill placeholders
            max_retries: Maximum number of attempts (default: 3)
            
        Returns:
            Tuple of (scaffold, error, progress_log, model_info)
            - scaffold: Complete TaskScaffold if successful
            - error: Error message if failed
            - progress_log: List of progress messages
            - model_info: String describing which model(s) generated the scaffold
        """
        last_error = None
        all_progress = []
        model_info = None
        
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                all_progress.append(f"\n{'='*80}")
                all_progress.append(f"🔄 Retry attempt {attempt}/{max_retries}")
                all_progress.append(f"{'='*80}\n")
            
            scaffold, error, progress, attempt_model_info = self._scaffold_attempt(instruction, task_id, execute)
            all_progress.extend(progress)
            
            if scaffold is not None:
                # Success!
                if attempt > 1:
                    all_progress.append(f"\n✅ Succeeded on attempt {attempt}/{max_retries}")
                model_info = attempt_model_info
                return scaffold, None, all_progress, model_info
            
            # Failed - save error for potential retry
            last_error = error
            
            if attempt < max_retries:
                all_progress.append(f"\n⚠️  Attempt {attempt} failed, retrying...")
        
        # All attempts failed
        all_progress.append(f"\n❌ All {max_retries} attempts failed")
        return None, last_error, all_progress, None
    
    def _scaffold_attempt(
        self,
        instruction: str,
        task_id: str,
        execute: bool
    ) -> Tuple[Optional[TaskScaffold], Optional[str], List[str], Optional[str]]:
        """
        Single attempt to generate a task scaffold.
        
        If multi-agent is enabled, generates with both R and R2 models
        and uses judge to pick the best one when they differ.
        
        Returns:
            Tuple of (scaffold, error, progress_log, model_info)
        """
        progress = []
        model_info = "Single Model"  # Default
        
        try:
            # Step 1: Map instruction to SOP chain
            progress.append("Step 1: Mapping instruction to SOP chain...")
            mapping = self.sop_mapper.map_instruction(instruction)
            
            sop_chain = mapping.primary_chain.sops
            progress.append(f"✓ Mapped to {len(sop_chain)} SOPs")
            
            if mapping.is_ambiguous:
                progress.append(f"⚠️  Warning: Instruction is ambiguous (confidence: {mapping.primary_chain.confidence}%)")
            
            # Step 2: Generate action sequence with placeholders
            if self.multi_agent_enabled:
                progress.append("\n🤖 Multi-Agent Mode: Generating with both R and R2 models...")
                
                # Generate with R model (primary)
                progress.append("  → Generating with Model R...")
                try:
                    scaffold_r = self._generate_scaffold(instruction, sop_chain, task_id)
                    progress.append(f"    ✓ Model R: {len(scaffold_r.actions)} actions")
                except Exception as e:
                    progress.append(f"    ❌ Model R failed: {str(e)}")
                    raise
                
                # Generate with R2 model
                progress.append("  → Generating with Model R2...")
                try:
                    # Temporarily switch LLM to R2 for generation
                    original_llm = self.llm
                    self.llm = self.llm_r2
                    scaffold_r2 = self._generate_scaffold(instruction, sop_chain, task_id)
                    self.llm = original_llm  # Restore original LLM
                    progress.append(f"    ✓ Model R2: {len(scaffold_r2.actions)} actions")
                except Exception as e:
                    self.llm = original_llm  # Restore on error
                    progress.append(f"    ❌ Model R2 failed: {str(e)}")
                    # Continue with R model only
                    scaffold = scaffold_r
                    model_info = "Model R (R2 failed)"
                    progress.append("  → Using Model R scaffold only")
                else:
                    # Both succeeded - compare and judge if needed
                    if self._scaffolds_are_equal(scaffold_r, scaffold_r2):
                        progress.append("  ✓ Both models agree!")
                        scaffold = scaffold_r
                        model_info = "Consensus (R + R2)"
                    else:
                        progress.append("  ⚠️  Models disagree - invoking judge...")
                        scaffold = self._judge_scaffolds(instruction, scaffold_r, scaffold_r2, progress)
                        # Determine which model was selected by checking if scaffold is R or R2
                        if scaffold is scaffold_r2:
                            model_info = "Judge → Model R2"
                        else:
                            model_info = "Judge → Model R"
            else:
                # Single model mode (current behavior)
                progress.append("\nStep 2: Generating action sequence...")
                try:
                    scaffold = self._generate_scaffold(instruction, sop_chain, task_id)
                    progress.append(f"✓ Generated {len(scaffold.actions)} actions")
                    model_info = "Single Model"
                except Exception as e:
                    progress.append(f"❌ Failed to generate scaffold: {str(e)}")
                    raise
            
            if not execute:
                return scaffold, None, progress, model_info
            
            # Step 3: Execute actions and fill placeholders
            progress.append("\nStep 3: Executing actions to fill placeholders...")
            filled_scaffold, error, final_result = self._fill_placeholders(scaffold, progress)
            
            if error:
                return None, error, progress, model_info
            
            # Step 4: Derive outputs from final action
            progress.append("\nStep 4: Deriving task outputs...")
            outputs = self._derive_outputs(filled_scaffold, final_result, progress)
            filled_scaffold.outputs = outputs
            
            progress.append(f"\n✅ Task scaffold complete!")
            return filled_scaffold, None, progress, model_info
            
        except Exception as e:
            import traceback
            error = f"Scaffolding failed: {str(e)}"
            progress.append(f"\n❌ {error}")
            progress.append(f"\nTraceback:\n{traceback.format_exc()}")
            return None, error, progress, None
    
    def _generate_scaffold(
        self,
        instruction: str,
        sop_chain: List[str],
        task_id: str
    ) -> TaskScaffold:
        """Generate initial scaffold with placeholders using LLM."""
        # Format SOP chain
        sop_chain_str = "\n".join([f"{i+1}. {sop}" for i, sop in enumerate(sop_chain)])
        
        # Get tools and rules
        tools_desc = self._get_tools_description()
        rules_summary = self._get_rules_summary()
        
        # Get full SOP definitions for the SOPs in the chain
        sop_definitions = self._get_sop_definitions(sop_chain)
        
        # Format prompt
        formatted_prompt = self.prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_str,
            sop_definitions=sop_definitions,
            tools_description=tools_desc,
            rules_summary=rules_summary,
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Get LLM response
        response = self.llm.invoke(formatted_prompt)
        
        # Parse scaffold
        try:
            scaffold = self.parser.parse(response.content)
        except Exception as e1:
            # Try cleaning response
            try:
                cleaned = self._clean_llm_response(response.content)
                scaffold = self.parser.parse(cleaned)
            except Exception as e2:
                # Show what we're trying to parse for debugging
                raise ValueError(
                    f"Failed to parse LLM response.\n"
                    f"First error: {e1}\n"
                    f"After cleaning: {e2}\n"
                    f"Cleaned content (first 500 chars): {cleaned[:500] if 'cleaned' in locals() else 'N/A'}"
                )
        
        # Set task ID and instruction
        scaffold.user_id = task_id
        scaffold.instruction = instruction
        scaffold.sop_chain = sop_chain
        
        # Post-process: Fix generic placeholders
        scaffold = self._fix_generic_placeholders(scaffold)
        
        return scaffold
    
    def _clean_llm_response(self, content: str) -> str:
        """Clean LLM response (remove reasoning tags, markdown, fix common issues)."""
        import re
        
        # Remove reasoning tags (common in models like DeepSeek, GPT-5)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        
        # Extract JSON from markdown code fences (```json {...} ```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # More aggressive: find ANY JSON object in the content
            # This handles cases where models output reasoning text before JSON
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
            if not json_match:
                # Try to find the largest brace-balanced JSON structure
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
        
        # Fix common JSON issues
        # Fix trailing commas
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Fix common typos from reasoning models
        content = re.sub(r'":-\s*', r'": ', content)
        content = re.sub(r'或少', r'', content)
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Final check: ensure it starts with { and ends with }
        if not content.startswith('{') or not content.endswith('}'):
            # Last resort: try to extract the JSON more aggressively
            lines = content.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                for char in line:
                    if char == '{':
                        if brace_count == 0:
                            in_json = True
                            json_lines = []
                        brace_count += 1
                    if in_json:
                        json_lines.append(char)
                    if char == '}':
                        brace_count -= 1
                        if brace_count == 0 and in_json:
                            # Found complete JSON
                            content = ''.join(json_lines)
                            break
                if brace_count == 0 and json_lines:
                    break
        
        return content
    
    def _fix_generic_placeholders(self, scaffold: TaskScaffold) -> TaskScaffold:
        """
        Post-process scaffold to replace generic placeholders with specific TRANSFORM syntax.
        
        Detects patterns like {{placeholder}}, {{value}}, {{data}} and replaces them with
        appropriate transformation descriptions based on parameter context.
        """
        import re
        
        # Generic placeholder keywords to detect
        generic_keywords = [
            'placeholder',
            'value',
            'data',
            'result',
            'output',
            'todo',
            'fixme',
        ]
        
        replacements_made = 0
        
        for action in scaffold.actions:
            for param_name, param_value in list(action.kwargs.items()):
                # Check if value is a string
                if isinstance(param_value, str):
                    # Detect generic placeholders in two ways:
                    # 1. Contains generic keywords like "placeholder", "value", etc.
                    # 2. Uses the parameter name itself as placeholder: {{param_name}}
                    value_lower = param_value.lower()
                    found_generic = False
                    
                    # Check for generic keywords
                    for keyword in generic_keywords:
                        if keyword in value_lower and '{{' in param_value and '}}' in param_value:
                            found_generic = True
                            break
                    
                    # Check if placeholder is just the parameter name itself
                    # e.g., parameter "available_periods" with value "{{available_periods}}"
                    if not found_generic and '{{' in param_value and '}}' in param_value:
                        # Extract placeholder content
                        import re
                        placeholder_match = re.search(r'\{\{([^}]+)\}\}', param_value)
                        if placeholder_match:
                            placeholder_content = placeholder_match.group(1).strip()
                            # If placeholder is exactly the parameter name, it's generic
                            if placeholder_content.lower() == param_name.lower():
                                found_generic = True
                    
                    if found_generic:
                        # Replace with context-aware TRANSFORM
                        replacement = self._generate_transform_for_parameter(
                            action.name, 
                            param_name,
                            param_value
                        )
                        action.kwargs[param_name] = replacement
                        replacements_made += 1
        
        return scaffold
    
    def _generate_transform_for_parameter(
        self, 
        action_name: str, 
        param_name: str,
        original_value: str
    ) -> str:
        """
        Generate appropriate TRANSFORM syntax for a parameter based on context.
        
        Args:
            action_name: Name of the action (e.g., 'validate_missing_data')
            param_name: Name of the parameter (e.g., 'available_periods')
            original_value: The original generic placeholder value
            
        Returns:
            TRANSFORM syntax string
        """
        # Common patterns for specific parameters
        transforms = {
            'available_periods': '{{TRANSFORM: extract period_label values from income_statements, balance_sheets, and cash_flow_statements}}',
            'periods': '{{TRANSFORM: extract period_label from available financial statements}}',
            'fiscal_years': '{{TRANSFORM: extract fiscal_year from income_statements}}',
            'period_labels': '{{TRANSFORM: get unique period_label values from stored statements}}',
            'normalized_records': '{{TRANSFORM: merge income_statements, balance_sheets, cash_flow_statements by period_label}}',
            'financial_data': '{{TRANSFORM: combine data from income_statements, balance_sheets, cash_flow_statements}}',
            'time_series_data': '{{TRANSFORM: extract time series values from financial statements}}',
            'historical_data': '{{TRANSFORM: aggregate historical financial data by period}}',
        }
        
        # Check for exact parameter name match
        if param_name.lower() in transforms:
            return transforms[param_name.lower()]
        
        # Check for partial matches
        param_lower = param_name.lower()
        if 'period' in param_lower:
            return '{{TRANSFORM: extract period identifiers from available statements}}'
        elif 'normalized' in param_lower or 'merged' in param_lower:
            return '{{TRANSFORM: merge and normalize data from multiple statement sources}}'
        elif 'records' in param_lower or 'data' in param_lower:
            return '{{TRANSFORM: combine relevant data from previous action outputs}}'
        elif 'year' in param_lower:
            return '{{TRANSFORM: extract fiscal year values from statements}}'
        elif 'series' in param_lower:
            return '{{TRANSFORM: create time series from financial data}}'
        
        # Default: generic transformation
        return f'{{{{TRANSFORM: derive {param_name} from previous outputs}}}}'
    
    def _fill_placeholders(
        self,
        scaffold: TaskScaffold,
        progress: List[str]
    ) -> Tuple[Optional[TaskScaffold], Optional[str], Any]:
        """
        Execute actions incrementally to fill placeholders.
        
        Returns:
            Tuple of (filled_scaffold, error, final_result)
        """
        context = {}  # Stores outputs from previous actions
        filled_actions = []
        final_result = None
        
        for i, action in enumerate(scaffold.actions):
            progress.append(f"\n  Action {i+1}/{len(scaffold.actions)}: {action.name}")
            
            # Fill placeholders in kwargs
            try:
                # Get tool schema for this action
                tool_schema = self._get_tool_schema(action.name)
                filled_kwargs = self._resolve_kwargs(action.name, action.kwargs, context, progress, tool_schema)
            except Exception as e:
                error = f"Failed to resolve kwargs for {action.name}: {e}"
                progress.append(f"  ❌ {error}")
                return None, error, None
            
            # Execute action
            try:
                result = self.executor.execute_action_by_name(action.name, filled_kwargs)
                progress.append(f"  ✓ Executed successfully")
                
                # Parse JSON string results
                parsed_result = result
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        parsed_result = result
                
                # Store result in context for next actions
                context[f"{action.name}_output"] = parsed_result
                context[f"action_{i}_output"] = parsed_result
                
                # Store specific fields if they exist
                if isinstance(parsed_result, dict):
                    for key, value in parsed_result.items():
                        context[key] = value
                
                # Store final result (last action)
                final_result = parsed_result
                
                # Create filled action
                filled_actions.append(TaskAction(
                    name=action.name,
                    kwargs=filled_kwargs,
                    comment=action.comment
                ))
                
            except Exception as e:
                error = f"Execution failed: {str(e)}"
                progress.append(f"  ❌ {error}")
                progress.append(f"  Context available: {list(context.keys())}")
                return None, error, None
        
        # Create filled scaffold
        filled_scaffold = TaskScaffold(
            user_id=scaffold.user_id,
            instruction=scaffold.instruction,
            actions=filled_actions,
            outputs=scaffold.outputs,
            sop_chain=scaffold.sop_chain
        )
        
        return filled_scaffold, None, final_result
    
    def _resolve_kwargs(
        self,
        action_name: str,
        kwargs: Dict[str, Any],
        context: Dict[str, Any],
        progress: List[str],
        tool_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve placeholders in kwargs.
        
        Sources (in order):
        1. Literal values (no placeholder)
        2. Context (previous action outputs)
        3. Deterministic calculation
        
        Raises exception if value cannot be resolved.
        """
        resolved = {}
        
        # Get parameter schemas if available
        param_schemas = {}
        if tool_schema and 'parameters' in tool_schema:
            param_schemas = tool_schema['parameters'].get('properties', {})
        
        for key, value in kwargs.items():
            param_schema = param_schemas.get(key)
            resolved[key] = self._resolve_value(key, value, context, progress, param_schema)
        
        return resolved
    
    def _resolve_value(
        self,
        key: str,
        value: Any,
        context: Dict[str, Any],
        progress: List[str],
        param_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Resolve a single value (handles nested structures)."""
        if isinstance(value, str):
            # Check for placeholder pattern {{...}} or {...}
            if (value.startswith('{{') and value.endswith('}}')) or \
               (value.startswith('{') and value.endswith('}') and not value.startswith('{{')):
                # Extract placeholder name
                if value.startswith('{{'):
                    placeholder = value[2:-2].strip()
                else:
                    placeholder = value[1:-1].strip()
                return self._resolve_placeholder(key, placeholder, context, progress, param_schema)
            # Check for embedded placeholders
            elif '{{' in value or ('{' in value and '}' in value):
                return self._resolve_embedded_placeholders(value, context, progress)
            else:
                return value
        
        elif isinstance(value, dict):
            return {k: self._resolve_value(k, v, context, progress, None) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self._resolve_value(key, item, context, progress, None) for item in value]
        
        else:
            return value
    
    def _resolve_placeholder(
        self,
        key: str,
        placeholder: str,
        context: Dict[str, Any],
        progress: List[str],
        param_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Resolve a placeholder to an actual value.
        
        Tries (in order):
        1. Direct context lookup
        2. Nested context lookup (dot notation)
        3. Transformation (with LLM code generation)
        4. Calculation expression
        5. Raises error if not resolvable
        """
        # Try direct lookup
        if placeholder in context:
            value = context[placeholder]
            progress.append(f"    Resolved {{{{{{{{placeholder}}}}}}}}: {placeholder} = {value}")
            return value
        
        # Try nested lookup (e.g., "extract_income_statements_output.data")
        if '.' in placeholder:
            parts = placeholder.split('.')
            obj = context
            for part in parts:
                if isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    break
            else:
                progress.append(f"    Resolved {{{{{{{{placeholder}}}}}}}}: {placeholder} = {obj}")
                return obj
        
        # Check for transformation markers FIRST - USE LLM TO GENERATE CODE!
        # (Must check BEFORE calculation, otherwise colons in TRANSFORM: will trigger calculation)
        transformation_markers = ['TRANSFORM:', 'CALCULATE:', 'MANUAL:']
        for marker in transformation_markers:
            if marker in placeholder.upper():
                # Extract the transformation description
                description = placeholder[placeholder.upper().index(marker) + len(marker):].strip()
                
                progress.append(f"    🔄 Transformation detected: {description}")
                progress.append(f"    🤖 Generating transformation code using LLM...")
                
                try:
                    # Use LLM to generate the transformation code
                    result = self._execute_transformation(description, context, progress, param_schema)
                    progress.append(f"    ✅ Transformation executed successfully")
                    return result
                except Exception as e:
                    raise ValueError(
                        f"Data transformation failed for parameter '{key}':\n"
                        f"  Transformation: {description}\n"
                        f"  Error: {str(e)}\n"
                        f"  Available data: {list(context.keys())}\n"
                    )
        
        # Try calculation (e.g., "revenue * 0.02")
        # (Checked AFTER transformations so TRANSFORM: syntax doesn't get misinterpreted)
        if any(op in placeholder for op in ['+', '-', '*', '/', '(', ')']):
            try:
                # Safe evaluation with context
                result = self._safe_eval(placeholder, context)
                progress.append(f"    Calculated {{{{{{{{placeholder}}}}}}}}: {placeholder} = {result}")
                return result
            except Exception as e:
                raise ValueError(f"Cannot calculate '{placeholder}': {e}")
        
        # Cannot resolve
        available = list(context.keys())
        raise ValueError(
            f"Cannot resolve placeholder '{{{{{{{{placeholder}}}}}}}}' for parameter '{key}'.\n"
            f"Available in context: {available}\n"
            f"Placeholder must be:\n"
            f"  1. A value from previous action output\n"
            f"  2. A deterministic calculation\n"
            f"  3. A default from rules.py\n"
            f"\n"
            f"If this parameter requires combining data from multiple sources,\n"
            f"use transformation syntax: {{{{{{{{TRANSFORM: description}}}}}}}}"
        )
    
    def _resolve_embedded_placeholders(
        self,
        template: str,
        context: Dict[str, Any],
        progress: List[str]
    ) -> str:
        """Resolve placeholders embedded in a string."""
        import re
        
        def replace_placeholder(match):
            placeholder = match.group(1).strip()
            return str(self._resolve_placeholder('embedded', placeholder, context, progress))
        
        # Try double braces first
        result = re.sub(r'\{\{(.*?)\}\}', replace_placeholder, template)
        # Then single braces if no double braces were found
        if '{{' not in template and '{' in result:
            result = re.sub(r'\{([^{}]+)\}', replace_placeholder, result)
        
        return result
    
    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate a mathematical expression with context."""
        # Replace placeholders in expression
        for key, value in context.items():
            if isinstance(value, (int, float)):
                expression = expression.replace(key, str(value))
        
        # Use eval with restricted globals (only math operations)
        allowed_globals = {
            '__builtins__': {},
            'int': int,
            'float': float,
            'abs': abs,
            'min': min,
            'max': max,
        }
        
        return eval(expression, allowed_globals, {})
    
    def _execute_transformation(
        self,
        description: str,
        context: Dict[str, Any],
        progress: List[str],
        target_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Use LLM to generate and execute transformation code.
        
        Args:
            description: Natural language description of transformation
            context: Available data context
            progress: Progress log
            
        Returns:
            Result of the transformation
        """
        # Build prompt for LLM to generate transformation code
        schema_section = ""
        if target_schema:
            schema_section = f"""
**Target Output Schema** (CRITICAL - must match exactly):
{json.dumps(target_schema, indent=2)}

The transformation MUST produce data matching this exact schema.
Extract the required fields from the available data sources.
"""
        
        prompt = f"""Generate Python code to perform the following data transformation:

**Transformation**: {description}

**Available Data** (in context dictionary):
{self._format_context_for_prompt(context)}
{schema_section}
**Requirements**:
1. Write a Python function called `transform()` that takes no arguments
2. The function should access data from the `context` dict (e.g., `context['income_statements']`)
3. Return the transformed result matching the target schema
4. Available modules: `re`, `datetime` (already imported as `from datetime import datetime`)
5. Handle edge cases (empty lists, missing fields, None values, etc.)
6. The code should be production-ready and deterministic
7. Use defensive programming - check types before accessing fields
8. **CRITICAL**: Extract ALL required fields from the source data as specified in the schema

**Example**:
```python
def transform():
    # Access data from context
    income_statements = context['income_statements']
    balance_sheets = context['balance_sheets']
    cash_flow_statements = context['cash_flow_statements']
    
    # Perform transformation
    normalized_records = []
    for i in range(len(income_statements)):
        normalized_records.append({{
            'period_label': income_statements[i]['period_label'],
            'revenue': income_statements[i]['revenue'],
            'net_income': income_statements[i]['net_income'],
            'total_assets': balance_sheets[i]['total_assets'],
            'operating_cashflow': cash_flow_statements[i]['operating_cashflow'],
            'free_cashflow': cash_flow_statements[i]['free_cashflow']
        }})
    
    return normalized_records
```

Generate ONLY the Python function code, no explanations.
"""
        
        # Get LLM to generate the code
        response = self.llm.invoke(prompt)
        code = self._extract_code_from_response(response.content)
        
        progress.append(f"    📝 Generated {len(code.split(chr(10)))} lines of transformation code")
        
        # Execute the code safely
        try:
            # Import necessary modules for transformation code
            import re
            from datetime import datetime
            
            # Create a restricted execution environment
            exec_globals = {
                '__builtins__': {
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'sorted': sorted,
                    'reversed': reversed,
                    'any': any,
                    'all': all,
                    '__import__': __import__,  # Needed for from/import statements
                    'globals': globals,  # Needed for defensive code patterns
                },
                'context': context,
                're': re,
                'datetime': datetime,
            }
            
            # Execute the function definition
            exec(code, exec_globals)
            
            # Call the transform function
            if 'transform' not in exec_globals:
                raise ValueError("Generated code does not define a 'transform()' function")
            
            result = exec_globals['transform']()
            return result
            
        except Exception as e:
            raise ValueError(f"Transformation execution failed: {str(e)}\n\nGenerated code:\n{code}")
    
    def _format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """Format context for LLM prompt (show structure, not full data)."""
        lines = []
        for key, value in context.items():
            if isinstance(value, list):
                if len(value) > 0:
                    lines.append(f"  {key}: list of {len(value)} items")
                    if isinstance(value[0], dict):
                        sample_keys = list(value[0].keys())[:5]
                        lines.append(f"    Sample keys: {sample_keys}")
                else:
                    lines.append(f"  {key}: empty list")
            elif isinstance(value, dict):
                lines.append(f"  {key}: dict with keys {list(value.keys())[:5]}")
            else:
                lines.append(f"  {key}: {type(value).__name__}")
        return "\n".join(lines)
    
    def _extract_code_from_response(self, content: str) -> str:
        """Extract Python code from LLM response."""
        import re
        
        # Remove reasoning tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        
        # Try to extract from code fences
        code_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code fence, look for function definition
        if 'def transform(' in content:
            # Extract from first def to end
            start = content.index('def transform(')
            return content[start:].strip()
        
        # Return as-is and hope for the best
        return content.strip()
    
    def _derive_outputs(
        self,
        scaffold: TaskScaffold,
        final_result: Any,
        progress: List[str]
    ) -> List[str]:
        """
        Derive task outputs from the final action's return value.
        
        Returns list of strings in format: "key: value"
        """
        if not scaffold.actions or final_result is None:
            return []
        
        outputs = []
        
        # Extract key fields from final result
        if isinstance(final_result, dict):
            # Skip standard fields that are not outputs
            skip_keys = {'success', 'error', 'message'}
            
            for key, value in final_result.items():
                if key in skip_keys:
                    continue
                
                # Format the output (JSON-like with quotes around keys)
                if isinstance(value, (list, dict)):
                    # For complex types, just note the count
                    if isinstance(value, list):
                        outputs.append(f'"{key}": {len(value)}')
                    else:
                        outputs.append(f'"{key}": {len(value)} fields')
                else:
                    outputs.append(f'"{key}": {value}')
        
        progress.append(f"  ✓ Derived {len(outputs)} output field(s)")
        
        return outputs if outputs else ['"success": true']

