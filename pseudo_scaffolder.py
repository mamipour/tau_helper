"""
Pseudo Scaffolder - Code-based task scaffolding with R/R2 roundtable.

This scaffolder has agents write Python code to generate actions, then
executes the code with real feedback. The code-based approach leverages
LLMs' strong code generation abilities for better reasoning about complex
task logic.

Flow:
1. Model R generates a Python plan (code that calls tools)
2. R2 reviews the plan structure (up to 5 rounds)
3. If no agreement, Judge mediates
4. Plan is executed line-by-line with real database feedback
5. Executed calls are converted to Action objects
"""

import importlib
import json
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import ast

from tau_helper.sop_mapper import SOPMapper


@dataclass
class ExecutedAction:
    """An action that was executed from the Python plan."""
    name: str
    kwargs: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    line_number: int = 0
    code_line: str = ""


@dataclass
class PlanReviewResult:
    """Result of R2's code review."""
    approved: bool
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    severity: str = "low"  # low, medium, high, critical


class PseudoScaffolder:
    """
    Code-based task scaffolder using Python plan generation.
    
    Agents write Python code that calls tool functions, which is then
    executed with real database feedback. This leverages LLMs' strong
    code generation abilities.
    """
    
    PLAN_GENERATION_PROMPT = """You are Model R, an expert Python programmer writing code to complete a task.

## Your Task

Write Python code that calls the available tools to complete the given instruction.
The code will be executed with real database feedback.

## Instruction (PRIMARY - This is what you MUST accomplish)

{instruction}

## Suggested SOP Chain (ADVISORY - Use as guidance, not gospel)

{sop_chain}

NOTE: The SOP chain above is a SUGGESTION based on pattern matching. 
**THE INSTRUCTION ALWAYS TAKES PRIORITY over the suggested chain.**

If the instruction contradicts the SOP chain, FOLLOW THE INSTRUCTION:
- Instruction says "no historical data" but chain includes historical SOPs → SKIP those SOPs
- Instruction says "simplified" or "basic" → SKIP complex/advanced SOPs
- Instruction explicitly excludes something → Don't include it regardless of chain

## Available Tools (as Python functions)

{tools_as_functions}

## Domain Rules

{rules}

## Code Requirements

1. **Use ONLY the available tool functions** - don't invent functions
2. **Store results in variables** - you'll need IDs from earlier calls
3. **Handle the data flow explicitly** - extract IDs from results before using
4. **INSTRUCTION IS KING** - The instruction defines what to do. Use SOP chain as guidance only.
5. **Skip SOPs that contradict instruction** - If instruction says "no X", don't include X-related SOPs
6. **No imports needed** - tools are already available
7. **Use clear variable names** - e.g., `result`, `item_id`, `list_result`

## CRITICAL: Instruction-Driven Completeness Check

Before finishing your code, verify:
- Does your code accomplish what the INSTRUCTION asks for?
- If instruction says "no X" or "without X", did you SKIP X-related operations?
- If instruction says "simplified" or "basic", did you skip complex operations?
- Does your code include a final write/notification step if required?
- If the instruction mentions "share", "post", "notify" - did you include that call?

## IMPORTANT: Data Structure Assumptions

- Tool returns are typically dicts with descriptive field names
- Common patterns: result["field"]["id"], result["items"], result["success"]
- If execution fails due to wrong field access, the error will show the actual structure
- The system will auto-fix key mismatches during execution

## CRITICAL: Parameter Names vs Response Field Names

- Tool PARAMETERS always use snake_case (e.g., board_id, sprint_id, channel_id)
- Tool RESPONSES may use camelCase in JSON fields
- When calling tools: use snake_case parameter names from the Available Tools section
- When reading results: use the field names shown in the response

Example: `some_tool(board_id=15)` - use snake_case for the parameter name

## IMPORTANT: Transition/Status Lookup Pattern

When looking for a transition to a target status:
- Transition NAME is not the same as target status name
- Example: To move to status "Impeded", look for transition where `to.name == "Impeded"` NOT transition.name == "Impeded"
- Use: `next(t for t in transitions if t["to"]["name"] == "TargetStatus")`

## CRITICAL: Follow ALL Domain Rules

- Read the Domain Rules section carefully - it contains formatting requirements
- If rules specify "whole numbers (no decimals)", use int() when formatting those values
- If rules specify exact templates, follow them exactly
- Apply ALL formatting rules from the domain when constructing output strings

## Example Pattern

```python
# Step 1: Call a tool and store the result
result = some_tool("param1", "param2")

# Step 2: Extract IDs from the result (check sample outputs for exact structure)
some_id = result["field"]["id"]  # Use the exact path from sample outputs

# Step 3: Use extracted IDs in subsequent calls
next_result = another_tool(some_id)

# Step 4: When iterating, check if items are dicts or strings
for item in result["items"]:
    # If items are dicts: item["key"]
    # If items are strings: just use item directly
    pass

# Step 5: For lookups, use next() with a condition
target_id = next(x["id"] for x in result["list"] if x["name"] == "target-name")

# Step 6: Apply formatting rules from Domain Rules when constructing output
# e.g., if rules say "whole numbers", convert float to int before formatting
value = int(some_float_value)  # Ensure no decimal places
```

## Output Format

Return ONLY valid Python code. No markdown, no explanation, just code.
The code should complete the entire task from start to finish.

```python
# Your code here
```
"""

    R2_CODE_REVIEW_PROMPT = """You are R2, reviewing Python code for TASK CORRECTNESS (not code quality).

## Task Context

**Instruction (PRIMARY):** {instruction}
**Suggested SOP Chain (ADVISORY):** {sop_chain}

## Available Tools

{tools_as_functions}

## Domain Rules

{rules}

## Code to Review

```python
{code}
```

## IMPORTANT: Focus on INSTRUCTION COMPLIANCE

You are reviewing whether the code will accomplish the INSTRUCTION correctly.
The SOP chain is just a suggestion - the INSTRUCTION is what matters!

**DO Review:**
1. **Tool Calls**: Are function names valid? Do parameters match tool schemas EXACTLY?
2. **Data Flow**: Are IDs from previous calls used correctly in subsequent calls?
3. **Instruction Compliance**: Does the code accomplish what the INSTRUCTION asks for?
4. **Hallucinated Values**: Are IDs INVENTED that don't appear in instruction?
5. **Domain Rule Compliance**: Does the code follow formatting rules from Domain Rules (e.g., number formatting, template formats)?

**NOTE: Return Key Access**
- We don't have sample tool outputs to verify exact return key names
- If execution fails due to wrong key access, live editing will fix it based on actual error
- Focus on tool parameters (from Available Tools), not return structures

**CRITICAL: Instruction Overrides SOP Chain!**
- If instruction says "no historical data" but SOP chain includes historical SOPs → CORRECT to skip them
- If instruction says "without X" or "simplified" → CORRECT to skip X-related operations
- If instruction excludes something, code SHOULD exclude it even if SOP chain suggests it
- The SOP chain is ADVISORY. The INSTRUCTION is KING.

**CRITICAL: Values from the Instruction are NOT Hallucinated!**
- If the instruction mentions a specific ID or name, using it in code is CORRECT
- Example: instruction says "item ABC-123" → using "ABC-123" is CORRECT
- Hallucination means inventing values that appear NOWHERE (not in instruction, not in tool results)
- Using values explicitly mentioned in the instruction is EXPECTED behavior

**DO NOT Review (these are irrelevant for scaffolding):**
- Error handling or try/catch patterns
- Dict access safety (KeyError guards)
- Logging or debugging statements
- Code style or formatting
- Variable naming conventions
- Production readiness concerns
- Whether values from instruction should be "validated" via read tools
- Exact return key names (live editing fixes these)

## Response Format

Respond with JSON:
{{
    "approved": true/false,
    "concerns": ["List of TASK CORRECTNESS issues only"],
    "suggestions": ["Suggested fixes for correctness issues"],
    "severity": "low/medium/high/critical"
}}

**Severity Guide:**
- "low": Minor issues that won't affect task completion
- "medium": Missing optional steps or suboptimal approach
- "high": Missing required SOP steps or wrong tool parameters
- "critical": Hallucinated IDs, completely wrong approach, or task won't complete

If the code will execute the correct tool calls in the right order with correct parameters, set approved=true.
"""

    R_REVISION_PROMPT = """You are Model R. R2 has reviewed your code and found issues.

## Original Instruction

{instruction}

## Your Previous Code

```python
{previous_code}
```

## R2's Concerns

{concerns}

## R2's Suggestions

{suggestions}

## Your Task

Revise the code to address R2's concerns. Return ONLY the corrected Python code.

```python
# Your revised code here
```
"""

    JUDGE_CODE_PROMPT = """You are the FINAL EDITOR. R and R2 could not agree after {rounds} rounds.
Your job is to WRITE THE FINAL WORKING CODE. This is the last step - whatever you produce will be executed.

## The Task

**Instruction (PRIMARY - accomplish THIS):** {instruction}
**Suggested SOP Chain (ADVISORY):** {sop_chain}

NOTE: The instruction is PRIMARY. If instruction contradicts SOP chain (e.g., "no X" but chain includes X), follow instruction!

## Available Tools

{tools_as_functions}

## Domain Rules (MUST FOLLOW)

{rules}

## R's Current Code (has issues)

```python
{code}
```

## R2's Concerns (what's wrong)

{concerns}

## R2's Suggestions (how to fix)

{suggestions}

## YOUR JOB: Write Final Working Code

You must produce code that:
1. RUNS WITHOUT ERRORS (no undefined variables, no syntax errors)
2. Completes the INSTRUCTION (not necessarily all SOPs if instruction excludes them!)
3. Follows ALL domain rules (especially formatting rules)
4. Uses correct tool calls with valid parameters

Fix the issues R2 identified. The main problems are usually:
- Undefined variables (define them!)
- Wrong return key access (live editing will fix if wrong)
- Missing steps (add them!)
- Format violations (use int() for numbers, correct templates)

**CRITICAL: Instruction Overrides SOP Chain:**
- If instruction says "no X" or "without X" → DO NOT include X-related code
- If instruction excludes something, SKIP it even if SOP chain suggests it

**Values from instruction are NOT hallucinated:**
- IDs/names from instruction → use them directly in code
- Numbers from instruction (e.g., "2 points") → use 2 directly

**IGNORE these (not real issues):**
- Error handling, try-catch
- Dict access safety, KeyError guards
- Code style
- Exact return key names (live editing fixes these)

OUTPUT: Return ONLY valid Python code. No explanation.

```python
# Your complete, working code here
```"""

    def __init__(
        self,
        domain: str,
        variation: str,
        llm: ChatOpenAI,
        llm_r2: ChatOpenAI,
        llm_judge: Optional[ChatOpenAI] = None,
        max_review_rounds: int = 5
    ):
        """
        Initialize the Pseudo Scaffolder.
        
        Args:
            domain: Domain name
            variation: Variation name
            llm: Primary LLM (Model R) for code generation
            llm_r2: Secondary LLM (R2) for code review
            llm_judge: Judge LLM for mediation (optional)
            max_review_rounds: Maximum R/R2 review rounds before Judge
        """
        self.domain = domain
        self.variation = variation
        self.llm = llm
        self.llm_r2 = llm_r2
        self.llm_judge = llm_judge
        self.max_review_rounds = max_review_rounds
        
        # Load domain resources
        self._load_tools()
        self._load_rules()
        
        # Initialize SOP mapper for chain detection
        self.sop_mapper = SOPMapper(llm, domain, variation, llm_r2=llm_r2, llm_judge=llm_judge)
        
        # Action executor for real execution
        self._load_action_executor()
    
    def _load_tools(self):
        """Load tools and format as Python function signatures."""
        tools_module = importlib.import_module(
            f"domains.{self.domain}.variations.{self.variation}.tools"
        )
        self.tools = getattr(tools_module, 'TOOLS', [])
        
        # Format tools as Python function signatures
        self.tools_as_functions = self._format_tools_as_functions()
        
        # Create a mapping of tool names to actual tool classes
        self.tool_map = {tool.get_info()['function']['name']: tool for tool in self.tools}
    
    def _load_rules(self):
        """Load domain rules."""
        try:
            rules_module = importlib.import_module(
                f"domains.{self.domain}.variations.{self.variation}.rules"
            )
            rules = getattr(rules_module, 'RULES', [])
            self.rules = "\n".join([f"- {rule}" for rule in rules])
        except Exception:
            self.rules = "(No rules defined)"
    
    def _load_action_executor(self):
        """Load the action executor for real database interaction."""
        try:
            from tau_helper.action_executor import ActionExecutor
            self.action_executor = ActionExecutor(self.domain, self.variation)
        except Exception as e:
            print(f"Warning: Could not load ActionExecutor: {e}")
            self.action_executor = None
    
    def _format_tools_as_functions(self) -> str:
        """Format tools as Python function signatures with docstrings and return schemas."""
        lines = []
        
        for tool in self.tools:
            info = tool.get_info()
            if 'function' not in info:
                continue
            
            func_info = info['function']
            name = func_info['name']
            desc = func_info.get('description', 'No description')
            params = func_info.get('parameters', {}).get('properties', {})
            required = func_info.get('parameters', {}).get('required', [])
            
            # Build function signature
            param_parts = []
            for param_name, param_info in params.items():
                param_type = param_info.get('type', 'Any')
                # Handle union types (e.g., ["string", "null"])
                if isinstance(param_type, list):
                    param_type = next((t for t in param_type if t != 'null'), 'Any')
                type_map = {'string': 'str', 'integer': 'int', 'boolean': 'bool', 
                           'array': 'list', 'object': 'dict', 'number': 'float'}
                py_type = type_map.get(param_type, 'Any')
                
                if param_name in required:
                    param_parts.append(f"{param_name}: {py_type}")
                else:
                    param_parts.append(f"{param_name}: {py_type} = None")
            
            sig = f"def {name}({', '.join(param_parts)}) -> dict:"
            
            # Build docstring
            doc_lines = [f'    """{desc}']
            if params:
                doc_lines.append("")
                doc_lines.append("    Args:")
                for param_name, param_info in params.items():
                    param_desc = param_info.get('description', '')
                    param_enum = param_info.get('enum', [])
                    if param_enum:
                        # Include valid enum values so model knows exact options
                        enum_str = ", ".join(f'"{v}"' for v in param_enum[:5])  # Limit to 5
                        if len(param_enum) > 5:
                            enum_str += ", ..."
                        doc_lines.append(f"        {param_name}: {param_desc} Valid values: [{enum_str}]")
                    else:
                        doc_lines.append(f"        {param_name}: {param_desc}")
            doc_lines.append("")
            doc_lines.append("    Returns:")
            doc_lines.append("        dict: Result of the operation")
            doc_lines.append('    """')
            
            lines.append(sig)
            lines.extend(doc_lines)
            lines.append("")
        
        return "\n".join(lines)
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Remove markdown code fences
        code = response.strip()
        
        # Handle ```python ... ``` blocks
        if "```python" in code:
            match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
            if match:
                code = match.group(1)
        elif "```" in code:
            match = re.search(r'```\s*(.*?)\s*```', code, re.DOTALL)
            if match:
                code = match.group(1)
        
        return code.strip()
    
    def _validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            compile(code, "<plan>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg} - '{e.text.strip() if e.text else ''}'"
    
    def _normalize_types(self, obj: Any) -> Any:
        """
        Normalize types in kwargs - convert floats that are whole numbers to ints.
        This ensures story_points=3.0 becomes story_points=3.
        """
        if isinstance(obj, dict):
            return {k: self._normalize_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_types(item) for item in obj]
        elif isinstance(obj, float):
            # Convert whole floats to ints (3.0 -> 3, but keep 3.5 as 3.5)
            if obj == int(obj):
                return int(obj)
            return obj
        else:
            return obj
    
    def _generate_plan(self, instruction: str, sop_chain: List[str], verbose: bool = False) -> str:
        """Have Model R generate a Python plan."""
        prompt = ChatPromptTemplate.from_template(self.PLAN_GENERATION_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=", ".join(sop_chain),
            tools_as_functions=self.tools_as_functions,
            rules=self.rules
        )
        
        response = self.llm.invoke(messages)
        return self._extract_code(response.content)
    
    def _review_plan(self, instruction: str, sop_chain: List[str], code: str) -> PlanReviewResult:
        """Have R2 review the Python plan."""
        prompt = ChatPromptTemplate.from_template(self.R2_CODE_REVIEW_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=", ".join(sop_chain),
            tools_as_functions=self.tools_as_functions,
            rules=self.rules,
            code=code
        )
        
        response = self.llm_r2.invoke(messages)
        
        # Parse JSON response
        try:
            content = response.content.strip()
            # Extract JSON from potential markdown
            if "```json" in content:
                match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)
            elif "```" in content:
                match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)
            
            # Find JSON object
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace != -1 and last_brace != -1:
                content = content[first_brace:last_brace + 1]
            
            result = json.loads(content)
            return PlanReviewResult(
                approved=result.get("approved", False),
                concerns=result.get("concerns", []),
                suggestions=result.get("suggestions", []),
                severity=result.get("severity", "medium")
            )
        except Exception as e:
            # If parsing fails, assume not approved
            return PlanReviewResult(
                approved=False,
                concerns=[f"Failed to parse R2 response: {str(e)}"],
                suggestions=[],
                severity="high"
            )
    
    def _revise_plan(self, instruction: str, previous_code: str, review: PlanReviewResult) -> str:
        """Have Model R revise the plan based on R2's feedback."""
        prompt = ChatPromptTemplate.from_template(self.R_REVISION_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            previous_code=previous_code,
            concerns="\n".join([f"- {c}" for c in review.concerns]),
            suggestions="\n".join([f"- {s}" for s in review.suggestions])
        )
        
        response = self.llm.invoke(messages)
        return self._extract_code(response.content)
    
    def _judge_code(
        self, 
        instruction: str, 
        sop_chain: List[str], 
        code: str, 
        review: PlanReviewResult,
        rounds: int,
        verbose: bool = False
    ) -> Optional[str]:
        """Have Judge produce final corrected code."""
        if not self.llm_judge:
            # No judge, return None to indicate failure
            return None
        
        prompt = ChatPromptTemplate.from_template(self.JUDGE_CODE_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=", ".join(sop_chain),
            tools_as_functions=self.tools_as_functions,
            rules=self.rules,
            code=code,
            concerns="\n".join([f"- {c}" for c in review.concerns]),
            suggestions="\n".join([f"- {s}" for s in review.suggestions]) if review.suggestions else "No specific suggestions",
            rounds=rounds
        )
        
        response = self.llm_judge.invoke(messages)
        content = response.content.strip()
        
        # Extract code from response
        if "```python" in content:
            code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
            if code_match:
                final_code = code_match.group(1).strip()
                if verbose:
                    print("\n--- Judge's Final Corrected Code ---")
                    for i, line in enumerate(final_code.split('\n'), 1):
                        print(f"    {i:3}| {line}")
                return final_code
        elif "```" in content:
            code_match = re.search(r'```\n?(.*?)```', content, re.DOTALL)
            if code_match:
                final_code = code_match.group(1).strip()
                if verbose:
                    print("\n--- Judge's Final Corrected Code ---")
                    for i, line in enumerate(final_code.split('\n'), 1):
                        print(f"    {i:3}| {line}")
                return final_code
        
        # If no code block, maybe the whole response is code
        if content.startswith("#") or "=" in content.split('\n')[0] or content.split('\n')[0].endswith(")"):
            if verbose:
                print("\n--- Judge's Final Corrected Code ---")
                for i, line in enumerate(content.split('\n'), 1):
                    print(f"    {i:3}| {line}")
            return content
        
        if verbose:
            print(f"   ⚠️ Could not extract code from Judge response")
        return None
    
    def _execute_plan(self, code: str, verbose: bool = False) -> List[ExecutedAction]:
        """
        Execute the Python plan with real database feedback.
        
        Parses the code and executes tool calls, capturing results.
        """
        executed_actions = []
        
        # Reset database for fresh state and get database reference
        if self.action_executor:
            self.action_executor.service.reset_database()
            self.action_executor.service.reset_tools()
            database = self.action_executor.service.database
        else:
            raise RuntimeError("ActionExecutor not initialized - cannot execute plan")
        
        # Create a namespace for execution
        namespace = {}
        
        # Add tool functions to namespace
        for tool_name, tool_instance in self.tool_map.items():
            # Get the parameter names for this tool
            tool_info = tool_instance.get_info()
            param_names = list(tool_info.get('function', {}).get('parameters', {}).get('properties', {}).keys())
            
            def make_tool_func(tool_inst, name, param_order, db, normalize_fn):
                def tool_func(*args, **kwargs):
                    # Convert positional args to kwargs using param order
                    for i, arg in enumerate(args):
                        if i < len(param_order):
                            kwargs[param_order[i]] = arg
                    
                    # Execute the tool (first arg is database)
                    try:
                        result = tool_inst.invoke(db, **kwargs)
                        
                        # Parse result if it's JSON string
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except:
                                pass
                        
                        # Normalize types (floats to ints where whole numbers)
                        result = normalize_fn(result)
                        
                        # Record the action
                        action = ExecutedAction(
                            name=name,
                            kwargs=kwargs,
                            result=result,
                            success=True
                        )
                        executed_actions.append(action)
                        
                        if verbose:
                            print(f"✓ {name}({json.dumps(kwargs)[:100]}...)")
                            print(f"  Result: {str(result)[:200]}...")
                        
                        return result
                    except Exception as e:
                        action = ExecutedAction(
                            name=name,
                            kwargs=kwargs,
                            result=None,
                            success=False,
                            error=str(e)
                        )
                        executed_actions.append(action)
                        
                        if verbose:
                            print(f"✗ {name}({json.dumps(kwargs)[:100]}...)")
                            print(f"  Error: {str(e)}")
                        
                        raise
                
                return tool_func
            
            namespace[tool_name] = make_tool_func(tool_instance, tool_name, param_names, database, self._normalize_types)
        
        # Validate syntax first
        try:
            compile(code, "<plan>", "exec")
        except SyntaxError as e:
            if verbose:
                print(f"\n❌ Syntax error in generated code: {str(e)}")
                print(f"   Line {e.lineno}: {e.text}")
            # Return empty but flag the error
            executed_actions.append(ExecutedAction(
                name="_syntax_error",
                kwargs={"error": str(e), "line": e.lineno},
                result=None,
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}"
            ))
            return executed_actions
        
        # Execute the code
        try:
            exec(code, namespace)
        except Exception as e:
            if verbose:
                print(f"\n❌ Execution error: {str(e)}")
                traceback.print_exc()
            # Add error action so we know what happened
            executed_actions.append(ExecutedAction(
                name="_execution_error",
                kwargs={},
                result=None,
                success=False,
                error=str(e)
            ))
        
        return executed_actions
    
    def _execute_with_live_editing(
        self, 
        code: str, 
        instruction: str,
        sop_chain: List[str],
        verbose: bool = False
    ) -> Tuple[List[ExecutedAction], str]:
        """
        Execute code step-by-step with R2 monitoring and live editing on failures.
        
        Returns:
            Tuple of (executed_actions, final_code)
        """
        executed_actions = []
        
        # Reset database
        if self.action_executor:
            self.action_executor.service.reset_database()
            self.action_executor.service.reset_tools()
            database = self.action_executor.service.database
        else:
            raise RuntimeError("ActionExecutor not initialized")
        
        # Parse code into an AST to identify tool calls
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            executed_actions.append(ExecutedAction(
                name="_syntax_error",
                kwargs={"error": str(e)},
                result=None,
                success=False,
                error=str(e)
            ))
            return executed_actions, code
        
        # Create namespace with tool functions
        namespace = {}
        tool_call_results = {}  # Track results for R2 review
        current_action_idx = [0]  # Mutable counter
        
        for tool_name, tool_instance in self.tool_map.items():
            tool_info = tool_instance.get_info()
            param_names = list(tool_info.get('function', {}).get('parameters', {}).get('properties', {}).keys())
            
            def make_tool_func(tool_inst, name, param_order, db, normalize_fn, actions_list, results_dict, idx_ref):
                def tool_func(*args, **kwargs):
                    # Convert positional args to kwargs
                    for i, arg in enumerate(args):
                        if i < len(param_order):
                            kwargs[param_order[i]] = arg
                    
                    idx_ref[0] += 1
                    action_num = idx_ref[0]
                    
                    try:
                        result = tool_inst.invoke(db, **kwargs)
                        
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except:
                                pass
                        
                        result = normalize_fn(result)
                        
                        action = ExecutedAction(
                            name=name,
                            kwargs=kwargs,
                            result=result,
                            success=True
                        )
                        actions_list.append(action)
                        results_dict[action_num] = {"name": name, "kwargs": kwargs, "result": result, "success": True}
                        
                        if verbose:
                            print(f"   [{action_num}] ✓ {name}")
                        
                        return result
                    except Exception as e:
                        action = ExecutedAction(
                            name=name,
                            kwargs=kwargs,
                            result=None,
                            success=False,
                            error=str(e)
                        )
                        actions_list.append(action)
                        results_dict[action_num] = {"name": name, "kwargs": kwargs, "error": str(e), "success": False}
                        
                        if verbose:
                            print(f"   [{action_num}] ✗ {name}: {str(e)[:80]}")
                        
                        raise
                
                return tool_func
            
            namespace[tool_name] = make_tool_func(
                tool_instance, tool_name, param_names, database, 
                self._normalize_types, executed_actions, tool_call_results, current_action_idx
            )
        
        # Execute the code
        if verbose:
            print("\n   Executing step-by-step...")
        
        try:
            exec(code, namespace)
            if verbose:
                print(f"\n   ✓ All {len(executed_actions)} actions executed successfully")
            return executed_actions, code
            
        except Exception as e:
            if verbose:
                print(f"\n   ⚠️ Execution failed at action {current_action_idx[0]}: {str(e)}")
            
            # Get R2 to diagnose and R to fix
            failed_action = executed_actions[-1] if executed_actions else None
            
            if not failed_action:
                return executed_actions, code
            
            # Ask R2 what went wrong
            # The actual exception message (str(e)) contains the real error info
            actual_error = str(e)
            tool_error = failed_action.error if failed_action.error else "Tool succeeded, but Python code failed after"
            
            diagnosis_prompt = f"""Execution failed. Diagnose the issue.

## Python Exception (THE ACTUAL ERROR)
{actual_error}

## Last Tool Called
Tool: {failed_action.name}
Args: {json.dumps(failed_action.kwargs)}
Tool Result: {json.dumps(failed_action.result)[:500] if failed_action.result else 'None'}
Tool Error: {tool_error}

## Successful Actions Before Failure
{chr(10).join([f"- {a.name}: {str(a.result)[:100]}" for a in executed_actions[:-1]])}

## Current Code
```python
{code}
```

IMPORTANT: The Python Exception shows the ACTUAL error. If it says "Missing expected keys" and shows the actual result dict, use that to identify the correct key names!

What went wrong? How should the code be fixed?
Respond with:
1. DIAGNOSIS: What's the root cause?
2. FIX: Specific code change needed
"""
            
            if verbose:
                print("\n   🔍 R2 diagnosing failure...")
            
            diagnosis_response = self.llm_r2.invoke([HumanMessage(content=diagnosis_prompt)])
            diagnosis = diagnosis_response.content
            
            if verbose:
                print(f"   R2 Diagnosis: {diagnosis[:200]}...")
            
            # Ask R to fix the code
            fix_prompt = f"""The code failed during execution. Fix it based on R2's diagnosis.

## Instruction
{instruction}

## SOP Chain
{', '.join(sop_chain)}

## Available Tools (USE EXACT PARAMETER NAMES)
{self.tools_as_functions}

## CRITICAL: Parameter names are ALWAYS snake_case
- Tool parameters: board_id, sprint_id, channel_id (snake_case)
- Response fields may be camelCase in JSON, but parameters are ALWAYS snake_case
- Example: jira_get_backlog_issues(board_id=15) NOT jira_get_backlog_issues(boardId=15)

## CRITICAL: Python Exception (THE ACTUAL ERROR)
{actual_error}

## R2's Diagnosis
{diagnosis}

## Failed Action Details
Tool: {failed_action.name}
Args: {json.dumps(failed_action.kwargs)}
Tool Result: {json.dumps(failed_action.result)[:500] if failed_action.result else 'None'}

## Successful Results (use these)
{chr(10).join([f"{a.name}: {json.dumps(a.result)[:200]}" for a in executed_actions[:-1]])}

## Current Code (fix this)
```python
{code}
```

IMPORTANT: The Python Exception shows the ACTUAL error. If it mentions "wacc_used" instead of "wacc", use the CORRECT key name from the actual result!

Fix the code and return the COMPLETE corrected Python code. Keep parameter names as snake_case!

```python
# Your fixed code here
```"""
            
            if verbose:
                print("   🔄 R fixing code based on diagnosis...")
            
            fix_response = self.llm.invoke([HumanMessage(content=fix_prompt)])
            fixed_code = self._extract_code(fix_response.content)
            
            if fixed_code and fixed_code != code:
                if verbose:
                    print("   ✓ Code fixed, re-executing from scratch...")
                
                # Re-execute with fixed code (recursive, but only once)
                # Reset and try again
                return self._execute_plan(fixed_code, verbose=verbose), fixed_code
            else:
                if verbose:
                    print("   ⚠️ Could not fix code")
                return executed_actions, code
    
    def scaffold(
        self,
        instruction: str,
        task_id: str = "task_new",
        verbose: bool = False
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], List[str]]:
        """
        Generate task actions using the pseudo scaffolding approach.
        
        Args:
            instruction: User-facing task instruction
            task_id: Task ID for logging
            verbose: Print detailed progress
            
        Returns:
            Tuple of (actions, error, progress_log)
        """
        progress = []
        
        def log(msg: str):
            progress.append(msg)
            if verbose:
                print(msg)
        
        log("=" * 80)
        log("PSEUDO SCAFFOLDER - Code-Based Task Generation")
        log("=" * 80)
        log(f"\nInstruction: {instruction[:200]}...")
        
        # Step 1: Map instruction to SOP chain
        log("\n📋 Step 1: Mapping instruction to SOP chain...")
        mapping = self.sop_mapper.map_instruction(instruction, verbose=verbose)
        sop_chain = mapping.primary_chain.sops
        log(f"   SOP Chain: {', '.join(sop_chain)}")
        
        # Step 2: Generate initial plan
        log("\n🖊️  Step 2: Model R generating Python plan...")
        code = self._generate_plan(instruction, sop_chain, verbose=verbose)
        log(f"   Generated {len(code.split(chr(10)))} lines of code")
        
        if verbose:
            log("\n--- Model R's Initial Code ---")
            for i, line in enumerate(code.split('\n'), 1):
                log(f"   {i:3}| {line}")
        
        # Step 3: R2 review rounds
        log(f"\n🔍 Step 3: R2 Code Review (max {self.max_review_rounds} rounds)...")
        
        final_code = code
        review = None
        
        for round_num in range(1, self.max_review_rounds + 1):
            log(f"\n   Round {round_num}/{self.max_review_rounds}:")
            
            # Check syntax before review
            syntax_ok, syntax_error = self._validate_syntax(final_code)
            if not syntax_ok:
                log(f"   ❌ Syntax error: {syntax_error}")
                if round_num < self.max_review_rounds:
                    log(f"   🔄 Model R fixing syntax...")
                    # Ask R to fix the syntax error
                    fix_review = PlanReviewResult(
                        approved=False,
                        concerns=[f"Python syntax error: {syntax_error}"],
                        suggestions=["Fix the syntax error and ensure valid Python code"],
                        severity="critical"
                    )
                    final_code = self._revise_plan(instruction, final_code, fix_review)
                    continue
                else:
                    return None, f"Code has syntax errors after {round_num} rounds", progress
            
            review = self._review_plan(instruction, sop_chain, final_code)
            
            if review.approved:
                log(f"   ✓ R2 APPROVED the code")
                break
            
            log(f"   ⚠️  R2 found concerns ({review.severity} severity):")
            for concern in review.concerns:
                log(f"      - {concern}")
            
            if round_num < self.max_review_rounds:
                log(f"   🔄 Model R revising code...")
                final_code = self._revise_plan(instruction, final_code, review)
                
                if verbose:
                    log(f"\n   --- Revised Code (Round {round_num}) ---")
                    for i, line in enumerate(final_code.split('\n'), 1):
                        log(f"      {i:3}| {line}")
        
        # Step 4: Judge if needed OR proceed with low/medium severity
        if review and not review.approved:
            # If severity is low/medium, proceed anyway - these are minor issues
            if review.severity in ["low", "medium"]:
                log(f"\n📋 Step 4: R2 has {review.severity} severity concerns - proceeding anyway")
                log(f"   (Low/medium severity concerns don't block execution)")
                for concern in review.concerns[:3]:  # Show first 3
                    log(f"   ⚠️  {concern[:100]}...")
            else:
                # High/critical severity - Judge produces final corrected code
                log(f"\n🏛️  Step 4: Judge as Final Editor (R2 has {review.severity} severity concerns)...")
                log(f"   Judge will produce the final corrected code...")
                
                judge_code = self._judge_code(
                    instruction, sop_chain, final_code, review, self.max_review_rounds, verbose=verbose
                )
                
                if judge_code:
                    log(f"   ✓ Judge produced final corrected code")
                    final_code = judge_code
                else:
                    log(f"   ⚠️ Judge could not produce corrected code - using R's last version")
        
        # Final syntax check before execution
        syntax_ok, syntax_error = self._validate_syntax(final_code)
        if not syntax_ok:
            log(f"\n❌ Final code has syntax error: {syntax_error}")
            if verbose:
                log("\n--- Final Code (with error) ---")
                for i, line in enumerate(final_code.split('\n'), 1):
                    log(f"   {i:3}| {line}")
            return None, f"Code has syntax errors: {syntax_error}", progress
        
        # Step 5: Execute the plan with live editing on failures
        log("\n⚡ Step 5: Executing Python plan with live editing...")
        
        if verbose:
            log("\n--- Executing Code ---")
            for i, line in enumerate(final_code.split('\n'), 1):
                log(f"   {i:3}| {line}")
            log("")
        
        # Use live editing execution - R2 monitors, R fixes failures
        executed_actions, final_code = self._execute_with_live_editing(
            final_code, instruction, sop_chain, verbose=verbose
        )
        
        log(f"\n   Executed {len(executed_actions)} tool calls")
        
        # Check for errors
        errors = [a for a in executed_actions if not a.success]
        if errors:
            log(f"\n   ⚠️  {len(errors)} execution errors:")
            for err in errors:
                log(f"      - {err.name}: {err.error}")
            if errors[0].name == "_syntax_error":
                return None, f"Code has syntax errors: {errors[0].error}", progress
        
        # Step 6: Convert to Action format
        log("\n📦 Step 6: Converting to Action objects...")
        
        actions = []
        for action in executed_actions:
            if action.success:
                # Normalize kwargs - convert floats that are whole numbers to ints
                normalized_kwargs = self._normalize_types(action.kwargs)
                actions.append({
                    "name": action.name,
                    "kwargs": normalized_kwargs
                })
        
        log(f"   Generated {len(actions)} actions")
        
        # Format output
        log("\n" + "=" * 80)
        log("Generated Task (Ready to Use)")
        log("=" * 80)
        
        output_lines = [
            f"Task(",
            f'    annotator="human",',
            f'    user_id="{task_id}",',
            f'    instruction="{instruction[:100]}...",',
            f'    actions=[',
        ]
        
        for i, action in enumerate(actions):
            output_lines.append(f'        # Action {i+1}')
            output_lines.append(f'        Action(')
            output_lines.append(f'            name="{action["name"]}",')
            output_lines.append(f'            kwargs={json.dumps(action["kwargs"])},')
            output_lines.append(f'        ),')
        
        output_lines.extend([
            '    ],',
            '    outputs=[],',
            '),'
        ])
        
        for line in output_lines:
            log(line)
        
        log("\n✅ Pseudo scaffolding complete!")
        
        return actions, None, progress

