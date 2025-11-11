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

## Database Context (For Reference Only - DO NOT USE FOR ID PARAMETERS)

{database_context}

**⚠️ CRITICAL - Database Context is OFF-LIMITS for ID Parameters**

When a SOP includes entity lookup operations:
- ❌ NEVER use database context to fill entity IDs directly
- ❌ NEVER hardcode IDs when a lookup SOP will provide them
- ✅ ALWAYS use placeholders from lookup SOP outputs
- ✅ This ensures EVERY ID is derived from tool execution, not pre-computed from context

**CRITICAL RULE: If SOP chain includes lookup/locate SOPs:**
- Assume ALL IDs will be provided by those SOPs
- Use ONLY placeholders to reference their outputs
- Do NOT look at database context to fill these parameters
- The database context is only for understanding structure, NOT for filling ID parameters

**Example:**
- Instruction mentions entities by name
- SOP chain includes: "Locate Source Entity" → step 3 returns source entity ID
- SOP chain includes: "Locate Target Entity" → step 3 returns target entity ID
- ❌ WRONG: source_id = "abc123xyz" (NO! Never pre-fill from context)
- ✅ CORRECT: source_id = "{{{{lookup_source_entity_output.id}}}}" (Use SOP output)
- ✅ CORRECT: target_id = "{{{{lookup_target_entity_output.id}}}}" (Use second SOP output)

## Example Actions from This Domain

{example_actions}

## Your Task

Generate a complete action sequence that implements the SOP chain, following the EXACT pattern and format shown in the example actions above.

### ⚠️ CRITICAL REQUIREMENT ⚠️

YOU MUST GENERATE ACTIONS FOR **ALL** SOPS IN THE CHAIN - NO EXCEPTIONS

The SOP Chain lists SOPs in order: SOP #1, SOP #2, SOP #3, etc.

**MANDATORY RULES:**
1. Start with SOP #1 - generate all its actions
2. Then SOP #2 - generate all its actions
3. Then SOP #3 - generate all its actions
4. Continue until ALL SOPs are covered
5. ❌ NEVER skip a SOP (even if you think it's redundant)
6. ❌ NEVER consolidate 2+ SOPs into 1 action
7. ❌ NEVER add actions not listed in an SOP
8. ✅ ALWAYS generate actions for SOP #1, #2, #3, #4, #5, ... in exact order

**Verification:**
- Count the SOPs in the chain (e.g., 5 SOPs)
- Your actions MUST include work for all 5 SOPs
- Each SOP may have 1-5 actions (depends on steps)
- If SOP #1 has 3 steps, generate 3 actions for it
- Then move to SOP #2, etc.

### Action Generation Rules (CRITICAL - READ CAREFULLY)

**MANDATORY RULE: Generate Actions for EVERY SINGLE SOP in the Chain**

YOU MUST generate actions for ALL SOPs in the exact order they appear in the SOP Chain.
- The SOP Chain is numbered 1, 2, 3, 4, 5, etc.
- You MUST generate actions starting from SOP #1, then SOP #2, then SOP #3, etc.
- ❌ NEVER skip a SOP (even if it seems redundant)
- ❌ NEVER consolidate multiple SOPs into one action
- ❌ NEVER add actions that are not part of a listed SOP
- ✅ ALWAYS generate actions for SOP #1 first, then SOP #2, then SOP #3, etc.

**If a SOP has multiple steps, generate one action per step in sequence:**
- SOP "Process Entity" with 3 steps → Generate 3 actions in order
- Example: Step 1: tool_a, Step 2: tool_b, Step 3: tool_c

1. **Follow SOP Steps EXACTLY**: Each SOP has numbered steps. Generate actions for ALL steps in EXACT order.
   - ❌ DO NOT skip any SOP from the chain
   - ❌ DO NOT skip steps within a SOP (even if you have the data from Database Context)
   - ❌ DO NOT reorder steps for "optimization"
   - ❌ DO NOT batch operations unless SOP explicitly does so
   - ❌ DO NOT "optimize" by hardcoding values that should come from intermediate steps
   - ✅ DO follow step-by-step: "Step 1: do X, then do Y" means action_X, then action_Y
   - ✅ DO generate actions for SOP #1, then SOP #2, then SOP #3, etc. in exact order
   - ✅ DO include ALL steps of multi-step SOPs even if you know the final result

**CRITICAL EXAMPLE - Multi-Step SOP:**
Even if Database Context shows entity details, you MUST still generate ALL steps of the SOP:
```
Action(name="tool_step_1", kwargs={{"param_a": "value1"}}),
Action(name="tool_step_2", kwargs={{"param_b": "value2", "param_c": "value3"}}),
Action(name="tool_step_3", kwargs={{"param_d": "value4", "param_e": "value5"}}),
```
DO NOT skip to: `tool_step_3(param_d="known_value")`

2. **Example - Multi-Step SOP Pattern**:
   ```
   Step 1: tool_a1 → tool_a2     ← TWO actions in sequence
   Step 2: tool_b1 → tool_b2     ← TWO actions in sequence
   Step 3: tool_c1 → tool_c2     ← TWO actions in sequence
   Step 4: tool_d                ← ONE action
   ```
   **CORRECT sequence**: tool_a1 → tool_a2 → tool_b1 → tool_b2 → tool_c1 → tool_c2 → tool_d (7 actions)
   
   **WRONG (DO NOT DO THIS)**: tool_a1 → tool_b1 → tool_c1 → tool_a2 → tool_b2 → tool_c2 (6 actions, missing tool_d, wrong order)

3. **Include Prerequisites**: All prerequisite actions must come first

4. **Use Exact Parameter Names**: CRITICAL - Use ONLY parameter names from the tool signatures above
   - ❌ WRONG: Adding parameters that are not in the tool's schema
   - ✅ CORRECT: Only use parameters exactly as defined in the tool signature

5. **No Extra Parameters**: DO NOT add parameters that are not in the tool signature
   - Even if the instruction mentions dates/timestamps, ONLY use parameters defined in the tool schema
   - If you think a parameter is needed but it's not in the schema, DON'T add it
   - The tool will use its own defaults for any omitted optional parameters

6. **Validate Every Parameter**: Before adding a parameter to kwargs, CHECK:
   - ✅ Is this parameter in the tool's "parameters" → "properties" list?
   - ✅ Is the parameter name spelled EXACTLY as in the schema?
   - ❌ If NO to either question, DO NOT include that parameter

7. **Placeholders**: Use {{placeholder_name}} for values that depend on previous actions

8. **Deterministic Defaults**: Use defaults from rules for known values

9. **Comments**: Add comments to explain what each action does (mention SOP step number)

### When to Use Placeholders vs Concrete Values (CRITICAL)

**Use CONCRETE VALUES for:**
- ✅ Values explicitly stated in the instruction (e.g., entity names, time periods, quantities)
- ✅ Simple string parameters that are context-specific (e.g., application names, descriptions)
- ✅ Configuration keys (e.g., parameter names from configuration)
- ✅ Enum values (e.g., status types, modes)
- ✅ Numeric constants (e.g., counts, amounts)

**Use PLACEHOLDERS for:**
- ❌ Values that come from PREVIOUS action outputs (e.g., {{{{spreadsheet_id}}}}, {{{{income_statements}}}})
- ❌ Values that need TRANSFORMATION (e.g., {{{{TRANSFORM: merge data}}}})
- ❌ Dynamic IDs or generated values

**Example (CORRECT)**:
```json
{{
  "name": "configure_api_identity",
  "kwargs": {{"application_name": "Analysis Tool"}},  // ✅ CONCRETE - descriptive name
  "comment": "Configure API identity"
}}
```

**Example (WRONG)**:
```json
{{
  "name": "configure_api_identity",
  "kwargs": {{"application_name": "{{{{application_name}}}}"}},  // ❌ WRONG - use concrete value
  "comment": "Configure API identity"
}}
```

### Placeholder Format (For Previous Action Outputs Only)

**CRITICAL: NEVER use generic placeholders like {{{{placeholder}}}} or {{{{value}}}}!**

**Direct field names** (when action output contains the field):
- {{{{entity_id}}}} - if previous action returned {{"entity_id": "..."}}
- {{{{data_records}}}} - if previous action returned {{"data_records": [...]}}

**Action output reference** (use the EXACT action name):
- {{{{tool_name_a_output}}}} - references the COMPLETE output from tool_name_a
- {{{{tool_name_b_output}}}} - references the COMPLETE output from tool_name_b
- {{{{tool_name_c_output}}}} - references the COMPLETE output from tool_name_c

**Nested access** (when you need a specific field from an action):
- {{{{tool_name_a_output.field_name}}}} - get field_name from tool_name_a output
- {{{{action_0_output.field_x}}}} - get field_x from first action
- {{{{tool_name_b_output.records[*].id}}}} - extract id from ALL records in the array
- {{{{tool_name_b_output.records[0].name}}}} - get name from first record
- {{{{action_2_output.nested.field}}}} - get nested.field from action 2

**Critical Rules**:
- ❌ NEVER use: {{{{placeholder}}}}, {{{{value}}}}, {{{{data}}}}, {{{{result}}}}
- ✅ ALWAYS use: {{{{action_name_output}}}} or {{{{field_name}}}}
- Use EXACT field names from tool outputs (check tool return schemas!)
- If a tool returns {{"success": true, "field_a": [...]}}, use {{{{field_a}}}}
- DON'T invent field names like {{{{extracted_data}}}} - use the actual field name
- DON'T prefix with "extracted_" or "data_" unless that's the actual field name

**Example - Correct placeholder usage:**
```
# Action 1: tool_build_filter returns a dict
Action(name="tool_build_filter", kwargs={{"param_a": "...", "param_b": [...]}})

# Action 2: tool_query needs that dict - use {{{{action_name_output}}}} format
Action(name="tool_query", kwargs={{"filter": "{{{{tool_build_filter_output}}}}"}})
```

**Example - Array field extraction:**
```
# Action 1: tool_lookup_entities returns {{entities: [...], missing: []}}
Action(name="tool_lookup_entities", kwargs={{"entity_names": ["Name1", "Name2"]}})

# Action 2: Extract entity IDs from the array using [*].id syntax
Action(name="tool_build_filter", kwargs={{
    "source_id": "abc123",
    "entity_ids": "{{{{tool_lookup_entities_output.entities[*].id}}}}"  # Extracts id from ALL entities
}})
```

**Key insight for array extraction:**
- Use `[*]` to extract a field from ALL elements in an array
- Use `[0]`, `[1]`, etc. to extract from a specific index
- Chain multiple levels: `output.data[*].nested.field`

**WHEN TO USE array extraction [*] vs TRANSFORM:**
✅ **USE [*] when:**
- Extracting a SINGLE field from all array elements
- Example: `entities[*].id` - just get the IDs
- Example: `records[*].value` - just get the values

❌ **USE TRANSFORM when:**
- Need to COMBINE data from MULTIPLE sources
- Need to FILTER or MATCH records based on conditions
- Need COMPLEX logic beyond simple field extraction

**Example - CORRECT usage:**
```
# CORRECT: Simple field extraction - use [*]
Action(name="tool_build_filter", kwargs={{
    "entity_ids": "{{{{tool_lookup_output.records[*].id}}}}"
}})

# WRONG: Don't use TRANSFORM for simple extraction
Action(name="tool_build_filter", kwargs={{
    "entity_ids": "{{{{TRANSFORM: extract IDs from records}}}}"  # ❌ Unnecessarily complex!
}})
```

**CRITICAL - Placeholder naming convention:**
- For dict/object parameters, ALWAYS use `{{{{action_name_output}}}}` format
- Match parameter name to previous action name
- Examples:
  - `{{{{"filter": "{{{{tool_build_filter_output}}}}"}}}}`
  - `{{{{"entities": "{{{{tool_query_entities_output}}}}"}}}}`
  - `{{{{"validated_data": "{{{{tool_validate_output}}}}"}}}}`
  - `{{{{"updates": "{{{{tool_compose_updates_output}}}}"}}}}`

**Pattern**: Use the EXACT tool name that produced the output

**WRONG - DO NOT DO THIS:**
```
Action(name="tool_query", kwargs={{"filter": "{{{{placeholder}}}}"}})  # ❌ Too generic!
Action(name="tool_query", kwargs={{"filter": "{{{{data}}}}"}})  # ❌ Too generic!
Action(name="tool_query", kwargs={{"filter": "{{{{filter}}}}"}})  # ❌ Same as param name!
```

**Data Transformation Detection** (CRITICAL):
When you need to COMBINE/MERGE/TRANSFORM data from multiple sources, use special syntax:

- {{{{TRANSFORM: merge dataset_a, dataset_b, dataset_c by key_field}}}}
- {{{{CALCULATE: combine field_x from dataset_a with field_y from dataset_b}}}}
- {{{{MANUAL: create normalized_records from multiple data sources}}}}

Use TRANSFORM/CALCULATE/MANUAL when:
1. Combining arrays from multiple sources into one
2. Extracting fields from different sources into new structure
3. Matching/joining records across multiple arrays
4. Any logic beyond simple field lookup

Example needing transformation:
```
persist_normalized_data(
  entity_id: "entity_123",
  normalized_records: {{{{TRANSFORM: merge dataset_a, dataset_b, dataset_c by period}}}}
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

For instruction "Extract entity data for analysis":
```json
{{
  "actions": [
    {{
      "name": "setup_environment",
      "kwargs": {{"env_name": "analysis_env", "env_type": "production"}},
      "comment": "SOP: Setup Environment"
    }},
    {{
      "name": "get_configuration",
      "kwargs": {{"param_name": "ApiKey"}},
      "comment": "SOP: Configure Access - Step 1"
    }},
    {{
      "name": "extract_primary_data",
      "kwargs": {{"entity_id": "entity_123", "num_records": 10}},
      "comment": "SOP: Extract Data - Step 1"
    }},
    {{
      "name": "store_primary_data",
      "kwargs": {{"entity_id": "entity_123", "data_records": "{{{{data_records}}}}"}},
      "comment": "SOP: Extract Data - Step 2 (NOTE: placeholder is QUOTED)"
    }}
  ]
}}
```

## Output Format

{format_instructions}

**CRITICAL - outputs field:**
- The `outputs` field must ALWAYS be an empty list: `[]`
- DO NOT add any natural language descriptions, summaries, or expected results
- The outputs field is reserved for structured data returned by the final action (NOT natural language)
- ❌ WRONG: `"outputs": ["Kevin Stein is now the owner..."]`
- ✅ CORRECT: `"outputs": []`

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
        """Get ALL rules from rules.py without truncation."""
        try:
            sops = self.sop_mapper.sops
            if isinstance(sops, list) and sops:
                lines = []
                
                # ALWAYS include SOP EXECUTION DISCIPLINE rule first (critical for scaffolding)
                discipline_rule = None
                for rule in sops:
                    if "SOP EXECUTION DISCIPLINE" in rule:
                        discipline_rule = rule
                        lines.append("**CRITICAL - SOP EXECUTION DISCIPLINE:**")
                        lines.append(rule.strip())
                        lines.append("")
                        break
                
                # Then add ALL other rules (no limit, no truncation)
                for rule in sops:
                    if rule != discipline_rule:  # Skip the one we already added
                        lines.append(rule.strip())
                        lines.append("")
                
                return "\n".join(lines) if lines else "See rules.py for details"
            return "See rules.py for details"
        except Exception:
            return "See rules.py for details"
    
    def _get_database_context(self, instruction: str) -> str:
        """
        Extract relevant database context based on the instruction.
        
        Looks for entity references (IDs, names, etc.) mentioned in the
        instruction and provides their current state from the database.
        """
        try:
            import re
            import json
            
            lines = []
            db = self.executor.service.database
            
            # Extract entity IDs from instruction (common patterns: 15-18 character alphanumeric)
            # This covers various ID formats: prefixed IDs, ObjectIds, UUIDs, etc.
            entity_ids = re.findall(r'\b[0-9a-zA-Z]{15,18}\b', instruction)
            
            if entity_ids:
                lines.append("**Entity IDs mentioned in instruction:**")
                
                # Try to find these IDs in any table
                found_entities = {}
                for table_name, table_data in db.items():
                    if not isinstance(table_data, list):
                        continue
                    
                    for entity_id in entity_ids:
                        for record in table_data:
                            if isinstance(record, dict) and record.get('Id') == entity_id:
                                if table_name not in found_entities:
                                    found_entities[table_name] = []
                                
                                # Extract key fields
                                entity_info = {'Id': entity_id}
                                for key in ['Name', 'Email', 'OwnerId', 'CreatedBy', 'Status']:
                                    if key in record:
                                        entity_info[key] = record[key]
                                
                                found_entities[table_name].append(entity_info)
                                break
                
                # Display found entities
                for table_name, entities in found_entities.items():
                    lines.append(f"  From table '{table_name}':")
                    for entity in entities:
                        id_str = entity.get('Id', 'unknown')
                        name_str = entity.get('Name', entity.get('Email', ''))
                        owner_str = entity.get('OwnerId', entity.get('CreatedBy', ''))
                        
                        if name_str:
                            lines.append(f"    - {name_str} (Id: {id_str})")
                        else:
                            lines.append(f"    - Id: {id_str}")
                        
                        if owner_str:
                            lines.append(f"      Owner/Creator: {owner_str}")
                
                lines.append("")
            
            # Extract person names from instruction (capitalized words)
            person_names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', instruction)
            if person_names:
                lines.append("**Person names mentioned in instruction:**")
                
                # Look in any table that might have Name field
                seen = set()
                for name in person_names:
                    if name in seen:
                        continue
                    seen.add(name)
                    
                    found = False
                    for table_name, table_data in db.items():
                        if not isinstance(table_data, list):
                            continue
                        
                        for record in table_data:
                            if isinstance(record, dict) and record.get('Name') == name:
                                lines.append(f"  - {name} (in '{table_name}' table)")
                                if 'Id' in record:
                                    lines.append(f"    Id: {record.get('Id')}")
                                if 'Email' in record:
                                    lines.append(f"    Email: {record.get('Email')}")
                                found = True
                                break
                        
                        if found:
                            break
                # Separator
                lines.append("")
            
            # Show database table names
            lines.append("**Available database tables:**")
            for table_name in sorted(db.keys()):
                table_data = db[table_name]
                if isinstance(table_data, list):
                    lines.append(f"  - {table_name}: {len(table_data)} records")
            
            return "\n".join(lines) if lines else "No specific database context extracted"
            
        except Exception as e:
            return f"Could not extract database context: {e}"
    
    def _get_example_actions(self) -> str:
        """
        Get example actions from task_001 to show the LLM the exact format expected.
        """
        try:
            tasks = self.executor.get_available_tasks()
            if 'task_001' in tasks:
                task = tasks['task_001']
                lines = ["Here are example actions from a similar task showing the EXACT format to follow:"]
                lines.append("")
                # Show first 5 actions as examples
                for i, action in enumerate(task.actions[:5]):
                    lines.append(f"Action(")
                    lines.append(f'    name="{action.name}",')
                    lines.append(f'    kwargs={{')
                    for key, value in action.kwargs.items():
                        if isinstance(value, str):
                            # Escape quotes and show exact format
                            escaped = value.replace('"', '\\"')
                            lines.append(f'        "{key}": "{escaped}",')
                        elif isinstance(value, list):
                            lines.append(f'        "{key}": {value},')
                        else:
                            lines.append(f'        "{key}": {value},')
                    lines.append(f'    }},')
                    lines.append(f'),')
                    if i < 4:
                        lines.append("")
                lines.append("")
                lines.append("**Key observations:**")
                lines.append("- JSON strings are properly escaped and formatted")
                lines.append("- All IDs use exact deterministic values from the database")
                lines.append("- Each SOP step has its own action")
                return "\n".join(lines)
            return "No example task available"
        except Exception as e:
            return f"Could not load example actions: {e}"
    
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
            
            # Find the SOPs section in rules (look for SOP definitions)
            sop_section = None
            for rule in sops:
                if "SOP " in rule and ("Step 1:" in rule or "need " in rule):
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
    
    def _validate_scaffold_sop_coverage(
        self,
        sop_chain: List[str],
        scaffold: 'TaskScaffold',
        progress: List[str]
    ) -> List[str]:
        """
        Validate that the generated scaffold has actions for ALL SOPs in the chain.
        
        This is a domain-agnostic check that ensures the LLM didn't skip any SOPs
        when generating actions.
        
        Args:
            sop_chain: List of SOP names that should be implemented
            scaffold: Generated scaffold with actions
            progress: Progress log
            
        Returns:
            List of validation issues (empty if all SOPs are covered)
        """
        issues = []
        
        # Extract SOP mentions from action comments
        action_comments = [action.comment or "" for action in scaffold.actions]
        all_comments = " ".join(action_comments).lower()
        
        # For each SOP in chain, check if it appears in action comments
        for sop in sop_chain:
            # Normalize SOP name for matching
            sop_normalized = sop.lower()
            
            # Extract key terms from SOP (e.g., "Create Quota For Period" -> ["create", "quota"])
            # Remove common words and parameters
            sop_clean = sop_normalized.split('(')[0].strip()  # Remove parameters
            
            # Check if SOP is mentioned in comments
            if sop_clean not in all_comments:
                # SOP not found - flag as missing
                # Note: We don't use lenient fallback matching because it causes false positives
                # (e.g., "Lookup Accounts by Name" matching "Lookup Account Roster For Transfer")
                issues.append(f"SOP '{sop}' not found in generated actions")
        
        # Also check by count - if we have way fewer actions than expected
        # Rough heuristic: each SOP should have at least 1-3 actions
        min_expected_actions = len(sop_chain)
        max_expected_actions = len(sop_chain) * 10  # Upper bound
        
        if len(scaffold.actions) < min_expected_actions:
            issues.append(f"Too few actions: {len(scaffold.actions)} actions for {len(sop_chain)} SOPs (expected at least {min_expected_actions})")
        
        if len(scaffold.actions) > max_expected_actions:
            issues.append(f"Too many actions: {len(scaffold.actions)} actions for {len(sop_chain)} SOPs (expected at most {max_expected_actions})")
        
        return issues
    
    def _validate_sop_completeness(
        self, 
        instruction: str, 
        sop_chain: List[str], 
        progress: List[str]
    ) -> None:
        """
        Validate that the SOP chain covers all operations mentioned in the instruction.
        
        Warns if the instruction mentions operations that don't appear to be covered
        by the generated SOP chain.
        
        Args:
            instruction: The task instruction
            sop_chain: List of SOP names in the chain
            progress: Progress log to append warnings to
        """
        instruction_lower = instruction.lower()
        sop_chain_str = ' '.join(sop_chain).lower()
        warnings = []
        
        # Check for quota operations
        quota_keywords = ['quota', 'target', 'goal', 'revenue target', 'sales target']
        if any(keyword in instruction_lower for keyword in quota_keywords):
            quota_sops = ['create quota', 'update quota', 'fetch quota', 'delete quota']
            if not any(sop in sop_chain_str for sop in quota_sops):
                warnings.append("⚠️  Instruction mentions quota/target but no quota SOP found in chain")
        
        # Check for transfer/reassignment operations
        transfer_keywords = ['transfer', 'reassign', 'move', 'shift', 'hand over', 'hand off']
        if any(keyword in instruction_lower for keyword in transfer_keywords):
            transfer_sops = ['reassign', 'transfer', 'update account owner', 'account roster']
            if not any(sop in sop_chain_str for sop in transfer_sops):
                warnings.append("⚠️  Instruction mentions transfer/reassign but no transfer SOP found in chain")
        
        # Check for performance/aggregation operations
        # Look for analysis verbs combined with performance indicators
        analysis_verbs = ['understand', 'review', 'check', 'analyze', 'see', 'view', 'assess', 'evaluate', 'examine']
        performance_indicators = ['performance', 'results', 'attainment', 'revenue', 'sales results', 'closed-won', 'aggregate', 'capacity', 'baseline']
        
        has_analysis_verb = any(verb in instruction_lower for verb in analysis_verbs)
        has_performance_indicator = any(indicator in instruction_lower for indicator in performance_indicators)
        
        # Also check for explicit performance keywords or baseline patterns
        explicit_performance = any(keyword in instruction_lower for keyword in ['aggregate', 'total revenue', 'sales results'])
        baseline_patterns = any(pattern in instruction_lower for pattern in ['to baseline', 'establish baseline', 'assess capacity'])
        
        if (has_analysis_verb and has_performance_indicator) or explicit_performance or baseline_patterns:
            perf_sops = ['aggregate', 'performance', 'closed-won', 'attainment']
            if not any(sop in sop_chain_str for sop in perf_sops):
                warnings.append("⚠️  Instruction mentions performance analysis/review but no aggregate performance SOP found in chain")
        
        # Check for account lookup operations
        import re
        
        # Check if accounts are mentioned
        account_keywords = ['account', 'accounts']
        has_account_keyword = any(keyword in instruction_lower for keyword in account_keywords)
        
        # Pattern for account names: capitalized multi-word names (e.g., "Delta Freight", "BlueCurve Analytics")
        # Look for proper nouns followed by transfer/move keywords
        account_name_patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:should move|should go|to move|to transfer)',
            r'(?:transfer|move|reassign)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+from',
            r'accounts?[—\-:]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
        ]
        
        has_account_names = any(re.search(pattern, instruction) for pattern in account_name_patterns)
        
        # If accounts are mentioned with names, check for lookup SOP
        if has_account_keyword and has_account_names:
            lookup_sops = ['lookup account', 'query account', 'find account', 'account by name']
            if not any(sop in sop_chain_str for sop in lookup_sops):
                warnings.append("⚠️  Instruction mentions specific account names but no account lookup SOP found in chain")
        
        # Check for multi-workflow indicators
        multi_workflow_keywords = ['also', 'additionally', 'furthermore', 'plus', 'as well as', 'and also']
        has_multi_workflow = any(keyword in instruction_lower for keyword in multi_workflow_keywords)
        
        if has_multi_workflow and len(warnings) > 0:
            warnings.insert(0, "⚠️  CRITICAL: Instruction contains multi-workflow indicator (e.g., 'also') - verify ALL workflows are included:")
        
        # Log all warnings
        for warning in warnings:
            progress.append(warning)
    
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
            execute: Whether to execute actions to fill placeholders (MUST BE False - execution mode is in development)
            max_retries: Maximum number of attempts (default: 3)
            
        Returns:
            Tuple of (scaffold, error, progress_log, model_info)
            - scaffold: Complete TaskScaffold if successful
            - error: Error message if failed
            - progress_log: List of progress messages
            - model_info: String describing which model(s) generated the scaffold
        """
        # CRITICAL: Execution mode is in development - users must use --no-execute
        if execute:
            error_msg = (
                "❌ ERROR: Execution mode is currently in development.\n"
                "\n"
                "Please use --no-execute flag:\n"
                "  python ../tau_helper/run.py scaffold DOMAIN --variation VARIATION --task TASK_ID --no-execute\n"
                "\n"
                "The --no-execute mode generates tasks with {{placeholders}}.\n"
                "Then use the action executor to fill placeholders manually:\n"
                "  python ../tau_helper/run.py execute DOMAIN --variation VARIATION --task TASK_ID --all-actions\n"
            )
            return None, error_msg, [error_msg], None
        
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
            
            # Validate SOP completeness
            self._validate_sop_completeness(instruction, sop_chain, progress)
            
            # Step 2: Generate action sequence with placeholders
            if self.multi_agent_enabled:
                progress.append("\n🤖 Multi-Agent Mode: Generating with both R and R2 models...")
                
                # Initialize validation tracking
                validation_issues_r = []
                validation_issues_r2 = []
                
                # Generate with R model (primary)
                progress.append("  → Generating with Model R...")
                try:
                    scaffold_r = self._generate_scaffold(instruction, sop_chain, task_id)
                    progress.append(f"    ✓ Model R: {len(scaffold_r.actions)} actions")
                    
                    # Note: Validation removed - causes too many false negatives
                    # The improved prompt with mandatory SOP rules is more reliable
                    validation_issues_r = []
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
                    
                    # Note: Validation removed - causes too many false negatives
                    validation_issues_r2 = []
                except Exception as e:
                    self.llm = original_llm  # Restore on error
                    progress.append(f"    ❌ Model R2 failed: {str(e)}")
                    # Continue with R model only
                    scaffold = scaffold_r
                    model_info = "Model R (R2 failed)"
                    progress.append("  → Using Model R scaffold only")
                else:
                    # Both succeeded - compare validation results
                    # Prefer the scaffold with fewer validation issues
                    if validation_issues_r and not validation_issues_r2:
                        progress.append("  → Model R2 has better SOP coverage, using R2")
                        scaffold = scaffold_r2
                        model_info = "R2 (better coverage)"
                    elif validation_issues_r2 and not validation_issues_r:
                        progress.append("  → Model R has better SOP coverage, using R")
                        scaffold = scaffold_r
                        model_info = "R (better coverage)"
                    elif self._scaffolds_are_equal(scaffold_r, scaffold_r2):
                        progress.append("  ✓ Both models agree!")
                        # Use R2 (Claude) by default as it's more reliable
                        scaffold = scaffold_r2
                        model_info = "Consensus (using R2)"
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
                    
                    # Note: Validation removed - causes too many false negatives
                    # The improved prompt with mandatory SOP rules is more reliable
                    
                    model_info = "Single Model"
                except Exception as e:
                    progress.append(f"❌ Failed to generate scaffold: {str(e)}")
                    raise
            
            # Scaffold is complete with placeholders!
            # Fix any generic placeholders BEFORE execution or return
            progress.append("\nStep 2b: Fixing generic placeholders and applying inference...")
            scaffold = self._fix_generic_placeholders(scaffold)
            progress.append("✓ Fixed placeholders")
            
            # NO-EXECUTE mode stops here
            if not execute:
                return scaffold, None, progress, model_info
            
            # EXECUTE mode: Test the scaffold to verify it works and collect final result
            progress.append("\nStep 3: Verifying scaffold by executing it...")
            try:
                # Execute all actions sequentially and capture resolved kwargs
                context = {}
                final_result = None
                resolved_kwargs_list = []  # Store the actual kwargs used for each action
                
                for i, action in enumerate(scaffold.actions):
                    progress.append(f"  Action {i+1}/{len(scaffold.actions)}: {action.name}")
                    
                    # Resolve kwargs
                    filled_kwargs = self._resolve_kwargs(
                        action.name, 
                        action.kwargs, 
                        context, 
                        progress,
                        self._get_tool_schema(action.name)
                    )
                    
                    # Store the resolved kwargs for later use
                    resolved_kwargs_list.append(filled_kwargs)
                    
                    # Execute
                    result = self.executor.execute_action_by_name(action.name, filled_kwargs)
                    progress.append(f"    ✓ Success")
                    
                    # Update context
                    context[f"{action.name}_output"] = result
                    context[f"action_{i}_output"] = result
                    
                    # Parse if JSON string
                    if isinstance(result, str):
                        try:
                            parsed = json.loads(result)
                            context[f"{action.name}_output_parsed"] = parsed
                            context[f"action_{i}_output_parsed"] = parsed
                            if isinstance(parsed, dict):
                                for key, value in parsed.items():
                                    context[key] = value
                        except:
                            pass
                    elif isinstance(result, dict):
                        context[f"{action.name}_output_parsed"] = result
                        for key, value in result.items():
                            context[key] = value
                    
                    final_result = result
                
                progress.append(f"  ✓ All actions executed successfully!")
                
            except Exception as e:
                error_msg = f"Scaffold execution failed: {str(e)}"
                progress.append(f"  ❌ {error_msg}")
                return None, error_msg, progress, model_info
            
            # Step 4: Fill any remaining generic placeholders with deterministic values from execution
            progress.append("\nStep 4: Filling deterministic JSON values from execution...")
            clean_scaffold = self._fill_json_from_execution(scaffold, resolved_kwargs_list, progress)
            
            # Derive outputs from final result
            progress.append("\nStep 5: Deriving task outputs...")
            if isinstance(final_result, dict):
                outputs = []
                skip_keys = {'success', 'error', 'message'}
                for key, value in final_result.items():
                    if key in skip_keys:
                        continue
                    if isinstance(value, list):
                        outputs.append(f'"{key}": {len(value)}')
                    elif isinstance(value, dict):
                        outputs.append(f'"{key}": {len(value)} fields')
                    else:
                        outputs.append(f'"{key}": {value}')
                clean_scaffold.outputs = outputs if outputs else []
                progress.append(f"  ✓ Derived {len(clean_scaffold.outputs)} output(s)")
            else:
                clean_scaffold.outputs = []
                progress.append(f"  ✓ No structured outputs")
            
            progress.append(f"\n✅ Task scaffold complete!")
            return clean_scaffold, None, progress, model_info
            
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
        
        # Get database context for entities mentioned in instruction
        database_context = self._get_database_context(instruction)
        
        # Get example actions to show exact format
        example_actions = self._get_example_actions()
        
        # Format prompt
        formatted_prompt = self.prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_str,
            sop_definitions=sop_definitions,
            tools_description=tools_desc,
            rules_summary=rules_summary,
            database_context=database_context,
            example_actions=example_actions,
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Get LLM response
        response = self.llm.invoke(formatted_prompt)
        
        # Parse scaffold
        try:
            # First normalize any double-escaped braces in the raw response
            normalized_content = self._normalize_placeholder_braces(response.content)
            scaffold = self.parser.parse(normalized_content)
        except Exception as e1:
            # Try cleaning response
            try:
                cleaned = self._clean_llm_response(response.content)
                cleaned = self._normalize_placeholder_braces(cleaned)
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
        
        # Post-process: Normalize placeholders in all actions (fix {{{{...}}}})
        for i, action in enumerate(scaffold.actions):
            before_kwargs = str(action.kwargs)
            action.kwargs = self._normalize_placeholders_in_dict(action.kwargs)
            after_kwargs = str(action.kwargs)
            if before_kwargs != after_kwargs:
                pass  # Normalization applied silently
        
        # Post-process: Remove non-existent tools
        scaffold = self._filter_nonexistent_tools(scaffold)
        
        # Post-process: Fix generic placeholders
        scaffold_before = scaffold
        scaffold = self._fix_generic_placeholders(scaffold)
        
        # Debug: Show if any placeholders were fixed
        for i, (action_before, action_after) in enumerate(zip(scaffold_before.actions, scaffold.actions)):
            for key in action_after.kwargs:
                before_val = str(action_before.kwargs.get(key, ''))
                after_val = str(action_after.kwargs.get(key, ''))
                if before_val != after_val and ('{{' in before_val or '{{' in after_val):
                    # Placeholder was fixed
                    pass  # Silently fix without logging
        
        return scaffold
    
    def _filter_nonexistent_tools(self, scaffold: TaskScaffold) -> TaskScaffold:
        """
        Remove actions that reference non-existent tools.
        
        Some LLMs hallucinate prerequisite tools (like setup_python_environment)
        that don't exist in all domains. This filters them out.
        """
        available_tools = set(self.executor.get_available_tools().keys())
        
        filtered_actions = []
        for action in scaffold.actions:
            if action.name in available_tools:
                filtered_actions.append(action)
        
        scaffold.actions = filtered_actions
        return scaffold
    
    def _normalize_placeholders_in_dict(self, obj):
        """Recursively normalize placeholder braces in dicts/lists/strings."""
        import re
        
        if isinstance(obj, dict):
            return {k: self._normalize_placeholders_in_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_placeholders_in_dict(item) for item in obj]
        elif isinstance(obj, str):
            # Fix quadruple braces: {{{{...}}}} → {{...}}
            result = obj
            if '{{{{' in result:
                result = re.sub(r'\{\{\{\{(.+?)\}\}\}\}', r'{{\1}}', result)
            return result
        else:
            return obj
    
    def _normalize_placeholder_braces(self, content: str) -> str:
        """
        Normalize double-escaped placeholder braces.
        
        Converts {{{{placeholder}}}} → {{placeholder}}
        
        This handles cases where LLMs escape braces in JSON output.
        """
        import re
        # Replace quadruple braces with double braces
        content = re.sub(r'\{\{\{\{([^}]+)\}\}\}\}', r'{{\1}}', content)
        return content
    
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
        Post-process scaffold to replace generic placeholders with specific references.
        
        Detects patterns like {{placeholder}}, {{value}}, {{data}} and replaces them with
        appropriate references to previous action outputs based on context.
        
        Also normalizes double-escaped placeholders ({{{{...}}}}) to proper format ({{...}}).
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
        
        for i, action in enumerate(scaffold.actions):
            for param_name, param_value in list(action.kwargs.items()):
                # Check if value is a string
                if isinstance(param_value, str):
                    original_value = param_value
                    
                    # First, fix double-escaped placeholders: {{{{...}}}} → {{...}}
                    # Use a more robust regex that handles any content between braces
                    if '{{{{' in param_value:
                        # Replace {{{{anything}}}} with {{anything}}
                        param_value = re.sub(r'\{\{\{\{(.+?)\}\}\}\}', r'{{\1}}', param_value)
                        action.kwargs[param_name] = param_value
                    
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
                        placeholder_match = re.search(r'\{\{([^}]+)\}\}', param_value)
                        if placeholder_match:
                            placeholder_content = placeholder_match.group(1).strip()
                            # If placeholder is exactly the parameter name, it's generic
                            if placeholder_content.lower() == param_name.lower():
                                found_generic = True
                            # Also check if this is a dict/object parameter that should reference previous action
                            # Common patterns: *_filter, *_data, *_request, *_updates, *_payload
                            elif any(suffix in param_name for suffix in ['_filter', '_data', '_request', '_updates', '_payload']):
                                found_generic = True
                            elif param_name.startswith(('validated_', 'formatted_', 'composed_', 'built_')):
                                found_generic = True
                    
                    # Also check for TRANSFORM that could be replaced with specific inference
                    if 'TRANSFORM' in param_value.upper() and not found_generic:
                        # Try to infer a specific placeholder instead of TRANSFORM
                        replacement = self._infer_placeholder_from_context(
                            action.name,
                            param_name,
                            param_value,
                            scaffold.actions[:i]
                        )
                        if replacement:
                            # Successfully inferred - use it instead of TRANSFORM
                            action.kwargs[param_name] = replacement
                            replacements_made += 1
                            found_generic = False  # Don't process again
                        else:
                            # Can't infer - let TRANSFORM be handled normally
                                found_generic = True
                    
                    if found_generic:
                        # Try to infer the correct placeholder from previous action
                        replacement = self._infer_placeholder_from_context(
                            action.name,
                            param_name,
                            param_value,
                            scaffold.actions[:i]  # Previous actions
                        )
                        if replacement:
                            action.kwargs[param_name] = replacement
                            replacements_made += 1
                        else:
                            # Fallback to transformation
                            replacement = self._generate_transform_for_parameter(
                                action.name, 
                                param_name,
                                param_value
                            )
                            action.kwargs[param_name] = replacement
                            replacements_made += 1
        
        return scaffold
    
    def _infer_array_field_extraction(
        self,
        param_name: str,
        previous_actions: List[TaskAction]
    ) -> Optional[str]:
        """
        Infer array field extraction for parameters ending with _ids, _list, etc.
        
        Strategy:
        1. Find previous action that returns an object with an array field
        2. Look for array field that contains objects with 'Id' or matching field
        3. Generate array extraction syntax: {{action_output.array_field[*].Id}}
        
        Example:
        - param_name: "account_ids"
        - Previous action "query_accounts_by_name" returns: {"accounts": [...], "missing": []}
        - Infer: {{query_accounts_by_name_output.accounts[*].Id}}
        """
        # Extract the entity type from param_name
        # E.g., "account_ids" -> "account", "user_ids" -> "user"
        if param_name.endswith('_ids'):
            entity_type = param_name[:-4]  # Remove "_ids"
        elif param_name.endswith('_list'):
            entity_type = param_name[:-5]  # Remove "_list"
        else:
            return None
        
        # Look for previous actions that might have queried this entity type
        for prev_action in reversed(previous_actions):
            action_name_lower = prev_action.name.lower()
            
            # Check if action name contains the entity type or common query patterns
            if (entity_type in action_name_lower or 
                'query' in action_name_lower or 
                'get' in action_name_lower or
                'fetch' in action_name_lower or
                'lookup' in action_name_lower):
                
                # Try common patterns for array field names
                # Plural form of entity: "accounts", "users", "items", etc.
                plural_forms = [
                    f"{entity_type}s",      # accounts, users
                    f"{entity_type}es",     # addresses, boxes
                    entity_type,            # data (no plural)
                    "items",                # generic
                    "results",              # generic
                    "data",                 # generic
                ]
                
                for array_field in plural_forms:
                    # Generate the extraction syntax
                    # Use _parsed suffix to access the dict version of the output
                    placeholder = f"{{{{{prev_action.name}_output_parsed.{array_field}[*].Id}}}}"
                    return placeholder
        
        return None
    
    def _infer_from_naming_convention(
        self,
        param_name: str,
        previous_actions: List[TaskAction]
    ) -> Optional[str]:
        """
        Infer placeholder from naming conventions.
        
        Patterns:
        - "transfer_filter" -> look for "build_*_filter" or "*_transfer_*"
        - "user_data" -> look for "get_user*" or "*_user_*"
        - "validated_*" -> look for "validate_*"
        - "formatted_*" -> look for "format_*"
        
        Returns: {{action_name_output}} if match found
        """
        import re
        
        # Extract key words from parameter name
        words = re.findall(r'[a-z]+', param_name.lower())
        
        # Look for previous actions with matching patterns
        for prev_action in reversed(previous_actions):
            action_name_lower = prev_action.name.lower()
            
            # Check if action name contains key words from parameter
            matches = 0
            for word in words:
                if word in action_name_lower:
                    matches += 1
            
            # If 2+ words match, or action contains param name, likely a match
            if matches >= 2 or param_name.lower() in action_name_lower:
                return f"{{{{{prev_action.name}_output}}}}"
            
            # Check for common verb patterns
            # E.g., "validated_accounts" <- "validate_account_*"
            if len(words) >= 2:
                first_word = words[0]
                verb_patterns = {
                    'validated': 'validate',
                    'formatted': 'format',
                    'composed': 'compose',
                    'built': 'build',
                    'created': 'create',
                    'updated': 'update',
                }
                
                if first_word in verb_patterns:
                    verb = verb_patterns[first_word]
                    if verb in action_name_lower:
                        # Check if rest of words match (handle singular/plural)
                        remaining_words = words[1:]
                        remaining_match = True
                        for word in remaining_words:
                            # Check both singular and plural forms
                            singular = word.rstrip('s') if word.endswith('s') else word
                            plural = word if word.endswith('s') else word + 's'
                            if singular not in action_name_lower and plural not in action_name_lower and word not in action_name_lower:
                                remaining_match = False
                                break
                        if remaining_match:
                            return f"{{{{{prev_action.name}_output}}}}"
        
        return None
    
    def _infer_placeholder_from_context(
        self,
        current_action_name: str,
        param_name: str,
        param_value: str,
        previous_actions: List[TaskAction]
    ) -> Optional[str]:
        """
        Infer the correct placeholder based on the current action and previous actions.
        
        For example, if query_accounts_for_transfer needs transfer_filter_json,
        and the previous action was build_account_transfer_filter,
        return {{build_account_transfer_filter_output}}.
        
        CRITICAL: For source_owner_id and target_owner_id, always use lookup SOP outputs
        if they've been executed.
        """
        import re
        
        # CRITICAL: Handle source_owner_id and target_owner_id from lookup SOPs
        # These should NEVER come from database context - they MUST come from tool outputs
        if param_name == 'source_owner_id':
            # Look for "query_single_user_record" actions from "Locate Source Owner" SOP
            if any(action.name == 'query_single_user_record' for action in previous_actions):
                # Use the FIRST query_single_user_record (source owner is first lookup SOP)
                return "{{query_single_user_record_output.Id}}"
        
        if param_name == 'target_owner_id':
            # Look for "query_single_user_record" actions from "Locate Sales Representative" SOP
            # This is typically the SECOND lookup SOP
            query_count = sum(1 for action in previous_actions if action.name == 'query_single_user_record')
            if query_count >= 2:
                # Use the second query_single_user_record (target owner is second lookup SOP)
                # We need to reference it by index
                return "{{query_single_user_record_output_1.Id}}"
            elif query_count >= 1:
                # If there's at least one query_single_user_record, use it
                # This handles cases where the target owner lookup might not have been generated yet
                # but we need a placeholder to try
                return "{{query_single_user_record_output.Id}}"
        
        # Try to infer array extraction for array-type parameters
        # Look for patterns like: param ends with "_ids" and previous action returns array
        if param_name.endswith('_ids') or param_name.endswith('_list'):
            array_field = self._infer_array_field_extraction(param_name, previous_actions)
            if array_field:
                return array_field
        
        # Try to match parameter to previous action by naming convention
        # E.g., "transfer_filter" might come from "build_transfer_filter"
        inferred = self._infer_from_naming_convention(param_name, previous_actions)
        if inferred:
            return inferred
        
        # If no specific mapping, try to find the immediately previous action
        if previous_actions:
            last_action = previous_actions[-1]
            # Check if the parameter name suggests it comes from the previous action
            if any(word in param_name.lower() for word in ['json', 'data', 'result', 'output']):
                return f"{{{{{last_action.name}_output}}}}"
        
        return None
    
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
    
    def _execute_and_collect_data(
        self,
        scaffold: TaskScaffold,
        progress: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute actions and collect all runtime data without modifying the scaffold.
        
        Returns:
            Tuple of (execution_data, error)
            execution_data contains: {
                'action_results': [{action_index, action_name, input_kwargs, output}],
                'final_result': final action output,
                'context': full context dict
            }
        """
        context = {}
        action_results = []
        final_result = None
        
        for i, action in enumerate(scaffold.actions):
            progress.append(f"  Action {i+1}/{len(scaffold.actions)}: {action.name}")
            
            # Resolve kwargs for this action
            try:
                tool_schema = self._get_tool_schema(action.name)
                filled_kwargs = self._resolve_kwargs(action.name, action.kwargs, context, progress, tool_schema)
            except Exception as e:
                error = f"Failed to resolve kwargs for {action.name}: {e}"
                progress.append(f"  ❌ {error}")
                return None, error
            
            # Execute action
            try:
                result = self.executor.execute_action_by_name(action.name, filled_kwargs)
                progress.append(f"  ✓ Executed successfully")
                
                # Parse JSON string results FOR CONTEXT ONLY
                # Keep original result (may be JSON string) for deterministic values
                parsed_result = result
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        parsed_result = result
                
                # Store action result with BOTH original and parsed
                action_results.append({
                    'action_index': i,
                    'action_name': action.name,
                    'input_kwargs': action.kwargs,  # Original kwargs with placeholders
                    'filled_kwargs': filled_kwargs,  # Resolved kwargs
                    'output': result,  # Original output (may be JSON string)
                    'parsed_output': parsed_result  # Parsed for context
                })
                
                # Update context for next actions
                # Store ORIGINAL result (JSON string if applicable) for placeholder resolution
                # Store PARSED result with _parsed suffix for field lookups
                context[f"{action.name}_output"] = result  # Original (may be JSON string)
                context[f"{action.name}_output_parsed"] = parsed_result  # For field lookups
                context[f"action_{i}_output"] = result
                context[f"action_{i}_output_parsed"] = parsed_result
                
                if isinstance(parsed_result, dict):
                    for key, value in parsed_result.items():
                        context[key] = value
                
                final_result = parsed_result
                
            except Exception as e:
                error = f"Execution failed: {str(e)}"
                progress.append(f"  ❌ {error}")
                return None, error
        
        execution_data = {
            'action_results': action_results,
            'final_result': final_result,
            'context': context
        }
        
        return execution_data, None
    
    def _fill_with_deterministic_values(
        self,
        scaffold: TaskScaffold,
        execution_data: Dict[str, Any],
        progress: List[str]
    ) -> TaskScaffold:
        """
        Use LLM to convert placeholders in kwargs to deterministic JSON values based on execution data.
        
        For parameters with placeholders that reference previous action outputs, replace them with
        the actual deterministic JSON values from those outputs.
        """
        action_results = execution_data['action_results']
        filled_actions = []
        
        for i, action in enumerate(scaffold.actions):
            # Get the execution result for this action
            exec_result = action_results[i]
            
            # Create new kwargs with deterministic values
            new_kwargs = {}
            for key, value in action.kwargs.items():
                if isinstance(value, str) and '{{' in value and '}}' in value:
                    # This is a placeholder - need to fill it with deterministic value
                    # Extract placeholder name
                    import re
                    placeholder_match = re.search(r'\{\{([^}]+)\}\}', value)
                    if placeholder_match:
                        placeholder = placeholder_match.group(1).strip()
                        
                        # Find the referenced action result
                        deterministic_value = self._get_deterministic_value(
                            placeholder, 
                            action_results[:i],  # Previous actions only
                            progress
                        )
                        new_kwargs[key] = deterministic_value
                    else:
                        new_kwargs[key] = value
                else:
                    # Not a placeholder, keep as-is
                    new_kwargs[key] = value
            
            filled_actions.append(TaskAction(
                name=action.name,
                kwargs=new_kwargs,
                comment=action.comment
            ))
        
        return TaskScaffold(
            user_id=scaffold.user_id,
            instruction=scaffold.instruction,
            actions=filled_actions,
            outputs=scaffold.outputs,
            sop_chain=scaffold.sop_chain
        )
    
    def _get_deterministic_value(
        self,
        placeholder: str,
        previous_results: List[Dict[str, Any]],
        progress: List[str]
    ) -> str:
        """
        Get the deterministic value for a placeholder based on execution results.
        
        For example, if placeholder is "build_account_transfer_filter_output",
        find that action's output and return it as a JSON string.
        """
        import json
        
        # Check if placeholder references an action output
        if placeholder.endswith('_output'):
            action_name = placeholder.replace('_output', '')
            
            # Find the action result
            for result in reversed(previous_results):
                if result['action_name'] == action_name:
                    output = result['output']
                    # Convert to JSON string if it's not already
                    if isinstance(output, str):
                        return output
                    else:
                        return json.dumps(output, separators=(',', ':'))
        
        # If not found, log and return the placeholder
        progress.append(f"    ⚠️  Could not resolve placeholder: {placeholder}")
        return f"{{{{{placeholder}}}}}"
    
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
            # Check for placeholder pattern {{...}}
            if value.startswith('{{') and value.endswith('}}'):
                # Extract placeholder name
                placeholder = value[2:-2].strip()
                return self._resolve_placeholder(key, placeholder, context, progress, param_schema)
            # Check for embedded placeholders (but NOT plain JSON!)
            elif '{{' in value:
                # Has double-brace placeholders embedded
                return self._resolve_embedded_placeholders(value, context, progress)
            elif value.startswith('{') and value.endswith('}'):
                # This looks like JSON - check if it's valid JSON first
                try:
                    import json
                    parsed = json.loads(value)
                    # Valid JSON - check if tool expects dict or string
                    if param_schema and param_schema.get('type') == 'object':
                        # Tool expects Dict - return parsed
                        return parsed
                    else:
                        # Tool expects string - return as-is
                        return value
                except:
                    # Not valid JSON - might be a placeholder without double braces
                    # Extract and try to resolve
                    placeholder = value[1:-1].strip()
                    return self._resolve_placeholder(key, placeholder, context, progress, param_schema)
            else:
                return value
        
        elif isinstance(value, dict):
            # Recursively resolve dict values
            # But if any nested value fails to resolve and is a placeholder, use TRANSFORM
            resolved_dict = {}
            has_unresolved = False
            
            for k, v in value.items():
                try:
                    resolved_dict[k] = self._resolve_value(k, v, context, progress, None)
                except ValueError as e:
                    # Check if this is a placeholder resolution error
                    if 'Cannot resolve placeholder' in str(e):
                        has_unresolved = True
                        # Keep the placeholder as-is for now
                        resolved_dict[k] = v
                    else:
                        raise
            
            # If there are unresolved placeholders in the dict, use TRANSFORM
            if has_unresolved:
                progress.append(f"    🔄 Dict parameter '{key}' has unresolved placeholders - using TRANSFORM")
                # Build a description of what we need
                description = f"construct {key} dict from available context with fields: {list(value.keys())}"
                try:
                    result = self._execute_transformation(description, context, progress, param_schema)
                    progress.append(f"    ✅ Dict constructed via transformation")
                    return result
                except Exception as e:
                    raise ValueError(f"Failed to construct dict '{key}' via transformation: {e}")
            
            return resolved_dict
        
        elif isinstance(value, list):
            return [self._resolve_value(key, item, context, progress, None) for item in value]
        
        else:
            return value
    
    def _resolve_nested_access(
        self,
        path: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        Resolve nested field access with array element extraction.
        
        Supports:
        - "output.field" - simple nested access
        - "output.array[*].field" - extract field from all array elements
        - "output.array[0].field" - extract field from specific index
        - "array[*].field" - extract field from array in context
        
        Examples:
        - "query_accounts_by_name_output.accounts[*].Id"
          → Extract Id from all accounts in the array
        - "action_2_output_parsed.user.Id"
          → Get user.Id from action_2_output_parsed
        """
        import re
        
        # Parse the path to handle array access
        # Pattern: field1.field2[*].field3 or field1[0].field2
        parts = []
        current = ""
        i = 0
        
        while i < len(path):
            char = path[i]
            
            if char == '.':
                if current:
                    parts.append(('field', current))
                    current = ""
                i += 1
            elif char == '[':
                # Found array access
                if current:
                    parts.append(('field', current))
                    current = ""
                
                # Extract index or wildcard
                j = path.index(']', i)
                index_str = path[i+1:j]
                
                if index_str == '*':
                    parts.append(('array_all', None))
                else:
                    try:
                        index = int(index_str)
                        parts.append(('array_index', index))
                    except ValueError:
                        return None
                
                i = j + 1
            else:
                current += char
                i += 1
        
        # Add final part
        if current:
            parts.append(('field', current))
        
        # Navigate through the parts
        obj = context
        
        for part_type, part_value in parts:
            if part_type == 'field':
                # Simple field access
                if isinstance(obj, dict) and part_value in obj:
                    obj = obj[part_value]
                else:
                    return None
            
            elif part_type == 'array_all':
                # Extract field from all array elements
                if not isinstance(obj, list):
                    return None
                # This is a marker - continue with next parts for each element
                # Collect remaining parts
                remaining_parts = parts[parts.index(('array_all', None)) + 1:]
                
                if not remaining_parts:
                    # No more parts, return the array itself
                    return obj
                
                # Extract the field from each element
                results = []
                for item in obj:
                    temp_obj = item
                    for rem_type, rem_value in remaining_parts:
                        if rem_type == 'field':
                            if isinstance(temp_obj, dict) and rem_value in temp_obj:
                                temp_obj = temp_obj[rem_value]
                            else:
                                temp_obj = None
                                break
                        else:
                            # Nested array access not supported yet
                            return None
                    
                    if temp_obj is not None:
                        results.append(temp_obj)
                
                return results if results else None
            
            elif part_type == 'array_index':
                # Access specific array index
                if isinstance(obj, list) and part_value < len(obj):
                    obj = obj[part_value]
                else:
                    return None
        
        return obj
    
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
        
        # Try nested lookup with array field extraction
        # Supports:
        # - "output.field" - simple nested access
        # - "output.array[*].field" - extract field from all array elements
        # - "output.array[0].field" - extract field from specific index
        # - "output.array.length" - get array length (JavaScript-style)
        if '.' in placeholder or '[' in placeholder:
            # Handle .length accessor (JavaScript-style)
            if placeholder.endswith('.length'):
                array_path = placeholder[:-7]  # Remove '.length'
                array_value = self._resolve_nested_access(array_path, context)
                if array_value is not None and isinstance(array_value, list):
                    progress.append(f"    Resolved {{{{{{{{placeholder}}}}}}}}: {placeholder} = {len(array_value)} (converted .length to len())")
                    return len(array_value)
            
            result = self._resolve_nested_access(placeholder, context)
            if result is not None:
                progress.append(f"    Resolved {{{{{{{{placeholder}}}}}}}}: {placeholder} = {result}")
                return result
        
        # FALLBACK: Handle generic placeholders (e.g., "placeholder", "data", "value")
        # If LLM generated a generic placeholder and the parameter name suggests it comes from
        # the previous action, use the previous action's output
        generic_placeholders = ['placeholder', 'data', 'value', 'result', 'output', 'input']
        placeholder_lower = placeholder.lower().strip()
        if placeholder_lower in generic_placeholders:
            # First, try to match the parameter name directly in context
            # E.g., if key is 'account_count' and context has 'count', use it
            if key in context:
                progress.append(f"    🔧 Generic placeholder for '{key}' - found exact match in context")
                return context[key]
            
            # Try variations of the parameter name
            # E.g., "account_count" -> try "count", "accountcount"
            variations = [
                key.split('_')[-1],  # Last word: "account_count" -> "count"
                key.replace('_', ''),  # Remove underscores: "account_count" -> "accountcount"
            ]
            for variation in variations:
                if variation in context:
                    progress.append(f"    🔧 Generic placeholder for '{key}' - found '{variation}' in context")
                    return context[variation]
            
            # Special case: If parameter is a dict/object type, try to construct it from context
            if param_schema and param_schema.get('type') == 'object':
                # Try to construct dict from context fields
                progress.append(f"    🔧 Generic placeholder for dict parameter - attempting auto-construction from context")
                constructed_dict = self._construct_dict_from_context(key, param_schema, context, progress)
                if constructed_dict:
                    progress.append(f"    ✅ Constructed dict with {len(constructed_dict)} fields")
                    return constructed_dict
            
            # Try to infer from parameter name and previous actions
            # Look for most recent action output that matches parameter naming
            action_outputs = [k for k in context.keys() if k.endswith('_output')]
            if action_outputs:
                # Use the most recent action output
                last_output_key = action_outputs[-1]
                value = context[last_output_key]
                progress.append(f"    ⚠️  Generic placeholder '{placeholder}' detected - using previous action output: {last_output_key}")
                return value
            else:
                raise ValueError(f"Generic placeholder '{placeholder}' detected but no previous actions available")
        
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
    
    def _construct_dict_from_context(
        self,
        param_name: str,
        param_schema: Dict[str, Any],
        context: Dict[str, Any],
        progress: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to construct a dict parameter from available context values.
        
        For example, if parameter is 'transfer_details' and context has:
        - source_owner_id
        - target_owner_id
        - account_count (or 'count')
        - account_ids
        
        This will construct: {"source_owner_id": "...", "target_owner_id": "...", ...}
        """
        # Get expected properties from schema
        properties = param_schema.get('properties', {})
        if not properties:
            # No schema properties defined, can't auto-construct
            return None
        
        constructed = {}
        missing_fields = []
        
        for field_name, field_schema in properties.items():
            # Try to find this field in context
            # Try exact match first
            if field_name in context:
                constructed[field_name] = context[field_name]
                continue
            
            # Try common variations
            # E.g., "account_count" might be "count" in context
            variations = [
                field_name,
                field_name.replace('_', ''),  # Remove underscores
                field_name.split('_')[-1],    # Last word only (e.g., "count" from "account_count")
            ]
            
            found = False
            for variation in variations:
                if variation in context:
                    constructed[field_name] = context[variation]
                    found = True
                    break
            
            if not found:
                # Field not found - check if it's required
                required_fields = param_schema.get('required', [])
                if field_name in required_fields:
                    missing_fields.append(field_name)
        
        if missing_fields:
            progress.append(f"    ⚠️  Missing required fields for dict construction: {missing_fields}")
            return None
        
        if not constructed:
            return None
        
        return constructed
    
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
    
    def _fill_json_from_execution(
        self,
        scaffold: TaskScaffold,
        resolved_kwargs_list: List[Dict[str, Any]],
        progress: List[str]
    ) -> TaskScaffold:
        """
        Replace placeholders in scaffold kwargs with the actual resolved values from execution.
        
        Instead of trying to re-resolve placeholders, we use the kwargs that were actually
        used during execution (after all TRANSFORM, placeholder resolution, etc.).
        """
        clean_actions = []
        filled_count = 0
        
        for i, action in enumerate(scaffold.actions):
            # Use the actual kwargs that were used during execution
            if i < len(resolved_kwargs_list):
                clean_kwargs = resolved_kwargs_list[i]
                
                # Count parameters that were filled
                for key, value in action.kwargs.items():
                    if str(clean_kwargs.get(key)) != str(value):
                        filled_count += 1
            else:
                # Shouldn't happen, but fallback to original
                clean_kwargs = action.kwargs
            
            clean_actions.append(TaskAction(
                name=action.name,
                kwargs=clean_kwargs,
                comment=action.comment
            ))
        
        progress.append(f"  ✓ Used {filled_count} resolved parameters from execution")
        
        return TaskScaffold(
            user_id=scaffold.user_id,
            instruction=scaffold.instruction,
            actions=clean_actions,
            outputs=scaffold.outputs,
            sop_chain=scaffold.sop_chain
        )
    
    def _fill_placeholder_value(
        self,
        value: Any,
        context: Dict[str, Any],
        progress: List[str]
    ) -> Any:
        """
        Recursively fill placeholders in a value structure.
        
        Handles:
        - Strings with {{placeholder}} -> lookup in context
        - Dicts -> recursively fill values
        - Lists -> recursively fill items
        - Other types -> return as-is
        """
        import re
        
        if isinstance(value, str):
            # Check for placeholder pattern {{...}}
            if '{{' in value and '}}' in value:
                # Extract placeholder(s)
                placeholders = re.findall(r'\{\{([^}]+)\}\}', value)
                
                if len(placeholders) == 1 and value == f"{{{{{placeholders[0]}}}}}":
                    # Pure placeholder - replace with actual value
                    placeholder = placeholders[0].strip()
                    
                    # Look up in context
                    if placeholder in context:
                        return context[placeholder]
                    
                    # Try nested access
                    if '.' in placeholder or '[' in placeholder:
                        nested_value = self._resolve_nested_access(placeholder, context)
                        if nested_value is not None:
                            return nested_value
                    
                    # Not found - keep placeholder
                    return value
                else:
                    # Mixed string with embedded placeholders - replace each
                    result = value
                    for placeholder in placeholders:
                        placeholder_full = f"{{{{{placeholder}}}}}"
                        placeholder_clean = placeholder.strip()
                        
                        # Look up value
                        if placeholder_clean in context:
                            replacement = str(context[placeholder_clean])
                            result = result.replace(placeholder_full, replacement)
                    
                    return result
            else:
                # No placeholders
                return value
        
        elif isinstance(value, dict):
            # Recursively fill dict values
            return {k: self._fill_placeholder_value(v, context, progress) for k, v in value.items()}
        
        elif isinstance(value, list):
            # Recursively fill list items
            return [self._fill_placeholder_value(item, context, progress) for item in value]
        
        else:
            # Other types - return as-is
            return value
    
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
        
        return outputs if outputs else []

