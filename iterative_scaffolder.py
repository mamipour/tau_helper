"""
Iterative Task Scaffolding - Step-by-Step Action Generation with Real Execution

This module implements a new approach to task scaffolding where:
1. Agent generates ONE action at a time
2. Action is executed immediately using ActionExecutor
3. Execution result is fed back to agent for next action
4. Process continues until task completion

Supports multi-agent mode with judge for consensus.
"""

import json
import sys
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

from tau_helper.sop_mapper import SOPMapper
from tau_helper.action_executor import ActionExecutor


class NextAction(BaseModel):
    """Represents the next action to execute."""
    done: bool = Field(description="True if task is complete, False if more actions needed")
    action_name: Optional[str] = Field(default=None, description="Tool/function name for next action")
    action_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Arguments for next action")
    reasoning: str = Field(description="Explanation of why this action is needed or why task is complete")
    sop_step: Optional[str] = Field(default=None, description="Which SOP step this action implements")


class IterativeScaffolder:
    """
    Iterative task scaffolder that generates actions step-by-step with real execution feedback.
    """

    NEXT_ACTION_PROMPT = """You are Model R, the primary task scaffolding agent for Warrior Tau-Bench.

## IMPORTANT: Your Authority

- YOU are in charge of this task
- An advisory agent (R2) provides feedback on your proposed actions
- R2 is NOT always 100% correct - you have the authority to disagree
- You make the FINAL decision on whether to revise your action or proceed
- Consider R2's feedback carefully, but trust your own judgment

Your job is to generate the NEXT action needed to complete this task.

## Task Instruction

{instruction}

## SOP Chain to Implement (CRITICAL: ONLY implement these SOPs, in this order)

{sop_chain}

**CRITICAL**: The SOP chain above is a COMPLETE list of SOPs you need to execute for this task.
- DO NOT execute SOPs that are not in this chain
- DO NOT add additional SOPs beyond what's listed
- The SOP mapper has already determined these are the ONLY SOPs needed for this instruction
- When ALL SOPs in this chain are complete, the task is DONE - set done=True

## Domain Rules & SOP Definitions

{rules_description}

**NOTE**: The rules above contain ALL domain rules and SOP definitions.
- Use this section to understand the details/steps of SOPs that appear in YOUR SOP chain above
- DO NOT execute SOPs from this section that are not in your SOP chain

## Available Tools (with detailed schemas)

{tools_description}

## Execution History

{execution_history}
{progress_tracker}

## Current Status

- Total SOPs in chain: {total_sops}
- SOPs completed: {completed_sops}
- Actions executed so far: {action_count}

## Your Task

Based on the execution history and SOP chain, determine the NEXT action to execute.

**DECISION FLOW (Follow this step-by-step for EVERY action):**
1. **Check for Prerequisites FIRST**: Look at Domain Rules for a "PREREQUISITES" section
   - If prerequisites exist: Compare the prerequisite list with execution history
   - Find the FIRST prerequisite NOT yet in history → That's your next action
   - Only proceed to step 2 when ALL prerequisites are complete
2. **Check SOP Chain Progress**: Once prerequisites are done, check which SOPs are complete
   - Identify the FIRST SOP not fully complete → Work on that SOP next
3. **Check SOP Steps**: Within the current SOP, check which steps are done
   - Find the FIRST step not executed → That's your next action
4. **All Done?**: If all prerequisites + all SOPs + all steps are complete → Set done=True

**CRITICAL RULES:**
1. Generate ONLY ONE action at a time
2. Use ACTUAL values from execution history (no placeholders!)
3. **ALWAYS Check PREREQUISITES First (Domain-Agnostic Pattern)**:
   - BEFORE starting ANY SOP in the chain, look for a "PREREQUISITES" section in Domain Rules
   - Prerequisites are listed as an ordered sequence (e.g., Step 1, Step 2, Step 3...)
   - **Execute prerequisites SEQUENTIALLY in the exact order listed**
   - **Check execution history**: Go through the prerequisite list ONE BY ONE - if ANY prerequisite is missing, execute it next
   - **Do NOT jump to the SOP chain until ALL prerequisites in the list are in execution history**
   - Example: If prerequisites are [setup_env, get_config, configure_api], you MUST run all 3 in order before starting SOP chain
   - This rule applies whether execution history is empty OR partially filled - always verify the COMPLETE prerequisite list is done
4. Follow the SOP chain steps in EXACT ORDER - never skip SOPs
5. Each SOP may have multiple REQUIRED STEPS (shown in SOP Definitions with numbers like "1. ...", "2. ...", "3. ...")
   - Execute ALL steps of an SOP in the exact order shown
   - Do NOT skip intermediate steps (e.g., can't do step 3 without doing step 2 first)
   - Complete ALL steps of SOP N before starting SOP N+1
6. Process ALL entities mentioned in the instruction (if instruction mentions multiple people/accounts, process ALL of them)
7. Match tool parameter types exactly: dict for "object", list for "array", string for "string" - check tool schemas!
8. When all SOPs are implemented AND all entities processed AND task is complete, set done=True

**OPTION 4 FIX: Simplified execution history usage & error recovery**
- Use ENTIRE output from previous actions
- If execution history shows an ERROR, fix it in your next action:
  * "Tool 'X' not found" → Use tool name, not SOP name (check SOP Definitions)
  * "'list'/'dict' errors" → Wrong type passed; check tool schema and wrap/extract accordingly
  * "Missing parameter 'X'" → Add it from execution history

**When to set done=True:**
Check ALL of these conditions before setting done=True:
1. ✓ All SOPs in YOUR SOP CHAIN have been implemented
   - Look at the "SOP Chain to Implement" section above - that's the COMPLETE list
   - Count how many SOPs are in the chain (e.g., if chain has 3 SOPs, all 3 must be done)
   - Do NOT add SOPs that aren't in your chain
   - Do NOT use SOPs from "SOP Definitions" that aren't in your chain
2. ✓ ALL required steps for each SOP have been executed in order (check SOP Definitions for numbered steps)
3. ✓ ALL entities mentioned in the instruction have been fully processed
   - Re-read the instruction and identify every person, account, record mentioned
   - Verify each entity went through all required SOPs
   - For multi-entity operations (e.g., "from A to B"), confirm BOTH A and B were processed
4. ✓ Task instruction requirements are FULLY accomplished (re-read instruction carefully)
5. ✓ No more work needed - everything is complete

**CRITICAL: If ALL conditions above are met, you MUST set done=True. Do NOT keep proposing actions if the task is complete.**

**When to set done=False:**
- ANY SOP in your assigned chain is not yet implemented
- Current SOP has more steps/tools remaining (check SOP Definitions for numbered steps)
- Task mentions multiple entities but you've only processed some
- ANY step within an SOP was skipped (e.g., did step 1 and 3 but not step 2)
- Execution history shows incomplete work

**OPTION 3 FIX: EXPLICIT COMPLETION SIGNAL**
After EVERY successful action, ask yourself:
1. "Did this action complete the LAST step of the LAST SOP in my chain?"
2. "Are ALL entities mentioned in the instruction fully processed?"
3. "Is there ANY more work needed?"

If answers are YES, YES, NO → Set done=True IMMEDIATELY. Do NOT continue.

**CRITICAL: Avoid Infinite Loops**
- If you notice you're repeating the same action, STOP and reconsider
- If the task is complete but you're still generating actions, set done=True
- If execution history shows {repetition_warning}
- If you see duplicate warnings in the execution history, either set done=True or try a completely different approach

{format_instructions}
"""

    JUDGE_PROMPT = """You are a judge evaluating two proposed next actions for a task.

## Task Context

Instruction: {instruction}
SOP Chain: {sop_chain}

## SOP Definitions (Detailed Breakdown)

{sop_definitions}

## Available Tools

{tools_description}

## Execution History

{execution_history}

## Action Proposal A (Model R)

{action_a}

## Action Proposal B (Model R2)

{action_b}

## Your Task

Evaluate which action is better for continuing this task. Consider:
1. Correctness - Does it follow the SOP chain?
2. Parameter accuracy - Are kwargs using correct values from execution history?
3. Completeness - Does it properly implement the next SOP step?
4. Logic - Does it make sense given the current state?
5. Tool validity - Is the proposed tool name valid (check tools_description)?

Return:
- "A" if Action A is better
- "B" if Action B is better
- "EQUAL" if both are essentially the same

Respond with ONLY one word: A, B, or EQUAL
"""

    VALIDATOR_PROMPT = """You are R2, an advisory validation agent providing feedback on proposed actions.

## IMPORTANT: Your Role

- You are an ADVISOR, not an enforcer
- The primary agent (Model R) is in charge and makes final decisions
- Your job is to identify POTENTIAL issues and concerns, not to block actions
- Model R may choose to accept your feedback, revise the action, or proceed anyway
- You are NOT always 100% correct - Model R has the authority to disagree with you

## IMPORTANT: Iterative Execution Mode

This is an ITERATIVE scaffolder - actions are generated and executed ONE AT A TIME.
- SOPs may span MULTIPLE actions (e.g., SOP with 3 steps = 3 separate actions)
- It is NORMAL and EXPECTED to execute just ONE step of an SOP per action
- Do NOT require all SOP steps to be executed in a single action
- Focus on whether THIS SPECIFIC action is valid given what's been done so far

## IMPORTANT: Validating Multi-Action Patterns (Retrospective Validation)

Some domain rules define patterns that span multiple actions (e.g., Extract-Store pattern requiring extract_* followed by store_*).
In this ITERATIVE system where Model R proposes ONE action at a time, you must validate these patterns RETROSPECTIVELY:

**RETROSPECTIVE VALIDATION (Check PREVIOUS actions for violations)**:
- When validating the CURRENT action, check if PREVIOUS actions violated multi-action patterns
- Example: If current action is store_data, check execution history to confirm the previous action was fetch_data
- If a PREVIOUS fetch/extract action was NOT followed by the corresponding store/save action, flag this as a violation

**DO NOT VALIDATE PROSPECTIVELY (Don't complain CURRENT action lacks FUTURE actions)**:
- DO NOT flag an action as invalid because it doesn't include a subsequent action that should come next
- Example: When validating fetch_data, DO NOT complain that it "doesn't include store_data"
- That store call will come in the NEXT action - this is expected in iterative execution
- Model R generates ONE action at a time, not batches of actions

**Pattern Reminders (Not Violations)**:
- If current action STARTS a multi-action pattern, you may include a reminder in missing_prerequisites
- Format: "Reminder: Next action must complete pattern by calling [tool_name]"
- This is informational guidance, NOT a violation of the current action
- Do NOT set valid=false based on a future requirement

**Examples**:

✅ CORRECT Retrospective Validation:
```
Current action: save_results
Execution history: [..., fetch_results (Action 5)]
Validation: ✓ Valid - Action 5 fetched data, current action saves it (pattern complete)
```

✅ CORRECT Pattern Reminder:
```
Current action: fetch_data
Execution history: [...]
Validation: ✓ Valid (set valid=true)
Missing prerequisites: ["Reminder: Next action must call save_data with fetched data"]
```

❌ INCORRECT Prospective Validation:
```
Current action: fetch_data
Validation: ✗ Invalid - "Action does not include required save_data call"
^ WRONG! This is prospective validation. The save call comes in the NEXT action.
```

❌ INCORRECT Missing Pattern Detection:
```
Current action: save_results
Execution history: [..., fetch_other_data (Action 7)]
Validation: ✗ Invalid - "Previous fetch_other_data (Action 7) was not followed by save_other_data"
^ CORRECT! This IS a pattern violation - Action 7 should have been followed by a save call.
```

**Summary**: Use domain rules to understand multi-action patterns, then validate RETROSPECTIVELY (check history) not PROSPECTIVELY (demand future actions in current call).

## Task Context

Instruction: {instruction}
SOP Chain: {sop_chain}

## SOP Definitions (Detailed Breakdown)

{sop_definitions}

## Domain Rules

{rules_description}

## Available Tools

{tools_description}

## Execution History (Values Available)

{execution_history}

## Proposed Action (To Validate)

Action: {action_name}
Parameters: {action_kwargs}
SOP Step: {sop_step}
Reasoning: {reasoning}

## Your Task

Check this proposed action for the following issues:

1. **Hallucinated Values**:
   - Check if the action uses IDs, values, or parameters that DO NOT appear in the execution history
   - Common hallucinations: made-up user IDs, account IDs, field values that weren't returned by previous actions
   - Exception: If execution history is empty (first action), any reasonable values from the instruction are OK

2. **Rule Violations**:
   - Check if the action violates any domain rules listed above
   - Ensure parameters follow the rules
   - Pay special attention to PREREQUISITES and SOP EXECUTION DISCIPLINE rules

3. **SOP Chain Adherence** (CRITICAL):
   - **First, check if the proposed action's SOP is even IN the SOP chain**
   - Parse the "SOP Step" field from the proposed action
   - Extract the SOP name from it (e.g., "SOP Step: 1. Validate Company" → SOP is "Validate Company")
   - Check if this SOP name appears ANYWHERE in the provided SOP Chain list
   - **If the SOP is NOT in the chain**: This is a CRITICAL rule violation - flag it immediately
     - Error message: "Action proposes SOP '[name]' which is NOT in the assigned SOP chain. Model R should ONLY execute SOPs from the chain."
     - Set valid=false
   - **If the SOP IS in the chain**: Then check progression (sequential order)
     - Look at execution history to determine which SOPs have been COMPLETED (all their steps executed)
     - Check if we're following the chain sequentially (SOP 1 → SOP 2 → SOP 3 → ...)
     - If SOP N has not been completed yet, the proposed action should be from SOP N, not SOP N+1 or later
     - Example: If chain is [A, B, C] and history shows [A step 1, A step 2, A step 3], next action should be from SOP B
     - Flag if the action appears to skip ahead in the SOP chain before completing earlier SOPs

4. **SOP Step Completeness** (for multi-step SOPs):
   - Consult the Domain Rules section above to find the detailed SOP definition with numbered steps
   - Check if ALL required steps for an SOP have been executed before moving to the next SOP
   - If the SOP definition shows steps 1, 2, 3 but execution history only shows steps 1 and 3, step 2 is missing
   - It's OK to execute ONE step at a time, but don't skip steps within an SOP
   - Example: Can't do step 3 (query) without first doing step 2 (resolve) for the same entity

5. **Entity Coverage from Instruction**:
   - Parse the instruction to identify ALL named entities that require processing (people, accounts, records, etc.)
   - Check if each mentioned entity has been properly looked up/processed according to the SOP chain
   - For operations involving multiple entities (e.g., "transfer from A to B", "update X and Y"), ensure ALL entities are handled
   - Flag if the proposed action jumps to a later SOP before all required entities from earlier SOPs have been processed
   - Example: If instruction mentions "Person A" and "Person B", both must be looked up before proceeding to later SOPs

6. **Missing Prerequisites**:
   - If action uses a value (like a user ID), check that a previous action retrieved that value
   - Example: Can't use an entity's ID if we never looked up that entity
   - Check the PREREQUISITES section in Domain Rules for specific prerequisite requirements

7. **Repetitive/Duplicate Actions & Task Completion**:
   - Check if the proposed action is repetitive (same action name + similar parameters as a recent action in execution history)
   - If the action is a duplicate, flag it in the issues list (e.g., "Duplicate action: store_data was already executed in Action 10")
   - **Task Completion Assessment**: If you detect duplicates AND observe that all SOPs in the SOP chain have been completed, you may set task_appears_complete=true
     - Check if all SOPs from the SOP chain have been executed (all their steps completed in execution history)
     - If yes, and current action is a duplicate/redundant, set task_appears_complete=true with reasoning explaining which SOPs were completed
     - Otherwise, set task_appears_complete=false
   - **IMPORTANT**: Only flag completion when BOTH conditions are met: (1) duplicate action detected, AND (2) all SOPs completed
   - Your assessment is advisory - Model R makes the final decision to set done=True

## Response Format

Respond with a JSON object:
{{
  "valid": true/false,
  "issues": [
    "Issue description 1",
    "Issue description 2"
  ],
  "missing_steps": [
    "SOP step that should have been executed first"
  ],
  "task_appears_complete": true/false,
  "completion_reasoning": "Brief explanation if task_appears_complete=true (1-2 sentences)"
}}

If valid=true, issues and missing_steps should be empty lists.
If valid=false, provide specific, actionable issues.
If task_appears_complete=true, provide reasoning explaining why the task seems done (e.g., "All SOPs completed: SOP1 done in Actions 1-3, SOP2 done in Actions 4-6, and current action is a duplicate").
"""

    R2_FEEDBACK_RESPONSE_PROMPT = """You are Model R, the primary agent responsible for completing this task.

Your advisory agent R2 has provided feedback on your proposed action. You need to make an informed decision: REVISE or PROCEED.

================================================================================
TASK CONTEXT
================================================================================

**Original Instruction:**
{instruction}

**Assigned SOP Chain** (steps you need to complete):
{sop_chain}

**Execution History** (what's been done so far):
{execution_history}

================================================================================
YOUR PROPOSED ACTION
================================================================================

Action: {action_name}
Parameters: {action_kwargs}
SOP Step: {sop_step}
Your Reasoning: {reasoning}

================================================================================
R2 VALIDATOR FEEDBACK
================================================================================

R2 identified the following concerns:

{r2_feedback}

================================================================================
AVAILABLE TOOLS
================================================================================

{available_tools}

================================================================================
YOUR DECISION
================================================================================

**Important Context:**
- R2's feedback is ADVISORY - carefully evaluate whether the concerns are valid
- You have full context about the task, history, and available tools
- Make your decision based on the complete picture, not just R2's feedback

**When to REVISE:**
- R2 correctly identified missing prerequisites or dependencies
- R2 correctly identified hallucinated tools or invalid parameters
- R2 correctly identified that the task is complete
- The proposed action would violate rules or constraints

**When to PROCEED:**
- R2's concerns are incorrect or based on misunderstanding
- The action is valid and appropriate for the current state
- R2 is being overly cautious without good reason

**If you choose REVISE:**
- Provide the CORRECTED action that addresses R2's valid concerns
- If R2 identified missing prerequisites, provide the FIRST prerequisite action
- **TASK COMPLETION**: If R2 raised a "task completion concern" and you AGREE the task is complete, set "done": true in the revised action
- For all other cases, provide a concrete, executable action with a valid tool name

**If you choose PROCEED:**
- Your original action will be executed despite R2's concerns
- Only choose this if R2's concerns are genuinely incorrect
- Be prepared to justify why R2's feedback is wrong

Respond with a JSON object:
{{
  "decision": "REVISE" or "PROCEED",
  "reasoning": "Detailed explanation of your decision (2-3 sentences explaining WHY you chose this, referencing specific aspects of the task context, execution history, or R2's feedback)",
  "revised_action": {{  // Only if decision is REVISE
    // For task completion: {{"done": true, "reasoning": "..."}}
    // For all other revisions: Must be a concrete action with valid tool name
    "action_name": "tool_name_here",  // Required (unless done=true for completion)
    "action_kwargs": {{...}},  // Required (unless done=true for completion)
    "sop_step": "...",  // Optional: SOP step description
    "reasoning": "...",  // Required: why this is the correct next action
    "done": true  // ONLY when task is truly complete
  }}
}}
"""

    def __init__(
            self,
            domain: str,
            variation: str,
            llm: ChatOpenAI,
            llm_r2: Optional[ChatOpenAI] = None,
            llm_judge: Optional[ChatOpenAI] = None
    ):
        """
        Initialize iterative scaffolder.

        Args:
            domain: Domain name
            variation: Variation name
            llm: Primary LLM (Model R)
            llm_r2: Secondary LLM (Model R2) for multi-agent mode
            llm_judge: Judge LLM for resolving disagreements
        """
        self.domain = domain
        self.variation = variation
        self.llm = llm
        self.llm_r2 = llm_r2
        self.llm_judge = llm_judge

        # Pass llm_r2 and llm_judge to SOPMapper for roundtable validation
        self.sop_mapper = SOPMapper(llm, domain, variation, llm_r2=llm_r2, llm_judge=llm_judge)
        self.action_executor = ActionExecutor(domain, variation)

        # Determine which mode to use based on environment variables:
        # - Multi-Agent Mode: DEFAULT_MODEL_R_JUDGE is set (R and R2 both generate, judge picks)
        # - R2 Validation with Roundtable: DEFAULT_MODEL_JUDGE is set (R generates, R2 validates, judge mediates)
        import os
        has_r_judge = os.getenv('DEFAULT_MODEL_R_JUDGE') is not None
        has_judge = os.getenv('DEFAULT_MODEL_JUDGE') is not None

        # Multi-agent mode: R and R2 both generate actions, judge picks the best
        self.multi_agent_enabled = llm_r2 is not None and llm_judge is not None and has_r_judge

        # R2 validation mode: R2 validates R's actions (optionally with judge for roundtable)
        self.validation_enabled = llm_r2 is not None and not self.multi_agent_enabled

        # Load tools, rules, and SOPs for prompting
        self._load_tools()
        self._load_rules()
        self._load_sop_definitions()

    def _load_tools(self):
        """Load tool descriptions with detailed schemas for prompting."""
        self.tools = self.action_executor.TOOLS

        # Format detailed tool descriptions with schemas
        tool_descs = []
        for tool_cls in self.tools:
            info = tool_cls.get_info()
            if 'function' in info:
                func_info = info['function']
                tool_descs.append(f"\n**{func_info['name']}**")
                if 'description' in func_info:
                    tool_descs.append(f"Description: {func_info['description']}")

                # Add detailed parameter information
                if 'parameters' in func_info and 'properties' in func_info['parameters']:
                    tool_descs.append("Parameters:")
                    props = func_info['parameters']['properties']
                    required = func_info['parameters'].get('required', [])

                    for param_name, param_info in props.items():
                        req_marker = " (required)" if param_name in required else " (optional)"
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        tool_descs.append(f"  - {param_name}: {param_type}{req_marker}")
                        if param_desc:
                            tool_descs.append(f"    {param_desc}")

                # Add related SOPs
                if 'related_sops' in info:
                    tool_descs.append(f"Related SOPs: {', '.join(info['related_sops'])}")

        self.tools_description = "\n".join(tool_descs)

    def _load_rules(self):
        """Load domain rules for prompting."""
        import importlib
        try:
            rules_module = importlib.import_module(
                f"domains.{self.domain}.variations.{self.variation}.rules"
            )
            self.rules = getattr(rules_module, 'RULES', [])
            self.rules_description = "\n".join([f"- {rule}" for rule in self.rules])
        except Exception:
            self.rules = []
            self.rules_description = "(No rules defined)"

    def _load_sop_definitions(self):
        """
        Load SOP definitions from rules.

        Instead of trying to parse SOPs separately, we pass all rules as the master prompt.
        The model will find and follow the SOPs defined within the rules naturally.
        This avoids fragile parsing logic that fails when rules format changes.
        """
        # Just use the rules description as-is - no parsing needed!
        # The model is smart enough to find and follow SOPs within the rules
        self.sop_definitions = self.rules_description if self.rules_description else "(No rules or SOPs defined)"

    def _log(self, message: str, progress_list: list, verbose: bool = False):
        """
        Log message to progress list and optionally print in real-time.

        Args:
            message: Message to log
            progress_list: Progress list to append to
            verbose: If True, print immediately with flush for real-time output
        """
        progress_list.append(message)
        if verbose:
            print(message, flush=True)
            sys.stdout.flush()  # Force immediate output to console

    def _validate_action_with_r2(
            self,
            next_action: NextAction,
            instruction: str,
            sop_chain: List[str],
            execution_history: List[Dict[str, Any]],
            repetition_warning: str = "no repetition detected"
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate action with R2 model for hallucinations and rule violations.

        Args:
            repetition_warning: Warning about repeated actions (if detected)

        Returns:
            Tuple of (is_valid, issues, missing_steps, task_complete, completion_reasoning)
        """
        # Format execution history for validation
        if execution_history:
            history_parts = []
            for h in execution_history:
                if 'error' not in h:
                    result = h['result']
                    history_parts.append(
                        f"Action {h['action_num']}: {h['action']}({json.dumps(h['kwargs'])})\n"
                        f"  Result: {json.dumps(result) if isinstance(result, (dict, list)) else str(result)}"
                    )
            history_str = "\n\n".join(history_parts)
        else:
            history_str = "(No previous actions - this is the first action)"

        # Format SOP chain
        sop_chain_str = "\n".join([f"{i + 1}. {sop}" for i, sop in enumerate(sop_chain)])

        # Create prompt
        prompt = ChatPromptTemplate.from_template(self.VALIDATOR_PROMPT)

        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_str,
            sop_definitions=self.sop_definitions,
            rules_description=self.rules_description,
            tools_description=self.tools_description,
            execution_history=history_str,
            action_name=next_action.action_name,
            action_kwargs=json.dumps(next_action.action_kwargs, indent=2),
            sop_step=next_action.sop_step or "N/A",
            reasoning=next_action.reasoning
        )

        response = self.llm_r2.invoke(messages)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            validation_result = json.loads(content)
            is_valid = validation_result.get("valid", True)
            issues = validation_result.get("issues", [])
            missing_steps = validation_result.get("missing_steps", [])
            task_complete = validation_result.get("task_appears_complete", False)
            completion_reasoning = validation_result.get("completion_reasoning", "")

            # Add repetition warning to issues if detected
            # This ensures repetition goes through roundtable discussion (R → R2 → Judge)
            if "REPETITION DETECTED" in repetition_warning or "CRITICAL" in repetition_warning:
                issues.append(f"Repetition concern: {repetition_warning}")
                is_valid = False  # Mark as invalid so it triggers roundtable discussion

            return is_valid, issues, missing_steps, task_complete, completion_reasoning

        except Exception as e:
            # If parsing fails, assume valid (don't block on validation errors)
            return True, [], [], False, ""

    def _build_progress_tracker(
            self,
            execution_history: List[Dict[str, Any]],
            sop_chain: List[str]
    ) -> str:
        """
        Build a progress tracker showing completed actions and recent failures/duplicates.
        Domain-agnostic - works with any execution history.

        OPTION 1 FIX: Make duplicate errors MORE VISIBLE so Model R can't miss them.
        """
        if not execution_history:
            return ""

        # Get successful actions (no errors)
        successful_actions = [
            h for h in execution_history
            if 'error' not in h and 'duplicate' not in h
        ]

        # Get recent duplicates/errors (last 5)
        recent_failures = [
            h for h in execution_history[-5:]
            if 'error' in h or 'duplicate' in h
        ]

        if not successful_actions and not recent_failures:
            return ""

        tracker = ""

        # OPTION 1: Show recent duplicates/errors PROMINENTLY at the top
        if recent_failures:
            tracker += "\n" + "="*80 + "\n"
            tracker += "⚠️  WARNING: RECENT DUPLICATE ACTIONS / ERRORS\n"
            tracker += "="*80 + "\n\n"
            tracker += "**YOU ARE REPEATING ACTIONS THAT WERE ALREADY EXECUTED!**\n\n"

            for failure in recent_failures:
                action_num = failure.get('action_num', '?')
                action_name = failure.get('action', 'unknown')
                action_params = failure.get('kwargs', {})

                if failure.get('duplicate'):
                    tracker += f"❌ Action {action_num}: {action_name}({json.dumps(action_params)})\n"
                    tracker += f"   ERROR: DUPLICATE - This exact action was already completed!\n"
                else:
                    error_msg = failure.get('error', 'Unknown error')
                    tracker += f"❌ Action {action_num}: {action_name}({json.dumps(action_params)})\n"
                    tracker += f"   ERROR: {error_msg[:150]}\n"
                tracker += "\n"

            tracker += "="*80 + "\n"
            tracker += "STOP REPEATING THESE ACTIONS! Try a different approach or set done=True.\n"
            tracker += "="*80 + "\n\n"

        # Show completed actions with parameters (OPTION 1: show params to distinguish calls)
        if successful_actions:
            tracker += "\n## COMPLETED ACTIONS (Do NOT repeat these)\n\n"

            for action in successful_actions[-10:]:  # Show last 10
                action_num = action.get('action_num', '?')
                action_name = action.get('action', 'unknown')
                action_params = action.get('kwargs', {})
                # Show parameters to help distinguish different calls to same function
                tracker += f"✅ Action {action_num}: {action_name}({json.dumps(action_params)})\n"

            if len(successful_actions) > 10:
                tracker += f"\n... and {len(successful_actions) - 10} more completed actions\n"

            tracker += f"\n**Total successful actions**: {len(successful_actions)}\n"
            tracker += "**DO NOT propose any action listed above - they are already complete.**\n"

        return tracker

    def _try_parse_prerequisites(self, rules_description: str) -> str:
        """
        Try to extract prerequisites from Domain Rules (natural language).
        Domain-agnostic - gracefully handles missing or varied formats.

        Returns empty string if no prerequisites found (safe fallback).
        """
        if not rules_description:
            return ""

        # Try to find prerequisite-related patterns (case-insensitive)
        prereq_section = ""
        lines = rules_description.split('\n')

        in_prereq_section = False
        prereq_lines = []

        for line in lines:
            line_lower = line.lower()

            # Start of prerequisites section (various formats)
            if any(keyword in line_lower for keyword in ['prerequisite', 'setup', 'before starting', 'initial steps']):
                in_prereq_section = True
                prereq_lines.append(line)
                continue

            # End of section (next major heading)
            if in_prereq_section and line.strip().startswith('##'):
                break

            # Collect lines in prerequisite section
            if in_prereq_section:
                prereq_lines.append(line)

        if prereq_lines:
            prereq_section = "\n## PREREQUISITES (Complete these FIRST before starting SOP chain)\n\n"
            prereq_section += "\n".join(prereq_lines)
            prereq_section += "\n\n**IMPORTANT**: Complete ALL prerequisites in order before starting SOP 1.\n"

        return prereq_section

    def _is_duplicate_action(
            self,
            action_name: str,
            action_kwargs: Dict[str, Any],
            execution_history: List[Dict[str, Any]],
            lookback: int = 5
    ) -> bool:
        """
        Domain-agnostic duplicate action detection.

        Checks if the proposed action matches any recent successful action.

        Args:
            action_name: Proposed action name
            action_kwargs: Proposed action kwargs
            execution_history: List of previously executed actions
            lookback: Number of recent actions to check (default: 5)

        Returns:
            True if duplicate found, False otherwise
        """
        # Only check successful actions (not errors or rejected actions)
        successful_actions = [
            h for h in execution_history[-lookback:]
            if 'error' not in h and 'revision_failed' not in h and 'judge_decision' not in h
        ]

        for hist_action in successful_actions:
            # Compare action name
            if hist_action.get('action') != action_name:
                continue

            # Compare kwargs (normalize for comparison)
            hist_kwargs = hist_action.get('kwargs', {})

            # Simple JSON comparison (order-independent)
            try:
                if json.dumps(action_kwargs, sort_keys=True) == json.dumps(hist_kwargs, sort_keys=True):
                    return True
            except:
                # If JSON serialization fails, do dict comparison
                if action_kwargs == hist_kwargs:
                    return True

        return False

    def _is_task_completion_feedback(self, r2_feedback: str) -> bool:
        """
        Domain-agnostic detection of task completion feedback from R2.

        Checks for keywords indicating R2 believes the task is complete.
        """
        completion_keywords = [
            "task appears complete",
            "task_appears_complete",
            "all sops complete",
            "all sops in",
            "sop chain.*complete",
            "already completed",
            "duplicate.*all.*done",
            "redundant.*completed"
        ]

        feedback_lower = r2_feedback.lower()
        for keyword in completion_keywords:
            import re
            if re.search(keyword, feedback_lower):
                return True
        return False

    def _ask_model_r_decision(
            self,
            proposed_action: NextAction,
            r2_feedback: str,
            instruction: str,
            sop_chain: List[str],
            execution_history: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[NextAction]]:
        """
        Ask Model R to decide whether to REVISE or PROCEED after R2's feedback.

        Args:
            proposed_action: The action Model R proposed
            r2_feedback: R2's feedback/concerns
            instruction: Original task instruction
            sop_chain: List of assigned SOPs
            execution_history: What's been done so far

        Returns:
            Tuple of (decision, revised_action)
            decision: "REVISE" or "PROCEED"
            revised_action: NextAction if REVISE, None if PROCEED
        """
        # Format SOP chain
        sop_chain_text = "\n".join([f"{i+1}. {sop}" for i, sop in enumerate(sop_chain)])

        # Format execution history (last 10 actions for context)
        if execution_history:
            recent_history = execution_history[-10:]
            history_text = ""
            for i, entry in enumerate(recent_history, 1):
                if entry.get("error"):
                    history_text += f"{i}. ❌ {entry.get('action')} - Error: {entry.get('error')[:100]}...\n"
                elif entry.get('duplicate'):
                    history_text += f"{i}. ⚠️  {entry.get('action')} - Duplicate action (skipped)\n"
                else:
                    result_preview = str(entry.get('result', ''))[:100]
                    history_text += f"{i}. ✓ {entry.get('action')}({json.dumps(entry.get('kwargs', {}))}) - Success\n"
        else:
            history_text = "(No actions executed yet)"

        # Format available tools with their schemas
        available_tools_text = ""
        for i, tool in enumerate(self.tools, 1):
            tool_info = tool.get_info()
            tool_name = tool_info['function']['name']
            tool_desc = tool_info['function'].get('description', 'No description')
            tool_params = tool_info['function'].get('parameters', {})

            available_tools_text += f"{i}. {tool_name}\n"
            available_tools_text += f"   Description: {tool_desc}\n"
            if tool_params.get('properties'):
                available_tools_text += f"   Parameters:\n"
                for param_name, param_info in tool_params['properties'].items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required = " (required)" if param_name in tool_params.get('required', []) else " (optional)"
                    available_tools_text += f"     - {param_name}: {param_type}{required} - {param_desc}\n"
            available_tools_text += "\n"

        # Create prompt
        prompt = ChatPromptTemplate.from_template(self.R2_FEEDBACK_RESPONSE_PROMPT)

        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_text,
            execution_history=history_text,
            action_name=proposed_action.action_name,
            action_kwargs=json.dumps(proposed_action.action_kwargs, indent=2),
            sop_step=proposed_action.sop_step or "N/A",
            reasoning=proposed_action.reasoning,
            r2_feedback=r2_feedback,
            available_tools=available_tools_text
        )

        response = self.llm.invoke(messages)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decision_result = json.loads(content)
            decision = decision_result.get("decision", "PROCEED")

            if decision == "REVISE":
                revised = decision_result.get("revised_action", {})

                # Check if this is a task completion scenario
                is_completion = self._is_task_completion_feedback(r2_feedback)

                # Allow done=True ONLY for task completion scenarios
                if revised.get("done", False):
                    if is_completion:
                        # Valid completion - Model R agrees task is done
                        revised_action = NextAction(
                            done=True,
                            action_name=None,
                            action_kwargs={},
                            sop_step=None,
                            reasoning=revised.get("reasoning", "Task complete - all SOPs executed")
                        )
                        return "REVISE", revised_action
                    else:
                        # Invalid - Model R trying to give up on real work
                        # Reject and treat as incomplete revision
                        return "PROCEED", None

                # Reject invalid revisions - Model R must provide a concrete action (unless done=True for completion)
                if not revised.get("action_name") or not revised.get("action_kwargs"):
                    return "PROCEED", None

                # Only inherit sop_step if action_name didn't change
                # If action changed, old sop_step is likely wrong
                revised_action_name = revised.get("action_name")
                if revised_action_name == proposed_action.action_name:
                    # Same action, keep sop_step
                    revised_sop_step = revised.get("sop_step", proposed_action.sop_step)
                else:
                    # Different action, don't inherit wrong sop_step
                    revised_sop_step = revised.get("sop_step", None)

                revised_action = NextAction(
                    done=False,
                    action_name=revised_action_name,
                    action_kwargs=revised.get("action_kwargs"),
                    sop_step=revised_sop_step,
                    reasoning=revised.get("reasoning", decision_result.get("reasoning", ""))
                )
                return "REVISE", revised_action
            else:
                return "PROCEED", None

        except Exception as e:
            # If parsing fails, default to PROCEED (trust Model R's original decision)
            return "PROCEED", None

    def scaffold(
            self,
            instruction: str,
            task_id: str = "task_new",
            max_actions: int = 100,
            verbose: bool = False
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], List[str], Optional[str]]:
        """
        Generate task actions iteratively with real execution feedback.

        Args:
            instruction: User-facing task instruction
            task_id: Task ID
            max_actions: Maximum actions to prevent infinite loops
            verbose: Print detailed progress

        Returns:
            Tuple of (actions, error, progress_log, model_info)
        """
        progress = []
        model_info = None

        try:
            # Step 1: Map instruction to SOP chain (with roundtable validation if enabled)
            self._log("Step 1: Mapping instruction to SOP chain...", progress, verbose)
            mapping = self.sop_mapper.map_instruction(instruction, verbose=verbose)
            sop_chain = mapping.primary_chain.sops
            self._log(f"✓ Mapped to {len(sop_chain)} SOPs", progress, verbose)

            if verbose:
                for i, sop in enumerate(sop_chain, 1):
                    self._log(f"  SOP #{i}: {sop}", progress, verbose)

            # Step 2: Reset database for fresh start
            self._log("\nStep 2: Resetting database...", progress, verbose)
            self.action_executor.service.reset_database()
            self.action_executor.service.reset_tools()
            self._log("✓ Database reset complete", progress, verbose)

            # Step 3: Iterative action generation and execution
            self._log(f"\nStep 3: Iterative action generation (max {max_actions} actions)...", progress, verbose)

            actions = []
            execution_history = []
            completed_sops = 0
            consecutive_failures = 0  # Circuit breaker: track failed revisions/rejections
            MAX_CONSECUTIVE_FAILURES = 3  # Stop after this many consecutive failures

            for action_num in range(1, max_actions + 1):
                self._log(f"\n--- Action {action_num} ---", progress, verbose)

                # Generate next action
                if self.multi_agent_enabled:
                    next_action, step_progress, step_model_info = self._generate_next_action_multi_agent(
                        instruction, sop_chain, execution_history, len(sop_chain), completed_sops, len(actions)
                    )
                    model_info = step_model_info
                    repetition_warning = "no repetition detected"  # Multi-agent doesn't use repetition detection
                else:
                    next_action, step_progress, repetition_warning = self._generate_next_action(
                        instruction, sop_chain, execution_history, len(sop_chain), completed_sops, len(actions)
                    )
                    model_info = "Single Model"

                progress.extend(step_progress)

                # Log Model R's initial proposal BEFORE R2 validation
                # This ensures the user sees what R proposed before R2's concerns
                if verbose:
                    for line in step_progress:
                        print(line, flush=True)

                # Check if done
                if next_action.done:
                    self._log(f"\n✅ Task complete! Generated {len(actions)} actions", progress, verbose)
                    self._log(f"Reasoning: {next_action.reasoning}", progress, verbose)
                    break

                # Duplicate Action Detection (domain-agnostic)
                if not next_action.done:
                    is_duplicate = self._is_duplicate_action(
                        next_action.action_name,
                        next_action.action_kwargs,
                        execution_history,
                        lookback=5
                    )
                    if is_duplicate:
                        self._log(
                            "⚠️  Duplicate action detected - this exact action was recently executed successfully",
                            progress, verbose)
                        self._log("  Skipping duplicate and asking for new action", progress, verbose)

                        # Add to history to help agent avoid repeating
                        execution_history.append({
                            "action_num": action_num,
                            "action": next_action.action_name,
                            "kwargs": next_action.action_kwargs,
                            "error": "DUPLICATE ACTION: This exact action (same name + parameters) was already executed successfully in recent history. Do not repeat it. If all work is done, set done=True to complete the task.",
                            "sop_step": next_action.sop_step,
                            "duplicate": True
                        })

                        # Increment failure counter
                        consecutive_failures += 1

                        # Circuit breaker: force completion if too many duplicates
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            self._log(
                                f"\n⚠️  Circuit breaker triggered: {consecutive_failures} consecutive failures (duplicates/rejections)",
                                progress, verbose)
                            self._log(f"  Task appears stuck in a loop. Forcing completion with current progress.",
                                      progress, verbose)
                            break

                        # Skip execution and continue to next iteration
                        continue

                # R2 Validation Loop: Check for hallucinations and rule violations
                # Keep validating until R2 approves or max rounds reached
                if self.validation_enabled and not self.multi_agent_enabled:
                    validation_round = 0
                    MAX_VALIDATION_ROUNDS = 3
                    validated = False
                    last_r2_concerns = None  # Track last round's concerns for judge consultation
                    action_rejected = False  # Flag to skip action if judge rejects it

                    while not validated and validation_round < MAX_VALIDATION_ROUNDS and not action_rejected:
                        validation_round += 1
                        if validation_round > 1:
                            self._log(
                                f"🔍 R2 Validator: Re-checking revised action (round {validation_round}/{MAX_VALIDATION_ROUNDS})...",
                                progress, verbose)
                        else:
                            self._log("🔍 R2 Validator: Checking for hallucinations and rule violations...", progress,
                                      verbose)

                        is_valid, issues, missing_steps, task_complete, completion_reasoning = self._validate_action_with_r2(
                            next_action, instruction, sop_chain, execution_history, repetition_warning
                        )

                        # Treat task completion as a concern (not automatic enforcement)
                        # This ensures completion goes through roundtable discussion (R → R2 → Judge)
                        if task_complete:
                            self._log("⚠️  R2 Validator: Task appears COMPLETE", progress, verbose)
                            self._log(f"  Reasoning: {completion_reasoning}", progress, verbose)
                            # Add completion as a concern - Model R will decide whether to agree or continue
                            issues.append(f"Task completion concern: {completion_reasoning}")
                            is_valid = False  # Mark as invalid to trigger roundtable discussion

                        if not is_valid:
                            self._log("⚠️  R2 Validator identified concerns:", progress, verbose)
                            for issue in issues:
                                self._log(f"  • {issue}", progress, verbose)
                            for step in missing_steps:
                                self._log(f"  • Missing prerequisite: {step}", progress, verbose)

                            # Present R2's feedback to Model R and ask for decision
                            self._log("💭 Asking Model R to decide: REVISE or PROCEED...", progress, verbose)

                            # Format feedback
                            feedback_text = "\n".join([f"- {i}" for i in issues + missing_steps])
                            # Store for potential judge consultation if validation exhausts all rounds
                            last_r2_concerns = feedback_text

                            # Ask Model R for decision
                            decision, revised_action = self._ask_model_r_decision(
                                next_action, feedback_text, instruction, sop_chain, execution_history
                            )

                            if decision == "REVISE":
                                self._log(f"✓ Model R decided to REVISE based on R2's feedback", progress, verbose)
                                if revised_action.done:
                                    self._log(f"  Action: Set task as complete (done=True)", progress, verbose)
                                    next_action = revised_action
                                    validated = True  # Task complete, exit loop
                                else:
                                    # Check if R2 flagged the action as out-of-chain
                                    is_out_of_chain = any(phrase in feedback_text.lower() for phrase in [
                                        "not in the assigned sop chain",
                                        "not in assigned sop chain",
                                        "not part of the assigned sop chain",
                                        "sop is not in the chain",
                                        "sop not in chain"
                                    ])

                                    # If out-of-chain and Model R didn't actually change the action, force completion
                                    if is_out_of_chain and revised_action.action_name == next_action.action_name:
                                        self._log(
                                            f"  ⚠️ WARNING: R2 flagged action as out-of-chain, but Model R returned same action",
                                            progress, verbose)
                                        self._log(f"  All SOPs in chain appear complete - setting done=True", progress,
                                                  verbose)
                                        next_action = NextAction(
                                            done=True,
                                            action_name=None,
                                            action_kwargs={},
                                            sop_step=None,
                                            reasoning="Model R attempted to execute actions outside assigned SOP chain. All assigned SOPs complete."
                                        )
                                        validated = True  # Task complete, exit loop
                                    else:
                                        # Accept the revision - will be re-validated by R2 in next loop iteration
                                        self._log(f"  Revised action: {revised_action.action_name}", progress, verbose)
                                        next_action = revised_action
                                        # Continue loop to re-validate with R2
                            else:  # PROCEED - R disagrees with R2
                                self._log(f"⚠️ Model R decided to PROCEED with original action (disagreed with R2)",
                                          progress, verbose)

                                # If judge is available, consult for roundtable decision
                                if self.llm_judge is not None:
                                    self._log(f"🏛️ Consulting Judge to resolve disagreement...", progress, verbose)

                                    # Judge decides: should R proceed or should R2's concerns be heeded?
                                    judge_decision = self._judge_r_vs_r2_feedback(
                                        instruction,
                                        sop_chain,
                                        execution_history,
                                        next_action,  # R's proposed action
                                        feedback_text,  # R2's concerns
                                        next_action.reasoning  # R's reasoning for PROCEED
                                    )

                                    if judge_decision == "R2":
                                        self._log(f"  ✓ Judge sided with R2 - asking R to revise", progress, verbose)
                                        # Force R to revise by asking again with stronger emphasis
                                        self._log(f"  🔄 Requesting mandatory revision from Model R...", progress,
                                                  verbose)

                                        # Ask R to revise with judge's backing
                                        revised_prompt = f"""The Judge has reviewed the disagreement and sided with R2's concerns.
You MUST revise your proposed action to address these validated concerns:

{feedback_text}

Please provide a revised action that properly addresses these issues."""

                                        _, forced_revision = self._ask_model_r_decision(
                                            next_action, revised_prompt, instruction, sop_chain, execution_history
                                        )
                                        if forced_revision:
                                            next_action = forced_revision
                                            if forced_revision.done:
                                                self._log(f"  ✓ Model R acknowledged task completion (done=True)", progress,
                                                          verbose)
                                            else:
                                                self._log(f"    Revised action: {next_action.action_name}", progress,
                                                          verbose)
                                                # Reset failure counter on successful revision
                                                consecutive_failures = 0
                                        else:
                                            # Judge sided with R2, but R failed to provide valid revision
                                            # Judge's decision is FINAL - we cannot proceed with rejected action
                                            self._log(f"    ⛔ Model R failed to provide valid revision after Judge ruling",
                                                      progress, verbose)
                                            self._log(f"    ❌ Action REJECTED - Judge's decision is final", progress,
                                                      verbose)
                                            self._log(
                                                f"    Skipping this action and asking for new action on next iteration",
                                                progress, verbose)

                                            # Add this failed attempt to execution history so R learns from it
                                            # CRITICAL: Include R2's concerns so Model R understands WHY action was rejected
                                            execution_history.append({
                                                "action_num": action_num,
                                                "action": next_action.action_name,
                                                "kwargs": next_action.action_kwargs,
                                                "error": f"REJECTED BY JUDGE: This action was rejected after roundtable discussion. R2 Validator raised concerns:\n{feedback_text}\n\nJudge sided with R2's assessment. Model R failed to provide valid revision. DO NOT retry this exact action - the concerns still apply. Move on to a DIFFERENT action or set done=True if task is complete.",
                                                "sop_step": next_action.sop_step,
                                                "judge_decision": "R2",
                                                "r2_concerns": feedback_text,
                                                "revision_failed": True
                                            })

                                            # Increment failure counter
                                            consecutive_failures += 1

                                            # Circuit breaker: force completion if too many rejections
                                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                                self._log(
                                                    f"\n⚠️  Circuit breaker triggered: {consecutive_failures} consecutive failures (duplicates/rejections)",
                                                    progress, verbose)
                                                self._log(
                                                    f"  Task appears stuck. Forcing completion with current progress.",
                                                    progress, verbose)
                                                break

                                            # Set flag to skip action and break out of validation loop
                                            action_rejected = True
                                            break
                                    else:  # Judge sided with R
                                        self._log(f"  ✓ Judge sided with Model R - proceeding with original action",
                                                  progress, verbose)
                                        validated = True  # Exit loop, proceed with action
                                else:
                                    # No judge available, R's decision stands
                                    self._log(f"  (No judge available - using R's decision)", progress, verbose)
                                    validated = True  # Exit loop, proceed with action
                        else:
                            self._log("✓ R2 Validator: No concerns identified", progress, verbose)
                            validated = True  # Exit loop, proceed with action

                    # Safety check: If we exhausted all validation rounds without approval
                    if not validated:
                        self._log(
                            f"⚠️  Warning: R2 validation exhausted all {MAX_VALIDATION_ROUNDS} rounds without approval",
                            progress, verbose)

                        # Consult judge to make final decision
                        if self.llm_judge is not None and last_r2_concerns:
                            self._log(f"🏛️ Consulting Judge to make final decision on disputed action...", progress, verbose)

                            judge_decision = self._judge_r_vs_r2_feedback(
                                instruction,
                                sop_chain,
                                execution_history,
                                next_action,  # Model R's final revised action
                                last_r2_concerns,  # R2's last concerns
                                "Model R revised the action multiple times but R2 validator still found concerns after 3 rounds."
                            )

                            if judge_decision == "R2":
                                # Judge sided with R2 - reject the action
                                self._log(f"  ✓ Judge sided with R2 - action REJECTED", progress, verbose)
                                self._log(f"  Skipping this action and asking for new action on next iteration", progress, verbose)

                                # Add to execution history so Model R learns from it
                                execution_history.append({
                                    "action_num": action_num,
                                    "action": next_action.action_name,
                                    "kwargs": next_action.action_kwargs,
                                    "error": f"REJECTED BY JUDGE: After {MAX_VALIDATION_ROUNDS} validation rounds, R2 still found concerns:\n{last_r2_concerns}\n\nJudge sided with R2's assessment. DO NOT retry this action - try a different approach.",
                                    "sop_step": next_action.sop_step,
                                    "judge_decision": "R2",
                                    "r2_concerns": last_r2_concerns,
                                    "validation_rounds_exhausted": True
                                })

                                # Increment failure counter
                                consecutive_failures += 1

                                # Circuit breaker: force completion if too many rejections
                                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                    self._log(
                                        f"\n⚠️  Circuit breaker triggered: {consecutive_failures} consecutive failures",
                                        progress, verbose)
                                    self._log(
                                        f"  Task appears stuck. Forcing completion with current progress.",
                                        progress, verbose)
                                    break

                                # Set flag to skip action
                                action_rejected = True
                            else:
                                # Judge sided with Model R - proceed with action
                                self._log(f"  ✓ Judge sided with Model R - proceeding with action despite R2's concerns", progress, verbose)
                        else:
                            # No judge available or no concerns tracked
                            if not self.llm_judge:
                                self._log(f"  (No judge available - proceeding with last revised action)", progress, verbose)
                            self._log(
                                f"  Proceeding with last revised action, but this may violate R2's concerns",
                                progress, verbose)

                # Check if action was rejected during validation
                if action_rejected:
                    # Skip execution of this action, continue to next iteration
                    continue

                # Validate action (allow empty if done=True)
                if not next_action.done and (not next_action.action_name or not next_action.action_kwargs):
                    error = "Agent generated invalid action (missing name or kwargs)"
                    self._log(f"❌ {error}", progress, verbose)
                    return None, error, progress, model_info

                # Pre-execution validation: Check if action_name is actually a tool name (skip if done=True)
                available_tool_names = [t.get_info()['function']['name'] for t in self.tools]
                if not next_action.done and next_action.action_name not in available_tool_names:
                    # Check if it matches an SOP name (common mistake)
                    # Convert snake_case action_name to Title Case to match SOP names
                    def snake_to_title(name):
                        return ' '.join(word.capitalize() for word in name.replace('_', ' ').split())

                    action_as_title = snake_to_title(next_action.action_name)
                    matching_sop = None
                    for sop in sop_chain:
                        # Check both substring match and title case conversion
                        if next_action.action_name in sop or action_as_title.lower() in sop.lower():
                            matching_sop = sop
                            break

                    if matching_sop:
                        validation_error = (
                            f"Invalid action_name: '{next_action.action_name}' appears to be an SOP name, not a tool name!\n"
                            f"SOP matched: '{matching_sop}'\n"
                            f"Available tools: {', '.join(available_tool_names[:5])}...\n"
                            f"Check SOP Definitions to find which tools implement this SOP."
                        )
                        self._log(f"⚠️ Validation error: {validation_error}", progress, verbose)

                        # Add validation error to execution history and retry
                        execution_history.append({
                            "action_num": action_num,
                            "action": next_action.action_name,
                            "kwargs": next_action.action_kwargs,
                            "error": validation_error,
                            "sop_step": next_action.sop_step
                        })

                        # Ask agent to fix
                        self._log(f"🔄 Asking agent to use correct tool name...", progress, verbose)
                        if self.multi_agent_enabled:
                            next_action, step_progress, step_model_info = self._generate_next_action_multi_agent(
                                instruction, sop_chain, execution_history, len(sop_chain), completed_sops, len(actions)
                            )
                            model_info = step_model_info
                        else:
                            next_action, step_progress, repetition_warning = self._generate_next_action(
                                instruction, sop_chain, execution_history, len(sop_chain), completed_sops, len(actions)
                            )
                        progress.extend(step_progress)
                        execution_history.pop()  # Remove error entry

                        # Validate again
                        if next_action.action_name not in available_tool_names:
                            error = f"Agent still using invalid tool name after correction: {next_action.action_name}"
                            self._log(f"❌ {error}", progress, verbose)
                            return None, error, progress, model_info

                # Check if task is done before executing
                if next_action.done:
                    self._log("✅ Task complete - exiting iteration loop", progress, verbose)
                    break

                # Execute action with retry mechanism
                self._log(f"Executing: {next_action.action_name}", progress, verbose)
                self._log(f"Parameters: {json.dumps(next_action.action_kwargs, indent=2)}", progress, verbose)

                execution_success = False
                retry_count = 0
                max_retries = 3

                while not execution_success and retry_count < max_retries:
                    try:
                        result = self.action_executor.execute_action(
                            next_action.action_name,
                            next_action.action_kwargs
                        )
                        execution_success = True
                        self._log(f"✓ Success", progress, verbose)
                        if verbose:
                            result_str = str(result)
                            if len(result_str) > 200:
                                self._log(f"Result: {result_str[:200]}...", progress, verbose)
                            else:
                                self._log(f"Result: {result_str}", progress, verbose)

                        # Reset failure counter on successful execution
                        consecutive_failures = 0

                        # Record action and result
                        actions.append({
                            "name": next_action.action_name,
                            "kwargs": next_action.action_kwargs,
                            "sop_step": next_action.sop_step,
                            "reasoning": next_action.reasoning
                        })

                        execution_history.append({
                            "action_num": action_num,
                            "action": next_action.action_name,
                            "kwargs": next_action.action_kwargs,
                            "result": result,
                            "sop_step": next_action.sop_step
                        })

                        # Update completed SOPs (simple heuristic based on SOP step mentions)
                        if next_action.sop_step:
                            # Extract SOP number from step description
                            import re
                            match = re.search(r'SOP #?(\d+)', next_action.sop_step, re.IGNORECASE)
                            if match:
                                sop_num = int(match.group(1))
                                if sop_num > completed_sops:
                                    # Check if this is the last step of this SOP
                                    # (crude heuristic: assume done if next action will be from different SOP)
                                    completed_sops = sop_num - 1  # Mark previous as complete

                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)

                        # Enhance error message based on error type
                        enhanced_error = error_msg

                        # Case 1: Tool not found error
                        if "not found" in error_msg.lower() and "tool" in error_msg.lower():
                            # Get all available tool names
                            available_tool_names = [t.get_info()['function']['name'] for t in self.tools]

                            # Try to find tools relevant to current SOP
                            relevant_tools = []
                            if next_action.sop_step:
                                # Search SOP definitions for this SOP
                                for line in self.sop_definitions.split('\n'):
                                    if next_action.sop_step in line and 'Tools:' in line:
                                        # Extract tool names after "Tools:"
                                        tools_part = line.split('Tools:')[1].strip()
                                        relevant_tools = [t.strip() for t in tools_part.split(',')]
                                        break

                            if relevant_tools:
                                enhanced_error = f"{error_msg}\n\nFor SOP '{next_action.sop_step}', the available tools are: {', '.join(relevant_tools)}\nYou MUST use one of these tool names."
                            else:
                                enhanced_error = f"{error_msg}\n\nAvailable tool names (check SOP Definitions): {', '.join(available_tool_names[:10])}{'...' if len(available_tool_names) > 10 else ''}"

                        # Case 2: Missing parameter or parameter structure error
                        elif "missing" in error_msg.lower() and (
                                "argument" in error_msg.lower() or "parameter" in error_msg.lower()):
                            # Find the tool definition to show correct parameter structure
                            tool_info = None
                            for tool in self.tools:
                                if tool.get_info()['function']['name'] == next_action.action_name:
                                    tool_info = tool.get_info()['function']
                                    break

                            if tool_info:
                                params = tool_info.get('parameters', {})
                                required_params = params.get('required', [])
                                properties = params.get('properties', {})

                                enhanced_error = f"{error_msg}\n\nTool '{next_action.action_name}' requires the following parameters:\n"
                                for param in required_params:
                                    param_info = properties.get(param, {})
                                    param_type = param_info.get('type', 'unknown')
                                    param_desc = param_info.get('description', 'No description')
                                    enhanced_error += f"  - {param} ({param_type}): {param_desc}\n"

                                enhanced_error += f"\nEnsure ALL required parameters are provided as separate top-level keys in the parameters object, not nested inside other parameters."

                        # Case 3: Catch-all for all other exception types
                        else:
                            # Provide comprehensive context for ANY exception
                            available_tool_names = [t.get_info()['function']['name'] for t in self.tools]

                            enhanced_error = f"{error_msg}\n\n"
                            enhanced_error += "=" * 80 + "\n"
                            enhanced_error += "CONTEXT TO HELP RESOLVE THIS ERROR\n"
                            enhanced_error += "=" * 80 + "\n\n"

                            # Show current SOP if available
                            if next_action.sop_step:
                                enhanced_error += f"Current SOP: {next_action.sop_step}\n\n"

                                # Try to find SOP definition
                                sop_lines = []
                                in_sop = False
                                for line in self.sop_definitions.split('\n'):
                                    if next_action.sop_step in line:
                                        in_sop = True
                                    if in_sop:
                                        sop_lines.append(line)
                                        # Stop at next SOP or after reasonable length
                                        if len(sop_lines) > 1 and (line.startswith('SOP ') or line.startswith('===')):
                                            break
                                        if len(sop_lines) > 15:  # Limit length
                                            break

                                if sop_lines:
                                    enhanced_error += "SOP Definition:\n"
                                    enhanced_error += '\n'.join(sop_lines[:15]) + "\n\n"

                            # Show attempted action
                            enhanced_error += f"Attempted Action: {next_action.action_name}\n"
                            enhanced_error += f"Parameters: {next_action.action_kwargs}\n\n"

                            # Show available tools (first 15)
                            enhanced_error += f"Available Tools ({len(available_tool_names)} total, showing first 15):\n"
                            for i, tool_name in enumerate(available_tool_names[:15], 1):
                                enhanced_error += f"  {i}. {tool_name}\n"
                            if len(available_tool_names) > 15:
                                enhanced_error += f"  ... and {len(available_tool_names) - 15} more\n"

                            enhanced_error += "\n" + "=" * 80 + "\n"
                            enhanced_error += "SUGGESTIONS:\n"
                            enhanced_error += "=" * 80 + "\n"
                            enhanced_error += "1. Check if the tool name matches exactly (case-sensitive)\n"
                            enhanced_error += "2. Verify parameter names and types against the SOP definition\n"
                            enhanced_error += "3. Ensure all required parameters are provided\n"
                            enhanced_error += "4. Check execution history to see if prerequisites were met\n"
                            enhanced_error += "5. Review the SOP definition above for correct tool names and parameter structures\n"

                        self._log(f"❌ Execution failed: {error_msg}", progress, verbose)

                        if retry_count < max_retries:
                            self._log(f"🔄 Retry {retry_count}/{max_retries - 1} - Asking agent to fix the error...",
                                      progress, verbose)

                            # Add enhanced error to execution history so agent can see it
                            execution_history.append({
                                "action_num": action_num,
                                "action": next_action.action_name,
                                "kwargs": next_action.action_kwargs,
                                "error": enhanced_error,
                                "sop_step": next_action.sop_step
                            })

                            # Ask agent to generate corrected action
                            if self.multi_agent_enabled:
                                next_action, step_progress, step_model_info = self._generate_next_action_multi_agent(
                                    instruction, sop_chain, execution_history, len(sop_chain), completed_sops, len(actions)
                                )
                                model_info = step_model_info
                                repetition_warning = "no repetition detected"  # Multi-agent doesn't use repetition detection
                            else:
                                next_action, step_progress, repetition_warning = self._generate_next_action(
                                    instruction, sop_chain, execution_history, len(sop_chain), completed_sops, len(actions)
                                )

                            progress.extend(step_progress)

                            # Remove error from history before next attempt
                            execution_history.pop()

                            self._log(f"Corrected action: {next_action.action_name}", progress, verbose)
                            self._log(f"Corrected parameters: {json.dumps(next_action.action_kwargs, indent=2)}", progress,
                                      verbose)
                        else:
                            error = f"Action execution failed after {max_retries} attempts: {error_msg}"
                            self._log(f"💥 {error}", progress, verbose)
                            return None, error, progress, model_info

            if len(actions) >= max_actions:
                error = f"Reached maximum action limit ({max_actions})"
                self._log(f"⚠️ {error}", progress, verbose)
                return None, error, progress, model_info

            return actions, None, progress, model_info

        except Exception as e:
            error = f"Scaffolding failed: {str(e)}"
            self._log(f"❌ {error}", progress, verbose)
            return None, error, progress, model_info


    def _generate_next_action(
        self,
        instruction: str,
        sop_chain: List[str],
        execution_history: List[Dict[str, Any]],
        total_sops: int,
        completed_sops: int,
        action_count: int
) -> Tuple[NextAction, List[str], str]:
        """Generate next action using primary model.

        Returns:
            Tuple of (next_action, progress, repetition_warning)
        """
        progress = []

        # Detect repetition (last 3 actions)
        repetition_warning = "no repetition detected"
        force_stop = False
        if len(execution_history) >= 3:
            recent_actions = [
                (h.get('action'), json.dumps(h.get('kwargs', {}), sort_keys=True))
                for h in execution_history[-3:]
                if 'error' not in h  # Only check successful actions
            ]
            # Check if all 3 are identical
            if len(recent_actions) == 3 and len(set(recent_actions)) == 1:
                repetition_warning = f"⚠️ REPETITION DETECTED: Action '{recent_actions[0][0]}' repeated 3 times! This likely means the task is complete or you're stuck in a loop. Strongly consider setting done=True."
                # Check if we've warned about this before (last 4 actions all the same)
                if len(execution_history) >= 4:
                    four_actions = [
                        (h.get('action'), json.dumps(h.get('kwargs', {}), sort_keys=True))
                        for h in execution_history[-4:]
                        if 'error' not in h
                    ]
                    if len(four_actions) == 4 and len(set(four_actions)) <= 2:  # Repeating pattern
                        force_stop = True
                        repetition_warning += "\n\n**CRITICAL: You are stuck in an infinite loop. Setting done=True.**"
            # Check if last 2 are identical
            elif len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]:
                repetition_warning = f"⚠️ WARNING: Action '{recent_actions[-1][0]}' repeated 2 times. Check if task is complete before continuing."

        # Format execution history with type information and errors
        if execution_history:
            history_parts = []
            for h in execution_history:
                # Check if this is an error entry
                if 'error' in h:
                    history_parts.append(
                        f"❌ PREVIOUS ATTEMPT FAILED:\n"
                        f"Action {h['action_num']}: {h['action']}({json.dumps(h['kwargs'])})\n"
                        f"  ERROR: {h['error']}\n"
                        f"  SOP Step: {h.get('sop_step', 'N/A')}\n"
                        f"\n**You MUST fix this error in your next action!**\n"
                        f"Analyze the error message and correct the action accordingly."
                    )
                else:
                    result = h['result']
                    # Add type information to help agent understand structure
                    if isinstance(result, dict):
                        result_str = f'{{"type": "dict", "value": {json.dumps(result)}}}'
                    elif isinstance(result, list):
                        result_str = f'{{"type": "list", "value": {json.dumps(result)}}}'
                    else:
                        result_str = f'{{"type": "string", "value": {json.dumps(str(result))}}}'

                    history_parts.append(
                        f"Action {h['action_num']}: {h['action']}({json.dumps(h['kwargs'])})\n"
                        f"  Result: {result_str}\n"
                        f"  SOP Step: {h.get('sop_step', 'N/A')}"
                    )
            history_str = "\n\n".join(history_parts)
        else:
            history_str = "(No actions executed yet - this is the first action)"

        # Format SOP chain
        sop_chain_str = "\n".join([f"{i + 1}. {sop}" for i, sop in enumerate(sop_chain)])

        # Build progress tracker (domain-agnostic)
        progress_tracker = self._build_progress_tracker(execution_history, sop_chain)

        # Create parser
        parser = PydanticOutputParser(pydantic_object=NextAction)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(self.NEXT_ACTION_PROMPT)

        # Generate
        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_str,
            rules_description=self.rules_description,
            tools_description=self.tools_description,
            execution_history=history_str,
            total_sops=total_sops,
            completed_sops=completed_sops,
            action_count=action_count,
            repetition_warning=repetition_warning,
            progress_tracker=progress_tracker,
            format_instructions=parser.get_format_instructions()
        )

        response = self.llm.invoke(messages)
        next_action = parser.parse(response.content)

        # Note: Repetition detection is now handled through roundtable discussion
        # (R2 validation → Model R decision → Judge mediation if needed)
        # No automatic completion - Judge decides if repetition means task is done

        progress.append(f"Agent decision: {'DONE' if next_action.done else 'CONTINUE'}")
        progress.append(f"Reasoning: {next_action.reasoning}")
        if next_action.sop_step:
            progress.append(f"SOP Step: {next_action.sop_step}")

            # NOTE: SOP validation is handled by R2 Validator which has full context
            # This simple string matching creates false positives when Model R rephrases SOP names
            # Disabled to avoid confusion - R2 will catch actual hallucinations

        return next_action, progress, repetition_warning


    def _generate_next_action_multi_agent(
        self,
        instruction: str,
        sop_chain: List[str],
        execution_history: List[Dict[str, Any]],
        total_sops: int,
        completed_sops: int,
        action_count: int
) -> Tuple[NextAction, List[str], str]:
        """Generate next action using multi-agent consensus."""
        progress = []
        progress.append("🤖 Multi-Agent Mode: Generating with R and R2...")

        # Generate with Model R
        progress.append("  → Model R generating...")
        action_r, prog_r = self._generate_next_action(
            instruction, sop_chain, execution_history, total_sops, completed_sops, action_count
        )
        progress.append(f"    R: {'DONE' if action_r.done else action_r.action_name}")

        # Generate with Model R2
        progress.append("  → Model R2 generating...")
        original_llm = self.llm
        self.llm = self.llm_r2
        try:
            action_r2, prog_r2 = self._generate_next_action(
                instruction, sop_chain, execution_history, total_sops, completed_sops, action_count
            )
            progress.append(f"    R2: {'DONE' if action_r2.done else action_r2.action_name}")
        finally:
            self.llm = original_llm

        # Compare actions
        if self._actions_equal(action_r, action_r2):
            progress.append("  ✓ Consensus: Both models agree")
            return action_r, progress, "Consensus (R + R2)"

        # Actions differ - use judge
        progress.append("  ⚠️ Models disagree, consulting judge...")

        winner = self._judge_actions(instruction, sop_chain, execution_history, action_r, action_r2)

        if winner == "A":
            progress.append("  ✓ Judge selected Model R")
            return action_r, progress, "Judge → Model R"
        elif winner == "B":
            progress.append("  ✓ Judge selected Model R2")
            return action_r2, progress, "Judge → Model R2"
        else:
            # Judge said equal, default to R
            progress.append("  = Judge says equal, using Model R")
            return action_r, progress, "Judge → Equal (using R)"


    def _actions_equal(self, action_a: NextAction, action_b: NextAction) -> bool:
        """Check if two actions are essentially the same."""
        if action_a.done != action_b.done:
            return False
        if action_a.done:  # Both done
            return True
        return (
                action_a.action_name == action_b.action_name and
                action_a.action_kwargs == action_b.action_kwargs
        )


    def _judge_actions(
        self,
        instruction: str,
        sop_chain: List[str],
        execution_history: List[Dict[str, Any]],
        action_a: NextAction,
        action_b: NextAction
) -> str:
        """Use judge model to select better action."""
        # Format execution history
        if execution_history:
            history_str = "\n".join([
                f"Action {h['action_num']}: {h['action']}({json.dumps(h['kwargs'])})"
                for h in execution_history[-5:]  # Last 5 for context
            ])
        else:
            history_str = "(No actions executed yet)"

        # Format SOP chain
        sop_chain_str = ", ".join(sop_chain)

        # Format actions
        action_a_str = json.dumps({
            "done": action_a.done,
            "action_name": action_a.action_name,
            "action_kwargs": action_a.action_kwargs,
            "reasoning": action_a.reasoning,
            "sop_step": action_a.sop_step
        }, indent=2)

        action_b_str = json.dumps({
            "done": action_b.done,
            "action_name": action_b.action_name,
            "action_kwargs": action_b.action_kwargs,
            "reasoning": action_b.reasoning,
            "sop_step": action_b.sop_step
        }, indent=2)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(self.JUDGE_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_str,
            sop_definitions=self.sop_definitions,
            tools_description=self.tools_description,
            execution_history=history_str,
            action_a=action_a_str,
            action_b=action_b_str
        )

        # Get judge decision
        response = self.llm_judge.invoke(messages)
        decision = response.content.strip().upper()

        # Validate decision
        if decision not in ["A", "B", "EQUAL"]:
            # Invalid response, default to A
            return "A"

        return decision


    def _judge_r_vs_r2_feedback(
        self,
        instruction: str,
        sop_chain: List[str],
        execution_history: List[Dict[str, Any]],
        r_action: NextAction,
        r2_concerns: str,
        r_reasoning: str
) -> str:
        """
        Judge decides between R's position (PROCEED) and R2's concerns (need revision).
        This creates a "roundtable" discussion where:
        - R proposes an action and explains why to proceed despite concerns
        - R2 raises concerns about the proposed action
        - Judge sees both perspectives and decides who's right

        Returns:
            "R" if R should proceed, "R2" if R2's concerns should be addressed
        """
        # Format execution history (last 10 actions for context, with result previews)
        if execution_history:
            recent_history = execution_history[-10:]
            history_str = ""
            for i, entry in enumerate(recent_history, 1):
                if entry.get("error"):
                    history_str += f"{i}. ❌ {entry.get('action')} - Error: {entry.get('error')[:100]}...\n"
                elif entry.get('duplicate'):
                    history_str += f"{i}. ⚠️  {entry.get('action')} - Duplicate action (skipped)\n"
                else:
                    result_preview = str(entry.get('result', ''))[:100]
                    history_str += f"{i}. ✓ {entry.get('action')}({json.dumps(entry.get('kwargs', {}))}) - Success\n"
        else:
            history_str = "(No actions executed yet)"

        # Format SOP chain as numbered list
        sop_chain_text = "\n".join([f"{i+1}. {sop}" for i, sop in enumerate(sop_chain)])

        # Create roundtable judge prompt with FULL context
        roundtable_prompt = """You are a judge mediating a disagreement between two AI agents working on a task.

================================================================================
TASK CONTEXT
================================================================================

**Original Instruction:**
{instruction}

**Assigned SOP Chain** (steps to complete):
{sop_chain}

**SOP Definitions** (detailed breakdown of what each SOP entails):

{sop_definitions}

**Execution History** (what's been done so far):
{execution_history}

================================================================================
AVAILABLE TOOLS
================================================================================

{available_tools}

================================================================================
DOMAIN RULES
================================================================================

{domain_rules}

================================================================================
ROUNDTABLE DISCUSSION
================================================================================

**Model R (Primary Agent)** proposed this action:
- Action: {r_action_name}
- Parameters: {r_action_kwargs}
- SOP Step: {r_sop_step}
- R's Reasoning for PROCEEDING despite concerns: {r_reasoning}

**Model R2 (Validator)** raised these concerns:
{r2_concerns}

================================================================================
YOUR DECISION
================================================================================

You must decide who is correct in this disagreement. Consider:
1. Are R2's concerns valid and important for correctness?
2. Is R's reasoning sound for why these concerns can be ignored?
3. What does the SOP chain, SOP definitions, and execution history tell you?
4. Would following R's approach lead to correct task completion?
5. Would addressing R2's concerns lead to a better outcome?
6. Does R's proposed action comply with domain rules and tool schemas?

**Important Context:**
- You have the FULL picture: instruction, SOP chain, SOP definitions, execution history, tools, and rules
- R is the primary agent responsible for task completion
- R2 is an advisor who may sometimes be overly cautious
- Base your decision on the complete context, not just the surface-level disagreement

Respond with ONLY one word:
- "R" if Model R is correct and should proceed with the original action
- "R2" if Model R2's concerns are valid and must be addressed

Decision:"""

        prompt = ChatPromptTemplate.from_template(roundtable_prompt)
        messages = prompt.format_messages(
            instruction=instruction,
            sop_chain=sop_chain_text,
            sop_definitions=self.sop_definitions,
            execution_history=history_str,
            available_tools=self.tools_description,
            domain_rules=self.rules,
            r_action_name=r_action.action_name or "(done=True)",
            r_action_kwargs=json.dumps(r_action.action_kwargs, indent=2) if r_action.action_kwargs else "{}",
            r_sop_step=r_action.sop_step or "(not specified)",
            r_reasoning=r_reasoning,
            r2_concerns=r2_concerns
        )

        # Get judge decision
        response = self.llm_judge.invoke(messages)
        decision = response.content.strip().upper()

        # Validate decision
        if "R2" in decision:
            return "R2"
        elif "R" in decision or decision == "A":  # "A" for backward compatibility
            return "R"
        else:
            # Default to R2 concerns if unclear (safer to address concerns)
            return "R2"
