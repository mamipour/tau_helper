"""
SOP Chain Mapper - Map task instructions to SOP chains.

This module analyzes task instructions and determines which SOPs (Standard Operating
Procedures) will be executed, detects ambiguities, and explains different interpretations.
"""

import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate


class SOPChain(BaseModel):
    """A single SOP chain interpretation."""
    
    sops: List[str] = Field(
        description="List of SOP names in execution order"
    )
    confidence: float = Field(
        description="Confidence score for this interpretation (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Explanation of why this SOP chain matches the instruction"
    )


class SOPMapping(BaseModel):
    """Result of mapping an instruction to SOP chains."""
    
    is_ambiguous: bool = Field(
        description="Whether the instruction has multiple valid interpretations"
    )
    primary_chain: SOPChain = Field(
        description="The most likely SOP chain"
    )
    alternative_chains: List[SOPChain] = Field(
        default_factory=list,
        description="Alternative SOP chain interpretations (if ambiguous)"
    )
    ambiguity_explanation: Optional[str] = Field(
        default=None,
        description="Explanation of why the instruction is ambiguous (if applicable)"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Information that would help disambiguate"
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested instruction fix (if ambiguity is fixable by minimal edits). Must be non-procedural, user-facing, and preserve the story/context."
    )


class SOPMapper:
    """
    Maps task instructions to SOP chains using LLM analysis with roundtable validation.

    Uses multi-agent roundtable:
    - Model R proposes SOP chain
    - Model R2 validates chain (checks for missing/incorrect SOPs)
    - Model R responds to concerns (REVISE or PROCEED)
    - Judge mediates disagreements

    Analyzes instructions against defined SOPs to determine execution paths,
    detect ambiguities, and explain different interpretations.
    """

    MAPPING_PROMPT = """You are an expert at analyzing task instructions and mapping them to Standard Operating Procedures (SOPs).

## Your Task

Given a task instruction and available SOPs, determine:
1. Which SOP chain(s) will be executed
2. Whether the instruction is ambiguous (multiple valid interpretations)
3. Alternative interpretations (if ambiguous)

## SOPs Available

{sops_description}

## Task Instruction

{instruction}

## Analysis Instructions

### 1. Identify ALL Complete SOP Chains

**CRITICAL: Multi-Workflow Detection**
- Instructions may contain MULTIPLE independent workflows
- Look for conjunction words: "also", "additionally", "and", "furthermore", "plus", "as well as"
- Look for sequential patterns: "before X, do Y", "want to understand X before Y", "need to check X then Y"
- Look for "then" clauses: "do X... Then Y" → Both X and Y need SOPs
- If instruction mentions MULTIPLE distinct operations, include SOPs for ALL of them
- List ALL SOPs for ALL workflows in logical execution order

**Don't Skip Analysis/Understanding Operations**
- Keywords: "want to understand", "need to check", "review", "analyze", "see", "assess", "view"
- + data/performance/results → This requires EXECUTION of analysis/aggregation SOPs
- "To baseline", "to establish baseline", "to assess capacity" → Also require analysis SOPs
- Pattern: "understand/check X... then Y" → Both X (analysis) and Y (action) need SOPs

**Entity Lookup Requirements (CRITICAL)**
- EVERY entity mentioned by NAME must be looked up BEFORE being used in ANY operation
- Named entities (people, accounts, records) require lookup SOPs to get their IDs
- NEVER assume IDs are available without lookup SOPs
- If multiple entities are mentioned (e.g., "from Person A to Person B"), include lookup SOPs for EACH

**Transfer/Reassignment Operations Require BOTH Lookups:**
- When instruction says "transfer/reassign X from Person A to Person B":
  * Person A (source) → needs "Locate Source Owner" or "Locate Sales Representative" SOP
  * Person B (target) → needs "Locate Sales Representative" SOP  
  * Entity X (account/record) → needs "Lookup Accounts by Name" or similar SOP
  * ALL THREE must be looked up BEFORE the transfer operation
- Example: "reassign BlueCurve from Ava to Chris"
  * SOP 1: Locate Source Owner (Ava)
  * SOP 2: Locate Sales Representative (Chris)
  * SOP 3: Lookup Accounts by Name (BlueCurve)
  * SOP 4-5: Transfer/Reassignment SOPs
- NEVER skip the target lookup - the target entity ALWAYS needs to be looked up

**Multi-Word Entity Detection**
- Multi-word proper nouns (e.g., "Delta Freight", "John Smith") are entity names
- Pattern: [Capitalized Words] + action verb → Entity name that needs lookup
- Entity lookups must happen BEFORE operations that use them

**Transfer/Reassignment SOP Prerequisites:**
- For ANY transfer/reassignment operation, the following SOPs must run in order:
  1. Lookup source and target entities
  2. Lookup items to be transferred (if by name)
  3. **Validation SOP (e.g., "Lookup Account Roster For Transfer")** - validates items belong to source
  4. Final transfer/reassignment SOP
- NEVER skip the validation SOP that runs between lookup and transfer
- Example complete chain: Locate Source → Locate Target → Lookup Items → Validate Transfer → Execute Transfer

**Common Multi-Workflow Patterns:**
- Operation 1 + Operation 2: "Do X... Also Y..."
- Analysis + Operation: "Understand/check/review... Then do..."
- Multiple entity operations: "Update entity A... Also reassign entity B..."
- Sequential workflows: "First X... Then also Y..."
- Before + After: "Before finalizing... check X... Then Y"

### 2. Check for Ambiguity

- Could the instruction be interpreted in multiple ways?
- Does it leave details unclear that would lead to DIFFERENT SOP chains?
- CRITICAL: If primary chain confidence is >85%, mark as NOT ambiguous (is_ambiguous=false)
- Only mark ambiguous if there are truly MULTIPLE REASONABLE interpretations

### 3. Explain Reasoning

- Why does this instruction map to this SOP chain?
- What keywords/phrases indicate which SOPs to use?

### 4. Alternative Interpretations (ONLY if ambiguous)

- Only provide if is_ambiguous=true
- Alternatives should have meaningful confidence (>30%)
- Each alternative must represent a genuinely different interpretation
- Confidence scores must be logically consistent:
  * If primary is >85%: NO alternatives (not ambiguous)
  * If primary is 60-85%: alternatives should be 40-60%
  * If primary is <60%: alternatives could be 40-55%

### 5. Missing Information Check

**Check for Missing REQUIRED Parameters:**
- Does instruction provide all parameters needed for the SOPs?
- CRITICAL: Check if lookup SOPs will provide missing parameters
  * If entity name given BUT no ID → NOT missing if lookup SOP exists in chain
  * If lookup SOP will get the parameter → NOT missing
- ONLY mark as missing if no SOP will provide it
- ONLY generate suggested_fix if parameters are TRULY missing

**If is_ambiguous=false:** set missing_information to empty list []

### 6. Suggest a Fix (ONLY if ambiguous)

- If is_ambiguous=false: set suggested_fix to null
- If is_ambiguous=true: provide complete suggested_fix
- Must be USER-FACING and NON-PROCEDURAL:
  * Written in second person ("You are...", "You want to...")
  * Describes WHAT, not HOW
  * NO function names or API calls
  * NO step-by-step instructions ("First...", "Then...")
  * Natural, conversational language
- Based on PRIMARY chain ONLY (not alternatives)
- Don't add features not in primary chain
- Keep same story, role, context - only add clarity

## Output Format

{format_instructions}

## Key Guidelines

### Confidence Score Logic
- Primary >85%: Not ambiguous, no alternatives
- Primary 60-85%: Ambiguous, alternatives 40-60%
- Primary <60%: Multiple valid interpretations, alternatives 40-55%
- Scores must be logically consistent

### General Rules
- SOPs are high-level operations (typically 1-3 API calls each)
- Tasks concatenate 2-5 SOPs to achieve goals
- Include parameters in SOP names when helpful
- If instruction doesn't mention something, don't assume it's needed
- Only mark ambiguous if instruction genuinely unclear (not just "we could also do X")
"""

    R2_VALIDATION_PROMPT = """You are R2, a validator checking if a proposed SOP chain is complete and correct.

## Task Context

**Instruction:** {instruction}

**Domain Rules and Available SOPs:**
{sops_description}

**Proposed SOP Chain (from Model R):**
{proposed_chain}

**Reasoning from Model R:**
{reasoning}

## Your Validation Task

The domain rules above contain Standard Operating Procedures (SOPs) that can be executed. SOPs may be defined in various formats - they might be explicitly labeled as "SOP [Name]", or they might be described as operational procedures, workflows, or process steps. Read the rules carefully to understand what operations are available.

Check the proposed SOP chain for these issues:

1. **Missing SOPs**: Does the instruction explicitly request operations NOT covered by the proposed chain?
   - Read through the domain rules to identify available operational procedures
   - Look for instruction keywords: "normalize", "persist", "store", "save", "export", "analyze", "aggregate", "validate", "transform", "extract", "pull", "create", "populate"
   - Match instruction intent to available procedures in the rules, regardless of how they're formatted
   - If instruction says "normalize the data", look for normalization/persistence procedures in the rules
   - If instruction says "extract data", look for extraction/pull procedures in the rules
   - Check if ALL explicitly requested operations have corresponding SOPs in the chain

2. **Incorrect SOP Ordering**: Are the SOPs in the wrong order?
   - Prerequisites must come before dependent SOPs
   - Data fetch/extract before data transform/normalize
   - Validate before persist

3. **Missing Prerequisites**: Are required prerequisite SOPs missing?
   - Check domain rules for required setup procedures

4. **Extra/Unnecessary SOPs**: Are there SOPs that don't match the instruction?
   - Only flag if clearly not requested

## Response Format

Return JSON:
{{
  "has_concerns": true/false,
  "concerns": "Detailed list of concerns (bullet points). If no concerns, empty string.",
  "severity": "high/medium/low" (high = missing critical SOPs, medium = ordering issues, low = minor improvements)
}}

**IMPORTANT**:
- SOPs may be defined in various natural language formats - use semantic understanding to identify them
- If instruction explicitly mentions an operation, search the rules for ANY procedure that performs that operation
- Be strict about completeness - if relevant procedures exist in the rules, they should be included in the chain
- Focus on what the instruction ASKS FOR, not on what Model R happened to propose
"""

    R_DECISION_PROMPT = """You are Model R, the primary SOP chain mapper.

R2 Validator has raised concerns about your proposed SOP chain. You must decide:
- **REVISE**: Acknowledge R2's concerns and provide a revised SOP chain
- **PROCEED**: Explain why R2's concerns are invalid and you should proceed with the original chain

## Context

**Instruction:** {instruction}

**Your Original SOP Chain:**
{proposed_chain}

**Your Original Reasoning:**
{reasoning}

**R2's Concerns:**
{r2_concerns}

## Your Decision

Consider:
1. Are R2's concerns valid based on the instruction?
2. Did you miss an explicitly requested operation?
3. Is the instruction genuinely ambiguous or did R2 misunderstand?

Return JSON:
{{
  "decision": "REVISE" or "PROCEED",
  "reasoning": "Explain your decision",
  "revised_chain": {{
    "sops": ["SOP 1", "SOP 2", ...],
    "confidence": 0.0-1.0,
    "reasoning": "Why this revised chain is better"
  }} (only if decision is REVISE, otherwise null)
}}
"""

    JUDGE_PROMPT = """You are a judge mediating a disagreement about an SOP chain mapping.

## Roundtable Discussion

**Model R (Primary Mapper)** proposed this SOP chain:
{proposed_chain}

R's Reasoning: {r_reasoning}

**Model R2 (Validator)** raised these concerns:
{r2_concerns}

**Model R decided to PROCEED** (disagreed with R2), with this reasoning:
{r_proceed_reasoning}

## Task Context

**Instruction:** {instruction}

**Available SOPs:**
{sops_description}

## Your Decision

You must decide who is correct. Consider:
1. Does the instruction explicitly request operations that the chain is missing?
2. Are there SOPs available that clearly match the instruction's intent?
3. Is R's chain actually complete, or did R miss something obvious?
4. Is R2 being overly strict, or are the concerns valid?

Respond with ONLY one word:
- "R" if Model R is correct and the chain is complete
- "R2" if Model R2's concerns are valid and the chain needs revision

Decision:"""

    def __init__(self, llm: ChatOpenAI, domain: str, variation: str = "variation_2", llm_r2: ChatOpenAI = None, llm_judge: ChatOpenAI = None):
        """
        Initialize SOP mapper with roundtable validation.

        Args:
            llm: LangChain ChatOpenAI instance (Model R - primary mapper)
            domain: Domain name (e.g., 'sec', 'airline')
            variation: Variation name (default: 'variation_2')
            llm_r2: Optional LangChain ChatOpenAI for R2 validator
            llm_judge: Optional LangChain ChatOpenAI for Judge (mediator)
        """
        self.llm = llm
        self.llm_r2 = llm_r2
        self.llm_judge = llm_judge
        self.domain = domain
        self.variation = variation

        # Roundtable mode enabled if R2 and Judge are provided
        self.roundtable_enabled = llm_r2 is not None and llm_judge is not None

        # Load SOPs from rules.py
        self.sops = self._load_sops()

        # Setup parser and prompt
        self.parser = PydanticOutputParser(pydantic_object=SOPMapping)
        self.prompt = ChatPromptTemplate.from_template(self.MAPPING_PROMPT)
    
    def _load_sops(self) -> Dict[str, Any]:
        """Load SOPs from domain's rules.py file."""
        try:
            # Import rules module
            rules_module = importlib.import_module(
                f"domains.{self.domain}.variations.{self.variation}.rules"
            )
            
            # Try to get RULES or similar
            if hasattr(rules_module, 'RULES'):
                return rules_module.RULES
            elif hasattr(rules_module, 'SOPS'):
                return rules_module.SOPS
            else:
                # Get all non-private attributes that might be SOPs
                sops = {}
                for name in dir(rules_module):
                    if not name.startswith('_'):
                        attr = getattr(rules_module, name)
                        if isinstance(attr, dict) or isinstance(attr, list):
                            sops[name] = attr
                return sops
        except Exception as e:
            raise ValueError(
                f"Could not load SOPs from {self.domain}/{self.variation}: {e}"
            )
    
    def _format_sops_description(self) -> str:
        """
        Format SOPs/rules into a description for the prompt.

        Returns the raw rules as-is, trusting the LLM to understand
        the natural language definitions regardless of format.
        """
        if not self.sops:
            return "No SOPs available (empty rules.py)"

        # Handle different SOP formats
        if isinstance(self.sops, dict):
            lines = []
            for name, sop in self.sops.items():
                if isinstance(sop, dict):
                    description = sop.get('description', sop.get('name', name))
                    lines.append(f"- **{name}**: {description}")
                elif isinstance(sop, str):
                    lines.append(f"- **{name}**: {sop}")
                else:
                    lines.append(f"- **{name}**")
            return "\n".join(lines)
        elif isinstance(self.sops, list):
            # For list format, join all rules as-is
            # The LLM can identify SOPs from natural language context
            return "\n\n".join(str(rule) for rule in self.sops if rule)
        else:
            return str(self.sops)
    
    def _clean_llm_response(self, content: str) -> str:
        """
        Clean LLM response by removing reasoning tags and markdown formatting.
        
        Some reasoning models output their thinking process in tags, followed
        by the actual JSON response wrapped in markdown code fences.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Cleaned content suitable for JSON parsing
        """
        import re
        
        # Remove <think>...</think> tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Also handle other common reasoning tags
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        
        # Remove markdown code fences (```json ... ``` or ``` ... ```)
        content = re.sub(r'^```(?:json)?\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Extract JSON object (remove text before and after)
        # Find the first { and last } to isolate the JSON
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            # Extract just the JSON object
            content = content[first_brace:last_brace + 1]
        elif last_brace != -1:
            # Fallback: just remove text after last brace
            content = content[:last_brace + 1]
        
        # Fix common JSON typos from reasoning models
        # Fix "key":- value (should be "key": value)
        content = re.sub(r'":- ', r'": ', content)
        content = re.sub(r'":-\s*', r'": ', content)
        
        # Fix other common typos
        # "或少" appears to be Chinese characters that sometimes appear instead of numbers
        content = re.sub(r'或少', r'', content)
        
        # Strip leading/trailing whitespace
        content = content.strip()

        return content

    def _ask_r2_validator(self, instruction: str, proposed_chain: SOPChain) -> Dict[str, Any]:
        """
        Ask R2 to validate the proposed SOP chain.

        Returns dict with:
            - has_concerns: bool
            - concerns: str
            - severity: str
        """
        sops_description = self._format_sops_description()

        prompt = ChatPromptTemplate.from_template(self.R2_VALIDATION_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            sops_description=sops_description,
            proposed_chain=", ".join(proposed_chain.sops),
            reasoning=proposed_chain.reasoning
        )

        response = self.llm_r2.invoke(messages)
        content = self._clean_llm_response(response.content)

        try:
            return json.loads(content)
        except:
            # If parsing fails, assume no concerns
            return {"has_concerns": False, "concerns": "", "severity": "low"}

    def _ask_model_r_decision(self, instruction: str, proposed_chain: SOPChain, r2_concerns: str) -> Dict[str, Any]:
        """
        Ask Model R to decide: REVISE or PROCEED after R2's concerns.

        Returns dict with:
            - decision: "REVISE" or "PROCEED"
            - reasoning: str
            - revised_chain: dict (if REVISE) or None
        """
        prompt = ChatPromptTemplate.from_template(self.R_DECISION_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            proposed_chain=", ".join(proposed_chain.sops),
            reasoning=proposed_chain.reasoning,
            r2_concerns=r2_concerns
        )

        response = self.llm.invoke(messages)
        content = self._clean_llm_response(response.content)

        try:
            return json.loads(content)
        except:
            # If parsing fails, default to PROCEED
            return {"decision": "PROCEED", "reasoning": "Failed to parse response", "revised_chain": None}

    def _consult_judge(self, instruction: str, proposed_chain: SOPChain, r2_concerns: str, r_proceed_reasoning: str) -> str:
        """
        Judge mediates disagreement between R and R2.

        Returns:
            "R" if R is correct, "R2" if R2's concerns are valid
        """
        sops_description = self._format_sops_description()

        prompt = ChatPromptTemplate.from_template(self.JUDGE_PROMPT)
        messages = prompt.format_messages(
            instruction=instruction,
            sops_description=sops_description,
            proposed_chain=", ".join(proposed_chain.sops),
            r_reasoning=proposed_chain.reasoning,
            r2_concerns=r2_concerns,
            r_proceed_reasoning=r_proceed_reasoning
        )

        response = self.llm_judge.invoke(messages)
        decision = response.content.strip().upper()

        # Validate decision
        if "R2" in decision:
            return "R2"
        else:
            return "R"

    def map_instruction(
        self,
        instruction: str,
        include_rules_context: bool = True,
        verbose: bool = False
    ) -> SOPMapping:
        """
        Map an instruction to SOP chains with optional roundtable validation.

        Roundtable flow (if enabled):
        1. Model R proposes SOP chain
        2. Model R2 validates (checks for missing/incorrect SOPs)
        3. If R2 has concerns:
           - Ask Model R to decide: REVISE or PROCEED
           - If R decides PROCEED (disagrees with R2), Judge mediates
           - Judge's decision is final
        4. Return final chain

        Args:
            instruction: The task instruction to analyze
            include_rules_context: Whether to include full rules context
            verbose: Print roundtable discussion details

        Returns:
            SOPMapping with chain analysis
        """
        # Step 1: Model R proposes SOP chain
        sops_description = self._format_sops_description()

        formatted_prompt = self.prompt.format_messages(
            sops_description=sops_description,
            instruction=instruction,
            format_instructions=self.parser.get_format_instructions()
        )

        response = self.llm.invoke(formatted_prompt)
        cleaned_content = self._clean_llm_response(response.content)
        mapping = self.parser.parse(cleaned_content)

        # If roundtable mode disabled, return immediately
        if not self.roundtable_enabled:
            if verbose:
                print(f"⚠️  SOP Mapper: Roundtable mode DISABLED (llm_r2={self.llm_r2 is not None}, llm_judge={self.llm_judge is not None})")
            return mapping

        # Step 2: Roundtable validation
        if verbose:
            print(f"\n🔍 R2 Validator: Checking proposed SOP chain...")
            print(f"   Proposed chain: {', '.join(mapping.primary_chain.sops)}")

        r2_result = self._ask_r2_validator(instruction, mapping.primary_chain)

        if not r2_result.get("has_concerns", False):
            if verbose:
                print(f"✓ R2 Validator: No concerns identified")
            return mapping

        # Step 3: R2 has concerns - ask Model R to decide
        r2_concerns = r2_result.get("concerns", "")
        severity = r2_result.get("severity", "medium")

        if verbose:
            print(f"⚠️  R2 Validator identified concerns ({severity} severity):")
            for line in r2_concerns.split("\n"):
                if line.strip():
                    print(f"   {line}")
            print(f"\n💭 Asking Model R to decide: REVISE or PROCEED...")

        r_decision_result = self._ask_model_r_decision(instruction, mapping.primary_chain, r2_concerns)
        decision = r_decision_result.get("decision", "PROCEED")
        r_reasoning = r_decision_result.get("reasoning", "")

        if decision == "REVISE":
            # Model R agreed with R2's concerns and provided revision
            if verbose:
                print(f"✓ Model R decided to REVISE based on R2's feedback")

            revised_chain_dict = r_decision_result.get("revised_chain")
            if revised_chain_dict and revised_chain_dict.get("sops"):
                revised_chain = SOPChain(
                    sops=revised_chain_dict["sops"],
                    confidence=revised_chain_dict.get("confidence", mapping.primary_chain.confidence),
                    reasoning=revised_chain_dict.get("reasoning", "")
                )
                mapping.primary_chain = revised_chain

                if verbose:
                    print(f"   Revised chain: {', '.join(revised_chain.sops)}")

            return mapping

        # Step 4: Model R decided PROCEED (disagreed with R2) - consult Judge
        if verbose:
            print(f"⚠️ Model R decided to PROCEED with original chain (disagreed with R2)")
            print(f"   R's reasoning: {r_reasoning}")
            print(f"\n🏛️ Consulting Judge to resolve disagreement...")

        judge_decision = self._consult_judge(instruction, mapping.primary_chain, r2_concerns, r_reasoning)

        if judge_decision == "R":
            # Judge sided with R - proceed with original chain
            if verbose:
                print(f"  ✓ Judge sided with R - proceeding with original chain")
            return mapping
        else:
            # Judge sided with R2 - force revision
            if verbose:
                print(f"  ✓ Judge sided with R2 - chain needs revision")
                print(f"  🔄 Forcing Model R to provide revised chain...")

            # Ask R again with judge's backing
            forced_prompt = f"""The Judge has reviewed the disagreement and sided with R2's concerns.

You MUST revise your SOP chain to address these concerns:
{r2_concerns}

The Judge has determined these concerns are valid and must be addressed."""

            r_forced_result = self._ask_model_r_decision(instruction, mapping.primary_chain, forced_prompt)

            revised_chain_dict = r_forced_result.get("revised_chain")
            if revised_chain_dict and revised_chain_dict.get("sops"):
                revised_chain = SOPChain(
                    sops=revised_chain_dict["sops"],
                    confidence=revised_chain_dict.get("confidence", mapping.primary_chain.confidence),
                    reasoning=revised_chain_dict.get("reasoning", "Judge-mandated revision")
                )
                mapping.primary_chain = revised_chain

                if verbose:
                    print(f"    ✓ Model R provided revised chain: {', '.join(revised_chain.sops)}")
            else:
                if verbose:
                    print(f"    ⚠️ Model R failed to provide valid revision - keeping original chain")

            return mapping
    
    def map_task(
        self, 
        task_id: str,
        domain: Optional[str] = None,
        variation: Optional[str] = None
    ) -> Tuple[str, SOPMapping]:
        """
        Map a specific task by ID to SOP chains.
        
        Args:
            task_id: Task identifier (e.g., 'task_001')
            domain: Override domain (uses self.domain if None)
            variation: Override variation (uses self.variation if None)
            
        Returns:
            Tuple of (instruction, SOPMapping)
        """
        domain = domain or self.domain
        variation = variation or self.variation
        
        # Load task
        tasks_module = importlib.import_module(
            f"domains.{domain}.variations.{variation}.tasks"
        )
        
        # Find the task
        tasks = {task.user_id: task for task in tasks_module.TASKS}
        
        if task_id not in tasks:
            raise ValueError(f"Task '{task_id}' not found in {domain}/{variation}")
        
        task = tasks[task_id]
        instruction = task.instruction
        
        # Map the instruction
        mapping = self.map_instruction(instruction)
        
        return instruction, mapping
    
    def compare_with_actual(
        self,
        task_id: str,
        domain: Optional[str] = None,
        variation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare predicted SOP chain with actual task actions.
        
        Args:
            task_id: Task identifier
            domain: Override domain
            variation: Override variation
            
        Returns:
            Comparison results with match analysis
        """
        domain = domain or self.domain
        variation = variation or self.variation
        
        # Get task
        tasks_module = importlib.import_module(
            f"domains.{domain}.variations.{variation}.tasks"
        )
        tasks = {task.user_id: task for task in tasks_module.TASKS}
        task = tasks[task_id]
        
        # Get predicted mapping
        instruction, mapping = self.map_task(task_id, domain, variation)
        
        # Get actual actions
        actual_actions = [action.name for action in task.actions]
        
        # Compare
        predicted_sops = mapping.primary_chain.sops
        
        return {
            'task_id': task_id,
            'instruction': instruction,
            'predicted_sops': predicted_sops,
            'actual_actions': actual_actions,
            'is_ambiguous': mapping.is_ambiguous,
            'alternative_chains': [
                chain.sops for chain in mapping.alternative_chains
            ],
            'match': self._analyze_match(predicted_sops, actual_actions),
            'full_mapping': mapping
        }
    
    def _analyze_match(
        self, 
        predicted_sops: List[str], 
        actual_actions: List[str]
    ) -> Dict[str, Any]:
        """Analyze how well predicted SOPs match actual actions."""
        # This is a simple heuristic - could be enhanced
        # Check if SOP names appear in action names or vice versa
        
        matches = 0
        for sop in predicted_sops:
            sop_lower = sop.lower().replace('_', '').replace('-', '')
            for action in actual_actions:
                action_lower = action.lower().replace('_', '').replace('-', '')
                if sop_lower in action_lower or action_lower in sop_lower:
                    matches += 1
                    break
        
        match_rate = matches / len(predicted_sops) if predicted_sops else 0
        
        return {
            'matches': matches,
            'predicted_count': len(predicted_sops),
            'actual_count': len(actual_actions),
            'match_rate': match_rate,
            'status': 'good' if match_rate > 0.7 else 'partial' if match_rate > 0.3 else 'poor'
        }


def get_sops_from_domain(domain: str, variation: str = "variation_2") -> Dict[str, Any]:
    """
    Helper function to load SOPs from a domain's rules.py
    
    Args:
        domain: Domain name
        variation: Variation name
        
    Returns:
        Dictionary of SOPs
    """
    try:
        rules_module = importlib.import_module(
            f"domains.{domain}.variations.{variation}.rules"
        )
        
        if hasattr(rules_module, 'RULES'):
            return rules_module.RULES
        elif hasattr(rules_module, 'SOPS'):
            return rules_module.SOPS
        else:
            return {}
    except Exception:
        return {}

