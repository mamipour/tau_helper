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
    Maps task instructions to SOP chains using LLM analysis.
    
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
    
    def __init__(self, llm: ChatOpenAI, domain: str, variation: str = "variation_2"):
        """
        Initialize SOP mapper.
        
        Args:
            llm: LangChain ChatOpenAI instance
            domain: Domain name (e.g., 'sec', 'airline')
            variation: Variation name (default: 'variation_2')
        """
        self.llm = llm
        self.domain = domain
        self.variation = variation
        
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
        """Format SOPs into a readable description for the prompt."""
        if not self.sops:
            return "No SOPs available (empty rules.py)"
        
        lines = []
        
        # Handle different SOP formats
        if isinstance(self.sops, dict):
            for name, sop in self.sops.items():
                if isinstance(sop, dict):
                    description = sop.get('description', sop.get('name', name))
                    lines.append(f"- **{name}**: {description}")
                elif isinstance(sop, str):
                    lines.append(f"- **{name}**: {sop}")
                else:
                    lines.append(f"- **{name}**")
        elif isinstance(self.sops, list):
            for sop in self.sops:
                if isinstance(sop, dict):
                    name = sop.get('name', 'Unknown')
                    description = sop.get('description', '')
                    lines.append(f"- **{name}**: {description}")
                else:
                    lines.append(f"- {sop}")
        
        return "\n".join(lines) if lines else str(self.sops)
    
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
    
    def map_instruction(
        self, 
        instruction: str,
        include_rules_context: bool = True
    ) -> SOPMapping:
        """
        Map an instruction to SOP chains.
        
        Args:
            instruction: The task instruction to analyze
            include_rules_context: Whether to include full rules context
            
        Returns:
            SOPMapping with chain analysis
        """
        # Format SOPs description
        sops_description = self._format_sops_description()
        
        # Format prompt
        formatted_prompt = self.prompt.format_messages(
            sops_description=sops_description,
            instruction=instruction,
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Get LLM response
        response = self.llm.invoke(formatted_prompt)
        
        # Clean response (remove reasoning tags)
        cleaned_content = self._clean_llm_response(response.content)
        
        # Parse structured output
        mapping = self.parser.parse(cleaned_content)
        
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

