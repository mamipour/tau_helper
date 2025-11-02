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
    
    MAPPING_PROMPT = """You are an expert at analyzing task instructions in the Warrior Tau-Bench system.

## Your Task

Given a task instruction and available SOPs (Standard Operating Procedures), determine:
1. Which SOP chain will be executed (INCLUDING prerequisites)
2. Whether the instruction is ambiguous (multiple valid interpretations)
3. Why it's ambiguous (if applicable)
4. Alternative interpretations

## SOPs Available

{sops_description}

## Task Instruction

{instruction}

## Instructions for Analysis

1. **Identify the COMPLETE SOP Chain**: 
   - List ALL steps that would be executed in order
   - INCLUDE prerequisites (setup_python_environment, get_configuration, configure_sec_api_identity, validate_company_data_access)
   - INCLUDE the actual SOPs (e.g., "SOP Pull Core Statements", "SOP Persist Normalized Financials")
   - Show parameters for SOPs where applicable (e.g., company_ticker=AAPL, num_periods=3)
   - Prerequisites always run BEFORE SOPs when SEC data extraction is needed

2. **Check for Ambiguity**: Could the instruction be interpreted in multiple ways?
   - Does it leave execution details unclear that would lead to DIFFERENT SOP chains?
   - Could it map to genuinely different SOP sequences?
   - CRITICAL: If primary chain confidence is >85%, mark as NOT ambiguous (is_ambiguous=false)
   - Only mark ambiguous if there are truly MULTIPLE REASONABLE interpretations
   
3. **Explain Reasoning**: Why does this instruction map to this SOP chain?
   - What keywords/phrases indicate which SOPs to use?
   - What policies/rules guide the chain?

4. **Alternative Interpretations**: ONLY if truly ambiguous
   - Only provide alternatives if is_ambiguous=true
   - Alternatives should have meaningful confidence (>30%)
   - Confidence scores should make logical sense: if primary is 60%, alternatives could be 40-50%
   - If primary is >85%, there should be NO alternatives (not ambiguous)
   - Each alternative must represent a genuinely different interpretation, not just "nice to have" features

5. **Missing Information**: What details would help disambiguate? (ONLY if ambiguous)
   - If is_ambiguous=false, set missing_information to empty list []
   - If is_ambiguous=true, list ONLY information that would help choose between primary and alternatives
   - DON'T list "nice to have" features that aren't mentioned in the instruction
   - DON'T list standard concepts that have deterministic meanings
   - DON'T list information available from domain rules/policies
   - Examples of truly missing: explicit output destination (DB vs spreadsheet), validation requirements beyond standard

6. **Suggest a Fix** (ONLY if ambiguous):
   - If is_ambiguous=false, set suggested_fix to null (instruction is already clear)
   - If is_ambiguous=true AND fixable by minimal edits, provide a COMPLETE suggested_fix
   - The suggested_fix must be a FULL REWRITE based ONLY on the PRIMARY chain
   - CRITICAL: The suggested fix must meet USER-FACING and NON-PROCEDURAL criteria (score 80-100):
     - **Written in second person**: Use "You are...", "You want to...", "You need to..."
     - **Describes WHAT, not HOW**: Focus on goals and outcomes, never implementation steps
     - **Natural, conversational language**: Like a colleague describing a task
     - **NO function names**: Never mention API calls, function names, or technical operations
     - **NO step-by-step instructions**: Never use "First...", "Then...", "Call...", "Execute..."
     - **NO procedural hints**: No "retrieve X then calculate Y" style instructions
   - CRITICAL RULES for suggested fix:
     - **Based on primary chain ONLY**: Include ONLY SOPs from the primary chain, nothing from alternatives
     - **Don't mention unasked features**: If the original doesn't ask for spreadsheets/time-series/etc, don't add them
     - **Concrete, not questions**: Provide a complete instruction, NEVER ask "Please specify..." or "Do you want..."
     - **Minimal changes**: Keep same story, role, context, tone - only add clarity for ambiguous parts
   
   - What to clarify in the fix:
     - If original says "for DCF model" (ambiguous), make it explicit: "stored in the financials database"
     - If original says "ensure consistency" (vague), be specific: "normalized and validated"
     - If original says "prepare data" (ambiguous), clarify: "extract, store, and normalize"
   
   - What NOT to add:
     - DON'T add features not in primary chain (e.g., if primary chain doesn't create spreadsheet, don't mention it)
     - DON'T ask questions ("specify whether...", "indicate if...")
     - DON'T add procedural steps ("first", "then", "call X then Y")
     - DON'T change the persona, role, or business context
   
   - Examples (USER-FACING and NON-PROCEDURAL):
     ✅ GOOD: "You are Sarah Chen. You want to extract Apple's income statements, balance sheets, and cash flow statements for the last 3 fiscal years, store them in the financials database, and normalize the data for future DCF analysis."
     ✅ GOOD: "You need to analyze Tesla's Q4 2023 financial performance and calculate their debt-to-equity ratio."
     ❌ BAD (procedural): "First, call get_balance_sheet() for Apple, then execute normalize_financials() with the results."
     ❌ BAD (asking questions): "Please specify whether you want the data in a spreadsheet or database."
     ❌ BAD (step-by-step): "First retrieve the balance sheet, then calculate ratios, then store the results."
   
   - Complete Example:
     Original (ambiguous): "You are Sarah Chen. You want to analyze Apple's financials by extracting their statements for DCF modeling."
     Primary chain: Extract + Store + Normalize (no spreadsheet, no time-series)
     Fixed (clear, user-facing, non-procedural): "You are Sarah Chen. You want to extract Apple's income statements, balance sheets, and cash flow statements for the last 3 fiscal years, store them in the financials database, and normalize the data for future DCF analysis."
     
   - When to set suggested_fix to null:
     - If is_ambiguous=false (already clear)
     - If ambiguity requires a true business decision between fundamentally different workflows

## Output Format

{format_instructions}

## Important Notes

### Prerequisites vs SOPs
- **Prerequisites** are setup steps that run BEFORE SOPs when needed:
  - setup_python_environment
  - get_configuration  
  - configure_sec_api_identity
  - validate_company_data_access
- **SOPs** are high-level business operations (e.g., "SOP Pull Core Statements")
- Your output should include BOTH prerequisites AND SOPs in the correct order

### Standard Concepts (NOT ambiguous)
- "Last N fiscal years" = most recent N fiscal years available in SEC data (deterministic)
- "Normalize data" = apply standard financial normalization rules per policy (deterministic)
- "Extract complete statements" = all three core statements (income, balance, cash flow)
- Company ticker lookup, standard metrics, default units/scales are all deterministic

### Confidence Score Logic (CRITICAL)
- If primary chain confidence is >85%, set is_ambiguous=false and provide NO alternatives
- If primary chain confidence is 60-85%, set is_ambiguous=true and alternatives should be 40-60%
- If primary chain confidence is <60%, there may be multiple equally valid interpretations (alternatives 40-55%)
- Confidence scores should be LOGICALLY CONSISTENT: 92% primary + 60% alternative = nonsense
- Example GOOD: Primary 95% (not ambiguous, no alternatives)
- Example GOOD: Primary 65%, Alternative 1: 50%, Alternative 2: 35% (ambiguous)
- Example BAD: Primary 92%, Alternative 1: 60%, Alternative 2: 45% (illogical)

### Ambiguity Sources (Real ambiguity ONLY)
- Whether to create a spreadsheet template or just persist to DB (if instruction mentions "spreadsheet" or "template" ambiguously)
- Whether to create time-series records in addition to normalized financials (if instruction mentions "time-series" or "tracking" ambiguously)
- Whether additional validation steps beyond built-in checks are needed (if instruction is unclear about validation)
- Custom assumptions vs policy defaults (if instruction mentions custom requirements)
- Specific output format (if instruction mentions outputs ambiguously)
- DON'T mark as ambiguous just because "we could also do X" - only if instruction genuinely unclear

### General Guidelines
- SOPs are high-level operations, typically 1-3 API calls each
- A task concatenates 2-5 SOPs to achieve a goal
- Tasks should be non-procedural (describe WHAT, not HOW)
- Include parameters in SOP names when they help clarify intent
- If instruction doesn't mention something, don't assume it's needed
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
        
        # Remove any explanation text after the JSON
        # Find the last } and truncate everything after it
        last_brace = content.rfind('}')
        if last_brace != -1:
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

