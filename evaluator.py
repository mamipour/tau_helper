"""
Instruction evaluation module for assessing task instructions.

Evaluates whether instructions are user-facing and non-procedural vs procedural.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate


class InstructionEvaluation(BaseModel):
    """Structured output for instruction evaluation."""
    
    score: int = Field(
        description="Score from 0-100. 100=fully user-facing and non-procedural, 0=highly procedural"
    )
    reasoning: str = Field(
        description="Detailed explanation of the score with specific examples from the instruction"
    )
    procedural_elements: list[str] = Field(
        description="List of procedural elements found (if any)"
    )
    user_facing_elements: list[str] = Field(
        description="List of user-facing elements found (if any)"
    )
    suggested_replacement: str = Field(
        description="Improved version of the instruction that is more user-facing and non-procedural"
    )


class InstructionEvaluator:
    """
    Evaluates task instructions for user-facing, non-procedural quality.
    
    User-facing and non-procedural instructions:
    ✅ "You are Ashley Wilson from HR. You want to onboard a new contractor."
    ✅ "You need to analyze Tesla's financial performance for Q4 2023."
    
    Procedural instructions:
    ❌ "Call get_user() then call update_user() with the new data."
    ❌ "First retrieve the balance sheet, then calculate the ratio."
    """
    
    EVALUATION_PROMPT = """You are an expert evaluator for Warrior Tau-Bench task instructions.

Your job is to evaluate whether a task instruction is USER-FACING and NON-PROCEDURAL or PROCEDURAL.

## Definitions

**USER-FACING and NON-PROCEDURAL (HIGH SCORE 80-100):**
- Written in second person ("You are...", "You want to...")
- Describes WHAT the user wants to achieve, not HOW
- Natural, conversational language
- Focuses on goals and context
- No mention of API calls, function names, or implementation steps
- Examples:
  * "You are Ashley Wilson from HR. You want to onboard a new contractor named John Doe."
  * "You need to analyze Tesla's Q4 2023 financial performance and calculate their debt-to-equity ratio."
  * "You're preparing a WACC analysis for Apple Inc. and need to gather their latest financial data."

**PROCEDURAL (LOW SCORE 0-40):**
- Contains explicit step-by-step instructions
- Mentions function names, API calls, or technical operations
- Uses imperative commands ("Call...", "First do...", "Then...")
- Focuses on HOW rather than WHAT
- Examples:
  * "Call get_user() then update_user() with the new data."
  * "First retrieve the balance sheet using get_balance_sheet(), then calculate ratios."
  * "Execute search_filings() followed by extract_data()."

**MIXED (MEDIUM SCORE 40-80):**
- Partially user-facing but includes some procedural hints
- May use technical terminology unnecessarily
- Could be improved by focusing more on goals

## Task

Evaluate the following instruction and provide:
1. A score from 0-100
2. Detailed reasoning with specific examples
3. List of procedural elements (if any)
4. List of user-facing elements (if any)
5. A suggested replacement that is fully user-facing and non-procedural

## Instruction to Evaluate

{instruction}

## Output Format

{format_instructions}
"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the evaluator with an LLM client.
        
        Args:
            llm: LangChain ChatOpenAI instance
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=InstructionEvaluation)
        
        self.prompt = ChatPromptTemplate.from_template(
            self.EVALUATION_PROMPT
        )
    
    def evaluate(self, instruction: str) -> InstructionEvaluation:
        """
        Evaluate an instruction for user-facing, non-procedural quality.
        
        Args:
            instruction: The instruction text to evaluate
        
        Returns:
            InstructionEvaluation object with score, reasoning, and suggestions
        
        Example:
            >>> evaluator = InstructionEvaluator(llm)
            >>> result = evaluator.evaluate("Call get_user() then update it")
            >>> print(result.score)  # Low score
            >>> print(result.suggested_replacement)
        """
        # Format the prompt
        formatted_prompt = self.prompt.format_messages(
            instruction=instruction,
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Get LLM response
        response = self.llm.invoke(formatted_prompt)
        
        # Parse the structured output
        evaluation = self.parser.parse(response.content)
        
        return evaluation
    
    def evaluate_batch(self, instructions: list[str]) -> list[InstructionEvaluation]:
        """
        Evaluate multiple instructions in batch.
        
        Args:
            instructions: List of instruction texts
        
        Returns:
            List of InstructionEvaluation objects
        """
        return [self.evaluate(instruction) for instruction in instructions]
    
    def get_summary_report(self, evaluation: InstructionEvaluation) -> str:
        """
        Generate a formatted summary report for an evaluation.
        
        Args:
            evaluation: InstructionEvaluation object
        
        Returns:
            Formatted string report
        """
        report_lines = [
            "=" * 80,
            "INSTRUCTION EVALUATION REPORT",
            "=" * 80,
            f"\n📊 SCORE: {evaluation.score}/100",
            "",
            "📝 REASONING:",
            evaluation.reasoning,
            "",
        ]
        
        if evaluation.procedural_elements:
            report_lines.extend([
                "❌ PROCEDURAL ELEMENTS FOUND:",
                *[f"  - {elem}" for elem in evaluation.procedural_elements],
                "",
            ])
        
        if evaluation.user_facing_elements:
            report_lines.extend([
                "✅ USER-FACING ELEMENTS:",
                *[f"  - {elem}" for elem in evaluation.user_facing_elements],
                "",
            ])
        
        report_lines.extend([
            "💡 SUGGESTED REPLACEMENT:",
            f"\"{evaluation.suggested_replacement}\"",
            "",
            "=" * 80,
        ])
        
        return "\n".join(report_lines)

