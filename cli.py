#!/usr/bin/env python3
"""
Main CLI entry point for Tau-Helper: Warrior Tau-Bench Helper Tools.

Provides various commands for task creation, validation, and management.
"""

import os
import click
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

from .llm import get_llm_client
from .evaluator import InstructionEvaluator

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    🛠️  Tau-Helper: Warrior Tau-Bench Helper Tools
    
    A comprehensive toolkit for creating, validating, and managing tasks, tools,
    and SOPs across all Tau-Bench domains.
    
    Available commands:
    - evaluate: Check if instructions are user-facing and non-procedural
    
    More features coming soon!
    """
    pass


@cli.command()
@click.argument("instruction", required=False)
@click.option(
    "-f", "--file",
    type=click.Path(exists=True),
    help="Read instruction from a file"
)
@click.option(
    "-m", "--model",
    help="Override default LLM model",
    default=None
)
@click.option(
    "--batch",
    is_flag=True,
    help="Evaluate multiple instructions (one per line from file)"
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON"
)
def evaluate(instruction, file, model, batch, output_json):
    """
    Evaluate if an instruction is user-facing and non-procedural.
    
    Scores from 0-100:
    - 100: Fully user-facing and non-procedural ✅
    - 0: Highly procedural ❌
    
    Examples:
    
        # Evaluate a single instruction
        $ helper evaluate "You are an analyst reviewing Tesla's financials"
        
        # Evaluate from file
        $ helper evaluate -f instruction.txt
        
        # Evaluate multiple instructions from file
        $ helper evaluate -f instructions.txt --batch
        
        # Use a different model
        $ helper evaluate "Some instruction" --model gpt-4-turbo
    """
    # Get instruction text
    if file:
        with open(file, "r") as f:
            if batch:
                instructions = [line.strip() for line in f if line.strip()]
            else:
                instruction = f.read().strip()
    elif not instruction:
        console.print("[red]Error:[/red] Provide an instruction as argument or use -f to read from file")
        raise click.Abort()
    
    # Initialize LLM and evaluator
    try:
        with console.status("[bold blue]Initializing LLM client..."):
            llm = get_llm_client(model=model)
            evaluator = InstructionEvaluator(llm)
    except Exception as e:
        console.print(f"[red]Error initializing LLM:[/red] {e}")
        raise click.Abort()
    
    # Evaluate
    if batch:
        _evaluate_batch(evaluator, instructions, output_json)
    else:
        _evaluate_single(evaluator, instruction, output_json)


def _evaluate_single(evaluator: InstructionEvaluator, instruction: str, output_json: bool):
    """Evaluate a single instruction."""
    console.print("\n[bold cyan]Evaluating instruction...[/bold cyan]\n")
    
    # Show the instruction
    console.print(Panel(
        instruction,
        title="📋 Instruction to Evaluate",
        border_style="blue",
        box=box.ROUNDED
    ))
    
    try:
        with console.status("[bold yellow]Analyzing with LLM..."):
            evaluation = evaluator.evaluate(instruction)
    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {e}")
        raise click.Abort()
    
    if output_json:
        import json
        console.print_json(json.dumps(evaluation.dict(), indent=2))
        return
    
    # Display results
    _display_evaluation_rich(evaluation)


def _evaluate_batch(evaluator: InstructionEvaluator, instructions: list[str], output_json: bool):
    """Evaluate multiple instructions."""
    console.print(f"\n[bold cyan]Evaluating {len(instructions)} instructions...[/bold cyan]\n")
    
    results = []
    
    with console.status("[bold yellow]Analyzing instructions...") as status:
        for i, instruction in enumerate(instructions, 1):
            status.update(f"[bold yellow]Analyzing instruction {i}/{len(instructions)}...")
            try:
                evaluation = evaluator.evaluate(instruction)
                results.append((instruction, evaluation))
            except Exception as e:
                console.print(f"[red]Error evaluating instruction {i}:[/red] {e}")
                continue
    
    if output_json:
        import json
        output = [
            {
                "instruction": instr,
                "evaluation": eval.dict()
            }
            for instr, eval in results
        ]
        console.print_json(json.dumps(output, indent=2))
        return
    
    # Display batch results
    _display_batch_results_rich(results)


def _display_evaluation_rich(evaluation):
    """Display evaluation results with rich formatting."""
    # Score panel with color coding
    if evaluation.score >= 80:
        score_color = "green"
        emoji = "✅"
    elif evaluation.score >= 50:
        score_color = "yellow"
        emoji = "⚠️"
    else:
        score_color = "red"
        emoji = "❌"
    
    console.print(Panel(
        f"[bold {score_color}]{evaluation.score}/100[/bold {score_color}]",
        title=f"{emoji} Score",
        border_style=score_color,
        box=box.DOUBLE
    ))
    
    # Reasoning
    console.print(f"\n[bold]📝 Reasoning:[/bold]")
    console.print(evaluation.reasoning)
    
    # Procedural elements
    if evaluation.procedural_elements:
        console.print(f"\n[bold red]❌ Procedural Elements Found:[/bold red]")
        for elem in evaluation.procedural_elements:
            console.print(f"  • {elem}")
    
    # User-facing elements
    if evaluation.user_facing_elements:
        console.print(f"\n[bold green]✅ User-Facing Elements:[/bold green]")
        for elem in evaluation.user_facing_elements:
            console.print(f"  • {elem}")
    
    # Suggested replacement
    console.print(f"\n[bold]💡 Suggested Replacement:[/bold]")
    console.print(Panel(
        evaluation.suggested_replacement,
        border_style="green",
        box=box.ROUNDED
    ))


def _display_batch_results_rich(results):
    """Display batch evaluation results in a table."""
    table = Table(title="📊 Batch Evaluation Results", box=box.ROUNDED)
    
    table.add_column("#", style="cyan", width=4)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Instruction Preview", style="dim", width=40)
    table.add_column("Status", justify="center", width=8)
    
    for i, (instruction, evaluation) in enumerate(results, 1):
        # Truncate instruction for preview
        preview = instruction[:60] + "..." if len(instruction) > 60 else instruction
        
        # Color code score
        if evaluation.score >= 80:
            score_str = f"[green]{evaluation.score}[/green]"
            status = "✅"
        elif evaluation.score >= 50:
            score_str = f"[yellow]{evaluation.score}[/yellow]"
            status = "⚠️"
        else:
            score_str = f"[red]{evaluation.score}[/red]"
            status = "❌"
        
        table.add_row(str(i), score_str, preview, status)
    
    console.print(table)
    
    # Summary statistics
    scores = [eval.score for _, eval in results]
    avg_score = sum(scores) / len(scores)
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  • Total instructions: {len(results)}")
    console.print(f"  • Average score: {avg_score:.1f}/100")
    console.print(f"  • High quality (≥80): {sum(1 for s in scores if s >= 80)}")
    console.print(f"  • Medium quality (50-79): {sum(1 for s in scores if 50 <= s < 80)}")
    console.print(f"  • Low quality (<50): {sum(1 for s in scores if s < 50)}")


@cli.command()
@click.argument("domain")
@click.option("--variation", required=True, help="Variation name (e.g., variation_1, variation_2)")
@click.option("--task", help="Task ID to execute")
@click.option("--action", type=int, help="Action index to execute")
@click.option("--action-range", help="Range of actions (e.g., 0-5)")
@click.option("--all-actions", is_flag=True, help="Execute all actions in task")
@click.option("--list-tasks", is_flag=True, help="List all tasks")
@click.option("--list-actions", is_flag=True, help="List actions in task")
@click.option("--inspect-db", is_flag=True, help="Inspect database")
@click.option("--table", help="Specific table to inspect")
@click.option("--reset-db", is_flag=True, help="Reset database")
def execute(domain, variation, task, action, action_range, all_actions, list_tasks, list_actions, inspect_db, table, reset_db):
    """
    Execute domain actions for testing and debugging.
    
    Examples:
    
        # List all tasks
        tau_helper execute sec --list-tasks
        
        # Execute single action
        tau_helper execute sec --task task_001 --action 0
        
        # Execute range of actions
        tau_helper execute sec --task task_001 --action-range 0-5
        
        # Execute all actions in task
        tau_helper execute sec --task task_001 --all-actions
    """
    from .action_executor import ActionExecutor, get_available_variations
    
    # Validate variation exists
    available_variations = get_available_variations(domain)
    if not available_variations:
        console.print(f"[red]Error:[/red] No variations found for domain '{domain}'")
        console.print(f"[yellow]Hint:[/yellow] Run 'python tau_helper/run.py list-domains --domain {domain}' to see available variations")
        raise click.Abort()
    
    if variation not in available_variations:
        console.print(f"[red]Error:[/red] Variation '{variation}' not found in domain '{domain}'")
        console.print(f"[yellow]Available variations:[/yellow] {', '.join(available_variations)}")
        raise click.Abort()
    
    try:
        with console.status(f"[bold blue]Loading {domain}/{variation}..."):
            executor = ActionExecutor(domain, variation)
    except Exception as e:
        console.print(f"[red]Error loading domain:[/red] {e}")
        raise click.Abort()
    
    # Handle different command modes
    if list_tasks:
        executor.list_tasks()
    
    elif reset_db:
        executor.reset_database()
    
    elif inspect_db:
        executor.inspect_database(table)
    
    elif task:
        if list_actions:
            executor.list_task_actions(task)
        
        elif all_actions:
            results = executor.execute_all_task_actions(task)
            
            # Summary
            success_count = sum(1 for r in results if r["success"])
            console.print(f"\n{'='*80}")
            console.print(f"[bold]SUMMARY:[/bold] {success_count}/{len(results)} actions succeeded")
            console.print(f"{'='*80}")
        
        elif action is not None:
            executor.execute_task_action(task, action)
        
        elif action_range:
            try:
                start, end = map(int, action_range.split("-"))
                results = executor.execute_task_actions_range(task, start, end)
                
                # Summary
                success_count = sum(1 for r in results if r["success"])
                console.print(f"\n{'='*80}")
                console.print(f"[bold]SUMMARY:[/bold] {success_count}/{len(results)} actions succeeded")
                console.print(f"{'='*80}")
            except ValueError:
                console.print("[red]Error:[/red] Invalid action range format. Use: 0-5")
                raise click.Abort()
        
        else:
            console.print("[red]Error:[/red] Specify --action, --action-range, --all-actions, or --list-actions")
            raise click.Abort()
    
    else:
        console.print("[yellow]Hint:[/yellow] Use --list-tasks to see available tasks")


@cli.command()
@click.argument("domain")
@click.option("--variation", required=True, help="Variation name (e.g., variation_1, variation_2)")
@click.option("--agent-json", help="Path to agent.json file")
@click.option("--task", help="Task ID to analyze")
@click.option("--list-tasks", is_flag=True, help="List all tasks")
@click.option("--stats", is_flag=True, help="Show statistics")
@click.option("--no-gt", is_flag=True, help="Don't show ground truth")
@click.option("--no-agents", is_flag=True, help="Don't show agent runs")
@click.option("--user-sim", is_flag=True, help="Show initial user simulation messages")
@click.option("--limit", type=int, help="Limit number of results (for --user-sim or --list-tasks)")
def agent_logs(domain, variation, agent_json, task, list_tasks, stats, no_gt, no_agents, user_sim, limit):
    """
    Analyze agent evaluation logs.
    
    Examples:
    
        # List all tasks
        tau_helper agent-logs sec --variation variation_2 --list-tasks
        
        # Analyze specific task
        tau_helper agent-logs sec --variation variation_2 --task task_072
        
        # Show statistics
        tau_helper agent-logs sec --variation variation_2 --stats
        
        # Show user simulation messages
        tau_helper agent-logs sec --variation variation_2 --user-sim
        
        # Show user sim for specific task
        tau_helper agent-logs sec --variation variation_2 --task task_001 --user-sim
        
        # Use custom agent.json location
        tau_helper agent-logs sec --variation variation_2 --agent-json /path/to/agent.json --task task_001
    """
    from .agent_log_reader import AgentLogReader, get_available_variations
    
    # Validate variation exists (unless custom agent-json path is provided)
    if not agent_json:
        available_variations = get_available_variations(domain)
        if not available_variations:
            console.print(f"[red]Error:[/red] No variations found for domain '{domain}'")
            console.print(f"[yellow]Hint:[/yellow] Run 'python tau_helper/run.py list-domains --domain {domain}' to see available variations")
            raise click.Abort()
        
        if variation not in available_variations:
            console.print(f"[red]Error:[/red] Variation '{variation}' not found in domain '{domain}'")
            console.print(f"[yellow]Available variations:[/yellow] {', '.join(available_variations)}")
            console.print(f"[dim]Or use --agent-json to specify a custom path[/dim]")
            raise click.Abort()
    
    try:
        with console.status(f"[bold blue]Loading agent logs for {domain}..."):
            reader = AgentLogReader(domain, variation, agent_json)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()
    
    if user_sim:
        # Show user simulation messages
        messages = reader.get_user_sim_messages(task_id=task, limit=limit)
        
        if not messages:
            console.print("[yellow]No user simulation messages found[/yellow]")
        else:
            # Group by task_id
            from collections import defaultdict
            by_task = defaultdict(list)
            for msg in messages:
                by_task[msg['task_id']].append(msg)
            
            for task_id, task_messages in by_task.items():
                console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
                console.print(f"[bold]Task:[/bold] {task_id} [dim]({len(task_messages)} agent attempts)[/dim]")
                console.print(f"[bold cyan]{'='*80}[/bold cyan]")
                
                for msg in task_messages:
                    status_color = "green" if msg['agent_succeeded'] else "red"
                    status_text = "✓ PASS" if msg['agent_succeeded'] else "✗ FAIL"
                    
                    console.print(Panel(
                        msg['user_sim_message'],
                        title=f"👤 Agent {msg['attempt_number']} [{status_color}]{status_text}[/{status_color}]",
                        border_style=status_color,
                        box=box.ROUNDED
                    ))
            
            total_attempts = len(messages)
            total_tasks = len(by_task)
            console.print(f"\n[dim]Showing {total_attempts} user simulation messages across {total_tasks} task(s)[/dim]")
    
    elif list_tasks:
        summaries = reader.list_tasks(limit=limit or 50)
        
        table = Table(title=f"Tasks in {domain}/{variation}", box=box.ROUNDED)
        table.add_column("Task ID", style="cyan")
        table.add_column("Pass Rate", justify="center")
        table.add_column("Passed", justify="center")
        table.add_column("Total", justify="center")
        
        for summary in summaries:
            pass_rate = summary['pass_rate']
            if pass_rate == 1.0:
                rate_style = "green"
            elif pass_rate >= 0.5:
                rate_style = "yellow"
            else:
                rate_style = "red"
            
            table.add_row(
                summary['task_id'],
                f"[{rate_style}]{pass_rate:.0%}[/{rate_style}]",
                str(summary['successful']),
                str(summary['total'])
            )
        
        console.print(table)
        
        if len(reader.results) > (limit or 50):
            console.print(f"\n[dim]... and {len(reader.results) - (limit or 50)} more tasks[/dim]")
    
    elif stats:
        statistics = reader.get_statistics()
        
        console.print(f"\n[bold cyan]Statistics for {domain}/{variation}[/bold cyan]")
        console.print("="*80)
        console.print(f"Total tasks: {statistics['total_tasks']}")
        console.print(f"Total attempts: {statistics['total_attempts']}")
        console.print(f"Total successes: {statistics['total_successes']}")
        console.print(f"Overall success rate: {statistics['overall_success_rate']:.1%}")
        console.print(f"Average task pass rate: {statistics['average_task_pass_rate']:.1%}")
        console.print(f"Tasks with 100% pass rate: {statistics['tasks_with_100_percent']}")
        console.print(f"Tasks with 0% pass rate: {statistics['tasks_with_0_percent']}")
    
    elif task:
        reader.print_task_analysis(
            task,
            include_ground_truth=not no_gt,
            include_agents=not no_agents,
            include_summary=True
        )
    
    else:
        console.print("[yellow]Hint:[/yellow] Use --list-tasks to see available tasks")


@cli.command()
@click.argument("domain")
@click.option("--variation", required=True, help="Variation name (e.g., variation_1, variation_2)")
@click.option("--instruction", help="Instruction text to analyze")
@click.option("--task", help="Task ID to analyze")
@click.option("--compare", is_flag=True, help="Compare predicted vs actual actions")
@click.option("--model", help="Override LLM model")
def map_sop(domain, variation, instruction, task, compare, model):
    """
    Map instructions to SOP chains and detect ambiguities.
    
    Examples:
    
        # Map instruction to SOPs
        tau_helper map-sop sec --instruction "You need to calculate WACC for Tesla"
        
        # Analyze specific task
        tau_helper map-sop sec --task task_001
        
        # Compare predicted vs actual
        tau_helper map-sop sec --task task_001 --compare
        
        # Use different model
        tau_helper map-sop sec --task task_001 --model gpt-4-turbo
    """
    from .sop_mapper import SOPMapper
    from .llm import get_reasoning_llm, get_llm_client
    from .action_executor import get_available_variations
    
    # Validate variation
    available_variations = get_available_variations(domain)
    if not available_variations:
        console.print(f"[red]Error:[/red] No variations found for domain '{domain}'")
        raise click.Abort()
    
    if variation not in available_variations:
        console.print(f"[red]Error:[/red] Variation '{variation}' not found")
        console.print(f"[yellow]Available:[/yellow] {', '.join(available_variations)}")
        raise click.Abort()
    
    # Initialize LLM and mapper
    try:
        with console.status("[bold blue]Initializing SOP mapper..."):
            # Use reasoning model by default, allow override with --model flag
            if model:
                llm = get_llm_client(model=model)
            else:
                llm = get_reasoning_llm()
            mapper = SOPMapper(llm, domain, variation)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()
    
    # Handle different modes
    if compare and task:
        # Compare mode
        try:
            with console.status(f"[bold yellow]Analyzing {domain}/{variation} - task {task}..."):
                result = mapper.compare_with_actual(task, domain=domain, variation=variation)
            
            console.print(f"\n[bold cyan]Task Analysis: {task} ({domain}/{variation})[/bold cyan]")
            console.print("="*80)
            
            console.print(f"\n[bold]Instruction:[/bold]")
            console.print(f"{result['instruction'][:200]}...")
            
            console.print(f"\n[bold]Predicted SOP Chain:[/bold]")
            for i, sop in enumerate(result['predicted_sops'], 1):
                console.print(f"  {i}. {sop}")
            
            console.print(f"\n[bold]Actual Actions:[/bold]")
            for i, action in enumerate(result['actual_actions'], 1):
                console.print(f"  {i}. {action}")
            
            match = result['match']
            status_color = "green" if match['status'] == 'good' else "yellow" if match['status'] == 'partial' else "red"
            console.print(f"\n[bold]Match Analysis:[/bold]")
            console.print(f"  Match Rate: [{status_color}]{match['match_rate']:.0%}[/{status_color}]")
            console.print(f"  Status: [{status_color}]{match['status'].upper()}[/{status_color}]")
            
            if result['is_ambiguous']:
                console.print(f"\n[yellow]⚠️  Instruction is ambiguous[/yellow]")
                for alt in result['alternative_chains']:
                    console.print(f"\n  Alternative: {' → '.join(alt)}")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise click.Abort()
    
    elif task:
        # Task analysis mode
        try:
            with console.status(f"[bold yellow]Analyzing {domain}/{variation} - task {task}..."):
                instruction_text, mapping = mapper.map_task(task, domain=domain, variation=variation)
            
            console.print(f"\n[dim]Domain: {domain}, Variation: {variation}[/dim]")
            _display_sop_mapping(instruction_text, mapping, console)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise click.Abort()
    
    elif instruction:
        # Instruction analysis mode
        try:
            with console.status(f"[bold yellow]Analyzing instruction for {domain}/{variation}..."):
                mapping = mapper.map_instruction(instruction)
            
            console.print(f"\n[dim]Domain: {domain}, Variation: {variation}[/dim]")
            _display_sop_mapping(instruction, mapping, console)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise click.Abort()
    
    else:
        console.print("[red]Error:[/red] Specify --instruction or --task")
        raise click.Abort()


def _display_sop_mapping(instruction: str, mapping, console):
    """Display SOP mapping results."""
    console.print("\n[bold cyan]Instruction Analysis[/bold cyan]")
    console.print("="*80)
    
    console.print(f"\n[bold]Instruction:[/bold]")
    console.print(f"{instruction}")
    
    # Ambiguity status
    if mapping.is_ambiguous:
        console.print(f"\n[yellow]⚠️  AMBIGUOUS - Multiple valid interpretations detected[/yellow]")
    else:
        console.print(f"\n[green]✓ CLEAR - Single SOP chain identified[/green]")
    
    # Primary chain
    console.print(f"\n[bold]Primary SOP Chain:[/bold]")
    console.print(f"Confidence: {mapping.primary_chain.confidence:.0%}")
    console.print(f"\nSOPs:")
    for i, sop in enumerate(mapping.primary_chain.sops, 1):
        console.print(f"  {i}. {sop}")
    
    console.print(f"\n[bold]Reasoning:[/bold]")
    console.print(f"{mapping.primary_chain.reasoning}")
    
    # Alternative chains
    if mapping.alternative_chains:
        console.print(f"\n[bold yellow]Alternative Interpretations:[/bold yellow]")
        for i, alt_chain in enumerate(mapping.alternative_chains, 1):
            console.print(f"\n  [bold]Alternative {i}:[/bold] (Confidence: {alt_chain.confidence:.0%})")
            console.print(f"  SOPs: {' → '.join(alt_chain.sops)}")
            console.print(f"  Reasoning: {alt_chain.reasoning}")
    
    # Ambiguity explanation
    if mapping.ambiguity_explanation:
        console.print(f"\n[bold]Why Ambiguous:[/bold]")
        console.print(f"{mapping.ambiguity_explanation}")
    
    # Missing information
    if mapping.missing_information:
        console.print(f"\n[bold]Missing Information (to disambiguate):[/bold]")
        for info in mapping.missing_information:
            console.print(f"  • {info}")
    
    # Suggested fix
    if mapping.suggested_fix:
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold green]💡 Suggested Instruction Fix[/bold green]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"\n[dim]Replace the original instruction with:[/dim]\n")
        console.print(f"[italic]{mapping.suggested_fix}[/italic]")
        console.print(f"\n[dim]This fix:[/dim]")
        console.print(f"[dim]  ✓ Removes ambiguity by making choices explicit[/dim]")
        console.print(f"[dim]  ✓ Keeps the same story, role, and context[/dim]")
        console.print(f"[dim]  ✓ Remains non-procedural and user-facing[/dim]")


@cli.command()
@click.option("--domain", help="Show variations for specific domain")
def list_domains(domain):
    """
    List available domains and their variations.
    
    Examples:
    
        # List all domains
        tau_helper list-domains
        
        # Show variations for specific domain
        tau_helper list-domains --domain sec
    """
    from .action_executor import get_available_domains, get_available_variations
    
    if domain:
        # Show variations for specific domain
        variations = get_available_variations(domain)
        
        if not variations:
            console.print(f"[red]No variations found for domain '{domain}'[/red]")
            console.print(f"[dim]Check that domain exists and has variations/ directory[/dim]")
            return
        
        console.print(f"\n[bold cyan]Variations in {domain}:[/bold cyan]")
        for var in variations:
            console.print(f"  • {var}")
        console.print(f"\nTotal: {len(variations)} variation(s)")
    else:
        # List all domains with their variations
        domains = get_available_domains()
        
        if not domains:
            console.print("[red]No domains found[/red]")
            return
        
        table = Table(title="Available Domains", box=box.ROUNDED)
        table.add_column("Domain", style="cyan")
        table.add_column("Variations", justify="right")
        table.add_column("Variation Names", style="dim")
        
        for dom in domains:
            variations = get_available_variations(dom)
            var_count = len(variations)
            var_names = ", ".join(variations) if variations else "none"
            table.add_row(dom, str(var_count), var_names)
        
        console.print(table)
        console.print(f"\n[dim]Total domains: {len(domains)}[/dim]")


@cli.command()
def info():
    """Display information about the tau-helper library."""
    info_text = """
    [bold cyan]🛠️  Tau-Helper: Warrior Tau-Bench Helper Library[/bold cyan]
    
    [bold]Version:[/bold] 0.1.0
    
    [bold]Current Features:[/bold]
    • Instruction evaluation (user-facing vs procedural)
    • SOP chain mapping with ambiguity detection
    • Action execution for testing
    • Agent log analysis
    • Domain/variation discovery
    
    [bold]Coming Soon:[/bold]
    • Task generation from SOPs
    • Tool validation
    • Full SOP compliance checking
    
    [bold]Configuration:[/bold]
    Models are configured in .env file:
    • DEFAULT_MODEL (used for evaluation & SOP mapping)
    • DEFAULT_MODEL1 (alternative model)
    
    [bold]Usage:[/bold]
    Run 'python tau_helper/run.py --help' for available commands
    """
    console.print(Panel(info_text, box=box.ROUNDED, border_style="cyan"))


@cli.command(hidden=True)  # Deprecated: use 'scaffold' instead
@click.argument("domain")
@click.option("--variation", required=True, help="Variation name (e.g., variation_2)")
@click.option("--instruction", help="Task instruction (if not using --task)")
@click.option("--task", help="Task ID to get instruction from (e.g., task_001)")
@click.option("--task-id", default="task_new", help="Task ID for generated task (default: task_new)")
@click.option("--model", help="Override LLM model for scaffolding")
@click.option("--max-actions", default=100, type=int, help="Maximum actions to prevent infinite loops (default: 100)")
@click.option("--verbose", is_flag=True, help="Show detailed progress logs including execution results")
def scaffold_legacy(domain, variation, instruction, task, task_id, model, max_actions, verbose):
    """
    [DEPRECATED] Legacy iterative scaffolder. Use 'scaffold' command instead.
    
    This command is hidden and kept for backwards compatibility only.
    """
    from .llm import get_llm_client, is_multi_agent_enabled, get_reasoning_llm_r2, get_judge_llm
    from .iterative_scaffolder import IterativeScaffolder

    console.print(f"[bold]Domain:[/bold] {domain}, [bold]Variation:[/bold] {variation}")
    console.print(f"[bold cyan]🚀 Iterative Scaffolding Mode[/bold cyan] - Actions generated step-by-step with real execution\n")

    # Get instruction
    if task and not instruction:
        # Load instruction from task
        try:
            from .action_executor import ActionExecutor
            executor = ActionExecutor(domain, variation)
            tasks = executor.get_available_tasks()
            if task not in tasks:
                console.print(f"[red]Error:[/red] Task '{task}' not found")
                return
            instruction = tasks[task].instruction
            console.print(f"[dim]Loaded instruction from {task}[/dim]\n")
        except Exception as e:
            console.print(f"[red]Error loading task:[/red] {e}")
            return

    if not instruction:
        console.print("[red]Error:[/red] Must provide --instruction or --task")
        return

    # Initialize LLM(s)
    try:
        if model:
            llm = get_llm_client(model=model)
        else:
            # Use temperature=0 for more deterministic scaffolding
            llm = get_llm_client(temperature=0.0, seed=42)

        # Check if multi-agent is enabled
        llm_r2 = None
        llm_judge = None
        if is_multi_agent_enabled():
            console.print("[bold cyan]🤖 Multi-agent mode enabled[/bold cyan]")
            console.print("[dim]Model R and R2 will generate actions independently, Judge will resolve conflicts[/dim]\n")
            llm_r2 = get_reasoning_llm_r2()
            llm_judge = get_judge_llm()
        elif os.getenv('DEFAULT_MODEL_R2'):
            # R2 validation mode (R2 exists, optionally with judge for roundtable)
            console.print("[bold green]🔍 R2 Validation mode enabled[/bold green]")
            llm_r2 = get_reasoning_llm_r2()

            # Load judge if available for roundtable discussion
            if os.getenv('DEFAULT_MODEL_JUDGE'):
                llm_judge = get_judge_llm()
                console.print("[dim]Model R generates actions, R2 validates, Judge mediates disagreements (roundtable mode)[/dim]\n")
            else:
                console.print("[dim]Model R generates actions, R2 validates for hallucinations and rule violations[/dim]\n")
        else:
            console.print("[dim]Single model mode (configure R2 in .env for validation, R2+Judge for multi-agent)[/dim]\n")

    except Exception as e:
        console.print(f"[red]Error initializing LLM:[/red] {e}")
        return

    # Initialize iterative scaffolder
    try:
        scaffolder = IterativeScaffolder(
            domain=domain,
            variation=variation,
            llm=llm,
            llm_r2=llm_r2,
            llm_judge=llm_judge
        )
    except Exception as e:
        console.print(f"[red]Error initializing scaffolder:[/red] {e}")
        return

    # Generate scaffold iteratively
    console.print(f"[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold green]Iterative Task Scaffolding[/bold green]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    console.print(f"[bold]Instruction:[/bold]")
    console.print(f"{instruction}\n")

    with console.status("[bold blue]Generating task actions iteratively..."):
        actions, error, progress, model_info = scaffolder.scaffold(
            instruction=instruction,
            task_id=task_id,
            max_actions=max_actions,
            verbose=verbose
        )

    # Show progress only if verbose or error occurred
    if verbose or error:
        console.print("\n[bold]Execution Log:[/bold]")
        for msg in progress:
            console.print(msg)
    else:
        # Show only summary
        # Count successes
        success_count = sum(1 for msg in progress if "✓ Success" in msg)
        retry_count = sum(1 for msg in progress if "🔄 Retry" in msg)
        consensus_count = sum(1 for msg in progress if "✓ Consensus" in msg)

        console.print(f"\n[dim]Generated {success_count} actions " +
                     (f"({retry_count} with retries)" if retry_count > 0 else "") +
                     (f" | {consensus_count} consensus" if consensus_count > 0 else "") +
                     "[/dim]")

    # Show result or error
    if error:
        console.print(f"\n[bold red]Scaffolding Failed[/bold red]")
        console.print(f"[red]{error}[/red]")
        if not verbose:
            console.print(f"[dim]Run with --verbose to see detailed error logs[/dim]")
        return

    # Display generated task
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold green]Generated Task (Ready to Use)[/bold green]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    # Format as Python code
    python_code = _format_iterative_task_as_python(task_id, instruction, actions)

    syntax = Syntax(python_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print(f"\n[bold green]✅ Iterative scaffolding complete![/bold green]")
    console.print(f"[bold green]Generated {len(actions)} actions with REAL values (no placeholders!)[/bold green]")
    console.print(f"[dim]Copy the above code to add this task to tasks.py[/dim]")

    # Show model selection summary
    if model_info:
        console.print(f"\n[dim]Model Strategy: {model_info}[/dim]")


def _format_iterative_task_as_python(task_id: str, instruction: str, actions: List[Dict[str, Any]]) -> str:
    """Format iteratively generated actions as Python code matching tasks.py format."""
    lines = []
    lines.append("Task(")
    lines.append('    annotator="human",')
    lines.append(f'    user_id="{task_id}",')
    lines.append(f'    instruction="{instruction}",')
    lines.append('    actions=[')

    for action in actions:
        # Add SOP comment if available
        if action.get('sop_step'):
            lines.append(f'        # {action["sop_step"]}')
        elif action.get('reasoning'):
            lines.append(f'        # {action["reasoning"][:80]}')

        # Format kwargs
        import json
        kwargs_str = json.dumps(action['kwargs'])

        lines.append(f'        Action(')
        lines.append(f'            name="{action["name"]}",')
        lines.append(f'            kwargs={kwargs_str},')
        lines.append(f'        ),')

    lines.append('    ],')
    lines.append('    outputs=[],')
    lines.append('),')

    return "\n".join(lines)


@cli.command()
@click.argument("domain")
@click.option("--variation", required=True, help="Variation name (e.g., variation_2)")
@click.option("--instruction", help="Task instruction (if not using --task)")
@click.option("--task", help="Task ID to get instruction from (e.g., task_001)")
@click.option("--task-id", default="task_new", help="Task ID for generated task (default: task_new)")
@click.option("--model", help="Override LLM model for scaffolding")
@click.option("--max-rounds", default=5, type=int, help="Maximum R/R2 review rounds (default: 5)")
@click.option("--verbose", is_flag=True, help="Show detailed progress logs")
def scaffold(domain, variation, instruction, task, task_id, model, max_rounds, verbose):
    """
    Generate a complete task scaffold using code-based execution with R/R2 roundtable.
    
    This approach has agents write Python code that calls tools. The code is reviewed
    in a roundtable (R generates, R2 reviews, Judge mediates), then executed with 
    real database feedback. Failed executions trigger automatic diagnosis and fixes.
    
    Features:
    - Code-based generation - forces explicit data flow
    - R2 reviews the full plan structure at once
    - Variables prevent ID hallucination (must be assigned from results)
    - Up to 5 rounds of refinement before execution
    - Live editing on execution failures (R2 diagnoses, R fixes)
    
    Examples:
    
        # Scaffold from instruction
        python tau_helper/run.py scaffold salesforce_performance_management \\
          --variation variation_2 \\
          --instruction "Transfer BlueCurve Analytics from Ava Lopez to Chris Sullivan"
        
        # Scaffold from existing task instruction  
        python tau_helper/run.py scaffold slack_jira_agile_automation \\
          --variation variation_2 --task task_010 --verbose
    """
    from .llm import get_llm_client, get_reasoning_llm_r2, get_judge_llm
    from .pseudo_scaffolder import PseudoScaffolder
    
    console.print(f"[bold]Domain:[/bold] {domain}, [bold]Variation:[/bold] {variation}")
    console.print(f"[bold magenta]🧪 PSEUDO SCAFFOLDER[/bold magenta] - Code-based task generation with R/R2 roundtable\n")
    
    # Get instruction
    if task and not instruction:
        # Load instruction from task
        try:
            from .action_executor import ActionExecutor
            executor = ActionExecutor(domain, variation)
            tasks = executor.get_available_tasks()
            if task not in tasks:
                console.print(f"[red]Error:[/red] Task '{task}' not found")
                return
            instruction = tasks[task].instruction
            console.print(f"[dim]Loaded instruction from {task}[/dim]\n")
        except Exception as e:
            console.print(f"[red]Error loading task:[/red] {e}")
            return
    
    if not instruction:
        console.print("[red]Error:[/red] Must provide --instruction or --task")
        return
    
    # Initialize LLMs
    try:
        if model:
            llm = get_llm_client(model=model)
        else:
            llm = get_llm_client(temperature=0.0, seed=42)
        
        # R2 is required for pseudo scaffolder
        llm_r2 = get_reasoning_llm_r2()
        if not llm_r2:
            console.print("[yellow]Warning:[/yellow] R2 model not configured. Using same model for R2.")
            llm_r2 = llm
        
        # Judge is optional but recommended
        llm_judge = get_judge_llm()
        if llm_judge:
            console.print("[dim]Judge available for mediation if R/R2 disagree[/dim]\n")
        else:
            console.print("[dim]No Judge configured - R2's final decision will be used[/dim]\n")
            
    except Exception as e:
        console.print(f"[red]Error initializing LLM:[/red] {e}")
        return
    
    # Initialize pseudo scaffolder
    try:
        scaffolder = PseudoScaffolder(
            domain=domain,
            variation=variation,
            llm=llm,
            llm_r2=llm_r2,
            llm_judge=llm_judge,
            max_review_rounds=max_rounds
        )
    except Exception as e:
        console.print(f"[red]Error initializing pseudo scaffolder:[/red] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate scaffold
    actions, error, progress = scaffolder.scaffold(
        instruction=instruction,
        task_id=task_id,
        verbose=verbose
    )
    
    # Show progress
    if verbose:
        for msg in progress:
            console.print(msg)
    
    # Show result or error
    if error:
        console.print(f"\n[bold red]Pseudo Scaffolding Failed[/bold red]")
        console.print(f"[red]{error}[/red]")
        return
    
    if not actions:
        console.print(f"\n[bold yellow]No actions generated[/bold yellow]")
        return
    
    console.print(f"\n[bold green]✅ Pseudo scaffolding complete![/bold green]")
    console.print(f"[bold green]Generated {len(actions)} actions from Python code execution[/bold green]")


if __name__ == "__main__":
    cli()

