"""
Domain-agnostic agent log reader for analyzing evaluation results.

This module reads and analyzes agent.json files from any domain.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class AgentLogReader:
    """Read and analyze agent evaluation logs."""
    
    def __init__(
        self, 
        domain: str, 
        variation: str = "variation_2",
        agent_json_path: Optional[str] = None
    ):
        """
        Initialize agent log reader.
        
        Args:
            domain: Domain name (e.g., 'sec', 'airline')
            variation: Variation name (default: 'variation_2')
            agent_json_path: Custom path to agent.json (optional)
        """
        self.domain = domain
        self.variation = variation
        self.agent_json_path = agent_json_path
        
        # Load logs
        self.logs = self._load_logs()
        self.results = self.logs.get('results', []) if isinstance(self.logs, dict) else self.logs
    
    def _load_logs(self) -> Dict[str, Any]:
        """Load agent evaluation logs from JSON file."""
        # Try multiple possible locations
        possible_paths = []
        
        if self.agent_json_path:
            possible_paths.append(Path(self.agent_json_path))
        
        # Try domain-specific paths
        possible_paths.extend([
            Path.cwd() / "domains" / self.domain / "agent.json",
            Path.cwd() / "domains" / self.domain / "variations" / self.variation / "agent.json",
            Path.cwd() / "agent.json",
        ])
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        
        raise FileNotFoundError(
            f"agent.json not found for domain '{self.domain}'. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    def list_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all tasks with their pass rates.
        
        Args:
            limit: Maximum number of tasks to show (optional)
            
        Returns:
            List of task summaries
        """
        summaries = []
        tasks_to_show = self.results[:limit] if limit else self.results
        
        for task in tasks_to_show:
            task_id = task.get('task_id', 'unknown')
            num_successful = task.get('number_of_successful_agent_attempts', 0)
            total = len(task.get('all_attempts', []))
            
            summaries.append({
                'task_id': task_id,
                'successful': num_successful,
                'total': total,
                'pass_rate': num_successful / total if total > 0 else 0
            })
        
        return summaries
    
    def get_task_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific task."""
        for task in self.results:
            if task.get('task_id') == task_id:
                return task
        return None
    
    def get_ground_truth(self, task_id: str) -> Optional[str]:
        """
        Load ground truth from tasks.py for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Ground truth task definition as string, or None if not found
        """
        try:
            # Find tasks.py
            tasks_path = (
                Path.cwd() / "domains" / self.domain / 
                "variations" / self.variation / "tasks.py"
            )
            
            if not tasks_path.exists():
                return None
            
            with open(tasks_path, 'r') as f:
                content = f.read()
                
                # Find the task
                task_start = content.find(f'user_id="{task_id}"')
                if task_start == -1:
                    return None
                
                # Find the Task( opening
                task_open = content.rfind('Task(', 0, task_start)
                
                # Find the matching closing - look for next Task( or end of list
                next_task = content.find('\n    Task(', task_start)
                if next_task == -1:
                    next_task = content.find('\n]', task_start)
                
                task_content = content[task_open:next_task]
                return task_content
                
        except Exception as e:
            return f"Error loading GT: {e}"
    
    def format_agent_run(
        self, attempt: Dict[str, Any], include_judge: bool = True
    ) -> str:
        """
        Format an agent attempt for display.
        
        Args:
            attempt: Agent attempt data
            include_judge: Whether to include judge comments
            
        Returns:
            Formatted string representation
        """
        lines = []
        agent_num = attempt.get('attempt_number', 0)
        
        lines.append(f"{'='*80}")
        lines.append(f"Agent {agent_num}:")
        lines.append("="*80)
        
        # Judge comments
        if include_judge:
            error_analysis = attempt.get('error_analysis')
            if error_analysis:
                lines.append("\nJudge said:")
                lines.append("-" * 80)
                category = error_analysis.get('error_category', 'N/A')
                lines.append(f"Category: {category}")
                if error_analysis.get('error_subcategory'):
                    lines.append(f"Subcategory: {error_analysis['error_subcategory']}")
                if error_analysis.get('description'):
                    lines.append(f"\n{error_analysis['description']}")
            else:
                succeeded = attempt.get('agent_succeeded', False)
                lines.append("\nJudge said:")
                lines.append("-" * 80)
                lines.append("✓ Agent succeeded - no errors" if succeeded else "No error analysis available")
        
        # Agent actions
        lines.append(f"\nAgent run:")
        lines.append("-" * 80)
        
        agent_trace = attempt.get('agent_trace', {})
        full_trace = agent_trace.get('full_trace', {})
        
        for turn in full_trace.get('turns', []):
            for tool_call in turn.get('tool_calls', []):
                name = tool_call.get('name', 'unknown')
                args = tool_call.get('arguments', {})
                
                # Format action
                lines.append(f"\n{name}(")
                for key, val in args.items():
                    if isinstance(val, (dict, list)):
                        val_str = json.dumps(val, indent=2)
                    else:
                        val_str = str(val)
                    lines.append(f"    {key}={val_str},")
                lines.append(")")
        
        return "\n".join(lines)
    
    def print_task_analysis(
        self, 
        task_id: str, 
        include_ground_truth: bool = True,
        include_agents: bool = True,
        include_summary: bool = True
    ):
        """
        Print comprehensive analysis of a task.
        
        Args:
            task_id: Task identifier
            include_ground_truth: Whether to show ground truth
            include_agents: Whether to show agent runs
            include_summary: Whether to show match rate summary
        """
        task_data = self.get_task_data(task_id)
        
        if not task_data:
            print(f"Error: Task {task_id} not found in logs")
            return
        
        # Ground truth
        if include_ground_truth:
            print("\n" + "="*80)
            print("GT (Ground Truth)")
            print("="*80)
            gt = self.get_ground_truth(task_id)
            if gt:
                print(gt)
            else:
                print(f"Ground truth not found for {task_id}")
        
        # Agent runs
        if include_agents:
            for attempt in task_data.get('all_attempts', []):
                print("\n" + self.format_agent_run(attempt))
        
        # Summary
        if include_summary:
            print("\n" + "="*80)
            print("Match Rate")
            print("="*80)
            
            num_successful = task_data.get('number_of_successful_agent_attempts', 0)
            total_attempts = len(task_data.get('all_attempts', []))
            match_rate = num_successful / total_attempts if total_attempts > 0 else 0
            
            print(f"\nSuccessful Agents: {num_successful}/{total_attempts}")
            print(f"Match Rate: {match_rate:.1%}")
            
            # Individual results
            print("\nAgent Results:")
            for attempt in task_data.get('all_attempts', []):
                agent_num = attempt.get('attempt_number', 0)
                succeeded = attempt.get('agent_succeeded', False)
                status = '✓ PASS' if succeeded else '✗ FAIL'
                print(f"  Agent {agent_num}: {status}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from the logs."""
        total_tasks = len(self.results)
        total_attempts = sum(len(t.get('all_attempts', [])) for t in self.results)
        total_successes = sum(
            t.get('number_of_successful_agent_attempts', 0) for t in self.results
        )
        
        # Calculate pass rates
        task_pass_rates = []
        for task in self.results:
            num_successful = task.get('number_of_successful_agent_attempts', 0)
            total = len(task.get('all_attempts', []))
            if total > 0:
                task_pass_rates.append(num_successful / total)
        
        avg_task_pass_rate = sum(task_pass_rates) / len(task_pass_rates) if task_pass_rates else 0
        
        return {
            'domain': self.domain,
            'variation': self.variation,
            'total_tasks': total_tasks,
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_attempts if total_attempts > 0 else 0,
            'average_task_pass_rate': avg_task_pass_rate,
            'tasks_with_100_percent': sum(1 for r in task_pass_rates if r == 1.0),
            'tasks_with_0_percent': sum(1 for r in task_pass_rates if r == 0.0),
        }


def find_agent_json(domain: str, variation: str = "variation_2") -> Optional[Path]:
    """
    Find agent.json file for a domain/variation.
    
    Args:
        domain: Domain name
        variation: Variation name
        
    Returns:
        Path to agent.json if found, None otherwise
    """
    possible_paths = [
        Path.cwd() / "domains" / domain / "agent.json",
        Path.cwd() / "domains" / domain / "variations" / variation / "agent.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def get_available_variations(domain: str) -> List[str]:
    """
    Get list of available variations for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        List of variation names (e.g., ['variation_1', 'variation_2'])
    """
    variations_dir = Path.cwd() / "domains" / domain / "variations"
    
    if not variations_dir.exists():
        return []
    
    variations = []
    for item in variations_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            # Check if it has both tasks.py and tools.py
            if (item / "tasks.py").exists() and (item / "tools.py").exists():
                variations.append(item.name)
    
    return sorted(variations)

