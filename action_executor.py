"""
Domain-agnostic action executor for testing task actions.

This module allows you to execute and test individual actions from any domain's tasks.
"""

import sys
import json
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional


class ActionExecutor:
    """Execute and test individual actions from domain tasks."""
    
    def __init__(self, domain: str, variation: str = "variation_2"):
        """
        Initialize action executor for a specific domain and variation.
        
        Args:
            domain: Domain name (e.g., 'sec', 'airline', 'banking_services')
            variation: Variation name (default: 'variation_2')
        """
        self.domain = domain
        self.variation = variation
        
        # Load domain modules
        self._load_domain_modules()
        
        # Initialize service
        self.service = self.ServiceClass(tools=self.TOOLS)
        
        # Index tasks by ID
        self.tasks = {task.user_id: task for task in self.TASKS}
    
    def _load_domain_modules(self):
        """Dynamically load domain-specific modules."""
        domain_module = f"domains.{self.domain}"
        
        try:
            # Load tasks, tools, and service
            tasks_module = importlib.import_module(
                f"{domain_module}.variations.{self.variation}.tasks"
            )
            tools_module = importlib.import_module(
                f"{domain_module}.variations.{self.variation}.tools"
            )
            service_module = importlib.import_module(f"{domain_module}.service")
            
            self.TASKS = tasks_module.TASKS
            self.TOOLS = tools_module.TOOLS
            
            # Find service class (usually ends with "System")
            self.ServiceClass = None
            for name in dir(service_module):
                obj = getattr(service_module, name)
                if isinstance(obj, type) and name.endswith("System"):
                    self.ServiceClass = obj
                    break
            
            if self.ServiceClass is None:
                raise ImportError(
                    f"Could not find service class in {domain_module}.service"
                )
                
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Could not load domain '{self.domain}' variation '{self.variation}': {e}"
            )
    
    def execute_action(self, action_name: str, kwargs: Dict[str, Any]) -> Any:
        """
        Execute a single action with given kwargs.
        
        Args:
            action_name: Name of the action/tool to execute
            kwargs: Keyword arguments for the action
            
        Returns:
            Result from the action execution
        """
        # Find the tool
        tool = None
        for t in self.TOOLS:
            if t.get_info()["function"]["name"] == action_name:
                tool = t
                break
        
        if not tool:
            raise ValueError(f"Tool '{action_name}' not found")
        
        # Execute the action
        result = tool.invoke(self.service.database, **kwargs)
        return result
    
    def execute_action_by_name(self, action_name: str, kwargs: Dict[str, Any]) -> Any:
        """
        Execute an action by name (alias for execute_action).
        
        Args:
            action_name: Name of the action/tool to execute
            kwargs: Keyword arguments for the action
            
        Returns:
            Result from the action execution
        """
        return self.execute_action(action_name, kwargs)
    
    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get all available tools in this domain/variation.
        
        Returns:
            Dictionary mapping tool names to tool objects
        """
        tools_dict = {}
        for tool in self.TOOLS:
            info = tool.get_info()
            tool_name = info["function"]["name"]
            tools_dict[tool_name] = tool
        return tools_dict
    
    def get_available_tasks(self) -> Dict[str, Any]:
        """
        Get all available tasks in this domain/variation.
        
        Returns:
            Dictionary mapping task IDs to task objects
        """
        return self.tasks
    
    def execute_task_action(
        self, task_id: str, action_index: int, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a specific action from a task.
        
        Args:
            task_id: Task identifier (e.g., 'task_001')
            action_index: Index of action to execute (0-based)
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with execution results
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        
        if action_index >= len(task.actions):
            raise ValueError(
                f"Action index {action_index} out of range "
                f"(task has {len(task.actions)} actions)"
            )
        
        action = task.actions[action_index]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Executing: {action.name}")
            print(f"Task: {task_id}")
            print(f"Action Index: {action_index}")
            print(f"{'='*80}")
            print(f"\nKwargs:")
            print(json.dumps(action.kwargs, indent=2))
        
        # Execute
        try:
            result = self.execute_action(action.name, action.kwargs)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"✅ SUCCESS")
                print(f"{'='*80}")
                print(f"\nResult:")
                if isinstance(result, str):
                    try:
                        # Try to parse and pretty-print JSON
                        parsed = json.loads(result)
                        print(json.dumps(parsed, indent=2))
                    except:
                        print(result)
                else:
                    print(json.dumps(result, indent=2) if result else "None")
            
            return {
                "success": True,
                "action": action.name,
                "kwargs": action.kwargs,
                "result": result
            }
        except Exception as e:
            if verbose:
                print(f"\n{'='*80}")
                print(f"❌ ERROR")
                print(f"{'='*80}")
                print(f"\n{type(e).__name__}: {str(e)}")
            
            return {
                "success": False,
                "action": action.name,
                "kwargs": action.kwargs,
                "error": str(e)
            }
    
    def execute_task_actions_range(
        self, task_id: str, start_idx: int, end_idx: int, verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute a range of actions from a task."""
        results = []
        
        for i in range(start_idx, end_idx + 1):
            result = self.execute_task_action(task_id, i, verbose=verbose)
            results.append(result)
            
            if not result["success"]:
                if verbose:
                    print(f"\n⚠️  Stopping at action {i} due to error")
                break
            
            if verbose and i < end_idx:
                print("\n" + "-"*80 + "\n")
        
        return results
    
    def execute_all_task_actions(
        self, task_id: str, verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute all actions from a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        num_actions = len(task.actions)
        
        if verbose:
            print(f"\nExecuting all {num_actions} actions from {task_id}")
            print(f"Task instruction: {task.instruction[:100]}...")
        
        return self.execute_task_actions_range(
            task_id, 0, num_actions - 1, verbose=verbose
        )
    
    def list_tasks(self):
        """List all available tasks."""
        print(f"\nAvailable tasks in {self.domain}/{self.variation} ({len(self.tasks)}):")
        print("="*80)
        for task_id, task in sorted(self.tasks.items()):
            num_actions = len(task.actions)
            instruction_preview = (
                task.instruction[:60] + "..." 
                if len(task.instruction) > 60 
                else task.instruction
            )
            print(f"{task_id:12} | {num_actions:2} actions | {instruction_preview}")
    
    def list_task_actions(self, task_id: str):
        """List all actions in a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        print(f"\nActions in {task_id} ({len(task.actions)} total):")
        print("="*80)
        for i, action in enumerate(task.actions):
            kwargs_preview = ", ".join(f"{k}=..." for k in action.kwargs.keys())
            print(f"[{i:2}] {action.name}({kwargs_preview})")
    
    def inspect_database(self, table: Optional[str] = None):
        """Inspect current database state."""
        if table:
            if table in self.service.database:
                print(f"\nTable: {table}")
                print("="*80)
                data = self.service.database[table]
                if isinstance(data, list) and len(data) > 0:
                    print(f"Total records: {len(data)}")
                    print("\nFirst few records:")
                    print(json.dumps(data[:3], indent=2))
                    if len(data) > 3:
                        print(f"\n... and {len(data) - 3} more")
                else:
                    print(json.dumps(data, indent=2))
            else:
                print(f"Table '{table}' not found")
                print(f"Available tables: {', '.join(self.service.database.keys())}")
        else:
            print(f"\nDatabase tables in {self.domain}:")
            print("="*80)
            for table_name, data in self.service.database.items():
                count = len(data) if isinstance(data, list) else "N/A"
                print(f"  {table_name:40} | {count} records")
    
    def reset_database(self):
        """Reset database to initial state."""
        self.service.reset_database()
        print(f"✅ Database reset to initial state for {self.domain}/{self.variation}")


def get_available_domains() -> List[str]:
    """Get list of available domains in the workspace."""
    domains_dir = Path.cwd() / "domains"
    if not domains_dir.exists():
        return []
    
    domains = []
    for item in domains_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            # Check if it has a service.py
            if (item / "service.py").exists():
                domains.append(item.name)
    
    return sorted(domains)


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

