"""
LLM client configuration and management using LangChain.

Supports any OpenAI-compatible API endpoint.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from helper/.env
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


def get_llm_client(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI client with configurable parameters.
    
    Args:
        model: Model name (defaults to DEFAULT_MODEL from .env)
        api_key: API key (defaults to DEFAULT_API_KEY from .env)
        base_url: API base URL (defaults to DEFAULT_BASE_URL from .env)
        temperature: Sampling temperature (default: 0.0 for deterministic outputs)
        max_tokens: Maximum tokens in response (optional)
        seed: Random seed for reproducibility (optional, defaults to 42 for reasoning models)
    
    Returns:
        Configured ChatOpenAI instance
    
    Example:
        >>> llm = get_llm_client()
        >>> response = llm.invoke("Hello, world!")
    """
    # Use provided values or fall back to environment variables
    model = model or os.getenv("DEFAULT_MODEL", "gpt-4o")
    api_key = api_key or os.getenv("DEFAULT_API_KEY")
    base_url = base_url or os.getenv("DEFAULT_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        raise ValueError(
            "API key not provided. Set DEFAULT_API_KEY in .env or pass api_key parameter."
        )
    
    # Configure ChatOpenAI
    config = {
        "model": model,
        "openai_api_key": api_key,
        "openai_api_base": base_url,
        "temperature": temperature,
    }
    
    if max_tokens:
        config["max_tokens"] = max_tokens
    
    # Add seed for reproducibility (especially important for reasoning models)
    # Pass seed as direct parameter, not in model_kwargs
    if seed is not None:
        config["seed"] = seed
    
    return ChatOpenAI(**config)


def get_reasoning_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    seed: int = 42,
) -> ChatOpenAI:
    """
    Get a reasoning LLM client (e.g., DeepSeek-R1) using DEFAULT_MODEL_R config.
    
    This is specifically for reasoning tasks like SOP mapping, instruction analysis, etc.
    Uses a default seed of 42 to improve reproducibility.
    
    Args:
        model: Model name (defaults to DEFAULT_MODEL_R from .env)
        api_key: API key (defaults to DEFAULT_API_KEY_R from .env)
        base_url: API base URL (defaults to DEFAULT_BASE_URL_R from .env)
        temperature: Sampling temperature (default: 0.0)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Configured ChatOpenAI instance
        
    Note:
        Even with temperature=0.0 and seed set, reasoning models may still show
        some variance due to their chain-of-thought reasoning process. For
        critical applications requiring 100% determinism, consider using a
        non-reasoning model.
    """
    model = model or os.getenv("DEFAULT_MODEL_R")
    api_key = api_key or os.getenv("DEFAULT_API_KEY_R")
    base_url = base_url or os.getenv("DEFAULT_BASE_URL_R")
    
    # Fallback to DEFAULT_MODEL if DEFAULT_MODEL_R is not set
    if not model:
        model = os.getenv("DEFAULT_MODEL", "gpt-4o")
        api_key = api_key or os.getenv("DEFAULT_API_KEY")
        base_url = base_url or os.getenv("DEFAULT_BASE_URL", "https://api.openai.com/v1")
    
    return get_llm_client(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
    )


def get_reasoning_llm_r2(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    seed: int = 42,
) -> Optional[ChatOpenAI]:
    """
    Get a second reasoning LLM client using DEFAULT_MODEL_R2 config.
    
    Used for multi-agent architecture where two reasoning models generate
    scaffolds independently and a judge picks the best one.
    
    Args:
        model: Model name (defaults to DEFAULT_MODEL_R2 from .env)
        api_key: API key (defaults to DEFAULT_API_KEY_R2 from .env)
        base_url: API base URL (defaults to DEFAULT_BASE_URL_R2 from .env)
        temperature: Sampling temperature (default: 0.0)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Configured ChatOpenAI instance, or None if R2 config not available
    """
    model = model or os.getenv("DEFAULT_MODEL_R2")
    api_key = api_key or os.getenv("DEFAULT_API_KEY_R2")
    base_url = base_url or os.getenv("DEFAULT_BASE_URL_R2")
    
    # Return None if R2 is not configured (multi-agent not enabled)
    if not model or not api_key:
        return None
    
    return get_llm_client(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
    )


def get_judge_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    seed: int = 42,
) -> Optional[ChatOpenAI]:
    """
    Get a judge LLM client using DEFAULT_MODEL_R_JUDGE config.
    
    The judge model evaluates outputs from R and R2 models and picks the best one
    when they produce different scaffolds.
    
    Args:
        model: Model name (defaults to DEFAULT_MODEL_R_JUDGE from .env)
        api_key: API key (defaults to DEFAULT_API_KEY_R_JUDGE from .env)
        base_url: API base URL (defaults to DEFAULT_BASE_URL_R_JUDGE from .env)
        temperature: Sampling temperature (default: 0.0)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Configured ChatOpenAI instance, or None if judge config not available
    """
    model = model or os.getenv("DEFAULT_MODEL_R_JUDGE")
    api_key = api_key or os.getenv("DEFAULT_API_KEY_R_JUDGE")
    base_url = base_url or os.getenv("DEFAULT_BASE_URL_R_JUDGE")
    
    # Return None if judge is not configured (multi-agent not enabled)
    if not model or not api_key:
        return None
    
    return get_llm_client(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
    )


def is_multi_agent_enabled() -> bool:
    """
    Check if multi-agent architecture is enabled.
    
    Multi-agent requires both R2 and JUDGE models to be configured.
    
    Returns:
        True if both DEFAULT_MODEL_R2 and DEFAULT_MODEL_R_JUDGE are configured
    """
    has_r2 = bool(os.getenv("DEFAULT_MODEL_R2") and os.getenv("DEFAULT_API_KEY_R2"))
    has_judge = bool(os.getenv("DEFAULT_MODEL_R_JUDGE") and os.getenv("DEFAULT_API_KEY_R_JUDGE"))
    return has_r2 and has_judge


def get_alternative_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """
    Get an alternative LLM client (e.g., DeepSeek) using DEFAULT_MODEL1 config.
    
    Args:
        model: Model name (defaults to DEFAULT_MODEL1 from .env)
        api_key: API key (defaults to DEFAULT_API_KEY1 from .env)
        base_url: API base URL (defaults to DEFAULT_BASE_URL1 from .env)
        temperature: Sampling temperature
    
    Returns:
        Configured ChatOpenAI instance
    """
    model = model or os.getenv("DEFAULT_MODEL1")
    api_key = api_key or os.getenv("DEFAULT_API_KEY1")
    base_url = base_url or os.getenv("DEFAULT_BASE_URL1")
    
    return get_llm_client(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

