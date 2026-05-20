"""LLM-based agents for tower defense.

Supports OpenAI (GPT-4o, o1, o3), Anthropic (Claude), and Google (Gemini).
Each agent takes observations, formats them for the LLM, and parses actions.

Install provider SDKs:
    pip install openai anthropic google-genai
"""

import json
import time
from typing import Dict, Any, Optional, List
import numpy as np

from ai_gym_td.agents import Agent
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.obs_format import (
    format_obs_for_llm,
    format_action_prompt,
    format_game_rules,
)


class LLMAgent(Agent):
    """Base class for LLM-powered agents."""
    
    def __init__(
        self,
        model: str,
        provider: str,
        env: Optional["TowerDefenseEnv"] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.verbose = verbose
        self.env = env
        
        # Build system prompt
        if system_prompt is None:
            system_prompt = self._default_system_prompt()
        self.system_prompt = system_prompt
        
        # Tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_latency = 0.0
        
        # History (optional, for multi-turn reasoning)
        self.history: List[Dict[str, str]] = []
    
    def _default_system_prompt(self) -> str:
        """Build default system prompt with game rules and action format."""
        parts = [
            "You are an expert tower defense strategist.",
            "You MUST respond with ONLY valid JSON, no explanations or reasoning.",
            "Do not include any text before or after the JSON object.",
            "",
            format_game_rules(),
            "",
            format_action_prompt(),
        ]
        return "\n".join(parts)
    
    def act(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        """Get action from LLM.
        
        Returns a numpy array [tower_idx, y, x] or [0, 0, 0] for pass.
        """
        # Format observation
        obs_text = self._format_observation(obs, info)
        
        # Call LLM
        t0 = time.time()
        response_text = self._call_llm(obs_text)
        latency = time.time() - t0
        self.total_latency += latency
        
        if self.verbose:
            print(f"\n[LLM call {self.total_calls + 1}]")
            print(f"Prompt:\n{obs_text}\n")
            print(f"Response:\n{response_text}\n")
            print(f"Latency: {latency:.2f}s")
        
        # Parse response
        action = self._parse_response(response_text, info)
        
        self.total_calls += 1
        return action
    
    def _format_observation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> str:
        """Format observation for LLM using rich formatter if env is available."""
        if self.env is not None:
            return format_obs_for_llm(self.env, obs, info)

        # Fallback: minimal format without env reference
        global_vec = obs["global"]
        gold_norm, lives_norm, wave_norm = global_vec[0], global_vec[1], global_vec[2]

        # Rough denormalization
        gold = int(gold_norm * 240)  # 120 * 2
        lives = int(lives_norm * 20)
        wave = int(wave_norm * 20)
        phase = "build" if global_vec[3] > 0.5 else "wave"

        lines = [
            f"## Current State",
            f"- Gold: {gold}",
            f"- Lives: {lives}",
            f"- Wave: {wave}/20",
            f"- Phase: {phase}",
            "",
            "Decide your next action.",
        ]
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API. Override in subclasses."""
        raise NotImplementedError
    
    def _parse_response(self, response: str, info: Dict[str, Any]) -> np.ndarray:
        """Parse LLM response into action array.
        
        Returns [tower_idx, y, x] where tower_idx=0 is pass.
        """
        # Extract JSON from response
        json_str = self._extract_json(response)
        if not json_str:
            if self.verbose:
                print(f"Warning: No JSON found in response. Passing.")
            return np.array([0, 0, 0], dtype=np.int64)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"Warning: Invalid JSON: {json_str}. Passing.")
            return np.array([0, 0, 0], dtype=np.int64)
        
        # Parse action
        action = data.get("action", "pass")
        if action == "pass":
            return np.array([0, 0, 0], dtype=np.int64)
        
        if action == "build":
            tower_name = data.get("tower_type", "").lower()
            x = data.get("x", 0)
            y = data.get("y", 0)
            
            # Map tower name to index
            tower_names = {"archer": 1, "cannon": 2, "ice": 3, "tesla": 4}
            tower_idx = tower_names.get(tower_name, 0)
            
            if tower_idx == 0:
                if self.verbose:
                    print(f"Warning: Unknown tower type '{tower_name}'. Passing.")
                return np.array([0, 0, 0], dtype=np.int64)
            
            return np.array([tower_idx, int(y), int(x)], dtype=np.int64)
        
        if self.verbose:
            print(f"Warning: Unknown action '{action}'. Passing.")
        return np.array([0, 0, 0], dtype=np.int64)
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text (handles markdown code blocks)."""
        # Try to find JSON in code blocks
        import re
        match = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find bare JSON object
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None
    
    def reset(self):
        """Reset agent state."""
        self.history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "model": self.model,
            "provider": self.provider,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_latency": self.total_latency / max(1, self.total_calls),
        }


class OpenAIAgent(LLMAgent):
    """OpenAI API agent (GPT-4o, o1, o3, etc.)."""
    
    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(model=model, provider="openai", **kwargs)
        
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI agent requires: pip install openai")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Pricing per 1K tokens (as of 2025-01)
        self.pricing = {
            "gpt-4o": {"input": 0.0025, "output": 0.010},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "o1": {"input": 0.015, "output": 0.060},
            "o3-mini": {"input": 0.0011, "output": 0.0044},
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Track tokens
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        self.total_tokens += input_tokens + output_tokens
        
        # Track cost
        if self.model in self.pricing:
            input_cost = (input_tokens / 1000) * self.pricing[self.model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[self.model]["output"]
            self.total_cost += input_cost + output_cost
        
        return response.choices[0].message.content


class AnthropicAgent(LLMAgent):
    """Anthropic API agent (Claude 3.5 Sonnet, Opus, etc.)."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(model=model, provider="anthropic", **kwargs)
        
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic agent requires: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Pricing per 1K tokens (as of 2025-01)
        self.pricing = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic API."""
        # Use Anthropic's native system parameter for the system prompt
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Track tokens
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.total_tokens += input_tokens + output_tokens
        
        # Track cost
        if self.model in self.pricing:
            input_cost = (input_tokens / 1000) * self.pricing[self.model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[self.model]["output"]
            self.total_cost += input_cost + output_cost
        
        return response.content[0].text


class GoogleAgent(LLMAgent):
    """Google Gemini API agent."""
    
    def __init__(self, model: str = "gemini-1.5-pro", **kwargs):
        super().__init__(model=model, provider="google", **kwargs)
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google agent requires: pip install google-generativeai")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        self.client = genai.GenerativeModel(model)
        
        # Pricing per 1K tokens (as of 2025-01)
        self.pricing = {
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call Gemini API."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        response = self.client.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )
        
        # Track tokens (Gemini doesn't always return usage stats)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            self.total_tokens += input_tokens + output_tokens
            
            # Track cost
            if self.model in self.pricing:
                input_cost = (input_tokens / 1000) * self.pricing[self.model]["input"]
                output_cost = (output_tokens / 1000) * self.pricing[self.model]["output"]
                self.total_cost += input_cost + output_cost
        
        return response.text


class OllamaAgent(LLMAgent):
    """Ollama agent for local/cloud models via OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", **kwargs):
        super().__init__(model=model, provider="ollama", **kwargs)

        try:
            import openai
        except ImportError:
            raise ImportError("Ollama agent requires: pip install openai")

        # Ollama uses OpenAI-compatible API
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="ollama"  # Ollama doesn't need a real API key
        )

        # No pricing for local models (cost = $0)
        self.pricing = {}

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama API."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Track tokens (Ollama provides usage stats)
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.total_tokens += input_tokens + output_tokens

        # No cost tracking for local models

        # Get response content (some models use 'reasoning' field)
        message = response.choices[0].message
        content = message.content

        # If content is empty, try reasoning field (used by some Ollama models)
        if not content and hasattr(message, 'reasoning'):
            content = message.reasoning

        return content or ""


def create_llm_agent(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMAgent:
    """Factory function to create LLM agents.
    
    Args:
        provider: "openai", "anthropic", "google", or "ollama"
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        api_key: API key (or use environment variables)
        **kwargs: Additional arguments passed to agent constructor
    
    Returns:
        LLMAgent instance
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIAgent(model=model, api_key=api_key, **kwargs)
    elif provider == "anthropic":
        return AnthropicAgent(model=model, api_key=api_key, **kwargs)
    elif provider == "google":
        return GoogleAgent(model=model, api_key=api_key, **kwargs)
    elif provider == "ollama":
        base_url = kwargs.pop("base_url", "http://localhost:11434/v1")
        return OllamaAgent(model=model, base_url=base_url, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: openai, anthropic, google, ollama")


__all__ = [
    "LLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "GoogleAgent",
    "OllamaAgent",
    "create_llm_agent",
]
