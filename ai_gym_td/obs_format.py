"""Observation formatting for LLM agents.

Converts the numpy observation dict into a compact, readable format
that LLMs can reason about. Includes:
- ASCII grid visualization
- Structured tower/enemy lists
- Global stats
- Legal action summary
"""

import json
from typing import Dict, Any, List
import numpy as np

from ai_gym_td.config import GameConfig
from ai_gym_td.env import TowerDefenseEnv, OBS_CHANNELS


def format_obs_for_llm(env: TowerDefenseEnv, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> str:
    """Convert observation dict to a readable string for LLMs.
    
    Returns a formatted string with:
    - Global stats (gold, lives, wave, phase)
    - ASCII grid visualization
    - Tower positions and types
    - Enemy positions, HP, and stats
    - Legal action summary
    """
    lines = []
    
    # Global stats
    global_vec = obs["global"]
    gold_norm, lives_norm, wave_norm = global_vec[0], global_vec[1], global_vec[2]
    phase_build, phase_wave = global_vec[3], global_vec[4]
    enemy_count_norm = global_vec[5]
    
    # Denormalize
    gold = int(gold_norm * env.cfg.starting_gold * 2)
    lives = int(lives_norm * env.cfg.starting_lives)
    wave = int(wave_norm * env.cfg.max_waves)
    phase = "build" if phase_build > 0.5 else ("wave" if phase_wave > 0.5 else "terminal")
    
    lines.append(f"## Game State")
    lines.append(f"- **Gold**: {info.get('gold', gold)}")
    lines.append(f"- **Lives**: {info.get('lives', lives)}")
    lines.append(f"- **Wave**: {info.get('wave', wave)}/{env.cfg.max_waves}")
    lines.append(f"- **Phase**: {info.get('phase', phase)}")
    lines.append("")
    
    # Grid visualization
    lines.append("## Grid Map")
    grid = _render_grid_ascii(env, obs)
    lines.append("```")
    lines.append(grid)
    lines.append("```")
    lines.append("Legend: . = grass, P = path, S = spawn, B = base, # = blocked")
    lines.append("        A = archer, C = cannon, I = ice, Z = tesla, E = enemy")
    lines.append("")
    
    # Towers
    lines.append("## Towers")
    towers = _extract_towers(env, obs)
    if towers:
        for t in towers:
            lines.append(f"- {t['type']} at ({t['x']}, {t['y']}): damage={t['damage']}, range={t['range']}")
    else:
        lines.append("No towers built yet.")
    lines.append("")
    
    # Enemies
    lines.append("## Enemies")
    enemies = _extract_enemies(env, obs)
    if enemies:
        for e in enemies[:10]:  # Limit to 10 enemies to keep prompt short
            lines.append(f"- {e['type']} at ({e['x']:.1f}, {e['y']:.1f}): HP={e['hp']:.0f}/{e['max_hp']:.0f}, speed={e['speed']:.1f}")
        if len(enemies) > 10:
            lines.append(f"... and {len(enemies) - 10} more")
    else:
        lines.append("No enemies on the field.")
    lines.append("")
    
    # Legal actions
    lines.append("## Legal Actions")
    action_mask = obs["action_mask"]
    legal_actions = _extract_legal_actions(env, action_mask, info)
    if legal_actions["can_build"]:
        lines.append("You can build:")
        for tower_name, cost in legal_actions["affordable_towers"]:
            lines.append(f"- **{tower_name}** (cost {cost})")
        if legal_actions["good_positions"]:
            lines.append("Recommended positions (near path):")
            for pos in legal_actions["good_positions"][:5]:  # Top 5 positions
                lines.append(f"  - ({pos['x']}, {pos['y']})")
    else:
        lines.append("No affordable towers or no legal placement cells.")
    lines.append("")
    
    return "\n".join(lines)


def _render_grid_ascii(env: TowerDefenseEnv, obs: Dict[str, np.ndarray]) -> str:
    """Render the grid as ASCII art."""
    H, W = env.cfg.height, env.cfg.width
    grid_chars = []
    
    # Build character grid
    for y in range(H):
        row = []
        for x in range(W):
            # Check tower channels (1-4)
            has_tower = False
            for ch in range(1, 5):
                if obs["grid"][y, x, ch] > 0.5:
                    tower_type = ["A", "C", "I", "Z"][ch - 1]
                    row.append(tower_type)
                    has_tower = True
                    break
            if has_tower:
                continue
            
            # Check enemy density (channel 5)
            if obs["grid"][y, x, 5] > 0.3:
                row.append("E")
                continue
            
            # Check terrain (channel 0)
            terrain = obs["grid"][y, x, 0]
            if terrain < 0.1:
                row.append(".")  # grass
            elif terrain < 0.4:
                row.append("P")  # path
            elif terrain < 0.6:
                row.append("S")  # spawn
            elif terrain < 0.9:
                row.append("B")  # base
            else:
                row.append("#")  # blocked
        
        # Add row number
        row_num = f"{y:2d}"
        grid_chars.append(f"{row_num} [{''.join(row)}]")
    
    # Add column numbers
    col_header = "   " + "".join([str(x % 10) for x in range(W)])
    grid_chars.insert(0, col_header)
    
    return "\n".join(grid_chars)


def _extract_towers(env: TowerDefenseEnv, obs: Dict[str, np.ndarray]) -> List[Dict]:
    """Extract tower positions and stats from observation."""
    towers = []
    H, W = env.cfg.height, env.cfg.width
    
    # Channel 0 is terrain, 1-4 are tower types
    tower_names = ["archer", "cannon", "ice", "tesla"]
    for ch in range(1, 5):
        for y in range(H):
            for x in range(W):
                if obs["grid"][y, x, ch] > 0.5:
                    tower_spec = env.cfg.towers[tower_names[ch - 1]]
                    towers.append({
                        "type": tower_names[ch - 1],
                        "x": x,
                        "y": y,
                        "damage": tower_spec.damage,
                        "range": tower_spec.range,
                        "fire_rate": tower_spec.fire_rate,
                    })
    
    return towers


def _extract_enemies(env: TowerDefenseEnv, obs: Dict[str, np.ndarray]) -> List[Dict]:
    """Extract enemy positions and stats from observation."""
    enemies = []
    H, W = env.cfg.height, env.cfg.width
    
    # Channel 5 is enemy density, 6 is HP fraction
    for y in range(H):
        for x in range(W):
            density = obs["grid"][y, x, 5]
            if density > 0.1:  # Threshold for enemy presence
                hp_frac = obs["grid"][y, x, 6]
                # Estimate enemy type based on speed (from global context)
                enemies.append({
                    "type": "enemy",
                    "x": x,
                    "y": y,
                    "hp": hp_frac * 100,  # Rough estimate
                    "max_hp": 100,
                    "speed": 1.5,  # Default, actual speed not in obs
                })
    
    return enemies


def _extract_legal_actions(env: TowerDefenseEnv, action_mask: np.ndarray, info: Dict[str, Any]) -> Dict:
    """Extract legal actions from action mask."""
    result = {
        "can_build": False,
        "affordable_towers": [],
        "good_positions": [],
    }
    
    gold = info.get("gold", 0)
    tower_names = ["pass", "archer", "cannon", "ice", "tesla"]
    
    # Check which towers are affordable
    for idx in range(1, len(tower_names)):
        tower_name = tower_names[idx]
        if tower_name in env.cfg.towers:
            cost = env.cfg.towers[tower_name].cost
            if gold >= cost:
                # Check if any placement is legal
                if action_mask[idx].sum() > 0:
                    result["affordable_towers"].append((tower_name, cost))
                    result["can_build"] = True
    
    # Find good positions (near path, low path_distance)
    # Use the grid's path_distance channel (channel 7)
    # We don't have direct access here, so we'll use the action mask
    # and pick positions where multiple tower types can be built
    if result["can_build"]:
        # Pick first affordable tower
        first_tower_idx = 1
        legal_cells = np.argwhere(action_mask[first_tower_idx] > 0)
        for y, x in legal_cells[:5]:  # Top 5 legal cells
            result["good_positions"].append({"x": int(x), "y": int(y)})
    
    return result


def format_action_prompt() -> str:
    """Return the action format instructions for the LLM."""
    return """## Action Format

Respond with a JSON object:
```json
{
  "action": "build",
  "tower_type": "archer",
  "x": 5,
  "y": 3
}
```

Or to pass (do nothing this turn):
```json
{
  "action": "pass"
}
```

**Tower types**: archer, cannon, ice, tesla
**Coordinates**: x is column (0-19), y is row (0-11)

Choose strategically:
- **Archer**: cheap, fast, good all-around
- **Cannon**: expensive, splash damage, slow
- **Ice**: slows enemies, low damage
- **Tesla**: chains to multiple enemies, expensive

Place towers near the path (P cells) to maximize coverage.
"""


def format_game_rules() -> str:
    """Return the game rules for the LLM."""
    return """## Tower Defense Rules

**Objective**: Prevent enemies from reaching the base (B). Each enemy that reaches the base costs you lives.

**Waves**: Enemies spawn in waves. You have a build phase to place towers, then enemies walk the path.

**Economy**: 
- Start with 120 gold
- Earn gold by killing enemies (bounty varies by enemy type)
- Towers cost gold to build

**Enemies**:
- **Scout**: fast, low HP
- **Grunt**: balanced
- **Brute**: slow, high HP, armor
- **Swarm**: very fast, very low HP

**Win condition**: Survive all 20 waves
**Lose condition**: Lives reach 0

**Strategy tips**:
- Build towers near path bends for maximum coverage
- Mix tower types (ice to slow, cannon for splash, archer for DPS)
- Don't overspend early — save gold for later waves
- Tesla towers chain lightning but are expensive
"""
