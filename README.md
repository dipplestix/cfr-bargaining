# Bargaining Game Nash Equilibrium Solver

Finding Nash equilibrium in a two-player bargaining game using Counterfactual Regret Minimization (CFR).

## The Bargaining Game

### Setup
- **Items**: 2 types of items, 2 of each type (4 items total)
- **Player valuations**: Each player privately values each item type at 0, 0.5, or 1
- **Walk-away value**: Each player has a reservation value drawn from {0, 1, ..., total_value} where total_value = 2*v1 + 2*v2
- **27 types per player**, creating 729 possible type profiles

### Actions
Players can:
- **Offer**: Propose giving X items of type 1 and Y items of type 2 to opponent
  - Possible offers: 00, 10, 20, 01, 02, 11, 22, 12, 21
- **Accept**: Accept the opponent's last offer (available after round 1)
- **Walk**: End negotiation, both players receive their walk-away value

### Timing
- 6 decision rounds: P1, P2, P1, P2, P1, P2
- Rounds 1-5: Can offer, walk, or accept (if offer exists)
- Round 6: Can only accept or walk

### Payoffs
- If **walk**: Both players get their walk-away value
- If **accept**: Accepting player gets offered bundle, offerer keeps remainder
- Bundle value = v1 * (items of type 1) + v2 * (items of type 2)

## Counterfactual Regret Minimization (CFR)

### Why CFR?
The game has ~170K info sets for P0 and ~560K for P1, making direct methods (sequence form LP, support enumeration) infeasible. CFR is an iterative algorithm designed for large extensive-form games with imperfect information.

### Algorithm Overview

CFR works by:
1. Tracking **regret** for each action at each information set
2. Using **regret matching** to compute strategies
3. Averaging strategies over iterations to converge to Nash equilibrium

### Regret Matching

At each info set, the strategy is computed from cumulative regrets:

```
strategy(a) = max(0, regret(a)) / sum(max(0, regret(a')) for all a')
```

If all regrets are non-positive, play uniformly.

### CFR Update

For each iteration:
1. Traverse the game tree
2. At terminal nodes, return payoffs
3. At decision nodes for the updating player:
   - Compute counterfactual value for each action
   - Update regrets: `regret(a) += value(a) - node_value`
4. Average the strategies over all iterations

### External Sampling MCCFR

We use **External Sampling Monte Carlo CFR** for efficiency:
- For the **updating player**: Explore ALL actions
- For the **opponent**: Sample ONE action according to their current strategy
- Sample type profiles rather than enumerating all 729

This reduces per-iteration cost from O(|game tree|) to O(depth * branching_factor).

### CFR+ Enhancements

We incorporate CFR+ improvements for faster convergence:

1. **Regret flooring**: Set negative regrets to 0
   ```python
   regret[action] = max(0, regret[action] + instant_regret)
   ```

2. **Linear averaging**: Weight iteration t's strategy by t
   ```python
   strategy_sum[action] += (iteration + 1) * strategy[action]
   ```

### Average Strategy

The Nash equilibrium strategy is the **average** of all strategies played:

```python
avg_strategy[action] = strategy_sum[action] / sum(strategy_sum.values())
```

This is what converges to equilibrium, not the current regret-matched strategy.

## Implementation

### Files

- `bargaining_game.py` - Game mechanics and utility computation
- `cfr_solver.py` - CFR implementation and exploitability computation
- `nash_solver.py` - Sequence form representation (documents game scale)

### Key Classes

```python
# Game representation
class PlayerType:
    v1: float           # Value for item type 1
    v2: float           # Value for item type 2
    walk_value: int     # Walk-away value

class GameState:
    round_num: int      # Current round (0-5)
    history: tuple      # Actions taken
    terminal: bool      # Game ended?
    outcome: tuple      # Terminal outcome

class BargainingGame:
    def get_actions(state) -> List[str]
    def step(state, action) -> GameState
    def get_payoffs(state, types) -> Tuple[float, float]
    def get_info_set(state, player, type) -> str
```

```python
# CFR Solver
class ExternalSamplingCFR:
    def get_strategy(info_set, actions) -> Dict[str, float]
    def external_sampling_cfr(state, types, updating_player) -> float
    def train(num_iterations) -> Tuple[strategy0, strategy1]
    def get_average_strategy() -> Dict[str, Dict[str, float]]
```

### Information Set Encoding

Info sets encode what a player knows:
```
"p{player}|({v1},{v2},{walk_value})|{action_history}"

Example: "p0|(1,0.5,2)|offer_11,offer_02"
```

## Results

### Action Pruning

We prune dominated actions to reduce the game tree:
- **Offers**: Only include offers where we keep ≥ walk_value
- **Accept**: Only include if offered bundle ≥ walk_value

This reduces the action space by ~30% on average, with larger reductions for high walk-away value types.

### Training Progress (1M iterations, with pruning)

| Iterations | Exploitability* |
|------------|-----------------|
| 200,000    | 0.00           |
| 400,000    | 0.39           |
| 600,000    | 0.28           |
| 800,000    | 0.77           |
| 1,000,000  | 0.08           |

*Estimated with 50 samples during training (high variance due to sampling)

### Final Results (500 sample evaluation)

| Metric | Value |
|--------|-------|
| **Total Exploitability** | **0.11** |
| P0 exploitability | 0.02 |
| P1 exploitability | 0.09 |
| P0 equilibrium utility | 1.66 |
| P1 equilibrium utility | 1.47 |
| P0 info sets discovered | 91,809 |
| P1 info sets discovered | 661,242 |
| Strategy file size | 18 MB (gzipped) |

### Effect of Pruning

| Metric | Without Pruning | With Pruning | Reduction |
|--------|-----------------|--------------|-----------|
| P0 info sets | 179,361 | 91,809 | **49%** |
| P1 info sets | 1,280,283 | 661,242 | **48%** |
| File size | 248 MB | 120 MB | **52%** |
| Exploitability | 0.20 | 0.11 | **45%** |

### Comparison with Uniform Random

| Strategy | Exploitability |
|----------|----------------|
| Uniform Random | 0.30 |
| CFR (1M iter) | 0.11 |
| **Improvement** | **63%** |

### Interpreting Exploitability

- **Exploitability = 0**: Exact Nash equilibrium
- **Exploitability = 0.11**: Neither player can gain more than ~0.09 by deviating to best response
- The CFR strategy is an **approximate Nash equilibrium**

## Sample Strategies

### Player 0 with type (v1=1, v2=1, walk=0)

After offer_00 -> offer_02 -> offer_20 -> offer_01:
```
accept:   62%
offer_20: 28%
offer_00:  4%
```

After offer_00 -> offer_02 -> offer_11 -> offer_01:
```
offer_20: 84%
offer_10:  9%
accept:    3%
```

### Strategy Interpretation

With action pruning, strategies are more concentrated because dominated actions are removed:
- **Type (1,1,0)**: With walk_value=0, all offers are viable, so strategy focuses on maximizing deal value
- **Accept decisions**: Accept when offered bundle exceeds walk_value significantly
- **Counter-offers**: Prefer offers that keep high-value items (offer_20 keeps 2 of type 1)

Key strategic patterns:
- **Aggressive offering**: Types with low walk_value make aggressive offers to close deals
- **Selective acceptance**: Only accept offers that beat walk_value
- **Value-aware counters**: Counter-offer based on item valuations

### Game-Theoretic Insights

With pruning, the game tree is reduced by ~50%. The near-Nash equilibrium (exploitability 0.11) shows:
- Both players get similar utility (P0: 1.66, P1: 1.47)
- P0 has a moderate first-mover advantage (~0.19 utility difference)
- Walk-away values provide significant bargaining power
- Dominated actions are never played in equilibrium

## Usage

### Loading Pre-trained Strategy

A pre-trained Nash equilibrium strategy (1M iterations) is saved in `nash_equilibrium.json.gz` (18MB compressed):

```python
from bargaining_game import BargainingGame
from cfr_solver import load_strategies, compute_exploitability

# Load pre-trained strategies (handles .gz automatically)
strategy0, strategy1, metadata = load_strategies("nash_equilibrium.json.gz")
print(f"Loaded strategies: {metadata}")

# Use with game
game = BargainingGame()
eu0, eu1 = game.compute_expected_utility(strategy0, strategy1)
print(f"Expected utilities: P0={eu0:.4f}, P1={eu1:.4f}")
```

### Training From Scratch

```python
from bargaining_game import BargainingGame
from cfr_solver import ExternalSamplingCFR, compute_exploitability, save_strategies

# Create game
game = BargainingGame()

# Train CFR (1M iterations for best results)
solver = ExternalSamplingCFR(game)
strategy0, strategy1 = solver.train(num_iterations=1000000)

# Save strategies
save_strategies(strategy0, strategy1, "my_strategy.json",
                metadata={"iterations": 1000000})

# Evaluate exploitability
exploit0, exploit1, total = compute_exploitability(
    game, strategy0, strategy1, num_samples=500
)
print(f"Exploitability: {total:.4f}")
```

## Running

```bash
# Install dependencies
uv sync

# Run CFR solver
uv run python cfr_solver.py

# Run game demo
uv run python bargaining_game.py
```

## References

- Zinkevich et al. (2007). "Regret Minimization in Games with Incomplete Information"
- Lanctot et al. (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games"
- Tammelin (2014). "Solving Large Imperfect Information Games Using CFR+"
- Brown & Sandholm (2019). "Solving Imperfect-Information Games via Discounted Regret Minimization"
