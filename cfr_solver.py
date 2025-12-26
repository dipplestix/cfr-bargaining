"""
Counterfactual Regret Minimization (CFR) Solver

CFR is an iterative algorithm that converges to Nash equilibrium in two-player
zero-sum games. In general-sum games (like this bargaining game), CFR converges
to a Coarse Correlated Equilibrium (CCE). For independent behavioral strategies,
an ε-CCE is equivalent to an ε-Nash equilibrium.
"""

import numpy as np
import json
import gzip
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from bargaining_game import BargainingGame, GameState, PlayerType, enumerate_types


def save_strategies(
    strategy0: Dict[str, Dict[str, float]],
    strategy1: Dict[str, Dict[str, float]],
    filepath: str,
    metadata: Optional[Dict] = None,
    compress: bool = True
):
    """Save strategies to a JSON file (optionally gzipped)."""
    data = {
        "strategy0": strategy0,
        "strategy1": strategy1,
        "metadata": metadata or {}
    }

    if compress or filepath.endswith('.gz'):
        if not filepath.endswith('.gz'):
            filepath += '.gz'
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"Strategies saved to {filepath}")


def load_strategies(filepath: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict]:
    """Load strategies from a JSON file (handles gzipped files)."""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)
    return data["strategy0"], data["strategy1"], data.get("metadata", {})


class OutcomeSamplingCFR:
    """
    Monte Carlo CFR with Outcome Sampling.

    Much faster than vanilla CFR for deep game trees.
    Samples a single trajectory through the game tree per iteration.
    """

    def __init__(self, game: BargainingGame):
        self.game = game
        self.types = game.types

        # Cumulative regrets and strategy sums
        self.regrets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.info_set_actions: Dict[str, List[str]] = {}

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        regrets = self.regrets[info_set]
        positive_regrets = {a: max(0, regrets[a]) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy across all iterations."""
        avg_strategy = {}
        for info_set, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                avg_strategy[info_set] = {a: s / total for a, s in action_sums.items()}
            elif info_set in self.info_set_actions:
                actions = self.info_set_actions[info_set]
                avg_strategy[info_set] = {a: 1.0 / len(actions) for a in actions}
        return avg_strategy

    def sample_action(self, strategy: Dict[str, float], epsilon: float = 0.6) -> str:
        """Sample action with epsilon-greedy exploration."""
        actions = list(strategy.keys())
        probs = list(strategy.values())

        # Epsilon-greedy: with prob epsilon, sample uniformly
        if np.random.random() < epsilon:
            return actions[np.random.randint(len(actions))]
        else:
            return np.random.choice(actions, p=probs)

    def outcome_sampling_cfr(
        self,
        state: GameState,
        types: Tuple[PlayerType, PlayerType],
        reach_probs: Tuple[float, float],
        sample_prob: float,
        updating_player: int
    ) -> float:
        """
        Outcome sampling CFR traversal.

        Only samples one action per decision point, making it much faster.
        """
        if state.terminal:
            payoffs = self.game.get_payoffs(state, types)
            return payoffs[updating_player] / sample_prob

        current_player = state.current_player
        player_type = types[current_player]
        info_set = self.game.get_info_set(state, current_player, player_type)
        # Use pruned actions based on player type
        actions = self.game.get_actions(state, player_type)

        if not actions:
            actions = ["walk"]

        if info_set not in self.info_set_actions:
            self.info_set_actions[info_set] = actions

        strategy = self.get_strategy(info_set, actions)

        # Sample one action
        epsilon = 0.6
        sampled_action = self.sample_action(strategy, epsilon)
        sample_action_prob = epsilon / len(actions) + (1 - epsilon) * strategy[sampled_action]

        # Recurse on sampled action only
        next_state = self.game.step(state, sampled_action)
        new_reach = list(reach_probs)
        new_reach[current_player] *= strategy[sampled_action]

        # Utility from this sample
        utility = self.outcome_sampling_cfr(
            next_state, types, tuple(new_reach),
            sample_prob * sample_action_prob, updating_player
        )

        # Update for updating player
        if current_player == updating_player:
            opponent = 1 - current_player
            W = utility * reach_probs[opponent]

            for action in actions:
                if action == sampled_action:
                    regret = W * (1 - strategy[action])
                else:
                    regret = -W * strategy[action]
                self.regrets[info_set][action] += regret

            # Update strategy sum
            for action in actions:
                self.strategy_sum[info_set][action] += reach_probs[current_player] * strategy[action] / sample_prob

        return utility

    def train(
        self,
        num_iterations: int = 10000,
        verbose: bool = True
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Train using outcome sampling."""
        num_types = len(self.types)

        for iteration in range(num_iterations):
            # Sample type profile
            type0 = self.types[np.random.randint(num_types)]
            type1 = self.types[np.random.randint(num_types)]
            types = (type0, type1)

            # Update both players
            for updating_player in [0, 1]:
                self.outcome_sampling_cfr(
                    self.game.initial_state(),
                    types,
                    (1.0, 1.0),
                    1.0,
                    updating_player
                )

            if verbose and (iteration + 1) % 1000 == 0:
                print(f"  Iteration {iteration + 1}/{num_iterations}")

        avg_strategy = self.get_average_strategy()
        strategy0 = {k: v for k, v in avg_strategy.items() if k.startswith("p0|")}
        strategy1 = {k: v for k, v in avg_strategy.items() if k.startswith("p1|")}

        return strategy0, strategy1


class ExternalSamplingCFR:
    """
    Monte Carlo CFR with External Sampling.

    Samples opponent actions, but explores all actions for the updating player.
    Good balance between speed and accuracy.
    Uses linear averaging (CFR+ style) for faster convergence.
    """

    def __init__(self, game: BargainingGame):
        self.game = game
        self.types = game.types

        self.regrets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.info_set_actions: Dict[str, List[str]] = {}
        self.iteration = 0  # Track iteration for linear averaging

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        regrets = self.regrets[info_set]
        positive_regrets = {a: max(0, regrets[a]) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy across all iterations."""
        avg_strategy = {}
        for info_set, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                avg_strategy[info_set] = {a: s / total for a, s in action_sums.items()}
            elif info_set in self.info_set_actions:
                actions = self.info_set_actions[info_set]
                avg_strategy[info_set] = {a: 1.0 / len(actions) for a in actions}
        return avg_strategy

    def external_sampling_cfr(
        self,
        state: GameState,
        types: Tuple[PlayerType, PlayerType],
        updating_player: int
    ) -> float:
        """
        External sampling CFR traversal.

        For updating player: explore all actions
        For opponent: sample one action according to strategy
        """
        if state.terminal:
            payoffs = self.game.get_payoffs(state, types)
            return payoffs[updating_player]

        current_player = state.current_player
        player_type = types[current_player]
        info_set = self.game.get_info_set(state, current_player, player_type)
        # Use pruned actions based on player type
        actions = self.game.get_actions(state, player_type)

        if not actions:
            # If no valid actions (shouldn't happen), just walk
            actions = ["walk"]

        if info_set not in self.info_set_actions:
            self.info_set_actions[info_set] = actions

        strategy = self.get_strategy(info_set, actions)

        if current_player == updating_player:
            # Explore all actions
            action_values = {}
            for action in actions:
                next_state = self.game.step(state, action)
                action_values[action] = self.external_sampling_cfr(
                    next_state, types, updating_player
                )

            # Compute node value
            node_value = sum(strategy[a] * action_values[a] for a in actions)

            # Update regrets (with CFR+ flooring at 0)
            for action in actions:
                regret = action_values[action] - node_value
                self.regrets[info_set][action] += regret
                # CFR+ style: floor regrets at 0
                self.regrets[info_set][action] = max(0, self.regrets[info_set][action])

            # Update strategy sum with linear averaging (weight by iteration)
            weight = self.iteration + 1
            for action in actions:
                self.strategy_sum[info_set][action] += weight * strategy[action]

            return node_value
        else:
            # Sample opponent action
            action = np.random.choice(actions, p=[strategy[a] for a in actions])
            next_state = self.game.step(state, action)
            return self.external_sampling_cfr(next_state, types, updating_player)

    def train(
        self,
        num_iterations: int = 1000,
        verbose: bool = True,
        eval_every: int = 0
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Train using external sampling.

        Args:
            num_iterations: Number of CFR iterations
            verbose: Print progress
            eval_every: Evaluate exploitability every N iterations (0 = disabled)
        """
        num_types = len(self.types)

        for iteration in range(num_iterations):
            self.iteration = iteration
            type0 = self.types[np.random.randint(num_types)]
            type1 = self.types[np.random.randint(num_types)]
            types = (type0, type1)

            for updating_player in [0, 1]:
                self.external_sampling_cfr(
                    self.game.initial_state(),
                    types,
                    updating_player
                )

            if verbose and (iteration + 1) % 10000 == 0:
                print(f"  Iteration {iteration + 1}/{num_iterations}")

            # Periodic evaluation
            if eval_every > 0 and (iteration + 1) % eval_every == 0:
                avg_strategy = self.get_average_strategy()
                s0 = {k: v for k, v in avg_strategy.items() if k.startswith("p0|")}
                s1 = {k: v for k, v in avg_strategy.items() if k.startswith("p1|")}
                # Exact exploitability (enumerates all 729 type profiles)
                from cfr_solver import compute_exploitability
                e0, e1, total = compute_exploitability(self.game, s0, s1)
                print(f"    Exploitability at iter {iteration+1}: {total:.4f}")

        avg_strategy = self.get_average_strategy()
        strategy0 = {k: v for k, v in avg_strategy.items() if k.startswith("p0|")}
        strategy1 = {k: v for k, v in avg_strategy.items() if k.startswith("p1|")}

        return strategy0, strategy1




class BestResponseComputer:
    """Computes best response strategies against a fixed opponent strategy."""

    def __init__(self, game: BargainingGame):
        self.game = game
        self.types = game.types

    def compute_best_response_value(
        self,
        state: GameState,
        types: Tuple[PlayerType, PlayerType],
        br_player: int,
        opponent_strategy: Dict[str, Dict[str, float]],
        cache: Dict[Tuple[str, int, int], float]
    ) -> float:
        """
        Compute the value of playing best response for br_player.

        Args:
            state: Current game state
            types: Type profile
            br_player: Player computing best response
            opponent_strategy: Fixed opponent strategy
            cache: Memoization cache

        Returns:
            Expected value for br_player when playing optimally
        """
        if state.terminal:
            payoffs = self.game.get_payoffs(state, types)
            return payoffs[br_player]

        current_player = state.current_player
        player_type = types[current_player]
        info_set = self.game.get_info_set(state, current_player, player_type)
        # Use pruned actions based on player type
        actions = self.game.get_actions(state, player_type)

        if not actions:
            actions = ["walk"]

        # Check cache
        cache_key = (info_set, id(types[0]), id(types[1]))
        if cache_key in cache:
            return cache[cache_key]

        if current_player == br_player:
            # Best response: take max over actions
            best_value = float('-inf')
            for action in actions:
                next_state = self.game.step(state, action)
                value = self.compute_best_response_value(
                    next_state, types, br_player, opponent_strategy, cache
                )
                best_value = max(best_value, value)
            cache[cache_key] = best_value
            return best_value
        else:
            # Opponent plays according to their strategy
            if info_set in opponent_strategy:
                probs = opponent_strategy[info_set]
            else:
                # Default to uniform
                probs = {a: 1.0 / len(actions) for a in actions}

            expected_value = 0.0
            for action in actions:
                prob = probs.get(action, 0)
                if prob > 0:
                    next_state = self.game.step(state, action)
                    value = self.compute_best_response_value(
                        next_state, types, br_player, opponent_strategy, cache
                    )
                    expected_value += prob * value

            cache[cache_key] = expected_value
            return expected_value

    def compute_best_response_utility(
        self,
        br_player: int,
        opponent_strategy: Dict[str, Dict[str, float]],
        type_profiles: Optional[List[Tuple[PlayerType, PlayerType]]] = None
    ) -> float:
        """
        Compute expected utility of best response for br_player.

        Args:
            br_player: Player computing best response
            opponent_strategy: Fixed opponent strategy
            type_profiles: List of type profiles to evaluate. If None, enumerates all.

        Returns:
            Expected utility when br_player plays best response
        """
        if type_profiles is None:
            # Enumerate all type profiles
            type_profiles = [(t0, t1) for t0 in self.types for t1 in self.types]

        total_value = 0.0

        for types in type_profiles:
            cache: Dict[Tuple[str, int, int], float] = {}
            value = self.compute_best_response_value(
                self.game.initial_state(),
                types,
                br_player,
                opponent_strategy,
                cache
            )
            total_value += value

        return total_value / len(type_profiles)


def compute_exploitability(
    game: BargainingGame,
    strategy0: Dict[str, Dict[str, float]],
    strategy1: Dict[str, Dict[str, float]],
    verbose: bool = False
) -> Tuple[float, float, float]:
    """
    Compute exploitability of a strategy profile.

    Exploitability measures how much each player can gain by deviating
    to their best response:
        exploit_i = BR_utility_i - current_utility_i

    For independent strategies (like behavioral strategies), this characterizes
    both ε-CCE and ε-Nash equilibrium, where ε = max(exploit0, exploit1).

    Enumerates all 729 type profiles for exact computation (no sampling variance).

    Args:
        game: The bargaining game
        strategy0: Player 0's strategy
        strategy1: Player 1's strategy
        verbose: Print progress

    Returns:
        (exploit0, exploit1, total_exploitability)
        - exploit0: How much P0 gains by deviating to BR
        - exploit1: How much P1 gains by deviating to BR
        - total_exploitability: exploit0 + exploit1
    """
    br_computer = BestResponseComputer(game)

    # Enumerate all type profiles (27 * 27 = 729)
    type_profiles = [(t0, t1) for t0 in game.types for t1 in game.types]

    # Current utilities (exact computation over all profiles)
    if verbose:
        print("  Computing current utilities (729 type profiles)...")

    current_u0 = 0.0
    current_u1 = 0.0

    for types in type_profiles:
        u0, u1 = game.play_with_strategies(strategy0, strategy1, types)
        current_u0 += u0
        current_u1 += u1

    current_u0 /= len(type_profiles)
    current_u1 /= len(type_profiles)

    if verbose:
        print(f"    Current utilities: P0={current_u0:.4f}, P1={current_u1:.4f}")

    # Best response utilities (using same type profiles)
    if verbose:
        print("  Computing P0 best response utility...")
    br_u0 = br_computer.compute_best_response_utility(0, strategy1, type_profiles)

    if verbose:
        print("  Computing P1 best response utility...")
    br_u1 = br_computer.compute_best_response_utility(1, strategy0, type_profiles)

    if verbose:
        print(f"    Best response utilities: P0={br_u0:.4f}, P1={br_u1:.4f}")

    # Exploitability is the gain from deviating
    exploit0 = max(0, br_u0 - current_u0)
    exploit1 = max(0, br_u1 - current_u1)
    total_exploit = exploit0 + exploit1

    return exploit0, exploit1, total_exploit


def main():
    """Demo the CFR solver."""
    print("CFR Nash Equilibrium Solver Demo")
    print("=" * 50)

    game = BargainingGame()

    print("\nTraining with External Sampling CFR+ (10000000 iterations)...")
    print("(Using action pruning + linear averaging + regret flooring)")
    solver = ExternalSamplingCFR(game)
    strategy0, strategy1 = solver.train(
        num_iterations=10000000,
        verbose=True,
        eval_every=200000  # Evaluate exploitability every 200000 iterations
    )

    print(f"\nStrategies computed:")
    print(f"  Player 0 info sets: {len(strategy0)}")
    print(f"  Player 1 info sets: {len(strategy1)}")

    # Save strategies (compressed)
    metadata = {
        "iterations": 10000000,
        "algorithm": "External Sampling CFR+",
        "p0_info_sets": len(strategy0),
        "p1_info_sets": len(strategy1)
    }
    save_strategies(strategy0, strategy1, "nash_equilibrium.json.gz", metadata)

    print("\nComputing exploitability (measures distance to CCE/Nash equilibrium)...")
    exploit0, exploit1, total_exploit = compute_exploitability(
        game, strategy0, strategy1, verbose=True
    )
    print(f"\nExploitability results:")
    print(f"  P0 can gain by deviating: {exploit0:.4f}")
    print(f"  P1 can gain by deviating: {exploit1:.4f}")
    print(f"  Total exploitability: {total_exploit:.4f}")
    epsilon = max(exploit0, exploit1)
    print(f"  ε = max(exploit0, exploit1) = {epsilon:.4f}")
    print(f"  This is a {epsilon:.4f}-CCE (and {epsilon:.4f}-Nash for independent strategies)")

    # Show some sample strategies
    print("\nSample strategy (first 5 info sets for P0):")
    for i, (info_set, probs) in enumerate(list(strategy0.items())[:5]):
        # Truncate info set for display
        short_info = info_set[:60] + "..." if len(info_set) > 60 else info_set
        # Show top actions
        top_actions = sorted(probs.items(), key=lambda x: -x[1])[:3]
        action_str = ", ".join(f"{a}:{p:.2f}" for a, p in top_actions)
        print(f"  {short_info}")
        print(f"    -> {action_str}")

    # Compare with uniform random baseline
    print("\n" + "=" * 50)
    print("Comparison with uniform random strategy:")
    uniform0 = {}  # Empty = uniform
    uniform1 = {}
    exploit0_u, exploit1_u, total_exploit_u = compute_exploitability(
        game, uniform0, uniform1, verbose=True
    )
    print(f"\nUniform strategy exploitability: {total_exploit_u:.4f}")
    print(f"CFR strategy exploitability: {total_exploit:.4f}")
    print(f"Improvement: {total_exploit_u - total_exploit:.4f}")


if __name__ == "__main__":
    main()
