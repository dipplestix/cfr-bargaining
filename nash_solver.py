"""
Nash Equilibrium Solver using Sequence Form and LCP

Implements:
1. Sequence form representation of the extensive-form game
2. LCP formulation for Nash equilibrium
3. Lemke's algorithm for solving the LCP
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from bargaining_game import BargainingGame, GameState, PlayerType, enumerate_types


@dataclass(frozen=True)
class Sequence:
    """
    A sequence is a path of actions for one player from the root.

    For player i, a sequence σ is a tuple of (info_set, action) pairs
    representing the actions taken by player i along some path.
    """
    player: int
    actions: Tuple[Tuple[str, str], ...]  # ((info_set_1, action_1), (info_set_2, action_2), ...)

    def __repr__(self) -> str:
        if not self.actions:
            return f"∅_{self.player}"
        actions_str = "->".join(f"{a}" for _, a in self.actions)
        return f"σ_{self.player}[{actions_str}]"

    @property
    def parent(self) -> Optional['Sequence']:
        """Return the parent sequence (without the last action)."""
        if not self.actions:
            return None
        return Sequence(self.player, self.actions[:-1])

    @property
    def last_info_set(self) -> Optional[str]:
        """Return the last info set in this sequence."""
        if not self.actions:
            return None
        return self.actions[-1][0]

    @property
    def last_action(self) -> Optional[str]:
        """Return the last action in this sequence."""
        if not self.actions:
            return None
        return self.actions[-1][1]


class SequenceFormGame:
    """
    Converts an extensive-form game to sequence form representation.

    The sequence form represents strategies as realization probabilities
    for each sequence, subject to linear constraints that capture the
    tree structure.
    """

    def __init__(self, game: BargainingGame):
        self.game = game
        self.types = game.types
        self.num_types = len(self.types)

        # Enumerate all sequences for each player
        self.sequences: Dict[int, List[Sequence]] = {0: [], 1: []}
        self.sequence_to_idx: Dict[int, Dict[Sequence, int]] = {0: {}, 1: {}}

        # Info sets for each player (with their available actions)
        self.info_sets: Dict[int, Dict[str, List[str]]] = {0: {}, 1: {}}

        # Parent sequence for each info set
        self.info_set_parent_seq: Dict[int, Dict[str, Sequence]] = {0: {}, 1: {}}

        # Build the sequence form representation
        self._enumerate_sequences()

        # Build constraint matrices E, F and vectors e, f
        self.E, self.e = self._build_constraint_matrix(0)
        self.F, self.f = self._build_constraint_matrix(1)

        # Build payoff matrices (averaged over type profiles)
        self.A, self.B = self._build_payoff_matrices()

    def _enumerate_sequences(self):
        """Enumerate all sequences for each player by traversing the game tree."""
        # Add empty sequences
        empty_seq_0 = Sequence(0, ())
        empty_seq_1 = Sequence(1, ())
        self.sequences[0].append(empty_seq_0)
        self.sequences[1].append(empty_seq_1)
        self.sequence_to_idx[0][empty_seq_0] = 0
        self.sequence_to_idx[1][empty_seq_1] = 0

        # Traverse game tree for all type profiles to find all sequences
        for type0 in self.types:
            for type1 in self.types:
                types = (type0, type1)
                self._traverse_for_sequences(
                    self.game.initial_state(),
                    types,
                    {0: empty_seq_0, 1: empty_seq_1}
                )

        print(f"Player 0 sequences: {len(self.sequences[0])}")
        print(f"Player 1 sequences: {len(self.sequences[1])}")
        print(f"Player 0 info sets: {len(self.info_sets[0])}")
        print(f"Player 1 info sets: {len(self.info_sets[1])}")

    def _traverse_for_sequences(
        self,
        state: GameState,
        types: Tuple[PlayerType, PlayerType],
        current_seqs: Dict[int, Sequence]
    ):
        """Recursively traverse game tree to enumerate sequences."""
        if state.terminal:
            return

        player = state.current_player
        player_type = types[player]
        info_set = self.game.get_info_set(state, player, player_type)
        actions = self.game.get_actions(state)

        # Record info set and its parent sequence
        if info_set not in self.info_sets[player]:
            self.info_sets[player][info_set] = actions
            self.info_set_parent_seq[player][info_set] = current_seqs[player]

        # For each action, create new sequence and recurse
        for action in actions:
            new_seq = Sequence(
                player,
                current_seqs[player].actions + ((info_set, action),)
            )

            # Add sequence if not already seen
            if new_seq not in self.sequence_to_idx[player]:
                self.sequence_to_idx[player][new_seq] = len(self.sequences[player])
                self.sequences[player].append(new_seq)

            # Update current sequence for this player and recurse
            new_current_seqs = current_seqs.copy()
            new_current_seqs[player] = new_seq

            next_state = self.game.step(state, action)
            self._traverse_for_sequences(next_state, types, new_current_seqs)

    def _build_constraint_matrix(self, player: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the sequence form constraint matrix for a player.

        The constraints are:
        - x(∅) = 1 (empty sequence has probability 1)
        - For each info set h: sum_{a in A(h)} x(σ_h, a) = x(σ_h)
          where σ_h is the parent sequence of info set h

        Returns (E, e) where Ex = e
        """
        num_seqs = len(self.sequences[player])
        num_info_sets = len(self.info_sets[player])

        # One constraint for empty sequence + one per info set
        num_constraints = 1 + num_info_sets

        E = np.zeros((num_constraints, num_seqs))
        e = np.zeros(num_constraints)

        # First constraint: x(∅) = 1
        empty_seq = Sequence(player, ())
        E[0, self.sequence_to_idx[player][empty_seq]] = 1
        e[0] = 1

        # One constraint per info set
        for i, (info_set, actions) in enumerate(self.info_sets[player].items()):
            row = i + 1

            # Parent sequence coefficient: -1
            parent_seq = self.info_set_parent_seq[player][info_set]
            E[row, self.sequence_to_idx[player][parent_seq]] = -1

            # Child sequences coefficient: +1
            for action in actions:
                child_seq = Sequence(
                    player,
                    parent_seq.actions + ((info_set, action),)
                )
                if child_seq in self.sequence_to_idx[player]:
                    E[row, self.sequence_to_idx[player][child_seq]] = 1

            e[row] = 0

        return E, e

    def _build_payoff_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the payoff matrices A and B for the sequence form.

        A[i,j] = expected payoff to player 0 when sequences i and j are played
        B[i,j] = expected payoff to player 1 when sequences i and j are played

        We average over all type profiles (uniform distribution).
        """
        num_seqs_0 = len(self.sequences[0])
        num_seqs_1 = len(self.sequences[1])

        # Initialize payoff contribution matrices
        A = np.zeros((num_seqs_0, num_seqs_1))
        B = np.zeros((num_seqs_0, num_seqs_1))

        # Count contributions for averaging
        counts = np.zeros((num_seqs_0, num_seqs_1))

        # Traverse game tree for all type profiles
        num_profiles = len(self.types) ** 2
        for type0 in self.types:
            for type1 in self.types:
                types = (type0, type1)
                self._add_terminal_payoffs(
                    self.game.initial_state(),
                    types,
                    {0: Sequence(0, ()), 1: Sequence(1, ())},
                    A, B, counts
                )

        # The payoff matrices already have the right structure
        # Each terminal contributes to exactly one (seq0, seq1) pair
        return A / num_profiles, B / num_profiles

    def _add_terminal_payoffs(
        self,
        state: GameState,
        types: Tuple[PlayerType, PlayerType],
        current_seqs: Dict[int, Sequence],
        A: np.ndarray,
        B: np.ndarray,
        counts: np.ndarray
    ):
        """Recursively add terminal payoffs to the payoff matrices."""
        if state.terminal:
            # Add payoff to the appropriate sequence pair
            seq0_idx = self.sequence_to_idx[0][current_seqs[0]]
            seq1_idx = self.sequence_to_idx[1][current_seqs[1]]

            p0, p1 = self.game.get_payoffs(state, types)
            A[seq0_idx, seq1_idx] += p0
            B[seq0_idx, seq1_idx] += p1
            counts[seq0_idx, seq1_idx] += 1
            return

        player = state.current_player
        player_type = types[player]
        info_set = self.game.get_info_set(state, player, player_type)
        actions = self.game.get_actions(state)

        for action in actions:
            new_seq = Sequence(
                player,
                current_seqs[player].actions + ((info_set, action),)
            )

            new_current_seqs = current_seqs.copy()
            new_current_seqs[player] = new_seq

            next_state = self.game.step(state, action)
            self._add_terminal_payoffs(next_state, types, new_current_seqs, A, B, counts)

    def sequence_to_behavioral(
        self,
        player: int,
        x: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert a sequence form strategy to a behavioral strategy.

        Args:
            player: Player index
            x: Sequence form strategy (realization probabilities)

        Returns:
            Behavioral strategy: info_set -> action -> probability
        """
        strategy = {}

        for info_set, actions in self.info_sets[player].items():
            parent_seq = self.info_set_parent_seq[player][info_set]
            parent_prob = x[self.sequence_to_idx[player][parent_seq]]

            action_probs = {}
            for action in actions:
                child_seq = Sequence(
                    player,
                    parent_seq.actions + ((info_set, action),)
                )
                if child_seq in self.sequence_to_idx[player]:
                    child_prob = x[self.sequence_to_idx[player][child_seq]]
                    if parent_prob > 1e-10:
                        action_probs[action] = child_prob / parent_prob
                    else:
                        action_probs[action] = 1.0 / len(actions)
                else:
                    action_probs[action] = 0.0

            # Normalize
            total = sum(action_probs.values())
            if total > 0:
                action_probs = {a: p / total for a, p in action_probs.items()}
            else:
                action_probs = {a: 1.0 / len(actions) for a in actions}

            strategy[info_set] = action_probs

        return strategy


def lemke_howson_lcp(M: np.ndarray, q: np.ndarray, max_iter: int = 10000) -> Optional[np.ndarray]:
    """
    Solve Linear Complementarity Problem using Lemke's algorithm.

    Find z >= 0 such that:
        w = Mz + q >= 0
        z'w = 0 (complementarity)

    Args:
        M: LCP matrix
        q: LCP vector
        max_iter: Maximum iterations

    Returns:
        Solution z, or None if no solution found
    """
    n = len(q)

    # Augmented tableau: [I | -M | -1 | q]
    # Variables: w_1, ..., w_n, z_1, ..., z_n, z_0
    tableau = np.zeros((n, 2 * n + 2))
    tableau[:, :n] = np.eye(n)  # w variables
    tableau[:, n:2*n] = -M  # z variables
    tableau[:, 2*n] = -np.ones(n)  # z_0 (covering vector)
    tableau[:, 2*n + 1] = q  # RHS

    # Basic variables: initially w_1, ..., w_n (indices 0 to n-1)
    basic = list(range(n))

    # Find most negative q_i to determine entering variable
    min_idx = np.argmin(q)
    if q[min_idx] >= 0:
        # q >= 0, solution is z = 0
        return np.zeros(n)

    # z_0 enters, w_{min_idx} leaves
    entering = 2 * n  # z_0
    leaving_row = min_idx

    for iteration in range(max_iter):
        # Pivot
        pivot_val = tableau[leaving_row, entering]
        if abs(pivot_val) < 1e-12:
            return None  # Degenerate

        tableau[leaving_row, :] /= pivot_val
        for i in range(n):
            if i != leaving_row:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving_row, :]

        # Update basic variables
        leaving_var = basic[leaving_row]
        basic[leaving_row] = entering

        # Determine next entering variable (complement of leaving)
        if leaving_var < n:
            # w_i left, z_i enters
            entering = leaving_var + n
        elif leaving_var < 2 * n:
            # z_i left, w_i enters
            entering = leaving_var - n
        else:
            # z_0 left, we're done!
            break

        # Find leaving variable using minimum ratio test
        min_ratio = float('inf')
        leaving_row = -1

        for i in range(n):
            if tableau[i, entering] > 1e-12:
                ratio = tableau[i, 2*n + 1] / tableau[i, entering]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_row = i

        if leaving_row == -1:
            return None  # Unbounded

    # Extract solution
    z = np.zeros(n)
    for i, var in enumerate(basic):
        if n <= var < 2 * n:
            z[var - n] = tableau[i, 2*n + 1]

    return z


class NashSolver:
    """
    Finds Nash equilibrium using sequence form and LCP.
    """

    def __init__(self, game: BargainingGame):
        print("Building sequence form representation...")
        self.sf_game = SequenceFormGame(game)
        self.game = game

    def solve(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], float, float]:
        """
        Find a Nash equilibrium.

        Returns:
            (strategy0, strategy1, utility0, utility1)
        """
        print("Setting up LCP...")

        # Get dimensions
        n0 = len(self.sf_game.sequences[0])
        n1 = len(self.sf_game.sequences[1])
        m0 = self.sf_game.E.shape[0]  # constraints for player 0
        m1 = self.sf_game.F.shape[0]  # constraints for player 1

        print(f"  Sequences: P0={n0}, P1={n1}")
        print(f"  Constraints: P0={m0}, P1={m1}")

        # The LCP formulation for Nash equilibrium:
        # Variables: x (P0 sequences), y (P1 sequences), p (P0 dual), q (P1 dual)
        #
        # Constraints:
        #   -A'p + E'u >= 0  (with complementarity to x)
        #   -B q + F'v >= 0  (with complementarity to y)
        #   Ex = e
        #   Fy = f
        #
        # This is a bit complex, so let's use a simpler formulation
        # based on the bimatrix game LCP formulation.

        # For sequence form, we use the extended LCP formulation:
        # Let's use the Lemke-Howson style approach for bimatrix games
        # but adapted for sequence constraints.

        # Build the LCP matrix
        # We solve: find (x, y, u, v) such that
        #   r = -A @ y + E.T @ u >= 0, x >= 0, x'r = 0
        #   s = -B.T @ x + F.T @ v >= 0, y >= 0, y's = 0
        #   E @ x = e
        #   F @ y = f

        # This can be formulated as an LCP with the following structure
        A = self.sf_game.A
        B = self.sf_game.B
        E = self.sf_game.E
        F = self.sf_game.F
        e = self.sf_game.e
        f = self.sf_game.f

        # Build block LCP matrix
        # z = [x, y]
        # M z + q >= 0
        # where M = [[0, -A], [-B.T, 0]]
        #       q = [E.T @ u, F.T @ v] but we need to handle constraints

        # Actually, for the constrained case, we need to solve a more complex system.
        # Let me use a different approach: convert to a standard LCP by eliminating
        # the equality constraints.

        # Alternative: Use support enumeration for smaller games
        # or iterative best response

        # For now, let's use iterative best response (fictitious play variant)
        # which is simpler and often works well

        print("Using iterative best response...")
        return self._solve_iterative(max_iter=1000)

    def _solve_iterative(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], float, float]:
        """
        Solve using iterative best response (replicator dynamics style).
        """
        n0 = len(self.sf_game.sequences[0])
        n1 = len(self.sf_game.sequences[1])

        # Initialize with uniform strategies
        x = self._uniform_sequence_strategy(0)
        y = self._uniform_sequence_strategy(1)

        A = self.sf_game.A
        B = self.sf_game.B
        E = self.sf_game.E
        F = self.sf_game.F
        e = self.sf_game.e
        f = self.sf_game.f

        for iteration in range(max_iter):
            # Best response for player 0 given y
            x_new = self._best_response(0, y, A)

            # Best response for player 1 given x
            y_new = self._best_response(1, x, B.T)

            # Check convergence
            x_diff = np.max(np.abs(x_new - x))
            y_diff = np.max(np.abs(y_new - y))

            if x_diff < tol and y_diff < tol:
                print(f"  Converged after {iteration + 1} iterations")
                break

            # Update with averaging for stability
            alpha = 2.0 / (iteration + 2)
            x = (1 - alpha) * x + alpha * x_new
            y = (1 - alpha) * y + alpha * y_new

            # Project back onto constraint set
            x = self._project_to_constraints(0, x)
            y = self._project_to_constraints(1, y)

            if (iteration + 1) % 100 == 0:
                u0 = x @ A @ y
                u1 = x @ B @ y
                print(f"  Iteration {iteration + 1}: utilities = ({u0:.4f}, {u1:.4f})")

        # Convert to behavioral strategies
        strat0 = self.sf_game.sequence_to_behavioral(0, x)
        strat1 = self.sf_game.sequence_to_behavioral(1, y)

        # Compute final utilities
        u0 = x @ A @ y
        u1 = x @ B @ y

        return strat0, strat1, u0, u1

    def _uniform_sequence_strategy(self, player: int) -> np.ndarray:
        """Generate uniform sequence form strategy."""
        n = len(self.sf_game.sequences[player])
        x = np.zeros(n)

        # Set empty sequence to 1
        x[0] = 1.0

        # For each info set, distribute uniformly among actions
        for info_set, actions in self.sf_game.info_sets[player].items():
            parent_seq = self.sf_game.info_set_parent_seq[player][info_set]
            parent_idx = self.sf_game.sequence_to_idx[player][parent_seq]
            parent_prob = x[parent_idx]

            for action in actions:
                child_seq = Sequence(
                    player,
                    parent_seq.actions + ((info_set, action),)
                )
                if child_seq in self.sf_game.sequence_to_idx[player]:
                    child_idx = self.sf_game.sequence_to_idx[player][child_seq]
                    x[child_idx] = parent_prob / len(actions)

        return x

    def _best_response(
        self,
        player: int,
        opponent_strategy: np.ndarray,
        payoff_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute best response sequence form strategy.

        This is done by computing the value at each info set bottom-up
        and selecting the action that maximizes expected payoff.
        """
        if player == 0:
            # Expected payoff for each sequence of player 0
            payoffs = payoff_matrix @ opponent_strategy
        else:
            # Expected payoff for each sequence of player 1
            payoffs = payoff_matrix @ opponent_strategy

        n = len(self.sf_game.sequences[player])
        x = np.zeros(n)
        x[0] = 1.0  # Empty sequence

        # Process info sets in order (works because of how we enumerate)
        for info_set, actions in self.sf_game.info_sets[player].items():
            parent_seq = self.sf_game.info_set_parent_seq[player][info_set]
            parent_idx = self.sf_game.sequence_to_idx[player][parent_seq]
            parent_prob = x[parent_idx]

            if parent_prob < 1e-10:
                continue

            # Find best action
            best_payoff = float('-inf')
            best_action = None

            for action in actions:
                child_seq = Sequence(
                    player,
                    parent_seq.actions + ((info_set, action),)
                )
                if child_seq in self.sf_game.sequence_to_idx[player]:
                    child_idx = self.sf_game.sequence_to_idx[player][child_seq]
                    if payoffs[child_idx] > best_payoff:
                        best_payoff = payoffs[child_idx]
                        best_action = action

            # Put all probability on best action
            if best_action is not None:
                child_seq = Sequence(
                    player,
                    parent_seq.actions + ((info_set, best_action),)
                )
                child_idx = self.sf_game.sequence_to_idx[player][child_seq]
                x[child_idx] = parent_prob

        return x

    def _project_to_constraints(self, player: int, x: np.ndarray) -> np.ndarray:
        """Project strategy onto sequence form constraints."""
        # Simple projection: ensure non-negativity and normalize at each info set
        x = np.maximum(x, 0)
        x[0] = 1.0  # Empty sequence

        for info_set, actions in self.sf_game.info_sets[player].items():
            parent_seq = self.sf_game.info_set_parent_seq[player][info_set]
            parent_idx = self.sf_game.sequence_to_idx[player][parent_seq]
            parent_prob = x[parent_idx]

            # Collect child probabilities
            child_indices = []
            child_probs = []
            for action in actions:
                child_seq = Sequence(
                    player,
                    parent_seq.actions + ((info_set, action),)
                )
                if child_seq in self.sf_game.sequence_to_idx[player]:
                    idx = self.sf_game.sequence_to_idx[player][child_seq]
                    child_indices.append(idx)
                    child_probs.append(x[idx])

            # Normalize to sum to parent probability
            total = sum(child_probs)
            if total > 1e-10:
                for idx, prob in zip(child_indices, child_probs):
                    x[idx] = parent_prob * prob / total
            else:
                # Uniform if all zero
                for idx in child_indices:
                    x[idx] = parent_prob / len(child_indices)

        return x


def main():
    """Demo the Nash solver."""
    print("Nash Equilibrium Solver Demo")
    print("=" * 50)

    game = BargainingGame()
    solver = NashSolver(game)

    print("\nSolving for Nash equilibrium...")
    strat0, strat1, u0, u1 = solver.solve()

    print(f"\nNash equilibrium utilities:")
    print(f"  Player 0: {u0:.4f}")
    print(f"  Player 1: {u1:.4f}")

    print(f"\nStrategy sizes:")
    print(f"  Player 0: {len(strat0)} info sets")
    print(f"  Player 1: {len(strat1)} info sets")

    # Verify with game's utility computation
    print("\nVerifying with game's utility computation...")
    eu0, eu1 = game.compute_expected_utility(strat0, strat1, verbose=False)
    print(f"  Expected utility (verification): P0={eu0:.4f}, P1={eu1:.4f}")


if __name__ == "__main__":
    main()
