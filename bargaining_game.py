"""
Bargaining Game Implementation

A two-player bargaining game where players negotiate over items with private valuations.
Supports sequence form strategies and expected utility computation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import itertools


# Constants
ITEM_VALUES = [0, 0.5, 1]  # Possible item valuations
OFFERS = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 2), (1, 2), (2, 1)]
TOTAL_ITEMS = (2, 2)  # 2 of each item type
NUM_ROUNDS = 6  # Total decision points


class Action(Enum):
    """Possible actions in the game."""
    WALK = "walk"
    ACCEPT = "accept"
    # Offers are represented as "offer_XY" where X is type1 count, Y is type2 count


@dataclass(frozen=True)
class PlayerType:
    """Represents a player's private type (valuations and walk-away value)."""
    v1: float  # Value for item type 1
    v2: float  # Value for item type 2
    walk_value: int  # Walk-away value (discrete)

    def bundle_value(self, n1: int, n2: int) -> float:
        """Compute value of a bundle with n1 items of type 1 and n2 items of type 2."""
        return self.v1 * n1 + self.v2 * n2

    def max_value(self) -> float:
        """Maximum possible value (getting all items)."""
        return self.bundle_value(2, 2)

    def __repr__(self) -> str:
        return f"Type(v1={self.v1}, v2={self.v2}, walk={self.walk_value})"


@dataclass
class GameState:
    """Represents the current state of the bargaining game."""
    round_num: int  # Current round (0-5)
    history: Tuple[str, ...]  # Sequence of actions taken
    terminal: bool = False
    outcome: Optional[Tuple] = None  # ('walk',) or ('accept', accepting_player, offer)

    @property
    def current_player(self) -> int:
        """Returns which player acts at this state (0 or 1)."""
        return self.round_num % 2

    @property
    def last_offer(self) -> Optional[Tuple[int, int]]:
        """Returns the last offer made, if any."""
        for action in reversed(self.history):
            if action.startswith("offer_"):
                offer_str = action[6:]  # Remove "offer_" prefix
                return (int(offer_str[0]), int(offer_str[1]))
        return None

    def copy(self) -> 'GameState':
        """Create a copy of this state."""
        return GameState(
            round_num=self.round_num,
            history=self.history,
            terminal=self.terminal,
            outcome=self.outcome
        )


def enumerate_types() -> List[PlayerType]:
    """Enumerate all possible player types (finite set of 27)."""
    types = []
    for v1 in ITEM_VALUES:
        for v2 in ITEM_VALUES:
            total = int(2 * v1 + 2 * v2)  # {0, 1, 2, 3, 4}
            for walk in range(total + 1):  # {0, ..., total}
                types.append(PlayerType(v1, v2, walk))
    return types


class BargainingGame:
    """
    Two-player bargaining game with private valuations.

    Game structure:
    - 2 item types, 2 of each (4 items total)
    - Each player has private values for each item type in {0, 0.5, 1}
    - Each player has a walk-away value in {0, 1, ..., total_value}
    - Players alternate: P1, P2, P1, P2, P1, P2 (6 rounds)
    - Round 1: P1 makes offer
    - Rounds 2-5: Accept, Walk, or Counter-offer
    - Round 6: Accept or Walk only
    """

    def __init__(self):
        self.types = enumerate_types()
        self.offers = OFFERS
        self.num_rounds = NUM_ROUNDS

    def initial_state(self) -> GameState:
        """Return the initial game state."""
        return GameState(round_num=0, history=())

    def get_actions(
        self,
        state: GameState,
        player_type: Optional[PlayerType] = None
    ) -> List[str]:
        """
        Get valid actions at the current state.

        Args:
            state: Current game state
            player_type: If provided, prune dominated actions based on type

        Returns list of action strings:
        - "walk": Walk away from negotiation
        - "accept": Accept the last offer (not available in round 0)
        - "offer_XY": Offer X items of type 1 and Y items of type 2 to opponent
        """
        if state.terminal:
            return []

        actions = ["walk"]

        # Accept is available if there's a previous offer to accept
        if state.round_num > 0 and state.last_offer is not None:
            # If we know our type, only include accept if it beats walking
            if player_type is not None:
                offer = state.last_offer
                accept_value = player_type.bundle_value(offer[0], offer[1])
                if accept_value >= player_type.walk_value:
                    actions.append("accept")
            else:
                actions.append("accept")

        # Offers are available in all rounds except the last
        if state.round_num < NUM_ROUNDS - 1:
            for offer in self.offers:
                # If we know our type, only include offers where we keep >= walk_value
                if player_type is not None:
                    # We keep the remainder after giving away the offer
                    keep_n1 = TOTAL_ITEMS[0] - offer[0]
                    keep_n2 = TOTAL_ITEMS[1] - offer[1]
                    keep_value = player_type.bundle_value(keep_n1, keep_n2)
                    if keep_value >= player_type.walk_value:
                        actions.append(f"offer_{offer[0]}{offer[1]}")
                else:
                    actions.append(f"offer_{offer[0]}{offer[1]}")

        return actions

    def step(self, state: GameState, action: str) -> GameState:
        """
        Apply an action and return the new state.

        Args:
            state: Current game state
            action: Action to take ("walk", "accept", or "offer_XY")

        Returns:
            New game state after the action
        """
        if state.terminal:
            raise ValueError("Cannot take action in terminal state")

        new_history = state.history + (action,)

        if action == "walk":
            return GameState(
                round_num=state.round_num,
                history=new_history,
                terminal=True,
                outcome=("walk",)
            )

        if action == "accept":
            last_offer = state.last_offer
            if last_offer is None:
                raise ValueError("Cannot accept without a previous offer")
            return GameState(
                round_num=state.round_num,
                history=new_history,
                terminal=True,
                outcome=("accept", state.current_player, last_offer)
            )

        # Must be an offer
        if not action.startswith("offer_"):
            raise ValueError(f"Invalid action: {action}")

        # Check if this is the last round (no offers allowed)
        if state.round_num >= NUM_ROUNDS - 1:
            raise ValueError("Cannot make offer in last round")

        return GameState(
            round_num=state.round_num + 1,
            history=new_history,
            terminal=False,
            outcome=None
        )

    def get_payoffs(
        self, state: GameState, types: Tuple[PlayerType, PlayerType]
    ) -> Tuple[float, float]:
        """
        Get payoffs for both players at a terminal state.

        Args:
            state: Terminal game state
            types: Tuple of (player0_type, player1_type)

        Returns:
            Tuple of (player0_payoff, player1_payoff)
        """
        if not state.terminal:
            raise ValueError("Can only get payoffs at terminal state")

        if state.outcome[0] == "walk":
            # Both players get their walk-away value
            return (float(types[0].walk_value), float(types[1].walk_value))

        # Accept case
        _, accepting_player, offer = state.outcome
        offering_player = 1 - accepting_player

        # Offer specifies what the offerer gives to the opponent
        # Accepter gets the offer, offerer gets the remainder
        offer_n1, offer_n2 = offer
        remainder_n1 = TOTAL_ITEMS[0] - offer_n1
        remainder_n2 = TOTAL_ITEMS[1] - offer_n2

        if accepting_player == 0:
            # Player 0 accepted, gets the offer
            p0_payoff = types[0].bundle_value(offer_n1, offer_n2)
            p1_payoff = types[1].bundle_value(remainder_n1, remainder_n2)
        else:
            # Player 1 accepted, gets the offer
            p0_payoff = types[0].bundle_value(remainder_n1, remainder_n2)
            p1_payoff = types[1].bundle_value(offer_n1, offer_n2)

        return (p0_payoff, p1_payoff)

    def get_info_set(
        self, state: GameState, player: int, player_type: PlayerType
    ) -> str:
        """
        Encode the information set for a player at a given state.

        Information set includes:
        - Player's own type (v1, v2, walk_value)
        - Full history of actions (all public)

        Args:
            state: Current game state
            player: Player index (0 or 1)
            player_type: The player's private type

        Returns:
            String encoding of the information set
        """
        type_str = f"({player_type.v1},{player_type.v2},{player_type.walk_value})"
        history_str = ",".join(state.history) if state.history else "start"
        return f"p{player}|{type_str}|{history_str}"

    def play_with_strategies(
        self,
        strategy0: Dict[str, Dict[str, float]],
        strategy1: Dict[str, Dict[str, float]],
        types: Tuple[PlayerType, PlayerType],
        state: Optional[GameState] = None
    ) -> Tuple[float, float]:
        """
        Compute expected payoffs for a fixed type profile with given strategies.

        Uses recursive tree traversal, weighting by action probabilities.

        Args:
            strategy0: Player 0's behavioral strategy (info_set -> action -> prob)
            strategy1: Player 1's behavioral strategy
            types: Fixed type profile (player0_type, player1_type)
            state: Current state (defaults to initial state)

        Returns:
            Expected payoffs (player0, player1) for this type profile
        """
        if state is None:
            state = self.initial_state()

        if state.terminal:
            return self.get_payoffs(state, types)

        current_player = state.current_player
        strategy = strategy0 if current_player == 0 else strategy1
        player_type = types[current_player]

        info_set = self.get_info_set(state, current_player, player_type)
        # Use pruned actions based on player type
        actions = self.get_actions(state, player_type)

        # Get action probabilities from strategy
        if info_set not in strategy:
            # Default to uniform random if info set not in strategy
            action_probs = {a: 1.0 / len(actions) for a in actions}
        else:
            action_probs = strategy[info_set]
            # Normalize in case probabilities don't sum to 1
            total = sum(action_probs.get(a, 0) for a in actions)
            if total == 0:
                action_probs = {a: 1.0 / len(actions) for a in actions}
            else:
                action_probs = {a: action_probs.get(a, 0) / total for a in actions}

        # Compute expected payoff over all actions
        expected_p0 = 0.0
        expected_p1 = 0.0

        for action in actions:
            prob = action_probs.get(action, 0)
            if prob > 0:
                next_state = self.step(state, action)
                p0, p1 = self.play_with_strategies(
                    strategy0, strategy1, types, next_state
                )
                expected_p0 += prob * p0
                expected_p1 += prob * p1

        return (expected_p0, expected_p1)

    def compute_expected_utility(
        self,
        strategy0: Dict[str, Dict[str, float]],
        strategy1: Dict[str, Dict[str, float]],
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Compute ex-ante expected utilities by integrating over all type profiles.

        Assumes uniform distribution over types.

        Args:
            strategy0: Player 0's behavioral strategy
            strategy1: Player 1's behavioral strategy
            verbose: If True, print progress updates

        Returns:
            Expected utilities (player0, player1)
        """
        total_p0 = 0.0
        total_p1 = 0.0
        num_profiles = 0
        total_profiles = len(self.types) ** 2

        for i, type0 in enumerate(self.types):
            for type1 in self.types:
                types = (type0, type1)
                p0, p1 = self.play_with_strategies(strategy0, strategy1, types)
                total_p0 += p0
                total_p1 += p1
                num_profiles += 1

            if verbose and (i + 1) % 9 == 0:
                print(f"  Progress: {num_profiles}/{total_profiles} profiles")

        # Average over all type profiles (uniform distribution)
        return (total_p0 / num_profiles, total_p1 / num_profiles)

    def enumerate_info_sets(self) -> Dict[int, List[str]]:
        """
        Enumerate all information sets for each player.

        Returns:
            Dictionary mapping player index to list of info set strings
        """
        info_sets = {0: set(), 1: set()}

        def traverse(state: GameState, types: Tuple[PlayerType, PlayerType]):
            if state.terminal:
                return

            player = state.current_player
            info_set = self.get_info_set(state, player, types[player])
            info_sets[player].add(info_set)

            for action in self.get_actions(state):
                next_state = self.step(state, action)
                traverse(next_state, types)

        # Traverse for all type profiles
        for type0 in self.types:
            for type1 in self.types:
                traverse(self.initial_state(), (type0, type1))

        return {p: sorted(list(s)) for p, s in info_sets.items()}

    def get_uniform_strategy(self, player: int) -> Dict[str, Dict[str, float]]:
        """
        Generate a uniform random strategy for a player.

        Returns a strategy that plays uniformly at random at every info set.
        """
        info_sets = self.enumerate_info_sets()[player]
        strategy = {}

        # For each info set, we need to know what actions are available
        # We'll traverse the game tree to find this
        info_set_actions = {}

        def traverse(state: GameState, player_type: PlayerType):
            if state.terminal:
                return

            current_player = state.current_player
            if current_player == player:
                info_set = self.get_info_set(state, player, player_type)
                actions = self.get_actions(state)
                if info_set not in info_set_actions:
                    info_set_actions[info_set] = set(actions)

            for action in self.get_actions(state):
                next_state = self.step(state, action)
                traverse(next_state, player_type)

        for player_type in self.types:
            traverse(self.initial_state(), player_type)

        for info_set, actions in info_set_actions.items():
            n = len(actions)
            strategy[info_set] = {a: 1.0 / n for a in actions}

        return strategy


def main():
    """Demo the bargaining game."""
    game = BargainingGame()

    print("Bargaining Game Demo")
    print("=" * 50)
    print(f"Number of player types: {len(game.types)}")
    print(f"Number of type profiles: {len(game.types) ** 2}")
    print(f"Number of offers: {len(game.offers)}")
    print(f"Number of rounds: {game.num_rounds}")

    print("\nSample types:")
    for t in game.types[:5]:
        print(f"  {t}")
    print("  ...")

    print("\nSample game play:")
    state = game.initial_state()
    types = (PlayerType(1, 0.5, 2), PlayerType(0.5, 1, 2))
    print(f"Types: P0={types[0]}, P1={types[1]}")

    actions_taken = []
    while not state.terminal:
        player = state.current_player
        actions = game.get_actions(state)
        print(f"  Round {state.round_num}, P{player} to act, actions: {actions}")
        action = actions[1] if len(actions) > 1 else actions[0]  # Pick second action for demo
        actions_taken.append(action)
        print(f"    -> P{player} chooses: {action}")
        state = game.step(state, action)

    print(f"  Game over: {state.outcome}")
    payoffs = game.get_payoffs(state, types)
    print(f"  Payoffs: P0={payoffs[0]}, P1={payoffs[1]}")

    print("\nTesting play_with_strategies for single type profile...")
    strat0 = {}  # Empty = uniform random
    strat1 = {}
    eu0, eu1 = game.play_with_strategies(strat0, strat1, types)
    print(f"  Expected utility with uniform strategies: P0={eu0:.4f}, P1={eu1:.4f}")

    print("\nComputing full expected utility over all 729 type profiles...")
    eu0, eu1 = game.compute_expected_utility(strat0, strat1, verbose=True)
    print(f"Ex-ante expected utility: P0={eu0:.4f}, P1={eu1:.4f}")


if __name__ == "__main__":
    main()
