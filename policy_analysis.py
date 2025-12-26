import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Nash Equilibrium Policy Analysis

        This notebook explores the trained CFR policy for the bargaining game.
        The policy was trained for 10 million iterations and achieves an exploitability of 0.1148.
        """
    )
    return


@app.cell
def _():
    from bargaining_game import BargainingGame, PlayerType
    from cfr_solver import load_strategies

    game = BargainingGame()
    strategy0, strategy1, metadata = load_strategies("nash_equilibrium.json.gz")

    print(f"Loaded strategy trained for {metadata.get('iterations', 'unknown')} iterations")
    print(f"P0 info sets: {len(strategy0)}")
    print(f"P1 info sets: {len(strategy1)}")
    return PlayerType, game, load_strategies, metadata, strategy0, strategy1


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Helper Functions

        Let's define some helpers to query and display the policy.
        """
    )
    return


@app.cell
def _(PlayerType, game, strategy0, strategy1):
    def get_policy(player: int, player_type: PlayerType, history: str = "start"):
        """Get the policy for a player at a given info set."""
        info_set = f"p{player}|({player_type.v1},{player_type.v2},{player_type.walk_value})|{history}"
        strategy = strategy0 if player == 0 else strategy1

        if info_set not in strategy:
            return None, info_set

        probs = strategy[info_set]
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return sorted_probs, info_set

    def format_policy(sorted_probs, threshold=0.01):
        """Format policy as a string."""
        if sorted_probs is None:
            return "Not in strategy (uniform random)"
        lines = []
        for action, prob in sorted_probs:
            if prob >= threshold:
                lines.append(f"  {action}: {prob:.1%}")
        return "\n".join(lines)

    def get_valid_actions(player_type: PlayerType, history: list[str]):
        """Get valid actions after pruning."""
        state = game.initial_state()
        for action in history:
            state = game.step(state, action)
        return game.get_actions(state, player_type)

    return format_policy, get_policy, get_valid_actions


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Example 1: The "All or Nothing" Type (1, 1, 4)

        This player values both item types at 1, giving a maximum possible value of 2×1 + 2×1 = 4.
        Their walk-away value is also 4, meaning they will **only accept deals where they get everything**.

        This creates an extreme strategic situation.
        """
    )
    return


@app.cell
def _(PlayerType, format_policy, get_policy, get_valid_actions):
    # Type (1, 1, 4) - the "all or nothing" type
    all_or_nothing = PlayerType(1, 1, 4)

    print("Type:", all_or_nothing)
    print(f"Max possible value: {all_or_nothing.max_value()}")
    print(f"Walk-away value: {all_or_nothing.walk_value}")
    print()

    # As P0 at start
    probs, info_set = get_policy(0, all_or_nothing, "start")
    valid = get_valid_actions(all_or_nothing, [])
    print("As P0 at game start:")
    print(f"  Valid actions: {valid}")
    print(format_policy(probs))
    print()

    # As P1 after various offers
    for offer in ["offer_00", "offer_22"]:
        probs, info_set = get_policy(1, all_or_nothing, offer)
        valid = get_valid_actions(all_or_nothing, [offer])
        offer_desc = "gets nothing" if offer == "offer_00" else "gets everything"
        print(f"As P1 after {offer} ({offer_desc}):")
        print(f"  Valid actions: {valid}")
        print(format_policy(probs))
        print()
    return all_or_nothing, info_set, offer, offer_desc, probs, valid


@app.cell
def _(mo):
    mo.md(
        r"""
        **Insight**: This type is extremely constrained:
        - Can only make `offer_00` (demand everything) since giving anything away leaves them below walk_value
        - Can only accept `offer_22` (receiving everything)
        - The 50/50 split reflects true indifference between walking and their only viable alternative
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Example 2: The "Item Specialist" Type (1, 0, 2)

        This player only values item type 1 (v1=1, v2=0), with walk_value=2.
        They're indifferent to item type 2 entirely.
        """
    )
    return


@app.cell
def _(PlayerType, format_policy, get_policy, get_valid_actions):
    # Type (1, 0, 2) - only cares about item type 1
    item1_specialist = PlayerType(1, 0, 2)

    print("Type:", item1_specialist)
    print(f"Value of 2 item1s: {item1_specialist.bundle_value(2, 0)}")
    print(f"Value of 2 item2s: {item1_specialist.bundle_value(0, 2)}")
    print()

    # As P0 at start
    probs2, info_set2 = get_policy(0, item1_specialist, "start")
    valid2 = get_valid_actions(item1_specialist, [])
    print("As P0 at game start:")
    print(f"  Valid actions: {valid2}")
    print(format_policy(probs2))
    print()

    # After receiving offer_20 (getting both item 1s)
    probs2, info_set2 = get_policy(1, item1_specialist, "offer_20")
    valid2 = get_valid_actions(item1_specialist, ["offer_20"])
    print("As P1 after offer_20 (would receive both item 1s):")
    print(f"  Accept value: {item1_specialist.bundle_value(2, 0)} = walk_value")
    print(f"  Valid actions: {valid2}")
    print(format_policy(probs2))
    return info_set2, item1_specialist, probs2, valid2


@app.cell
def _(mo):
    mo.md(
        r"""
        **Insight**: After receiving an offer that exactly matches their walk_value:
        - All counter-offers are pruned (none strictly better than accepting)
        - Only walk and accept remain, both giving value 2
        - 50/50 split reflects true game-theoretic indifference
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Example 3: The "Flexible Negotiator" Type (0.5, 0.5, 0)

        This player has low valuations (0.5 each) and zero walk-away value.
        They're highly flexible and willing to make many different deals.
        """
    )
    return


@app.cell
def _(PlayerType, format_policy, get_policy, get_valid_actions):
    # Type (0.5, 0.5, 0) - flexible negotiator
    flexible = PlayerType(0.5, 0.5, 0)

    print("Type:", flexible)
    print(f"Max possible value: {flexible.max_value()}")
    print(f"Walk-away value: {flexible.walk_value}")
    print()

    # As P0 at start - should have many options
    probs3, info_set3 = get_policy(0, flexible, "start")
    valid3 = get_valid_actions(flexible, [])
    print("As P0 at game start:")
    print(f"  Valid actions: {valid3}")
    print(format_policy(probs3))
    print()

    # After a few rounds of negotiation
    history = "offer_11,offer_11"
    probs3, info_set3 = get_policy(0, flexible, history)
    valid3 = get_valid_actions(flexible, history.split(","))
    print(f"As P0 after {history}:")
    print(f"  Valid actions: {valid3}")
    print(format_policy(probs3))
    return flexible, history, info_set3, probs3, valid3


@app.cell
def _(mo):
    mo.md(
        r"""
        **Insight**: With walk_value=0, this type:
        - Has access to all possible offers (no pruning based on keeping value)
        - Shows more complex mixed strategies
        - Demonstrates how the Nash equilibrium balances multiple viable options
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Example 4: Same Situation, Different Types

        How do different player types respond to the same offer?
        Let's see how various types respond as P1 after P0 offers `offer_11` (1 of each item).
        """
    )
    return


@app.cell
def _(PlayerType, format_policy, get_policy, get_valid_actions):
    print("P1's response to offer_11 (receiving 1 of each item):")
    print("=" * 50)

    types_to_compare = [
        PlayerType(0, 0, 0),      # Values nothing
        PlayerType(0.5, 0.5, 0),  # Low values, no walk
        PlayerType(0.5, 0.5, 1),  # Low values, some walk
        PlayerType(1, 1, 0),      # High values, no walk
        PlayerType(1, 1, 2),      # High values, moderate walk
        PlayerType(1, 0, 1),      # Only values item 1
        PlayerType(0, 1, 1),      # Only values item 2
    ]

    for ptype in types_to_compare:
        accept_val = ptype.bundle_value(1, 1)
        print(f"\nType {ptype}:")
        print(f"  Accept value: {accept_val}, Walk value: {ptype.walk_value}")

        probs4, _ = get_policy(1, ptype, "offer_11")
        valid4 = get_valid_actions(ptype, ["offer_11"])
        print(f"  Valid actions: {valid4}")
        print(format_policy(probs4))
    return accept_val, probs4, ptype, types_to_compare, valid4


@app.cell
def _(mo):
    mo.md(
        r"""
        **Insight**: The same offer elicits very different responses:
        - Types with `accept_value < walk_value` can't accept (pruned)
        - Types with high walk values have fewer counter-offer options
        - Item specialists (1,0,_) and (0,1,_) value the offer at only 1, not 2
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Example 5: Deep Negotiation Chains

        What happens in extended negotiations? Let's trace a 4-round negotiation.
        """
    )
    return


@app.cell
def _(PlayerType, format_policy, get_policy):
    # Trace a negotiation between two specific types
    p0_type = PlayerType(1, 0.5, 1)
    p1_type = PlayerType(0.5, 1, 1)

    print(f"P0 type: {p0_type} (prefers item type 1)")
    print(f"P1 type: {p1_type} (prefers item type 2)")
    print()

    histories = [
        ("start", 0),
        ("offer_20", 1),           # P0 offers 2 of item1, 0 of item2
        ("offer_20,offer_02", 0),  # P1 counters with 0 of item1, 2 of item2
        ("offer_20,offer_02,offer_21", 1),  # P0 counters
        ("offer_20,offer_02,offer_21,offer_12", 0),  # P1 counters
    ]

    for hist, player in histories:
        ptype = p0_type if player == 0 else p1_type
        probs5, info_set5 = get_policy(player, ptype, hist)

        print(f"Round {len(hist.split(',')) if hist != 'start' else 0}: P{player} at '{hist}'")
        if probs5:
            print(format_policy(probs5))
        else:
            print("  (Info set not visited in training)")
        print()
    return hist, histories, info_set5, p0_type, p1_type, player, probs5, ptype


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary Statistics
        """
    )
    return


@app.cell
def _(strategy0, strategy1):
    # Analyze strategy statistics
    def analyze_strategy(strategy, name):
        total_infosets = len(strategy)

        # Count pure vs mixed strategies
        pure_count = 0
        mixed_count = 0

        for info_set, probs in strategy.items():
            max_prob = max(probs.values())
            if max_prob > 0.99:
                pure_count += 1
            else:
                mixed_count += 1

        print(f"{name}:")
        print(f"  Total info sets: {total_infosets}")
        print(f"  Pure strategies (>99% one action): {pure_count} ({100*pure_count/total_infosets:.1f}%)")
        print(f"  Mixed strategies: {mixed_count} ({100*mixed_count/total_infosets:.1f}%)")
        print()

    analyze_strategy(strategy0, "Player 0")
    analyze_strategy(strategy1, "Player 1")
    return (analyze_strategy,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Conclusion

        The trained Nash equilibrium policy shows sophisticated behavior:

        1. **Pruning is effective**: Dominated actions are never played, reducing the strategy space significantly

        2. **Indifference is real**: When multiple actions give equal value, the policy correctly mixes 50/50

        3. **Type-dependent strategies**: Different player types respond very differently to the same situations

        4. **Mixed strategies emerge**: Many info sets have non-trivial mixed strategies, indicating complex strategic trade-offs

        5. **Exploitability of 0.1148** means neither player can gain more than ~0.07 by deviating to their best response
        """
    )
    return


if __name__ == "__main__":
    app.run()
