RANDOM_SEED = 20251208


class Actions:
    # column names in table
    fold = "Fold"
    check_call = "CheckCall"
    bet_raise = "BetRaise"

    @staticmethod
    def get_all_actions() -> list[str]:
        return [Actions.fold, Actions.check_call, Actions.bet_raise]
