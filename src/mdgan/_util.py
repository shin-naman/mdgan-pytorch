"""Small internal helpers shared across the package."""


def positive_int(name: str, value) -> int:
    """Return `value` if it is a positive int, else raise ValueError naming `name`.

    bool is rejected even though it subclasses int (True would otherwise pass as 1).
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value