def to_s(minutes: float, seconds: float) -> float:
    return (minutes * 60) + seconds


def s_to_ms(seconds: float) -> float:
    return 1000 * seconds


def to_ms(minutes: float, seconds: float):
    return 1000 * to_s(minutes, seconds)
