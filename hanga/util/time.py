from functools import singledispatch


def to_s(minutes: int, seconds: float) -> float:
    return (minutes * 60) + seconds


@singledispatch
def to_ms():
    raise NotImplementedError


@to_ms.register
def _(minutes: int, seconds: float) -> float:
    return 1000 * to_s(minutes, seconds)


@to_ms.register
def _(seconds: float) -> float:
    return 1000 * seconds
