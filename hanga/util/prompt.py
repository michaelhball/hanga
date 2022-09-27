from typing import List, Optional, Union


def construct_prompts(
    prompts: Union[str, List[str]],
    prefixes: Optional[List[str]] = None,
    suffixes: Optional[List[str]] = None,
):
    """"""

    if isinstance(prompts, str):
        prompts = [prompts]
    prefixes = prefixes or []
    suffixes = suffixes or []

    _prompts = []
    for prompt in prompts:
        if len(prefixes) == 0 and len(suffixes) == 0:
            _prompts.append(prompt)
        elif len(prefixes) != 0 and len(suffixes) != 0:
            for prefix in prefixes:
                for suffix in suffixes:
                    _prompts.append(f"{prefix} {prompt} {suffix}")
        elif len(prefixes) != 0:
            for prefix in prefixes:
                _prompts.append(f"{prefix} {prompt}")
        else:
            for suffix in suffixes:
                _prompts.append(f"{prompt} {suffix}")

    return _prompts
