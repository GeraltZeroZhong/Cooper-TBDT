from __future__ import annotations


def format_bin_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def build_bin_ranges(edges: list[float], last_label: str = "gt") -> list[tuple[float, float | None, str]]:
    if len(edges) < 2:
        raise ValueError("Bin edges must include at least 2 values.")
    normalized = [float(edge) for edge in edges]
    if any(high <= low for low, high in zip(normalized[:-1], normalized[1:])):
        raise ValueError("Bin edges must be strictly increasing.")

    ranges: list[tuple[float, float | None, str]] = []
    for low, high in zip(normalized[:-1], normalized[1:]):
        ranges.append((low, high, f"{format_bin_value(low)}to{format_bin_value(high)}"))
    ranges.append((normalized[-1], None, f"{last_label}{format_bin_value(normalized[-1])}"))
    return ranges


def parse_float_edges(raw: str) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError("At least two bin edges are required.")
    if any(high <= low for low, high in zip(values[:-1], values[1:])):
        raise ValueError("Bin edges must be strictly increasing.")
    return values
