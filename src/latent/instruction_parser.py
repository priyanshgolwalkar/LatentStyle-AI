import os
import re
from dataclasses import dataclass
from typing import Optional

from src.utils.config import CLASS_NAMES


@dataclass(frozen=True)
class ParsedInstruction:
    source: Optional[int]
    target: Optional[int]
    alpha: Optional[float]
    source_text: str
    target_text: str


_ALIASES = {
    "tshirt": 0, "t-shirt": 0, "t shirt": 0,
    "trouser": 1, "trousers": 1, "pants": 1,
    "pullover": 2, "sweater": 2,
    "dress": 3,
    "coat": 4, "jacket": 4,
    "sandal": 5, "sandals": 5,
    "shirt": 6,
    "sneaker": 7, "shoe": 7, "sneakers": 7,
    "bag": 8, "handbag": 8,
    "ankle boot": 9, "ankle boots": 9, "boot": 9, "boots": 9,
}


def parse_instruction(text: str, default_alpha: float = 1.0) -> ParsedInstruction:
    """Parse classes and alpha with a deterministic local parser."""
    normalized = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    matches = []
    for alias, class_id in sorted(_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_normalized = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
        escaped = re.escape(alias_normalized)
        for match in re.finditer(rf"\b{escaped}\b", normalized):
            matches.append((match.start(), match.end(), alias, class_id))
    matches.sort(key=lambda item: item[0])

    filtered_matches = []
    for start, end, alias, class_id in matches:
        if any(
            start < other_end and end > other_start and class_id != other_id
            for other_start, other_end, _, other_id in filtered_matches
        ):
            continue
        filtered_matches.append((start, end, alias, class_id))

    # Collapse aliases that refer to the same class and preserve phrase order.
    ordered_ids = []
    for _, _, _, class_id in filtered_matches:
        if not ordered_ids or ordered_ids[-1] != class_id:
            ordered_ids.append(class_id)
    source = ordered_ids[0] if ordered_ids else None
    target = ordered_ids[1] if len(ordered_ids) > 1 else None

    alpha_match = re.search(
        r"(?:alpha|strength)\s*(?:=|:)?\s*(2(?:\.0+)?|1(?:\.[0-9]+)?|0(?:\.[0-9]+)?)",
        normalized,
    )
    alpha = float(alpha_match.group(1)) if alpha_match else default_alpha
    if not 0.0 <= alpha <= 2.0:
        raise ValueError("Editing strength must be between 0 and 2.")
    return ParsedInstruction(
        source=source, target=target, alpha=alpha,
        source_text=CLASS_NAMES[source] if source is not None else "",
        target_text=CLASS_NAMES[target] if target is not None else "",
    )
