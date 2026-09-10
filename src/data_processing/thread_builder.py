"""Multi-turn conversation thread parser."""
import re
from typing import Dict, List, Optional, Tuple
from src.data_processing.cleaner import clean_tweet_text, is_valid_text


def parse_conversation_thread(raw_conversation: str) -> Optional[Dict[str, str]]:
    """
    Parses a multi-turn conversation string containing 'Customer:' and 'Support:' turns.
    Extracts:
    - customer_message: The initial customer inquiry
    - support_response: The brand's direct resolution or response
    - context: Any prior/subsequent turns for multi-turn awareness
    - full_dialogue: Cleaned complete dialogue
    """
    if not raw_conversation or not isinstance(raw_conversation, str):
        return None

    # Split turns by Customer: or Support:
    turns = re.split(r"(?=(?:Customer|Support):)", raw_conversation.strip())
    customer_turns = []
    support_turns = []

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        if turn.startswith("Customer:"):
            msg = turn[len("Customer:"):].strip()
            cleaned = clean_tweet_text(msg)
            if cleaned:
                customer_turns.append(cleaned)
        elif turn.startswith("Support:"):
            msg = turn[len("Support:"):].strip()
            cleaned = clean_tweet_text(msg)
            if cleaned:
                support_turns.append(cleaned)

    if not customer_turns or not support_turns:
        return None

    initial_customer_msg = customer_turns[0]
    initial_support_response = support_turns[0]

    # Validate length and content
    if not is_valid_text(initial_customer_msg, min_chars=15) or not is_valid_text(initial_support_response, min_chars=10):
        return None

    # Build context if multi-turn
    context = ""
    if len(customer_turns) > 1 or len(support_turns) > 1:
        context_parts = []
        for i, ct in enumerate(customer_turns[:2]):
            context_parts.append(f"Customer: {ct}")
            if i < len(support_turns):
                context_parts.append(f"Support: {support_turns[i]}")
        context = " | ".join(context_parts)

    return {
        "customer_message": initial_customer_msg,
        "support_response": initial_support_response,
        "context": context if context else f"Customer: {initial_customer_msg}",
        "num_customer_turns": len(customer_turns),
        "num_support_turns": len(support_turns)
    }
