"""Render optional tools without the checkpoint's mandatory-call preamble."""
import json


def render_prompt(tokenizer, task):
    messages = [dict(message) for message in task['messages']]
    if task['tools']:
        schemas = '\n\nAvailable tools (JSON schemas):\n' + json.dumps(task['tools'])
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] += schemas
        else:
            messages.insert(0, {'role': 'system', 'content': schemas.strip()})
    # Passing tools, even [], activates the checkpoint's forced-call instructions.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def loop_evidence(text, finish_reason):
    """Require both nontermination and substantial repeated reasoning text."""
    from collections import Counter

    words = text.lower().split()[-384:]
    phrases = [tuple(words[i:i + 12]) for i in range(max(0, len(words) - 11))]
    repeated_phrase_count = max(Counter(phrases).values(), default=0)
    grams = [tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    repeated_fraction = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
    return {
        'detected': finish_reason == 'length' and repeated_phrase_count >= 3 and repeated_fraction >= 0.30,
        'tail_repeated_fraction': repeated_fraction,
        'most_repeated_12_word_phrase_count': repeated_phrase_count,
    }
