#!/usr/bin/env python3
"""Sync vLLM's DeepSeek V4.1 chat encoder with DeepSeek's updated reference encoder (shipped as
encoding/encoding.py in DeepSeek-V4.1-Flash-1017 and in the base repo since commit 314113f):
  * a user message holding only tool results no longer starts a new turn, so the reasoning of the
    assistant message that issued the tool calls is kept while their results come back (before, it
    was dropped mid-turn when no `tools` were declared);
  * tool results given as content-block lists stay structured; list-valued user content is kept;
  * the context is merged/sorted in the same order as the reference, and the last-user index is
    computed once over the full message list;
  * a literal <｜deepseek_image｜> in message text is a request error (400), not an image slot.
Upstream: vllm-project/vllm#62196 and #62199 (open). Anchors target the 0909 image's files.
usage: <tokenizers/deepseek_v41_encoding.py> <tokenizers/deepseek_v41.py>"""
import sys

def sub(s, old, new, n=1):
    assert s.count(old) == n, (s.count(old), old[:80])
    return s.replace(old, new)

enc_p, tok_p = sys.argv[1], sys.argv[2]
s = open(enc_p).read()
if "dsv41 1017 encoding" in s:
    print("already patched", enc_p)
else:
    s = sub(s, '''        if role == "tool":
            # Convert tool message to a user message with tool_result block
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
''', '''        if role == "tool":
            # Convert tool message to a user message with tool_result block
            content = msg.get("content", "")  # dsv41 1017 encoding: keep content-block tool results
            if msg.get("content_blocks") and not isinstance(msg.get("content"), str):
                content = msg["content_blocks"]
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
''')
    s = sub(s, '''            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content_blocks = [{"type": "text", "text": msg.get("content", "")}]
''', '''            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content = msg.get("content", "")
                content_blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
''')
    s = sub(s, '''    messages for the purposes of the assistant generation header.
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role in ["user", "developer"] or (role == "system" and idx > 0):
            last_user_index = idx
            break
''', '''    messages for the purposes of the assistant generation header.
    User messages containing only tool results do not start a new turn.
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role in ["user", "developer"] or (role == "system" and idx > 0):
            content_blocks = messages[idx].get("content_blocks")
            if content_blocks and not any(
                block.get("type") in ("text", "image", "image_url")
                for block in content_blocks
            ):
                continue
            last_user_index = idx
            break
''')
    s = sub(s, '''    drop_thinking: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    """
    Render a single message at the given index into its V4.1 encoded string form.
    """
''', '''    drop_thinking: bool = True,
    reasoning_effort: Union[str, int, None] = None,
    last_user_idx: Optional[int] = None,
) -> str:
    """
    Render a single message at the given index into its V4.1 encoded string form.
    """
''')
    s = sub(s, '''    msg = messages[index]
    last_user_idx = find_last_user_index(messages)
''', '''    msg = messages[index]
    last_user_idx = (
        find_last_user_index(messages) if last_user_idx is None else last_user_idx
    )
''')
    s = sub(s, '''    """Encode preprocessed (text-only) messages into the V4.1 prompt format."""
    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)
''', '''    """Encode preprocessed (text-only) messages into the V4.1 prompt format."""
    # Preprocess: merge tool messages and sort tool results (reference order)
    context = merge_tool_messages(context or [])
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
''')
    s = sub(s, '''        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )
''', '''        num_to_render = len(messages)
        context_len = len(context)

    last_user_idx = find_last_user_index(full_messages)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
            last_user_idx=last_user_idx,
        )
''')
    open(enc_p, "w").write(s); print("patched", enc_p)

t = open(tok_p).read()
if "_reject_image_placeholder" in t:
    print("already patched", tok_p)
else:
    t = sub(t, '''def _normalize_messages(
''', '''def _reject_image_placeholder(text: Any) -> None:
    # dsv41 1017 encoding: the reference encoder asserts; vLLM answers 400
    if isinstance(text, str) and IMAGE_PLACEHOLDER in text:
        raise ValueError(
            f"DeepSeek V4.1 message text must not contain {IMAGE_PLACEHOLDER!r}; "
            "send images as image content parts"
        )


def _normalize_messages(
''')
    t = sub(t, '''        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type == "text":
                    parts.append(block.get("text", ""))
''', '''        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        _reject_image_placeholder(message.get("reasoning_content"))
        content = message.get("content")
        _reject_image_placeholder(content)
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type == "text":
                    text = block.get("text", "")
                    _reject_image_placeholder(text)
                    parts.append(text)
''')
    open(tok_p, "w").write(t); print("patched", tok_p)

# --- reasoning-effort budgets and the latest_reminder role ---------------------------------------
# The 0909 image maps low/high/xhigh/max to 25/50/75/100; DeepSeek's reference (0909 and 1017 alike)
# maps low/high/max to 50/75/100 with "high" the default, so the image under-budgeted every default
# thinking request (50 instead of 75). Upstream main now has 50/75/75/100 (xhigh = high); #62226 adds
# the OpenAI aliases minimal (= low) and medium (= high). The reference also renders a
# `latest_reminder` role (the image's encoder does too) that the wrapper's role whitelist rejected.
s = open(enc_p).read()
if "dsv41 effort budgets" in s:
    print("already patched (effort)", enc_p)
else:
    s = sub(s, '''REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 25,
    "high": 50,
    "xhigh": 75,
    "max": 100,
}
''', '''REASONING_EFFORT_MAPPINGS: Dict[str, int] = {  # dsv41 effort budgets: reference values
    "minimal": 50,
    "low": 50,
    "medium": 75,
    "high": 75,
    "xhigh": 75,
    "max": 100,
}
''')
    open(enc_p, "w").write(s); print("patched (effort)", enc_p)
t = open(tok_p).read()
if "latest_reminder" in t:
    print("already patched (roles)", tok_p)
else:
    t = sub(t, '''        if role not in ("system", "developer", "user", "assistant", "tool"):''',
               '''        if role not in ("system", "developer", "user", "assistant", "tool", "latest_reminder"):''')
    t = sub(t, '''                    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "''',
               '''                    "DeepSeek V4.1 reasoning_effort must be minimal, low, medium, high, xhigh, max, "''')
    open(tok_p, "w").write(t); print("patched (roles)", tok_p)
