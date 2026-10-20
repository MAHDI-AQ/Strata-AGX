"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    the template's XML form (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`),
    never emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Normalize tool_calls arguments from JSON string to dict for Jinja template compatibility
        norm_messages = []
        for m in messages:
            if isinstance(m, dict) and m.get("tool_calls") and isinstance(m["tool_calls"], list):
                m_copy = dict(m)
                norm_tcs = []
                for tc in m_copy["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_copy = dict(tc)
                        fn = tc_copy.get("function")
                        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                            fn_copy = dict(fn)
                            try:
                                fn_copy["arguments"] = json.loads(fn_copy["arguments"])
                            except Exception:
                                pass
                            tc_copy["function"] = fn_copy
                        elif isinstance(tc_copy.get("arguments"), str):
                            try:
                                tc_copy["arguments"] = json.loads(tc_copy["arguments"])
                            except Exception:
                                pass
                        norm_tcs.append(tc_copy)
                    else:
                        norm_tcs.append(tc)
                m_copy["tool_calls"] = norm_tcs
                norm_messages.append(m_copy)
            else:
                norm_messages.append(m)

        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  STRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        messages = [m for i, m in enumerate(norm_messages)
                    if i == len(norm_messages) - 1 or os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1" or not (isinstance(m, dict) and m.get("role") == "assistant"
                                                      and not _text_of(m.get("content")).strip()
                                                      and not _text_of(m.get("reasoning_content")).strip()
                                                      and not _has_image(m.get("content")) and not m.get("tool_calls"))]
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)


# ------------------------------------------------------------------------------------------------ requests
def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   ("text", "input_text", None))


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  The model's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto those four.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "xhigh"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in ("text", "input_text", None) and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# #537: a literal <think> / </think> inside a message's text is plain text, not the model's reasoning markers.  The
# tokenizer matches those two strings as their special tokens everywhere (GGUF token type 4, as llama.cpp does), so
# a user quoting "</think>" used to hand the model a real end-of-reasoning token.  Before the template is rendered
# they are swapped for these private-use characters, and the server encodes the spans they mark as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
# #554: the vision markers the same way.  They are special tokens only where the template writes them for an image
# item; the same strings in a message's text (an agent reading chat_template.jinja, a tool result quoting it) became
# the same special ids, so text with <|vision_start|><|image_pad|> before a picture took that picture's embeddings
VISION_TAGS = {"<|vision_start|>": "\U000F0E03", "<|image_pad|>": "\U000F0E04", "<|vision_end|>": "\U000F0E05",
               "<|video_pad|>": "\U000F0E06"}
TOOL_TAGS = {"<tool_call>": "\U000F0E07", "</tool_call>": "\U000F0E08", "<function=": "\U000F0E09", "</function>": "\U000F0E0A"}
LITERAL_TAGS = {**THINK_TAGS, **VISION_TAGS, **TOOL_TAGS}
THINK_MARKS = {v: k for k, v in LITERAL_TAGS.items()}
CONTROL_MARK0 = 0xF0E10      # the control tokens' marks start here, clear of the two sets above


def literal_tags(controls) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each control token's text (`controls`: the tokenizer's CONTROL literals,
    <|im_start|>, <|im_end|>, <|endoftext|>, ...).  The rendered prompt is encoded with those literals parsed, so one
    written inside a message - a file an agent reads, a pasted chat template - opened or ended a turn there.  Longest
    first, as the tokenizer matches them: a literal inside a longer one is not marked before it."""
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(c for c in controls
                                                                                if c not in LITERAL_TAGS)}}
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS):
    """#537: (messages, tools) with every literal <think> / </think> (#554: and vision marker) in their text swapped for
    LITERAL_TAGS' marks (or `tags`': literal_tags() adds the control tokens' texts), and
    whether there was one (None: no change, the same objects back - a prompt without them renders as it always did).
    An assistant message whose content opens with a whole <think>...</think> block (clients that send the reasoning
    inline) keeps that one block as the model's markers, as before."""
    if not _has_tag(messages, tags) and not _has_tag(tools, tags):
        return messages, tools, False
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v, tags)
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = (content[:i] + _mark(content[i:j], tags) + "</think>" +
                            _mark(content[j + len("</think>"):], tags))
        out.append(m)
    return out, _mark_deep(tools, tags), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with THINK_TAGS' marks turned back into the tags' text, and the (start, end) spans of those
    tags in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    marks = THINK_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    mark_re = _THINK_MARK_RE if tags is LITERAL_TAGS else re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    converted = [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]
    merged = []
    for m in converted:
        if merged and merged[-1].get("role") == "user" and m.get("role") == "user":
            prev_c = merged[-1].get("content", "")
            curr_c = m.get("content", "")
            if isinstance(prev_c, str) and isinstance(curr_c, str):
                merged[-1]["content"] = prev_c + chr(10) + chr(10) + curr_c
            elif isinstance(prev_c, list) and isinstance(curr_c, list):
                merged[-1]["content"] = prev_c + curr_c
            elif isinstance(prev_c, list) and isinstance(curr_c, str):
                merged[-1]["content"] = prev_c + [{"type": "text", "text": curr_c}]
            elif isinstance(prev_c, str) and isinstance(curr_c, list):
                merged[-1]["content"] = [{"type": "text", "text": prev_c}] + curr_c
            else:
                merged.append(m)
        else:
            merged.append(m)
    return merged


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
    return tools


def tool_arguments(raw) -> dict:
    """A call's arguments from a client's history -> the mapping the template renders (both APIs).  A JSON object
    (as text or already parsed) is used as it is; empty is {}; anything else (cut-off or malformed JSON, a list, a
    number) is kept as text under "arguments" instead of refusing the whole conversation (#510)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for m in _object_list(req.get("messages"), "messages"):
        role = m.get("role")
        if role == "developer":
            role = "system"
        out = {"role": role, "content": _parts_of(m.get("content")) if role in ("user", "tool", "assistant") else _text_of(m.get("content"))}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                calls.append({"function": {"name": fn.get("name"), "arguments": tool_arguments(fn.get("arguments"))}})
            out["tool_calls"] = calls
        messages.append(out)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))

    # Bug 3: Auto-disable thinking on short utility requests (e.g. title/summary generation)
    # When max_tokens <= 128 and no tools, or single-line prompt without explicit thinking opt-in
    max_tok = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)
    has_explicit_thinking = bool(req.get("reasoning_effort") or req.get("reasoning") or 
                                 (isinstance(req.get("chat_template_kwargs"), dict) and "enable_thinking" in req["chat_template_kwargs"]))
    if not tools and 0 < max_tok <= 128 and not has_explicit_thinking:
        kwargs["enable_thinking"] = False

    return _late_system_to_user(messages), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=2.1.170.bf4;
    cc_entrypoint=sdk-cli; cch=b145e;`.  cch changes on EVERY request and the version's 4th part on every session, so
    the prompt changed ~22K tokens in (after the tool list) on every turn and the conversation cache could only reuse
    up to its last 16K checkpoint: half of every agent prompt was read again.  Both stamps are pinned to f's, as
    llama.cpp does (ggml-org/llama.cpp#21793); only a header at the very start of the system text is touched, and
    only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31; False
    renders it without thinking (#278, the config's "anthropic_thinking": "on_request")."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": pin_billing_stamp(_text_of(system))})
    for m in _object_list(req.get("messages"), "messages"):
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))}})
            elif kind == "tool_result":
                # A tool's image (Claude Code's Read of a picture) reaches the encoder like a user's, as the OpenAI
                # path's tool messages already do; a text-only result is one string, as before.
                messages.append({"role": "tool", "content": _parts_of(block.get("content"))})
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts) if _has_image(parts) else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in _tool_list(req.get("tools"), None)] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # Opt-in (the config's "anthropic_thinking": "on_request"; the default thinks as 0.1.31 did, since a
        # client that never asks would otherwise lose the thinking on every turn).  Anthropic's thinking is
        # opt-in there. Claude Code's helper calls (a session title, a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    return _late_system_to_user(messages), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"


PARAM_END = "</parameter>"
FUNC_START = "<function="
FUNC_END = "</function>"


def sanitize_pua(text: str) -> str:
    """Decode byte-fallback Private Use Area characters and scrub remaining PUA junk.
    0xF000..0xF0FF -> decoded UTF-8 bytes (0x00..0xFF).
    Scrub: 0xE000..0xF8FF (excluding 0xF000..0xF0FF already mapped), 0xF0000..0xFFFFD, 0x100000..0x10FFFD.
    Preserve internal LITERAL_TAGS marks (0xF0E01..0xF0E0A) and CONTROL_MARK0+ marks."""
    out = []
    byte_buf = bytearray()

    def flush_bytes():
        if byte_buf:
            out.append(byte_buf.decode("utf-8", errors="replace"))
            byte_buf.clear()

    for c in text:
        cp = ord(c)
        if 0xF000 <= cp <= 0xF0FF:
            byte_buf.append(cp - 0xF000)
            continue
        flush_bytes()
        if 0xE000 <= cp <= 0xF8FF:
            continue
        elif 0xF0000 <= cp <= 0xFFFFD:
            if 0xF0E01 <= cp <= 0xF0EFF:
                out.append(c)
            else:
                continue
        elif 0x100000 <= cp <= 0x10FFFD:
            continue
        else:
            out.append(c)
    flush_bytes()
    return "".join(out)



PARAM_PREFIX_RE = re.compile(
    r"<\s*(?:parameter|param|arg)(?:\s*=\s*|\s+name\s*=\s*['\"]?|:|\s+name\s*>)",
    re.IGNORECASE
)

PARAM_START_RE = re.compile(
    r"<\s*(?:parameter|param|arg)(?:\s*=\s*|\s+name\s*=\s*['\"]?|:|\s+name\s*>)(?P<name>[a-zA-Z0-9_\.\-]+)['\"]?\s*>",
    re.IGNORECASE
)

PARAM_UNIVERSAL_RE = re.compile(
    r"<\s*(?:parameter|param|arg)"
    r"(?:\s*=\s*|\s+name\s*=\s*['\"]?|:|\s+name\s*>)"
    r"(?P<name>[a-zA-Z0-9_\.\-]+)['\"]?\s*>"
    r"(?P<value>[\s\S]*?)"
    r"(?:<\s*/\s*(?:parameter|param|arg)\s*>|"
    r"(?=<\s*(?:parameter|param|arg)(?:=|\s+name|:|\s+name\s*>))|"
    r"(?=<\s*/\s*(?:function|call)\s*>)|"
    r"(?=<\s*(?:tool_call|function)))",
    re.IGNORECASE
)

PARAM_UNCLOSED_RE = re.compile(
    r"<\s*(?:parameter|param|arg)(?:=|\s+name\s*=\s*['\"]?|:|\s+name\s*>)"
    r"(?P<name>[a-zA-Z0-9_\.\-]+)['\"]?\s*>"
    r"(?P<value>[\s\S]+)$",
    re.IGNORECASE
)

FUNC_HEADER_RE = re.compile(
    r"<\s*(?:function|invoke|call)(?:\s*=\s*|\s+name\s*=\s*['\"]?)(?P<name>[a-zA-Z0-9_\.\-]+)['\"]?\s*>",
    re.IGNORECASE
)


def param_end(text: str, final: bool = False) -> int:
    """Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
    `<parameter=`, `</function>`, or `</tool_call>` - the same text inside a value (a file that documents the call format, #210) is
    part of the value.  -1: none yet; -2: a candidate whose follower has not arrived (streaming; `final` accepts it).
    CRIT-INVARIANT: A parameter value MUST NEVER swallow a subsequent <tool_call>, <function=, </function>, </tool_call>,
    or <|im_start|>. If a hard boundary appears before a valid </parameter>, the parameter ends immediately before that boundary.
    """
    at = text.find(PARAM_END)

    hard_boundaries = []
    for tag in (CALL_START, "<function=", FUNC_END, CALL_END, "<|im_start|>", "<|im_end|>"):
        idx = text.find(tag)
        if idx >= 0:
            hard_boundaries.append(idx)
    m_p = PARAM_PREFIX_RE.search(text)
    if m_p:
        hard_boundaries.append(m_p.start())

    first_hb = min(hard_boundaries) if hard_boundaries else -1

    if first_hb >= 0 and (at < 0 or first_hb < at):
        return first_hb

    while at >= 0:
        after = text[at + len(PARAM_END):].lstrip()
        if after.startswith(("<parameter=", "<param=", "<arg=", FUNC_END, CALL_END)) or PARAM_PREFIX_RE.match(after):
            return at
        if not after or "<parameter=".startswith(after) or FUNC_END.startswith(after) or CALL_END.startswith(after):
            return at if final else -2
        next_at = text.find(PARAM_END, at + 1)
        if first_hb >= 0 and first_hb > at and (next_at < 0 or first_hb < next_at):
            return first_hb
        at = next_at

    if final and text:
        return len(text)
    return -1


def extract_call_name(body: str) -> str | None:
    s = body.strip()
    m = FUNC_HEADER_RE.search(s)
    if m:
        return m.group("name").strip()
    if s.startswith("{") or '"name"' in s:
        try:
            d = json.loads(s)
            if isinstance(d, dict) and "name" in d:
                return str(d["name"]).strip()
        except Exception:
            pass
        m = re.search(r'"name"\s*:\s*"([^"]+)"', s)
        if m:
            return m.group(1).strip()
    return None


def call_end(text: str) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>` or JSON object,
    found by walking its parameters with param_end.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if s.startswith("{") and not s.startswith("<function="):
        return text.find(CALL_END)
    m_fn = FUNC_HEADER_RE.search(text[pos:])
    if not m_fn or m_fn.start() != 0:
        return text.find(CALL_END) if not any("<function=".startswith(s[:i]) for i in range(1, len("<function="))) else -1
    pos += m_fn.end()
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        m_p = PARAM_PREFIX_RE.match(s)
        if m_p:
            gt = text.find(">", pos + m_p.start())
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:])
            if end < 0:
                return -1
            skip = len(PARAM_END) if text[gt + 1 + end:].startswith(PARAM_END) else 0
            pos = gt + 1 + end + skip
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or any("<parameter=".startswith(s[:i]) for i in range(1, len("<parameter="))) or FUNC_END.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


def fill_missing_tool_arguments(name: str, args: dict, schema: dict | None, raw_body: str = "") -> dict:
    """Restricts argument autocompletion strictly to schema properties that explicitly
    define a 'default' field. Never synthesizes dummy or fabricated values for omitted required fields.
    Anti-empty-dict guard: If a tool has required arguments and args is empty but raw_body has content,
    salvages the argument from raw_body."""
    if not isinstance(args, dict):
        args = {}
    params = (schema or {}).get("parameters") or {}
    required = params.get("required") or []
    properties = params.get("properties") or {}
    filled = dict(args)

    for req in required:
        if req in filled and filled[req] is not None and filled[req] != "":
            continue
        prop_def = properties.get(req) or {}
        if "default" in prop_def:
            filled[req] = prop_def["default"]

    # Salvage engine: If a required argument is still missing and raw_body has content
    for req in required:
        if req not in filled or filled[req] is None or filled[req] == "":
            clean_body = re.sub(r"</?(?:function|tool_call|parameter)[^>]*>", "", raw_body).strip()
            if clean_body:
                kv_m = re.search(rf"(?:{re.escape(req)}|cmd)\s*[=:]\s*([^\n\r]+)", clean_body, re.IGNORECASE)
                if kv_m:
                    val = kv_m.group(1).strip("\"' \t")
                    if val:
                        filled[req] = val
                        continue
                if len(required) == 1 and properties.get(req, {}).get("type") == "string":
                    val = clean_body.strip("\"' \t")
                    if val:
                        filled[req] = val

    return filled


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>` or `{"name": ..., "arguments": ...}` -> ToolCall.
    Values are JSON-decoded when the tool's schema says the parameter is not a string."""
    body = body.strip()

    # 1. JSON tool call
    if body.startswith("{"):
        try:
            data = json.loads(body)
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", data.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if not isinstance(args, dict):
                    args = {}
                return ToolCall(name=data["name"], arguments=fill_missing_tool_arguments(data["name"], args, schema, body))
        except Exception:
            pass
        m = re.search(r'"name"\s*:\s*"([^"]+)"', body)
        if m:
            name = m.group(1).strip()
            args = {}
            ma = re.search(r'"(?:arguments|parameters)"\s*:\s*(\{[\s\S]*\})', body)
            if ma:
                try:
                    args = json.loads(ma.group(1))
                except Exception:
                    pass
            return ToolCall(name=name, arguments=fill_missing_tool_arguments(name, args, schema, body))

    # 2. XML tool call
    name = extract_call_name(body)
    if not name:
        raise ValueError("malformed tool call: " + body[:80])

    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}

    header_m = FUNC_HEADER_RE.search(body)
    param_body = body[header_m.end():] if header_m else body

    matched = False
    for m in PARAM_UNIVERSAL_RE.finditer(param_body):
        matched = True
        pname = m.group("name").strip()
        val = m.group("value")
        if val.startswith("\n"):
            val = val[1:]
        if val.endswith("\n"):
            val = val[:-1]
        val = sanitize_pua(val)
        if not re.match(r"^[a-zA-Z0-9_\.\-]+$", pname):
            continue
        declared = (props.get(pname) or {}).get("type")
        if declared == "string":
            args[pname] = val
        else:
            try:
                args[pname] = json.loads(val)
            except (ValueError, TypeError):
                args[pname] = val

    if not matched:
        unclosed = PARAM_UNCLOSED_RE.search(param_body)
        if unclosed:
            pname = unclosed.group("name").strip()
            val = unclosed.group("value").rstrip()
            if val.startswith("\n"):
                val = val[1:]
            val = sanitize_pua(val)
            declared = (props.get(pname) or {}).get("type")
            if declared == "string":
                args[pname] = val
            else:
                try:
                    args[pname] = json.loads(val)
                except (ValueError, TypeError):
                    args[pname] = val

    # Anti-empty-dict fallback
    if not args and param_body.strip() and not param_body.strip().startswith("</function>"):
        jm = re.search(r"(\{[\s\S]*\})", param_body)
        if jm:
            try:
                candidate = json.loads(jm.group(1))
                if isinstance(candidate, dict):
                    args = candidate
            except Exception:
                pass

    return ToolCall(name=name, arguments=fill_missing_tool_arguments(name, args, schema, param_body))


RCALL_MAX = 256  # Task 4.1: Cap unclosed CoT tool call buffering to 256 chars        # the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)


_TOOL_SYNTAX_RE = re.compile(r"</?tool_call>|<function=[^>]*>|</function>", re.IGNORECASE)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # #804/#1058: calls found inside the reasoning wait here as [raw text, ToolCall | None] until the turn shows
        # they were acts: only whitespace (or more calls) after them, then the end of the turn or `</think>`.  The
        # reasoning text before them is tracked (code fence, inline code, current line) to tell an act from a quote.
        self.pending = []
        self.rescued = 0             # calls delivered from the reasoning
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        self._reset_scan()
        self.precall_buf = []
        self.precall_len = 0
        self.call_seen = False
        self.holding_active = bool(self.schemas)

    def _flush_precall(self, as_reasoning: bool) -> list[Event]:
        out = []
        for text in self.precall_buf:
            out.append(Event("reasoning" if as_reasoning else "content", text))
        self.precall_buf = []
        self.precall_len = 0
        self.holding_active = False
        return out

    def _track(self, text: str) -> str:
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                s = (self.line + part).strip()
                if not self.fence:
                    if (s.startswith("```") and "`" not in s[3:]) or (s.startswith("~~~") and "~" not in s[3:]):
                        self.fence = s[:3]
                elif s.startswith(self.fence) and s[len(self.fence):] == "":
                    self.fence = ""
                self.line = ""
                if not self.fence:
                    self.ticks = 0
            else:
                self.line += part
                if not self.fence:
                    self.ticks = (self.ticks + part.count("`")) % 2
        return text

    def _release(self, as_calls: bool) -> list[Event]:
        out = []
        for raw, call in self.pending:
            if as_calls and call is not None:
                self.rescued += 1
                out.append(Event("tool_call", call=call))
            else:
                if call is not None:
                    self.refused += 1
                out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _reset_scan(self):
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}
        self.seen_params = {}

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                m_fn = FUNC_HEADER_RE.search(rest)
                if not m_fn:
                    if rest.lstrip().startswith("{"):
                        name = extract_call_name(rest)
                        if name:
                            self.scall = ToolCall(name=name, arguments={})
                            props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                            self.sdeclared = {k: (v or {}).get("type") for k, v in props.items()}
                            out.append(Event("tool_start", call=self.scall))
                            args("{")
                            self.sp += rest.find("{") + 1
                            self.ss = "between"
                            continue
                    return out
                name = m_fn.group("name").strip()
                self.scall = ToolCall(name=name, arguments={})
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.sdeclared = {k: (v or {}).get("type") for k, v in props.items()}
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += rest.find(m_fn.group(0)) + len(m_fn.group(0))
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                m_p = PARAM_START_RE.match(stripped)
                if m_p:
                    pname = m_p.group("name").strip()
                    self.sp += m_p.end()
                    props = ((self.schemas.get(self.scall.name if self.scall else "") or {}).get("parameters") or {}).get("properties") or {}
                    if not re.match(r"^[a-zA-Z0-9_\.\-]+$", pname) or (props and pname not in props):
                        if props and len(props) == 1 and pname in ("cmd", "file", "filename", "arg"):
                            pname = next(iter(props.keys()))
                        elif props and pname not in props:
                            self.cur_pname = None
                            self.ss = "skip_param"
                            continue
                    self.cur_pname = pname
                    self.seen_params[pname] = []
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    if self.sdeclared.get(pname) == "string":
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    name = self.scall.name if self.scall else ""
                    schema = self.schemas.get(name) or {}
                    seen = getattr(self, "seen_params", {})
                    curr_args = {p: "".join(parts) for p, parts in seen.items()} if isinstance(seen, dict) else {p: "" for p in seen}
                    if self.scall is not None:
                        self.scall.arguments = dict(curr_args)
                    filled = fill_missing_tool_arguments(name, curr_args, schema, self.buf)
                    for k, v in filled.items():
                        if k not in curr_args:
                            args(("" if self.sfirst else ",") + json.dumps(k) + ":" + json.dumps(v, ensure_ascii=False))
                            self.sfirst = False
                            if isinstance(seen, dict):
                                seen[k] = [str(v)]
                            if self.scall is not None:
                                self.scall.arguments[k] = v
                    args("}")
                    self.sp += len("</function>")
                    self.ss = "done"
                else:
                    return out
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                end = param_end(rest)
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    if getattr(self, "cur_pname", None) in self.seen_params:
                        self.seen_params[self.cur_pname].append(value)
                        if self.scall is not None:
                            self.scall.arguments[self.cur_pname] = "".join(self.seen_params[self.cur_pname])
                    args(json.dumps(value)[1:-1] + '"')
                    rest_at_end = rest[end:]
                    if rest_at_end.startswith(PARAM_END):
                        self.sp += end + len(PARAM_END)
                        self.ss = "between"
                        continue
                    elif rest_at_end.startswith(("<tool_call>", "</tool_call>", "</function>", "<|im_start|>")):
                        args("}")
                        self.sp += end
                        self.ss = "done"
                        return out
                    else:
                        self.sp += end
                        self.ss = "between"
                        continue
                hold_tags = (PARAM_END, FUNC_END, CALL_END, CALL_START, "<function=", "<parameter=", "<param=", "<arg=", "<|im_start|>", "<|im_end|>")
                safe = rest.find(PARAM_END) if end == -2 else len(rest) - self._hold(rest, hold_tags)
                if safe > 0 and rest[safe - 1] == "\n":
                    safe -= 1
                while safe > 0 and rest[safe - 1] == '\\':
                    safe -= 1
                if safe > 0:
                    if getattr(self, "cur_pname", None) in self.seen_params:
                        self.seen_params[self.cur_pname].append(rest[:safe])
                        if self.scall is not None:
                            self.scall.arguments[self.cur_pname] = "".join(self.seen_params[self.cur_pname])
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest)
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                try:
                    v = json.loads(value)
                except ValueError:
                    v = value
                if getattr(self, "cur_pname", None) in self.seen_params:
                    self.seen_params[self.cur_pname].append(str(v))
                    if self.scall is not None:
                        self.scall.arguments[self.cur_pname] = v
                args(json.dumps(v, ensure_ascii=False))
                rest_at_end = rest[end:]
                if rest_at_end.startswith(PARAM_END):
                    self.sp += end + len(PARAM_END)
                    self.ss = "between"
                elif rest_at_end.startswith(("<tool_call>", "</tool_call>", "</function>", "<|im_start|>")):
                    args("}")
                    self.sp += end
                    self.ss = "done"
                    return out
                else:
                    self.sp += end
                    self.ss = "between"
            elif self.ss == "skip_param":
                end = param_end(rest)
                if end < 0:
                    return out
                skip_len = len(PARAM_END) if rest[end:].startswith(PARAM_END) else 0
                self.sp += end + skip_len
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """A streamed call that ended without a clean </function>: close its JSON so clients can still parse it."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            raw_val = self.buf[self.sp:].strip("\n")
            if raw_val:
                try:
                    tail += json.dumps(json.loads(raw_val))
                except ValueError:
                    tail += json.dumps(raw_val)
            else:
                tail += '""'
        name = self.scall.name if self.scall else ""
        schema = self.schemas.get(name) or {}
        seen = getattr(self, "seen_params", {})
        curr_args = {p: "".join(parts) for p, parts in seen.items()} if isinstance(seen, dict) else {p: "" for p in seen}
        if self.scall is not None:
            self.scall.arguments = dict(curr_args)
        filled = fill_missing_tool_arguments(name, curr_args, schema, self.buf)
        for k, v in filled.items():
            if k not in curr_args:
                tail += ("," if not tail.endswith("{") else "") + json.dumps(k) + ":" + json.dumps(v, ensure_ascii=False)
                if isinstance(seen, dict):
                    seen[k] = [str(v)]
                if self.scall is not None:
                    self.scall.arguments[k] = v
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    best = max(best, n)
        return best

    def _feed_raw(self, delta: str) -> list[Event]:
        self.buf += sanitize_pua(delta)
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.
                body = self.buf[len(CALL_START):]
                name = extract_call_name(body)
                if name and name in self.schemas:
                    out += self._release(False)
                    out.append(Event("reasoning", "\n</think>\n"))
                    self.buf = body
                    self._reset_scan()
                    self.state, self.lead = "call", False
                    continue
                end, think = call_end(body), body.find(THINK_END)
                if end >= 0 and (think < 0 or think >= end):
                    call = None
                    try:
                        name = extract_call_name(body[:end])
                        if name and name in self.schemas:
                            call = parse_tool_call(body[:end], self.schemas.get(name))
                    except (ValueError, Exception):
                        pass
                    raw = self.buf[:len(CALL_START) + end + len(CALL_END)]
                    if (call is not None and not self.fence and not self.ticks and self.line.strip() == "") or self.pending:
                        self.pending.append([raw, call])    # settled by what follows it (see __init__)
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf = body[end + len(CALL_END):]
                    self.state = "reasoning"
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:len(CALL_START) + think])))
                    self.buf = body[think:]
                    self.state = "reasoning"
                elif len(body) > RCALL_MAX:          # a tag in the prose that never closes: stop holding the thinking back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf = ""
                    self.state = "reasoning"
                else:
                    return out
            elif self.state == "reasoning":
                i = self.buf.find(THINK_END)
                tool = self.buf.find(CALL_START) if self.schemas else -1
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self.buf[:tool]))
                    self.buf = self.buf[tool:]
                    self.state = "rcall"
                    continue
                if i < 0:
                    keep = self._hold(self.buf, (THINK_END, CALL_START) if self.schemas else (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                if i:
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                else:
                    out += self._release(True)
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
            elif self.state == "content":
                # Strip duplicate leading or orphan closing tool tags between calls or leading think closure
                while True:
                    s_buf = self.buf.lstrip()
                    if s_buf.startswith(CALL_END):
                        self.buf = s_buf[len(CALL_END):].lstrip()
                    elif s_buf.startswith(THINK_END):
                        self.buf = s_buf[len(THINK_END):]
                        self.lead = True
                    elif s_buf.startswith(CALL_START):
                        after_probe = s_buf[len(CALL_START):].lstrip()
                        if after_probe.startswith(CALL_START):
                            self.buf = after_probe
                        else:
                            break
                    else:
                        break
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START, THINK_END, CALL_END))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self.buf[:j]))
                        self.buf = self.buf[j:]
                    return out
                # A call is `<tool_call>` and then (after whitespace) `<function=` or `{`; the tag with anything else after
                # it is prose that names the format ("I'll use a <tool_call> block") - content, not a malformed call
                # that ends the request.  Until its follower has arrived it is held, like a partial tag.
                after = self.buf[i + len(CALL_START):].lstrip()
                is_call_start = after.startswith((FUNC_START, "{"))
                may_be_call = not after or FUNC_START.startswith(after) or "{\n".startswith(after)
                if after and not is_call_start and not may_be_call:
                    out.append(Event("content", self.buf[:i + len(CALL_START)]))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if not is_call_start:
                    j = i
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self.buf[:j]))
                        self.buf = self.buf[j:]
                    return out
                if i and self.buf[:i].strip():
                    out.append(Event("content", self.buf[:i].rstrip("\n")))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            else:
                i = call_end(self.buf)
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if self.stream_tools and self.ss == "done" and self.scall is not None and i < 0:
                    stripped_after = self.buf[self.sp:].lstrip()
                    if stripped_after.startswith((CALL_START, "<|im_start|>")):
                        body = self.buf[:self.sp]
                        self.buf = stripped_after
                        name = self.scall.name
                        schema = self.schemas.get(name) or {}
                        self.scall.arguments = fill_missing_tool_arguments(name, self.scall.arguments, schema, body)
                        out.append(Event("tool_call", call=self.scall))
                        self._reset_scan()
                        self.state, self.lead = "content", True
                        continue
                if i < 0:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + len(CALL_END):]
                name = extract_call_name(body)
                call = parse_tool_call(body, self.schemas.get(name))
                if self.scall is not None:
                    call.id = self.scall.id
                out.append(Event("tool_call", call=call))
                self._reset_scan()
                self.state, self.lead = "content", True

    def _finish_raw(self) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content; one that was
        already announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211)."""
        out = []
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            out += self._close_scan()
            if self.scall is not None:
                seen = getattr(self, "seen_params", {})
                curr_args = {p: "".join(parts) for p, parts in seen.items()} if isinstance(seen, dict) else {p: "" for p in seen}
                schema = self.schemas.get(self.scall.name) or {}
                self.scall.arguments = fill_missing_tool_arguments(self.scall.name, curr_args, schema, self.buf)
                out.append(Event("tool_call", call=self.scall))
            self.buf = ""
            self._reset_scan()
            return out
        if self.pending:
            out += self._release(False)
        if self.buf:
            # Check if self.buf holds a complete tool call that merely missed </tool_call> at end of stream
            if self.state in ("rcall", "call"):
                raw_body = self.buf[len(CALL_START):] if self.buf.startswith(CALL_START) else self.buf
                call_name = extract_call_name(raw_body)
                if call_name and call_name in self.schemas:
                    # CRIT-06: Attempt direct parse, then synthesized closing repair
                    schema = self.schemas.get(call_name)
                    candidates = [raw_body]
                    # Synthesize missing closing braces / XML tags for truncated stream
                    if "<function=" in raw_body and "</function>" not in raw_body:
                        candidates.append(raw_body + "\n</function>")
                    trimmed = raw_body.strip()
                    if trimmed.startswith("{") and not trimmed.endswith("}"):
                        candidates.append(trimmed + "}")
                    for cand in candidates:
                        try:
                            salvaged_call = parse_tool_call(cand, schema)
                            if salvaged_call:
                                out.append(Event("tool_call", call=salvaged_call))
                                self.buf = ""
                                self._reset_scan()
                                return out
                        except Exception:
                            continue
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = self.buf if self.state != "call" else CALL_START + self.buf
            text = _TOOL_SYNTAX_RE.sub("", text)
            # Bug 1 Guard: Never leak raw orphan tool delimiters as content deltas
            if self.state == "call":
                cleaned_text = text.strip()
                if kind == "content" and cleaned_text in (CALL_START, CALL_END, f"{CALL_START}{CALL_END}"):
                    text = ""
            if text:
                out.append(Event(kind, text))
            self.buf = ""
        return out

    def feed(self, delta: str) -> list[Event]:
        raw = self._feed_raw(delta)
        if not self.holding_active:
            return raw
        out = []
        for ev in raw:
            if ev.kind in ("tool_start", "tool_call", "tool_args"):
                self.call_seen = True
                if self.precall_buf:
                    out.extend(self._flush_precall(as_reasoning=True))
                out.append(ev)
            elif ev.kind == "content":
                if self.call_seen:
                    out.append(ev)
                else:
                    self.precall_buf.append(ev.text)
                    self.precall_len += len(ev.text)
                    if self.precall_len > 400:
                        out.extend(self._flush_precall(as_reasoning=False))
            else:
                out.append(ev)
        return out

    def finish(self) -> list[Event]:
        raw = self._finish_raw()
        if not self.holding_active:
            # Strip trailing orphaned delimiters from any remaining raw events
            for ev in raw:
                if ev.kind in ("content", "reasoning") and ev.text:
                    ev.text = _TOOL_SYNTAX_RE.sub("", ev.text)
            return [ev for ev in raw if ev.kind not in ("content", "reasoning") or ev.text.strip()]
        out = []
        for ev in raw:
            if ev.kind in ("tool_start", "tool_call", "tool_args"):
                self.call_seen = True
                if self.precall_buf:
                    out.extend(self._flush_precall(as_reasoning=True))
                out.append(ev)
            elif ev.kind == "content":
                cleaned_text = _TOOL_SYNTAX_RE.sub("", ev.text)
                if not cleaned_text.strip():
                    continue
                ev.text = cleaned_text
                if self.call_seen:
                    out.append(ev)
                else:
                    self.precall_buf.append(ev.text)
            else:
                out.append(ev)
        if self.precall_buf:
            out.extend(self._flush_precall(as_reasoning=self.call_seen))
        # Strip any orphaned tool delimiters from final flushed text
        final_out = []
        for ev in out:
            if ev.kind in ("content", "reasoning") and ev.text:
                ev.text = _TOOL_SYNTAX_RE.sub("", ev.text)
                if not ev.text.strip():
                    continue
            final_out.append(ev)
        return final_out
