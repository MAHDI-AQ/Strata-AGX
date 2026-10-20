"""Unit tests for serve.frontend (Part 3 architecture)."""
import unittest

from serve.frontend import (
    CONTROL_MARK0,
    LITERAL_TAGS,
    THINK_TAGS,
    TOOL_TAGS,
    VISION_TAGS,
    OutputParser,
    Event,
    _TOOL_SYNTAX_RE,
)


class TestFrontendTags(unittest.TestCase):
    def test_tool_tags_in_literal_tags(self):
        expected_tool_tags = {
            "<tool_call>": "\U000F0E07",
            "</tool_call>": "\U000F0E08",
            "<function=": "\U000F0E09",
            "</function>": "\U000F0E0A",
        }
        self.assertEqual(TOOL_TAGS, expected_tool_tags)
        for tag, mark in expected_tool_tags.items():
            self.assertIn(tag, LITERAL_TAGS)
            self.assertEqual(LITERAL_TAGS[tag], mark)

        for tag, mark in THINK_TAGS.items():
            self.assertIn(tag, LITERAL_TAGS)
            self.assertEqual(LITERAL_TAGS[tag], mark)

        for tag, mark in VISION_TAGS.items():
            self.assertIn(tag, LITERAL_TAGS)
            self.assertEqual(LITERAL_TAGS[tag], mark)

    def test_control_mark0_unchanged(self):
        self.assertEqual(CONTROL_MARK0, 0xF0E10)
        # All literal marks must be below CONTROL_MARK0
        for tag, mark in LITERAL_TAGS.items():
            self.assertLess(ord(mark), CONTROL_MARK0, f"Mark for {tag} exceeds CONTROL_MARK0")


class TestOutputParserRCallTransition(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                },
            },
            {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        ]

    def test_rcall_triggers_explicit_reasoning_closure(self):
        parser = OutputParser(thinking=True, tools=self.tools)
        # In reasoning state, model emits tool call
        events = parser.feed("Thinking about the math... <tool_call><function=calc>")
        self.assertEqual(parser.state, "call")
        reasoning_events = [e for e in events if e.kind == "reasoning"]
        self.assertTrue(len(reasoning_events) >= 2)
        # The last reasoning event before switching to call must be the closure "\n</think>\n"
        self.assertEqual(reasoning_events[-1].text, "\n</think>\n")

    def test_rcall_with_chunks(self):
        parser = OutputParser(thinking=True, tools=self.tools)
        stream = [
            "Let me check",
            " the answer. ",
            "<tool_",
            "call>",
            "<function=calc>",
            "<parameter=expr>2+2</parameter>",
            "</function>",
            "</tool_call>",
        ]
        all_events = []
        for chunk in stream:
            all_events.extend(parser.feed(chunk))
        all_events.extend(parser.finish())

        # Check explicit reasoning closure was emitted
        reasoning_texts = [e.text for e in all_events if e.kind == "reasoning"]
        self.assertIn("\n</think>\n", reasoning_texts)

        # Check tool_call event was produced
        tool_calls = [e.call for e in all_events if e.kind == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "calc")
        self.assertEqual(tool_calls[0].arguments, {"expr": "2+2"})


class TestOutputParserFinishStripping(unittest.TestCase):
    def setUp(self):
        self.tools = [{"name": "test_fn", "parameters": {}}]

    def test_finish_strips_tool_syntax_re(self):
        cases = [
            "<tool_call>",
            "</tool_call>",
            "<function=unknown>",
            "</function>",
            "<tool_call><function=foo>",
            "trailing text <tool_call>",
        ]
        for c in cases:
            with self.subTest(raw=c):
                parser = OutputParser(thinking=False, tools=self.tools)
                # Feed in content mode
                parser.feed(c)
                events = parser.finish()
                for ev in events:
                    if ev.kind in ("content", "reasoning"):
                        self.assertFalse(
                            bool(_TOOL_SYNTAX_RE.search(ev.text)),
                            f"Tool syntax leaked in {ev.kind}: {ev.text!r}",
                        )

    def test_finish_clean_empty_on_pure_delimiters(self):
        parser = OutputParser(thinking=True, tools=self.tools)
        parser.feed("<tool_call>")
        events = parser.finish()
        # Raw tool call tag alone should not produce reasoning or content
        for ev in events:
            self.assertNotIn("<tool_call>", ev.text or "")


if __name__ == "__main__":
    unittest.main()

class TestPrecallChatterGuard(unittest.TestCase):
    def test_precall_conversational_chatter_routed_to_reasoning(self):
        tools = [{"name": "session_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
        parser = OutputParser(thinking=True, tools=tools, stream_tools=True)
        raw = (
            "<think>\nNeed to search.\n</think>\n"
            "Let me search the session first.\n"
            "<tool_call>\n"
            "<function=session_search>\n"
            "<parameter=query>\n"
            "20261016_224622_eabc18\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        events = []
        for i in range(0, len(raw), 5):
            events.extend(parser.feed(raw[i:i + 5]))
        events.extend(parser.finish())

        content_events = [e for e in events if e.kind == "content"]
        reasoning_events = [e for e in events if e.kind == "reasoning"]
        tool_events = [e for e in events if e.kind in ("tool_start", "tool_call")]

        self.assertEqual(len(content_events), 0, f"Expected 0 content events before tool call, got: {content_events}")
        reasoning_text = "".join(e.text for e in reasoning_events)
        self.assertIn("Let me search the session first.", reasoning_text)
        self.assertTrue(len(tool_events) > 0)

    def test_direct_answer_emits_content_when_tools_declared(self):
        tools = [{"name": "session_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
        parser = OutputParser(thinking=True, tools=tools, stream_tools=True)
        raw = "<think>\nDirect answer plan.\n</think>\nHere is the answer to your query."
        events = []
        for i in range(0, len(raw), 5):
            events.extend(parser.feed(raw[i:i + 5]))
        events.extend(parser.finish())

        content_events = [e for e in events if e.kind == "content"]
        content_text = "".join(e.text for e in content_events)
        self.assertIn("Here is the answer to your query.", content_text)


class TestUniversalDialectRecovery(unittest.TestCase):
    def setUp(self):
        self.tools = [{
            "name": "search_files",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "target": {"type": "string"},
                }
            }
        }]

    def test_turn17_double_bracket_and_param_name_recovery(self):
        # Exact reproduction of greeting-hermes-assistant-20261020.json Turn 17
        parser = OutputParser(thinking=True, tools=self.tools, stream_tools=True)
        raw = (
            "<think>\nNeed to search Obsidian.\n</think>\n"
            "<<function_calls>\n"
            "<parameter=limit>\n50\n</parameter>\n"
            "<parameter=path>\n.\n</parameter>\n"
            "<parameter=pattern>\n*\n</parameter>\n"
            "<parameter=target>\nfiles\n</parameter>\n"
            "<parameter=name>\nsearch_files\n</parameter>"
        )
        events = []
        for i in range(0, len(raw), 7):
            events.extend(parser.feed(raw[i:i + 7]))
        events.extend(parser.finish())

        # Must not emit tool calls as content!
        content_events = [e for e in events if e.kind == "content"]
        self.assertEqual(len(content_events), 0, f"Expected 0 content events, got: {content_events}")

        # Must emit tool_call with parsed arguments
        tool_call_events = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(tool_call_events), 1)
        call = tool_call_events[0].call
        self.assertEqual(call.name, "search_files")
        self.assertEqual(call.arguments.get("limit"), 50)
        self.assertEqual(call.arguments.get("path"), ".")
        self.assertEqual(call.arguments.get("pattern"), "*")
        self.assertEqual(call.arguments.get("target"), "files")
        self.assertNotIn("name", call.arguments)

    def test_plural_function_calls_with_closing_tag(self):
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        raw = (
            "<function_calls>\n"
            "<function=search_files>\n"
            "<parameter=path>.</parameter>\n"
            "</function>\n"
            "</function_calls>"
        )
        events = []
        for i in range(0, len(raw), 4):
            events.extend(parser.feed(raw[i:i + 4]))
        events.extend(parser.finish())

        tool_call_events = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(tool_call_events), 1)
        self.assertEqual(tool_call_events[0].call.name, "search_files")
        self.assertEqual(tool_call_events[0].call.arguments.get("path"), ".")

    def test_precall_chatter_with_function_calls(self):
        parser = OutputParser(thinking=True, tools=self.tools, stream_tools=True)
        raw = (
            "<think>\nPlanning search.\n</think>\n"
            "I will search for the markdown files now.\n"
            "<<function_calls>\n"
            "<parameter=name>search_files</parameter>\n"
            "<parameter=path>.</parameter>\n"
            "</function_calls>"
        )
        events = parser.feed(raw)
        events.extend(parser.finish())

        content_events = [e for e in events if e.kind == "content"]
        self.assertEqual(len(content_events), 0)
        reasoning_events = [e for e in events if e.kind == "reasoning"]
        reasoning_text = "".join(e.text for e in reasoning_events)
        self.assertIn("I will search for the markdown files now.", reasoning_text)

        tool_calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].call.name, "search_files")
