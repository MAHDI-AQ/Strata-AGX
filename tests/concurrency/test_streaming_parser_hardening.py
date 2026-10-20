import sys
import unittest
import json

sys.path.insert(0, "/tmp")
from serve.frontend import (
    OutputParser,
    parse_tool_call,
    param_end,
    call_end,
    extract_call_name,
    fill_missing_tool_arguments,
    ToolCall
)

class TestStreamingParserHardening(unittest.TestCase):
    def setUp(self):
        self.terminal_tool = {
            "name": "terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
        self.read_file_tool = {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
        self.tools = [self.terminal_tool, self.read_file_tool]

    def test_standard_streaming(self):
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        raw = "<tool_call>\n<function=terminal>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
        events = []
        for i in range(0, len(raw), 4):
            events.extend(parser.feed(raw[i:i+4]))
        events.extend(parser.finish())
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "terminal")
        self.assertEqual(calls[0].arguments, {"command": "ls -la"})

    def test_syntax_variation_attr_quotes(self):
        # <parameter name="command">
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        raw = '<tool_call>\n<function=terminal>\n<parameter name="command">\nls -la\n</parameter>\n</function>\n</tool_call>'
        events = []
        for i in range(0, len(raw), 4):
            events.extend(parser.feed(raw[i:i+4]))
        events.extend(parser.finish())
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "terminal")
        self.assertEqual(calls[0].arguments, {"command": "ls -la"})

    def test_syntax_variation_attr_no_quotes(self):
        # <parameter name=command>
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        raw = '<tool_call>\n<function=terminal>\n<parameter name=command>\nls -la\n</parameter>\n</function>\n</tool_call>'
        events = []
        for i in range(0, len(raw), 4):
            events.extend(parser.feed(raw[i:i+4]))
        events.extend(parser.finish())
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "terminal")
        self.assertEqual(calls[0].arguments, {"command": "ls -la"})

    def test_syntax_variation_colon(self):
        # <parameter:command>
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        raw = '<tool_call>\n<function=terminal>\n<parameter:command>\nls -la\n</parameter>\n</function>\n</tool_call>'
        events = []
        for i in range(0, len(raw), 4):
            events.extend(parser.feed(raw[i:i+4]))
        events.extend(parser.finish())
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "terminal")
        self.assertEqual(calls[0].arguments, {"command": "ls -la"})

    def test_turn7_runaway_parameter_does_not_swallow_subsequent_call(self):
        # Turn 7 from hello-hermes-20261020.json:
        # First call has unclosed parameter 'Personal-', followed immediately by second tool call
        raw = (
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\n"
            "C:/Users/mahdi/Documents/mahdi/Personal-\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\n"
            "C:/Users/mahdi/Documents/mahdi/cok-server/Execution/2026-10-16-recovery-newcomer-sprint-goal.md\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        events = []
        for i in range(0, len(raw), 4):
            events.extend(parser.feed(raw[i:i+4]))
        events.extend(parser.finish())
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertTrue(len(calls) >= 1)
        # Crucial invariant: path of any call MUST NOT contain literal <tool_call>
        for c in calls:
            self.assertNotIn("<tool_call>", c.arguments.get("path", ""))
            self.assertNotIn("<function=", c.arguments.get("path", ""))

    def test_turn11_parameter_name_cmd_salvage(self):
        # Turn 11 syntax where model emitted cmd=...
        body = '<function=terminal>\n<parameter name>cmd=cd /app && ls -la\n</function>'
        call = parse_tool_call(body, self.terminal_tool)
        self.assertEqual(call.name, "terminal")
        self.assertIn("ls -la", call.arguments.get("command", ""))
        self.assertNotEqual(call.arguments, {})

    def test_unclosed_stream_eos_salvage(self):
        # Stream truncated at max_tokens mid-parameter
        parser = OutputParser(thinking=False, tools=self.tools, stream_tools=True)
        raw = '<tool_call>\n<function=terminal>\n<parameter=command>\ngit status'
        events = parser.feed(raw) + parser.finish()
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "terminal")
        self.assertEqual(calls[0].arguments, {"command": "git status"})

    def test_anti_empty_dict_guard_never_emits_empty_for_terminal(self):
        # If model emitted body with command but non-standard tags, never emit {}
        body = '<function=terminal>\npython script.py --test\n</function>'
        call = parse_tool_call(body, self.terminal_tool)
        self.assertEqual(call.name, "terminal")
        self.assertEqual(call.arguments, {"command": "python script.py --test"})

if __name__ == "__main__":
    unittest.main()
