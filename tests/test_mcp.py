"""Unit tests for the Kohort Model Context Protocol (MCP) Server."""

from __future__ import annotations

import json
import unittest

from kohort.mcp_server import (
    MCP_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    TOOLS,
    handle_rpc_request,
)


class McpServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_handshake(self):
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_VERSION,
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["result"]["protocolVersion"], MCP_VERSION)
        self.assertEqual(resp["result"]["serverInfo"]["name"], SERVER_NAME)
        self.assertEqual(resp["result"]["serverInfo"]["version"], SERVER_VERSION)
        self.assertIn("tools", resp["result"]["capabilities"])

    async def test_notifications_initialized(self):
        req = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        resp = await handle_rpc_request(req)
        self.assertIsNone(resp)

    async def test_ping(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["id"], 2)
        self.assertEqual(resp["result"], {})

    async def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["id"], 3)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("concept_test", tool_names)
        self.assertIn("competitive_teardown", tool_names)
        self.assertIn("find_advantage", tool_names)
        self.assertIn("build_audience", tool_names)
        self.assertIn("list_audiences", tool_names)
        self.assertIn("explore_report", tool_names)

    async def test_tool_call_list_audiences(self):
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "list_audiences",
                "arguments": {},
            },
        }
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertFalse(resp["result"]["isError"])
        content_text = resp["result"]["content"][0]["text"]
        data = json.loads(content_text)
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(data["count"], 1)

    async def test_tool_call_competitive_teardown_mock(self):
        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "competitive_teardown",
                "arguments": {"target": "getminds.ai", "mock": True},
            },
        }
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertFalse(resp["result"]["isError"])
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["status"], "success")
        self.assertIn("run_dir", data)
        self.assertIn("pricing_summary", data)

    async def test_tool_call_build_audience_mock(self):
        req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "build_audience",
                "arguments": {"spec": "sa-tech-founders", "personas": 6, "mock": True},
            },
        }
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertFalse(resp["result"]["isError"])
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["sample_size"], 6)

    async def test_tool_call_concept_test_mock(self):
        req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "concept_test",
                "arguments": {
                    "brief_path": "briefs/example_loopa.yaml",
                    "personas": 6,
                    "mock": True,
                },
            },
        }
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertFalse(resp["result"]["isError"])
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["status"], "success")
        self.assertIn("run_dir", data)
        self.assertIn("trial_estimate", data)

        # Explore the run we just made
        explore_req = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "explore_report",
                "arguments": {"run_dir": data["run_dir"]},
            },
        }
        exp_resp = await handle_rpc_request(explore_req)
        self.assertIsNotNone(exp_resp)
        self.assertFalse(exp_resp["result"]["isError"])
        exp_data = json.loads(exp_resp["result"]["content"][0]["text"])
        self.assertEqual(exp_data["type"], "concept_test")

    async def test_tool_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}},
        }
        resp = await handle_rpc_request(req)
        self.assertIsNotNone(resp)
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("Unknown tool", resp["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
