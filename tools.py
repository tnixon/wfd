"""
MCP Client and tool conversion utilities for the LangGraph Claude demo.
"""

import json
import requests
from typing import Dict, Any, List


class MCPClient:
    """Client for communicating with MCP server"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = None
        self.tools = {}

    def _parse_sse_response(self, response_text: str) -> dict:
        """Parse Server-Sent Events response format"""
        lines = response_text.strip().split('\n')
        data_line = None
        
        for line in lines:
            if line.startswith('data: '):
                data_line = line[6:]  # Remove 'data: ' prefix
                break
        
        if data_line:
            return json.loads(data_line)
        else:
            raise ValueError("No data found in SSE response")

    def initialize(self) -> bool:
        """Initialize MCP session and get available tools"""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "langgraph-claude-agent",
                    "version": "1.0.0"
                }
            }
        }
        
        # Set proper headers including Accept header for both JSON and SSE
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        try:
            response = requests.post(self.server_url, json=init_request, headers=headers)
            response.raise_for_status()
            
            # Parse SSE response
            init_data = self._parse_sse_response(response.text)
            
            # Extract session ID from response headers
            self.session_id = response.headers.get('mcp-session-id')
            print(f"Initialized with session ID: {self.session_id}")
            print(f"Server info: {init_data.get('result', {}).get('serverInfo', {})}")
            
            # Get list of available tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            if self.session_id:
                headers['mcp-session-id'] = self.session_id
            
            tools_response = requests.post(self.server_url, json=tools_request, headers=headers)
            tools_response.raise_for_status()
            
            # Parse SSE response for tools
            tools_data = self._parse_sse_response(tools_response.text)
            
            if 'result' in tools_data and 'tools' in tools_data['result']:
                for tool in tools_data['result']['tools']:
                    self.tools[tool['name']] = tool
                    
            print(f"Available tools: {list(self.tools.keys())}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize MCP client: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return False
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.session_id:
            return {"error": "MCP session not initialized"}
        
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not available"}
        
        tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self.session_id
        }
        
        try:
            response = requests.post(self.server_url, json=tool_request, headers=headers)
            response.raise_for_status()
            
            # Parse SSE response
            result = self._parse_sse_response(response.text)
            
            if 'result' in result:
                return result['result']
            else:
                return {"error": "No result in response"}
                
        except Exception as e:
            return {"error": f"Tool call failed: {str(e)}"}


def convert_mcp_tools_to_claude_format(mcp_tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions to Claude's expected format"""
    claude_tools = []
    
    for tool_name, tool_def in mcp_tools.items():
        claude_tool = {
            "name": tool_name,
            "description": tool_def.get("description", ""),
            "input_schema": tool_def.get("inputSchema", {"type": "object", "properties": {}})
        }
        claude_tools.append(claude_tool)
    
    return claude_tools


def validate_and_clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clean messages for Claude API compatibility"""
    cleaned_messages = []
    
    for msg in messages:
        # Skip messages that don't have the required structure
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
            
        role = msg["role"]
        content = msg["content"]
        
        # Only allow valid roles
        if role not in ["user", "assistant"]:
            continue
            
        # Ensure content is a string for basic messages
        if isinstance(content, str):
            cleaned_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle complex content (like tool results)
            cleaned_messages.append({"role": role, "content": content})
        else:
            # Convert other types to string
            cleaned_messages.append({"role": role, "content": str(content)})
    
    return cleaned_messages