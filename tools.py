"""
MCP Client and tool conversion utilities for the LangGraph Claude demo.
"""

import json
import requests
from typing import Dict, Any, List, Optional, Callable, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model


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


def create_pydantic_model_from_schema(tool_name: str, input_schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from MCP input schema"""
    properties = input_schema.get("properties", {})
    required_fields = input_schema.get("required", [])
    
    fields = {}
    for field_name, field_def in properties.items():
        field_type = str  # Default to string
        default_value = ... if field_name in required_fields else None
        
        # Map JSON Schema types to Python types
        if field_def.get("type") == "number":
            field_type = float
        elif field_def.get("type") == "integer":
            field_type = int
        elif field_def.get("type") == "boolean":
            field_type = bool
        elif field_def.get("type") == "array":
            field_type = List[str]  # Simplified - assume array of strings
        
        # Create field with description
        if default_value is ...:
            fields[field_name] = (field_type, Field(description=field_def.get("description", "")))
        else:
            fields[field_name] = (Optional[field_type], Field(default=default_value, description=field_def.get("description", "")))
    
    return create_model(f"{tool_name}Input", **fields)


class MCPTool(BaseTool):
    """LangChain Tool wrapper for MCP tools"""
    
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    mcp_client: MCPClient
    tool_name: str
    
    def __init__(self, tool_name: str, tool_def: Dict[str, Any], mcp_client: MCPClient):
        # Create Pydantic model for arguments
        input_schema = tool_def.get("inputSchema", {})
        args_schema = None
        
        if input_schema and input_schema.get("properties"):
            try:
                args_schema = create_pydantic_model_from_schema(tool_name, input_schema)
            except Exception as e:
                print(f"Warning: Could not create schema for {tool_name}: {e}")
        
        super().__init__(
            name=tool_name,
            description=tool_def.get("description", f"MCP tool: {tool_name}"),
            args_schema=args_schema,
            mcp_client=mcp_client,
            tool_name=tool_name
        )
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool"""
        try:
            # Filter out None values to avoid sending empty parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            print(f"Debug: Calling MCP tool {self.tool_name} with args: {filtered_kwargs}")
            result = self.mcp_client.call_tool(self.tool_name, filtered_kwargs)
            
            if isinstance(result, dict) and "error" in result:
                return f"Error: {result['error']}"
            
            # Return formatted JSON result
            return json.dumps(result, indent=2)
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            print(f"Debug: {error_msg}")
            return error_msg
    
    async def _arun(self, **kwargs) -> str:
        """Async version - just calls sync version for now"""
        return self._run(**kwargs)


def create_langchain_tools_from_mcp(mcp_client: MCPClient) -> List[MCPTool]:
    """Convert MCP tools to LangChain Tool objects"""
    langchain_tools = []
    
    for tool_name, tool_def in mcp_client.tools.items():
        try:
            langchain_tool = MCPTool(tool_name, tool_def, mcp_client)
            langchain_tools.append(langchain_tool)
            print(f"Created LangChain tool: {tool_name}")
        except Exception as e:
            print(f"Failed to create tool {tool_name}: {e}")
    
    return langchain_tools


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