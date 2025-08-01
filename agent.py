"""
LangGraph agent setup and utility functions for Claude MCP integration.
"""

import json
import uuid
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from anthropic import Anthropic

from tools import MCPClient, convert_mcp_tools_to_claude_format, validate_and_clean_messages


# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    claude_response: str
    mcp_session_id: str
    available_tools: Dict[str, Any]


class ClaudeMCPAgent:
    """LangGraph agent with Claude and MCP integration"""
    
    def __init__(self, anthropic_api_key: str, mcp_server_url: str, model_id: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model_id = model_id
        self.mcp_client = MCPClient(mcp_server_url)
        self.memory = MemorySaver()
        self.app = None
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("initialize_mcp", self._initialize_mcp_session)
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("call_claude", self._call_claude_with_tools)
        
        # Define the flow
        workflow.set_entry_point("initialize_mcp")
        workflow.add_edge("initialize_mcp", "process_input")
        workflow.add_edge("process_input", "call_claude")
        workflow.add_edge("call_claude", END)
        
        # Compile with memory
        self.app = workflow.compile(checkpointer=self.memory)
    
    def _initialize_mcp_session(self, state: State) -> State:
        """Initialize MCP session and load available tools - only run once"""
        # Check if MCP is already initialized
        if state.get("mcp_session_id") and state.get("available_tools"):
            print("MCP already initialized, skipping...")
            return state
        
        success = self.mcp_client.initialize()
        
        if success:
            print(f"MCP session initialized successfully!")
            print(f"Available tools: {list(self.mcp_client.tools.keys())}")
            return {
                "mcp_session_id": self.mcp_client.session_id or "",
                "available_tools": self.mcp_client.tools,
                "messages": state.get("messages", [])  # Preserve existing messages
            }
        else:
            print("Failed to initialize MCP session")
            return {
                "mcp_session_id": "",
                "available_tools": {},
                "messages": state.get("messages", [])  # Preserve existing messages
            }
    
    def _process_input(self, state: State) -> State:
        """Process the user input before sending to Claude"""
        user_input = state["user_input"]
        
        # Add some basic preprocessing
        processed_input = user_input.strip()
        
        # Preserve all existing state and only update the processed input
        return {
            **state,  # Keep all existing state
            "user_input": processed_input
        }
    
    def _call_claude_with_tools(self, state: State) -> State:
        """Call Claude API with the user input and available MCP tools"""
        user_message = state["user_input"]
        existing_messages = state.get("messages", [])
        
        try:
            # Convert MCP tools to Claude format
            claude_tools = convert_mcp_tools_to_claude_format(state.get("available_tools", {}))
            
            # Clean and validate existing messages
            cleaned_messages = validate_and_clean_messages(existing_messages)
            
            # Prepare messages - use cleaned conversation history and add new user message
            messages = cleaned_messages.copy()
            messages.append({"role": "user", "content": user_message})
            
            # Call Claude with tools if available
            if claude_tools:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=2000,
                    messages=messages,
                    tools=claude_tools
                )
            else:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=2000,
                    messages=messages
                )
            
            # Handle tool use in response
            if hasattr(response, 'content') and len(response.content) > 0:
                content_block = response.content[0]
                
                # Check if Claude wants to use a tool
                if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                    tool_name = content_block.name
                    tool_args = content_block.input
                    
                    # Execute the tool via MCP
                    tool_result = self.mcp_client.call_tool(tool_name, tool_args)
                    
                    # Call Claude again with the tool result
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user", 
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": json.dumps(tool_result, indent=2)
                        }]
                    })
                    
                    # Get final response from Claude
                    final_response = self.client.messages.create(
                        model=self.model_id,
                        max_tokens=2000,
                        messages=messages,
                        tools=claude_tools if claude_tools else None
                    )
                    
                    claude_response = final_response.content[0].text if final_response.content else "No response"
                    # Add the final assistant response to conversation history
                    messages.append({"role": "assistant", "content": claude_response})
                else:
                    claude_response = content_block.text if hasattr(content_block, 'text') else str(content_block)
                    # Add the assistant response to conversation history
                    messages.append({"role": "assistant", "content": claude_response})
            else:
                claude_response = "No response content"
                messages.append({"role": "assistant", "content": claude_response})
            
            # Return updated state with full conversation history
            return {
                "messages": messages,
                "claude_response": claude_response
            }
            
        except Exception as e:
            error_msg = f"Error calling Claude API: {str(e)}"
            print(f"Debug - Error details: {e}")
            
            # Even for errors, maintain conversation history
            cleaned_messages = validate_and_clean_messages(existing_messages)
            messages = cleaned_messages.copy()
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": error_msg})
            
            return {
                "messages": messages,
                "claude_response": error_msg
            }
    
    def chat(self, user_input: str, thread_id: str = "default_conversation") -> str:
        """Run a single chat interaction with memory"""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current conversation state from memory
        try:
            current_state = self.app.get_state(config)
            if current_state.values:
                # Continue existing conversation
                input_state = {
                    **current_state.values,
                    "user_input": user_input
                }
            else:
                # Start new conversation
                input_state = {
                    "messages": [],
                    "user_input": user_input,
                    "claude_response": "",
                    "mcp_session_id": "",
                    "available_tools": {}
                }
        except Exception as e:
            print(f"Debug - Exception getting state: {e}")
            # Fallback for new conversation
            input_state = {
                "messages": [],
                "user_input": user_input,
                "claude_response": "",
                "mcp_session_id": "",
                "available_tools": {}
            }
        
        result = self.app.invoke(input_state, config)
        return result["claude_response"]
    
    def interactive_chat(self, thread_id: str = "interactive_session"):
        """Run an interactive chat session"""
        print("Chat with Claude (powered by MCP restaurant booking tools)!")
        print("Memory-enabled: Claude will remember our conversation!")
        print("Available commands:")
        print("- Ask for restaurant recommendations")
        print("- Request restaurant details") 
        print("- Check availability and make reservations")
        print("- Type 'quit' to exit")
        print("- Type 'reset' to start a new conversation")
        print("-" * 50)
        
        current_thread_id = thread_id
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'reset':
                current_thread_id = f"session_{str(uuid.uuid4())[:8]}"
                print(f"Started new conversation (thread: {current_thread_id})")
                continue
            
            try:
                response = self.chat(user_input, current_thread_id)
                print(f"\nClaude: {response}")
            except Exception as e:
                print(f"\nError: {e}")
    
    def show_conversation_history(self, thread_id: str = "default_conversation"):
        """Display the conversation history for a thread"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.app.get_state(config)
            if state.values and "messages" in state.values:
                messages = state.values["messages"]
                print(f"\nConversation History (Thread: {thread_id}):")
                print("=" * 50)
                for i, msg in enumerate(messages, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        print(f"{i}. {role.upper()}: {content[:200]}{'...' if len(content) > 200 else ''}")
                    else:
                        print(f"{i}. {role.upper()}: [Complex content]")
                print("=" * 50)
            else:
                print(f"No conversation history found for thread: {thread_id}")
        except Exception as e:
            print(f"Error retrieving history: {e}")
    
    def demo_tools(self):
        """Demonstrate the MCP tools functionality with memory"""
        print("MCP Tools Demo with Memory")
        print("=" * 50)
        
        examples = [
            "Find me romantic Italian restaurants in Taipei",
            "Tell me more about the first restaurant",
            "What about availability for tomorrow at 7pm?",
            "Can you help me make a reservation?"
        ]
        
        print("Try these example conversations to see memory in action:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        
        print("\nUse agent.interactive_chat() to start chatting!")
        print("Use agent.show_conversation_history() to see saved conversations!")


def create_agent(anthropic_api_key: str, mcp_server_url: str, model_id: str = "claude-sonnet-4-20250514") -> ClaudeMCPAgent:
    """Factory function to create a Claude MCP agent"""
    return ClaudeMCPAgent(anthropic_api_key, mcp_server_url, model_id)