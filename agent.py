"""
LangGraph agent setup and utility functions for Claude MCP integration using ToolsNode pattern.
"""

import uuid
import datetime as dt
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from tools import MCPClient, create_langchain_tools_from_mcp

import mlflow
from mlflow.entities import SpanType

# The default system message for the agent
DEFAULT_SYSTEM_MESSAGE = """You are a helpful restaurant booking assistant. 
You will assist a user in locating restaurants that meet their desired requirements, locating details
of the restaurants and helping them however you can in helping them book a table at their desired times.
You will have a number of tools available to you to help with this, including:
- Finding restaurants based on user preferences
- Retrieving restaurant details
- Checking availability for reservations
- Making reservations on behalf of the user

Today's date is {today_date}. If not specified by the user, assume that any bookings are for tomorrow {tmrw_date}.

Please exercise judgement in terms of which tools to use for which step in the process.
Always try to be helpful and polite, and remember that you are a virtual assistant."""

# Define the state structure using LangChain message format
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class WFDAgent:
    """LangGraph agent with Claude and MCP integration using ToolsNode pattern"""
    
    def __init__(self,
                 anthropic_api_key: str,
                 mcp_server_url: str,
                 model_id: str = "claude-3-5-sonnet-20241022",
                 system_message: str = DEFAULT_SYSTEM_MESSAGE):
        self.model_id = model_id
        self.mcp_client = MCPClient(mcp_server_url)
        self.memory = MemorySaver()
        self.sys_msg = system_message
        self.app = None
        self.tools = []
        
        # Initialize ChatAnthropic
        self.llm = ChatAnthropic(
            api_key=anthropic_api_key,
            model=model_id,
            max_tokens=2000,
            temperature=0
        )
        
        self._initialize_mcp_and_build_workflow()

    def system_message(self) -> SystemMessage:
        """Return the system message for the agent"""
        today_date = dt.datetime.now()
        tomorrow_date = today_date + dt.timedelta(days=1)
        formatted_message = self.sys_msg.format(
            today_date=today_date.strftime("%Y-%m-%d"),
            tmrw_date=tomorrow_date.strftime("%Y-%m-%d")
        )
        return SystemMessage(content=formatted_message)
    
    def _initialize_mcp_and_build_workflow(self):
        """Initialize MCP and build the LangGraph workflow"""
        # Initialize MCP client
        success = self.mcp_client.initialize()
        
        if success:
            print(f"MCP session initialized successfully!")
            print(f"Available tools: {list(self.mcp_client.tools.keys())}")
            
            # Create LangChain tools from MCP tools
            self.tools = create_langchain_tools_from_mcp(self.mcp_client)
            
            # Bind tools to the LLM
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            
        else:
            print("Failed to initialize MCP session - proceeding without tools")
            self.llm_with_tools = self.llm
        
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow with ToolsNode"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        
        if self.tools:
            # Create ToolsNode from LangChain tools
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
        
        # Define the flow
        workflow.set_entry_point("agent")
        
        if self.tools:
            # Add conditional edges for tool usage
            workflow.add_conditional_edges(
                "agent",
                tools_condition,
                # self._should_continue,
                # {
                #     "continue": "tools",
                #     "end": END,
                # }
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        # Compile with memory
        self.app = workflow.compile(checkpointer=self.memory)
    
    def _call_model(self, state: State) -> State:
        """Call the language model"""
        messages = state["messages"]
        response = self.llm_with_tools.invoke([self.system_message()] + messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: State) -> str:
        """Determine if we should continue to tools or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, then we route to the "tools" node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        # Otherwise, we stop (reply to the user)
        return "end"
    
    @mlflow.trace(span_type=SpanType.AGENT, name="WFDAgent.chat")
    def chat(self, user_input: str, thread_id: str = "default_conversation") -> str:
        """Run a single chat interaction with memory"""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create human message
        input_message = HumanMessage(content=user_input)
        
        # Invoke the workflow
        result = self.app.invoke({"messages": [input_message]}, config)
        
        # Get the last message from the result
        last_message = result["messages"][-1]
        
        if isinstance(last_message, AIMessage):
            return last_message.content
        else:
            return str(last_message.content)
    
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
                    if isinstance(msg, HumanMessage):
                        print(f"{i}. USER: {msg.content[:200]}{'...' if len(str(msg.content)) > 200 else ''}")
                    elif isinstance(msg, AIMessage):
                        content = msg.content if msg.content else "[Tool calls]"
                        print(f"{i}. ASSISTANT: {str(content)[:200]}{'...' if len(str(content)) > 200 else ''}")
                    else:
                        print(f"{i}. {type(msg).__name__}: {str(msg)[:200]}...")
                print("=" * 50)
            else:
                print(f"No conversation history found for thread: {thread_id}")
        except Exception as e:
            print(f"Error retrieving history: {e}")
    
    def demo_tools(self):
        """Demonstrate the MCP tools functionality with memory"""
        print("MCP Tools Demo with ToolsNode Pattern")
        print("=" * 50)
        
        if self.tools:
            print("Available MCP Tools:")
            for tool in self.tools:
                print(f"- {tool.name}: {tool.description}")
            print()
        
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
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]


def create_agent(anthropic_api_key: str, mcp_server_url: str, model_id: str = "claude-3-5-sonnet-20241022") -> WFDAgent:
    """Factory function to create a Claude MCP agent"""
    return WFDAgent(anthropic_api_key, mcp_server_url, model_id)