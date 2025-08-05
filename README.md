# WFD - What's For Dinner?
What's For Dinner? A simple AI Agent to help you find and plan your dinner. This is a toy project
used to explore the capabilities of AI agents, MCP and various logging and evaluation tools. You can 
chat with the agent to discover local restaurants, 

# Software stack
- Python 3.11
- [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://python.langchain.com/) for building the agent
- [Claude](https://claude.ai/) for the LLM
- [MCP](https://modelcontextprotocol.io/docs/getting-started/intro) for managing the agent's tools
- [MLflow](https://mlflow.org/) for logging and evaluation

# Tools used
I make use of the [mcp-booking](https://github.com/tnixon/mcp-booking) (a modified fork of a MCP server by [Sam Wang](https://github.com/samwang0723)) module as a tool to provide
access to the Google Places API. We import it as a submodule in this repository. This provide our agent with the following tools:
- `search_restaurants`: search for restaurants in a given area based on a range of criteria (e.g. cuisine, price range, event type, mood, etc.)
- `get_restaurant_details`: get details about a specific restaurant, including its address, phone number, and website
- `get_booking_instructions`: get instructions on how to book a reservation at a specific restaurant
- `check_availability`: check the availability of a restaurant for a specific date and time (*mock implementation*)
- `make_reservation`: make a reservation at a specific restaurant for a specific date and time (*mock implementation*)

The last two tools are mock implementations, as the Google Places API does not provide a way to check availability or make reservations directly. 
In the future, it would be nice integrate with a real booking system, such as OpenTable or Resy, to provide this functionality, as well as meal-delivery services like Uber Eats or DoorDash.
Unfortunately, most of these services do not provide public APIs, or have very limited functionality, and require a partnership to access their APIs.
The simulated booking process of making a reservation was good enough for the purposes of this PoC project.

# Agent Design

The agent is built using LangGraph as a simple React style Agent, which gives is quite a bit of autonomy in how it can respond to user queries, and what tools it will choose to use.
The tools are provided by the MCP server, and are discovered by the agent at runtime. These are simple tools that are just an interface to the Google Places API, and a few mocked out
deterministic functions to simulate the process of checking availability and making reservations. It would have been simpler to just add them as native python fucntions, but I was
interested in exploring the capabilities of MCP and how it can be used to manage tools in an AI agent. 
The agent uses the standard LangGraph [MemorySaver](https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/?h=memorysaver#1-create-a-memorysaver-checkpointer) checkpointer
to save its state between conversational turns, allowing it to maintain context and remember previous interactions with the user. In order for this to work, the agent needs to be given a consistent thread-id 
for each conversation, which is used to index the conversation state in memory. This is done by passing the `thread_id` parameter to the agent on each invocation (ie. turn of the conversation).
I chose to use Claude as the LLM, as it is a powerful model that can handle complex queries and provide detailed responses, and I just happen to like it over other comparable models.
The agent is traced and evaluations are logged to experiments in MLflow. I chose this because it is both powerful and simple to integrate with. Also as a former Databricks employee, 
I am quite familiar with it and it is a tool I like to use. I could have used LangSmith for tracing, but I thought it best to use a tool that is less specific to LangGraph and more general purpose.

The agent code is in the following files:
- `agent.py`: the implementation of the agent as a LangGraph React style agent.
- `tools.py`: provides an MCP client that connects to the MCP server and then returns a set of LangChain tool objects based on the tools discovered

I developed all the code in [PyCharm](https://www.jetbrains.com/pycharm/) with the assistance of [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview).

# Running the agent
To run the agent, you will need to have Python 3.11 installed, and the required dependencies. You can install the dependencies using pip:
```bash
pip install -r requirements.txt
```

You will need to follow the instructions in the `tools/mcp-booking/README.md` file to set up the MCP server and the Google Places API credentials.
The MCP server can be run locally with the following command:
```bash
npm run dev
```
This will start the MCP server on port 3000 by default. The agent code assumes that the MCP server is running on `http://localhost:3000`, so you will need to change this if you are running it on a different host or port.

Additionally you will need to create a `.env` file in the root of the project with the following content:
```
ANTHROPIC_API_KEY=<your_anthropic_api_key>
OPENAI_API_KEY=<your_openai_api_key>
MLFLOW_DIR=./mlflow_data
```

You can run the agent in any of the following Jupyter notebooks:
- `langgraph_claude_demo.ipynb`: a simple notebook to load the agent and run it in a loop, allowing you to chat with it and see how it responds to your queries.
- `simple_tracing.ipynb`: run some simple tracing experiments, to evaluate how the agent will handle different queries, what tools it will use, and how long it takes to respond.
- `evaluation.ipynb`: runs a number of evaluations on queries with the agent, and logs the results to MLflow experiments. Currently it calculates the following metrics:
  - `token_count`: the number of tokens used by the agent in the response
  - `toxicity`: the toxicity score of the response, based on https://huggingface.co/spaces/evaluate-measurement/toxicity
  - `flesh_kincaid_grade_level`: the reading-level or complexity of the language used in the response, based on the Flesch-Kincaid Grade Level formula
  - `ari_grade_level`: the ARI (Automated Readability Index) grade level of the response, which is another measure of readability
  - `answer_relevance`: the relevance of the answer to the query, based on using OpenAI's GPT-4o model to evaluate the response against the query (LLM-as-a-judge)

# Experimental results

Although very preliminary, the most useful results I found was in the tracing experiments. I was able to quickly debug various issues with the agent,
such as it not having the correct date, and complaining about attempting to make a reservation on a date that was in the past. The solution to this issue
was to provide the current data in the system prompt, a relatively simple fix. Other issues that came up in traces included periodic issues with the MCP server
not connecting correctly to the Google Places API, which I was visible in the traces, but not in the MCP server logs.

The evaluation metrics mostly returned expected results. It was nice to be confirmed that the agent was not producing any toxic responses, and that the relevance of the answers was mostly high.
Although, the relevance scores could have been better if the MCP server was able to provide better booking functionality, and not just mock implementations. The number of examples
that were evaluated was small, however, so the results here should be seen as a technical validation, but not a comprehensive evaluation of the agent's capabilities and limitations. 
More extensive evaluation with a larger set of queries that cover a wider range of scenarios would be needed to get a better understanding of the agent's performance. Also, queries that
are intentionally adversarial or designed to trick the agent into producing incorrect or nonsensical responses would be useful to test the robustness of the agent, and whehter it could be
made to produce harmful or toxic responses.

# Production risks and limitations
This is a toy project, and not intended for production use. It is a proof of concept to explore the capabilities of AI agents, MCP and various logging and evaluation tools. 
A number of areas would need to be improved before this could be used in production, including:
- Integration with a real booking system, such as OpenTable or Resy, to provide the ability to check availability and make reservations
- More comprehensive evaluation of the agent's capabilities and limitations, including adversarial queries and edge cases
- The MCP server currently uses a single Google Places API key, which is shared by all users of the server. This is not a good practice for production use, as it could lead to rate limiting or abuse of the API. A better approach would be to use a more robust authentication protocol such as OAuth2, so that each user has their own API access level.
- Both the agent and the MCP server should be hardened against common security vulnerabilities, and need better error handling and logging to ensure that they can handle unexpected situations gracefully.
- Both the agent and the MCP server should be tested for performance and scalability, to ensure that they can handle a large number of concurrent users and queries without crashing or slowing down.
- Both the agent and the MCP server should be deployed to scalable infrastructure, ideally with auto-scaling capabilities, to ensure that they can handle spikes in traffic and load without downtime or performance degradation.

# Thoughts on a great evaluation and observability framework
A great evaluation and observability framework for AI agents should provide the following features:
- **Ease of Use**: Simple to set up and use, with minimal configuration required to get started.
- **Integration**: Ability to integrate with various AI agents, LLMs, and tools, regardless of the underlying technology stack.
- **Metrics**: Should come with a large set of the most common metrics used to evaluate AI agents, such as relevance, toxicity, readability, correctness, and performance.
  - **Custom Metrics**: Ability to define custom metrics and evaluation criteria, tailored to the specific use case, and business requirements
- **Human-in-the-loop**: Ability to incorporate human feedback into the evaluation process, allowing for more nuanced and context-aware assessments of the agent's performance.
- **Tracing**: Ability to trace the execution of the agent, including the tools used, the inputs and outputs of each tool, and the time taken for each step.
- **Evaluation**: Ability to evaluate the agent's responses against a set of predefined metrics, such as relevance, toxicity, readability, and correctness.
- **Logging**: Ability to log the agent's interactions with the user, including the queries, responses, and any errors or exceptions that occur.
- **Visualization**: Ability to visualize the agent's performance over time, including trends in the metrics, the distribution of tool usage
- **Alerting**: Ability to set up alerts based on the agent's performance, such as when the relevance of the responses drops below a certain threshold, or if the distribution of a metric deviates significantly from the expected range.
- **Scalability**: Should be able to handle a large number of agents and queries, and be able to scale horizontally to handle increased load.
- **Integrate into the training loop**: Should be able to integrate into the training loop of the agent, allowing for continuous improvement and refinement of the agent's capabilities based on the evaluation results.
- **Well organized**: Should be well organized and easy to navigate, with clear documentation and examples to help users get started quickly.

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.