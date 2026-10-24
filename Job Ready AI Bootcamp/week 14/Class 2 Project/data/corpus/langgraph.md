# LangGraph

## Core model
LangGraph builds agents as graphs. A StateGraph has a typed state (usually a TypedDict), nodes that are Python functions returning partial state updates, and edges that decide which node runs next. Conditional edges choose the next node from the current state, which is how loops and retries are expressed.

## Reducers
By default a node's update overwrites a state key. A reducer, such as operator.add on a list, tells LangGraph to combine updates instead, for example to append to a history list. Two parallel nodes writing the same key without a reducer raise an InvalidUpdateError.

## Persistence and human-in-the-loop
A checkpointer saves the state after every step, keyed by a thread_id. InMemorySaver loses everything when the process stops, while SqliteSaver and PostgresSaver persist to disk or a database. Calling interrupt() inside a node pauses the graph; the run continues when it is invoked again with Command(resume=value). On resume the interrupted node runs again from its beginning.

## Version notes
LangGraph 1.0 was released in October 2025. In LangGraph 1.x the prebuilt create_react_agent is deprecated in favour of create_agent from langchain.agents, and the default recursion limit is no longer 25, so loops should carry their own counters.
