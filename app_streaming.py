#uvicorn app_streaming:app --host 127.0.0.1 --port 8001 --reload

import os
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import List, Literal, Optional
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults, WikipediaQueryRun
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from psycopg_pool import ConnectionPool
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import uuid
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import prompts_file

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

recall_vector_store = PGVector(
    embeddings=hf,
    collection_name="long_term_memory",
    connection="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

app = FastAPI(
    title="Inventory Management Chapt API", 
)

# FastAPI server base URL
API_BASE_URL = "http://127.0.0.1:8000/api/products"

class InputPayload(BaseModel):
    user_message: str
    user_id: str
    thread_id: str

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    print("Memories to save:", memory)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    print("Memory Updated")
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)
    
    # Filter as a dictionary
    filter_dict = {"user_id": user_id}


    documents = recall_vector_store.similarity_search(
        query, k=3, filter=filter_dict
    )
    return [document.page_content for document in documents]

@tool
def ddg(search_string: str):
    """
    Retrieves information from the internet based on the provided query.
    """
    search = DuckDuckGoSearchResults()
    ddg_results = search.invoke(search_string)
    return ddg_results    
    
@tool
def wikipedia_tool(search_string: str):
    """
    Retrieves information from wikipedia based on the provided query.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wiki_results = wikipedia.run(search_string)
    return wiki_results

@tool
def create_product(name: str, price: float, quantity: int):
    """
    Create a new product in the inventory system.

    Parameters:
        name (str): Name of the product.
        price (float): Price of the product.
        quantity (int): Quantity of the product in stock.

    Returns:
        str: A message with the ID of the created product.
    """
    payload = {
        "name": name,
        "price": price,
        "quantity": quantity
    }
    try:
        response = requests.post(f"{API_BASE_URL}/create", json=payload)
        if response.status_code == 201:
            product = response.json()
            product_id = product["id"]
            return f"Product '{name}' created successfully with ID: {product_id}."
        else:
            return f"Failed to create product. Status code: {response.status_code}, Message: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

@tool
def update_product(product_id: int, name: str = None, price: float = None, quantity: int = None):
    """
    Update an existing product's information.

    Parameters:
        product_id (int): ID of the product to update.
        name (str, optional): New name of the product.
        price (float, optional): New price of the product.
        quantity (int, optional): New quantity of the product in stock.

    Returns:
        str: A message indicating the result of the update attempt.
    """
    payload = {
        "name": name,
        "price": price,
        "quantity": quantity
    }
    # Remove keys with None values to avoid sending unnecessary data
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.put(f"{API_BASE_URL}/{product_id}/update", json=payload)
        if response.status_code == 200:
            return f"Product '{product_id}' updated successfully! Response: {response.json()}"
        else:
            return f"Failed to update product. Status code: {response.status_code}, Message: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

@tool
def get_product(product_id: int):
    """
    Retrieve a product by its ID to confirm its creation.

    Parameters:
        product_id (int): ID of the product to retrieve.

    Returns:
        str: Product details if found, otherwise a message indicating it was not found.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/{product_id}")
        if response.status_code == 200:
            return f"Product found: {response.json()}"
        elif response.status_code == 404:
            return "Product not found."
        else:
            return f"Failed to retrieve product. Status code: {response.status_code}, Message: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    
@tool
def list_all_products():
    """
    Retrieve a list of all products.

    Returns:
        str: List of products if successful, otherwise an error message.
    """
    try:
        url = f"{API_BASE_URL}"  # Ensure there's no duplication in the path
        response = requests.get(url)
        
        if response.status_code == 200:
            products = response.json()
            return f"Products list: {products}"
        elif response.status_code == 404:
            return "Endpoint not found. Verify the URL and API path."
        else:
            return f"Failed to retrieve products. Status code: {response.status_code}, Message: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    

tools = [wikipedia_tool, ddg, create_product, update_product, get_product, list_all_products]

class State(MessagesState):
    recall_memories: List[str]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompts_file.agent_prompt,
        ),
        ("placeholder", "{messages}"),
    ]
)

model_with_tools = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'], temperature=0, top_p=0.5).bind_tools(tools)

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response."""
        
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories."""
    
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    print("Recalled Memories:", recall_memories)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph."""

    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END

#Presrequisites: brew update -> brew install postgres -> brew services start postgresql -> psql postgres -> CREATE DATABASE my_database; -> CREATE USER my_user WITH PASSWORD 'my_password'; -> GRANT ALL PRIVILEGES ON DATABASE my_database TO my_user; -> \q
DB_URI = DB_URI = "postgresql://my_user:my_password@localhost:5432/my_database?sslmode=disable"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# Create the graph and add nodes
builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

def response_generator(user_message, user_id, thread_id):
    with ConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        # Set up the PostgresSaver with the pool
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        # Compile the graph with the checkpointer
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"user_id":user_id, "thread_id": thread_id}}

        # Wrap the user input in a HumanMessage
        input_message = {"messages": [HumanMessage(content=user_message)]}

        # Stream messages from the graph
        for msg, metadata in graph.stream(input_message, config, stream_mode="messages"):
            if (
                isinstance(msg, AIMessage)  # Check if msg is of type AIMessage
                and metadata["langgraph_node"] == "agent"
            ):
                yield msg.content
                
@app.post("/api/v1/agent_main")
def get_agent_response(payload: InputPayload):
    user_message = payload.user_message
    user_id = payload.user_id
    thread_id = payload.thread_id
    return StreamingResponse(
        response_generator(user_message, user_id, thread_id),
        media_type="text/plain"
    )

# Run FastAPI server on an alternative port, e.g., 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)