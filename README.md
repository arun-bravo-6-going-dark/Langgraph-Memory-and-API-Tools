# Langgraph Memory and API Tools: Inventory Management 

## Project Overview
This project contains a FastAPI-based **Product Inventory Management System** and an **AI Agent** that interacts with the inventory API. The AI Agent uses tool calling to interact with the inventory and has the ability to store long-term and short-term memories in a PostgreSQL database.

### Key Components
1. **dummy_get_put_app.py**: A RESTful FastAPI application that implements inventory management.
2. **app_streaming.py**: An AI agent that calls the inventory APIs for interaction.
3. **langgraph_longterm_memory_postgres.ipynb**: Notebook showcasing integration with PostgreSQL for memory management.
4. **prompts_file.py**: Contains prompt logic or utility functions for the AI agent.

---

## 1. dummy_get_put_app.py - Product Inventory Management API
### **Description**
A lightweight API to manage products in an inventory, including CRUD operations:
- **Create a Product**: Add new products with name, price, and quantity.
- **Update a Product**: Modify specific fields of a product.
- **Get a Product**: Retrieve product details by ID.
- **List Products**: View all products.

### **Key Features**
- **FastAPI Framework**: Leveraging Pydantic for data validation.
- **In-Memory Storage**: Uses a simple dictionary for demonstration purposes.
- **Automatic Documentation**: OpenAPI and Swagger UI available.

### **API Endpoints**
| Method | Endpoint                        | Description              |
|--------|---------------------------------|--------------------------|
| POST   | `/api/products/create`         | Create a new product     |
| PUT    | `/api/products/{product_id}/update` | Update an existing product |
| GET    | `/api/products/{product_id}`   | Retrieve a specific product |
| GET    | `/api/products`                | List all products        |

### **Running the App**
Run the inventory management API locally:
```bash
uvicorn dummy_get_put_app:app --host 127.0.0.1 --port 8000 --reload
```

---

## 2. app_streaming.py - AI Agent with Tool Calling
### **Description**
The AI Agent interacts with the inventory API by calling its endpoints. The agent can store and retrieve both **short-term** and **long-term memories** using a PostgreSQL database.

### **Key Features**
- **Tool Calling**: Makes API requests to interact with the inventory.
- **Memory Management**: Supports PostgreSQL for persistent storage of memories.

### **Running the AI Agent**
Run the AI agent locally:
```bash
uvicorn app_streaming:app --host 127.0.0.1 --port 8001 --reload
```

### **Workflow**
1. Start the Inventory API on port 8000.
2. Start the AI Agent on port 8001.
3. The AI Agent communicates with the Inventory API for product management tasks.

---

## 3. Memory Management - PostgreSQL
The agent integrates long-term and short-term memories using PostgreSQL. Ensure the database is set up before running the AI agent.

**Database Setup**:
- Install PostgreSQL.
- Create the required tables for storing memories.

---

## Project Dependencies
- **Python 3.10+**
- **FastAPI**
- **Uvicorn**
- **Pydantic**
- **PostgreSQL**
- **LangChain** (for AI agent integration)

### Install Requirements
Run the following command to install dependencies:
```bash
pip install fastapi uvicorn pydantic psycopg2 langchain
```

---

## Running the Full System
1. Start the Inventory Management API:
   ```bash
   uvicorn dummy_get_put_app:app --host 127.0.0.1 --port 8000 --reload
   ```
2. Start the AI Agent:
   ```bash
   uvicorn app_streaming:app --host 127.0.0.1 --port 8001 --reload
   ```

The AI Agent will interact with the Inventory API via the tool-calling mechanism while managing memories in PostgreSQL.

---

## Notes
- This project is for educational and testing purposes.
- The inventory data is not persistent as it uses in-memory storage.
- PostgreSQL is required only for the AI agent's memory management.

---

## Author
**Arun Vignesh Nedunchezhian**

---

## License
This project is open-source and available under the MIT License.
