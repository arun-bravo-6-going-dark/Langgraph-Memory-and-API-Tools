"""
App Summary:
This FastAPI application serves as a basic product inventory management system, simulating the creation, retrieval, updating, and listing of products. It is designed as a lightweight, in-memory RESTful API to test and demonstrate API tool integration, particularly for language model-based agents like LangChain or similar frameworks.

Key Features:
1. **Create Product**: Allows adding a new product to the inventory with attributes such as name, price, and quantity.
2. **Update Product**: Supports partial updates to existing products using their unique ID.
3. **Retrieve a Product**: Fetches detailed information about a specific product by its ID.
4. **List Products**: Returns a list of all products in the inventory.
5. **In-Memory Storage**: Products are stored in an in-memory dictionary for simplicity, making the app stateless and easy to reset.

Intended Use:
This app is ideal for testing API integrations and learning purposes. It provides clear, self-contained endpoints for CRUD operations while using FastAPI's robust features like Pydantic for data validation and automatic OpenAPI documentation generation.

Note:
Since this app uses in-memory storage (`products` dictionary), data will reset each time the application restarts. It's not intended for production use but rather for educational and testing purposes.


Command to run the app: uvicorn dummy_get_put_app:app --host 127.0.0.1 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

# In-memory storage for products
products = {}
next_id = 1

# Define a Product schema with an `id` field
class Product(BaseModel):
    id: int
    name: str
    price: float
    quantity: int

# Define a schema for creating products without `id`
class CreateProduct(BaseModel):
    name: str
    price: float
    quantity: int

# Define an UpdateProduct schema for partial updates
class UpdateProduct(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None

@app.post("/api/products/create", response_model=Product, status_code=201)
def create_product(product: CreateProduct):
    global next_id
    product_data = product.dict()
    product_data["id"] = next_id
    products[next_id] = product_data
    next_id += 1
    return product_data  # Returns the product with `id`

@app.put("/api/products/{product_id}/update", response_model=Product)
def update_product(product_id: int, product: UpdateProduct):
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    
    stored_product_data = products[product_id]
    update_data = product.dict(exclude_unset=True)
    updated_product = {**stored_product_data, **update_data}
    products[product_id] = updated_product
    return updated_product

@app.get("/api/products/{product_id}", response_model=Product)
def get_product(product_id: int):
    """
    Retrieve a product by its ID.
    """
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    return products[product_id]

@app.get("/api/products", response_model=List[Product])
def list_products():
    """
    Retrieve a list of all products.
    """
    return list(products.values())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
