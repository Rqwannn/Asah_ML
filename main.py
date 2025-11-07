import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:ai_apps", 
        host="0.0.0.0",
        port=8020,
        reload=False
    )