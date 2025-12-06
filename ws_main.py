import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "routes.routes:ai_apps", 
        host="0.0.0.0",
        port=8021,
        reload=False
    )