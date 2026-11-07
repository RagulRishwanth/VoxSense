from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app=FastAPI(title="VoxSense API",version="1.0.0",description="Research and educational voice biomarker analytics. Not a diagnostic device.")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:3000"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)
