
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

uri = os.getenv("MONGODB_URL")

if not uri:
    raise ValueError("MONGODB_URL not found in .env")

client = MongoClient(uri, serverSelectionTimeoutMS=10000)

try:
    client.admin.command("ping")
    print("MongoDB connected successfully!")
except Exception as e:
    print("Connection failed:", e)
finally:
    client.close()
