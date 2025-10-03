#!/usr/bin/env python3
"""
ZION API Server Runner
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from api import app
    import uvicorn
    print("🚀 Starting ZION 2.7.1 API Server...")
    print(f"📍 Routes loaded: {len(app.routes)}")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()