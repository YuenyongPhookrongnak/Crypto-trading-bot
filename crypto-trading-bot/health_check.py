# utils/health_check.py

import requests
import openai
import logging
import platform
import sys
import pkg_resources  # สำหรับดึงเวอร์ชันของทุก installed package


def check_binance():
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Binance API Health Check failed: {e}")
        return False

def check_openai(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()  # Simple request to test
        return True
    except Exception as e:
        logging.error(f"OpenAI Health Check failed: {e}")
        return False

def run_all(openai_key):
    print("🔎 Running API/AI Health Check...")
    
    binance_ok = check_binance()
    openai_ok = check_openai(openai_key)

    if not binance_ok:
        raise SystemExit("❌ Binance API ไม่พร้อมใช้งาน")

    if not openai_ok:
        raise SystemExit("❌ OpenAI ไม่พร้อมใช้งาน")

    print("✅ Health Check Passed")


def log_versions():
    print("\n🧾 System & Library Versions")
    print(f"🔹 Python Version: {platform.python_version()}")
    print(f"🔹 Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    important_libs = ["openai", "requests", "pandas", "numpy", "ta", "ccxt"]

    for lib in important_libs:
        try:
            version = pkg_resources.get_distribution(lib).version
            print(f"✅ {lib}: v{version}")
        except Exception:
            print(f"⚠️ {lib}: not installed")