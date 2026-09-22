"""
Environment Validation Script for AirSpark.
Checks Python version, Java installation, PySpark readiness, and essential packages.
"""
import sys
import os
import shutil
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.logging import get_logger

logger = get_logger("AirSpark.EnvironmentValidator")


def check_python_version() -> bool:
    """Verify Python >= 3.10."""
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"[*] Checking Python Version: {sys.version.split()[0]} ... ", end="")
    if major >= 3 and minor >= 10:
        print("[OK]")
        return True
    print(f"[FAIL] Python >= 3.10 required (found {major}.{minor})")
    return False


def check_java_installed() -> bool:
    """Verify Java Runtime is accessible."""
    print("[*] Checking Java (JDK/JRE) installation ... ", end="")
    java_cmd = shutil.which("java")
    if not java_cmd:
        print("[FAIL]")
        print("    -> Action Required: Install Java (OpenJDK 11 or 17 recommended) and add to PATH.")
        return False

    try:
        res = subprocess.run(["java", "-version"], capture_output=True, text=True)
        version_line = res.stderr.splitlines()[0] if res.stderr else res.stdout.splitlines()[0]
        print(f"[OK] ({version_line.strip()})")
        return True
    except Exception as e:
        print(f"[FAIL] Error invoking java: {e}")
        return False


def check_pyspark() -> bool:
    """Verify PySpark can be imported and initialized."""
    print("[*] Checking PySpark library & local cluster launch ... ", end="")
    try:
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        import pyspark
        from pyspark.sql import SparkSession
        
        spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("AirSpark-Env-Check")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        test_df = spark.createDataFrame([(1, "OK")], ["id", "status"])
        cnt = test_df.count()
        spark.stop()
        print(f"[OK] (PySpark {pyspark.__version__} operational)")
        return True
    except Exception as e:
        print(f"[FAIL] PySpark initialization failed: {e}")
        return False


def check_python_packages() -> bool:
    """Verify required Python packages."""
    required = [
        "pandas", "numpy", "pyarrow", "yaml", "streamlit", "plotly", "scipy", "sklearn", "pytest"
    ]
    all_ok = True
    print("[*] Checking core Python dependencies:")
    for pkg in required:
        try:
            __import__(pkg)
            print(f"    - {pkg:<15} [OK]")
        except ImportError:
            print(f"    - {pkg:<15} [MISSING] -> Run: pip install {pkg}")
            all_ok = False
    return all_ok


def main():
    print("\n=======================================================")
    print("      AirSpark Environment & Dependency Validator       ")
    print("=======================================================\n")
    
    p_ok = check_python_version()
    j_ok = check_java_installed()
    pk_ok = check_python_packages()
    s_ok = check_pyspark() if (p_ok and j_ok) else False

    print("\n-------------------------------------------------------")
    if p_ok and j_ok and pk_ok and s_ok:
        print(" [SUCCESS] All requirements satisfied! AirSpark is ready to run.")
        print("-------------------------------------------------------\n")
        return 0
    else:
        print(" [WARNING] Environment has issues. Please resolve missing dependencies above.")
        print("-------------------------------------------------------\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
