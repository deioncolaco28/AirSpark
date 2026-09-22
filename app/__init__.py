"""
AirSpark Root Application Package.
Configures cross-platform environment defaults for PySpark and Hadoop on Windows.
"""
import os
import sys
from pathlib import Path

# Ensure PySpark workers locate the active Python interpreter on Windows/Linux/macOS
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Auto-configure local Hadoop directory on Windows if available
root_dir = Path(__file__).resolve().parent.parent
hadoop_dir = root_dir / "hadoop"
if hadoop_dir.exists() and (hadoop_dir / "bin" / "winutils.exe").exists():
    os.environ["HADOOP_HOME"] = str(hadoop_dir)
    os.environ["hadoop.home.dir"] = str(hadoop_dir)
    # Add hadoop/bin to PATH
    hadoop_bin = str(hadoop_dir / "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
