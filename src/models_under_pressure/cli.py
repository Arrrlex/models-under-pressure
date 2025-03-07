import subprocess
import sys
from pathlib import Path


def dash_command():
    """Run the Streamlit dashboard with any provided arguments."""
    dashboard_path = Path(__file__).parent / "dashboard.py"

    # Get any arguments passed after the dash command
    args = sys.argv[1:]

    # Use Streamlit's Python API to run the dashboard
    subprocess.run(["streamlit", "run", str(dashboard_path), "--"] + args)
