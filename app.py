"""
Main entry point for AI Solar Planning & Financial Analysis System.
Supports both `streamlit run app.py` and `streamlit run ML_Solar.py`.
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("ML_Solar.py", run_name="__main__")
