"""
src package initialization.

This package contains modules for:
- Data ingestion (fetching from market APIs, cleaning)
- Database management (InfluxDB operations)
- Indicators and feature engineering
- ML models (Neural SDE, CTRNN, training scripts)
- Orchestration (Prefect flows, monitoring)
- Visualization (Dash/Plotly apps)
"""

# version
__version__ = "0.1.0"

# setup basic logging and environment loading
import logging
logging.basicConfig(level=logging.INFO)