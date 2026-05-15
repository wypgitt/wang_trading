"""FastAPI backend-for-frontend (BFF) for the Wang Trading web app.

See docs/web_app_design_v2.md and docs/api_contracts_v2.md for the contract.
This package is the only place the React frontend talks to. It must never
expose secrets and must always wrap responses in the standard envelope.
"""
