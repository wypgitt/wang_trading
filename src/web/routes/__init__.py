"""HTTP routes.

Each module exposes an :class:`fastapi.APIRouter` named ``router`` that
:mod:`src.web.app` mounts under ``/api/v1``. Routes are thin: they
parse / validate, call the service layer, wrap in :func:`envelope`.
"""
