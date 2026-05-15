"""Service layer.

Each service:
- accepts request filters
- calls existing repository / pipeline modules under :mod:`src`
- normalizes response shape into a :mod:`src.web.dtos` model
- applies pagination / limits
- strips sensitive fields
- adds freshness metadata when constructing the envelope upstream

Services do not import FastAPI — they return plain DTOs so they can be
unit-tested without spinning up an HTTP layer.
"""
