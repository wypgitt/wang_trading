"""Standalone Prometheus metrics endpoint.

The trading pipelines normally own their MetricsCollector. This tiny server
exists so supervisor/systemd configs have a valid monitoring target even
before the full trading runner is online.
"""

from __future__ import annotations

import argparse
import logging
import time

from src.bootstrap import load_runtime_config
from src.monitoring.metrics import MetricsCollector

log = logging.getLogger(__name__)


def main() -> int:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser("metrics_server")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--addr", type=str, default="0.0.0.0")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config, default_name="monitoring")
    monitoring = cfg.get("monitoring", cfg)
    port = int(args.port or monitoring.get("metrics_port", 9090))

    logging.basicConfig(level=getattr(logging, str(monitoring.get("log_level", "INFO"))))
    metrics = MetricsCollector()
    metrics.start_server(port=port, addr=args.addr)
    log.info("Prometheus metrics endpoint listening on %s:%d", args.addr, port)
    while True:
        time.sleep(3600)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
