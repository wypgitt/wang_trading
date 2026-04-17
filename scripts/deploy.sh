#!/usr/bin/env bash
# Deploy the wang_trading stack to a Linux host. Does NOT start the live
# trading service — operator runs preflight then `systemctl start
# wang-live-trading` explicitly.
#
# Usage:  sudo ./scripts/deploy.sh [--repo /path/to/repo]

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
INSTALL_DIR="/opt/wang_trading"
LOG_DIR="/var/log/wang_trading"
SERVICE_USER="wang"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$(id -u)" -ne 0 ]]; then
    echo "deploy.sh must run as root (sudo)." >&2
    exit 1
fi

echo "==> Repo dir   : ${REPO_DIR}"
echo "==> Install dir: ${INSTALL_DIR}"
echo "==> Log dir    : ${LOG_DIR}"

# 1. Service user
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    echo "==> Creating service user '${SERVICE_USER}'"
    useradd --system --home "${INSTALL_DIR}" --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

# 2. Directories
echo "==> Creating directories"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0755 "${INSTALL_DIR}"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0755 "${INSTALL_DIR}/logs"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0755 "${INSTALL_DIR}/data"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0755 "${LOG_DIR}"

# 3. Copy source
echo "==> Copying source into ${INSTALL_DIR}"
rsync -a --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='.venv' --exclude='venv' \
    --exclude='logs/*' --exclude='data/*' \
    "${REPO_DIR}/" "${INSTALL_DIR}/"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

# 4. Python venv + deps
echo "==> Setting up virtualenv"
if [[ ! -d "${INSTALL_DIR}/venv" ]]; then
    sudo -u "${SERVICE_USER}" "${PYTHON_BIN}" -m venv "${INSTALL_DIR}/venv"
fi
sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/venv/bin/pip" install --upgrade pip
sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/venv/bin/pip" install \
    -r "${INSTALL_DIR}/requirements.txt"

# 5. Config: seed from examples but do not overwrite operator-edited files
echo "==> Seeding config from examples (no overwrite)"
for f in settings live_trading futures_contracts paper_trading; do
    ex="${INSTALL_DIR}/config/${f}.example.yaml"
    yml="${INSTALL_DIR}/config/${f}.yaml"
    if [[ -f "${ex}" && ! -f "${yml}" ]]; then
        cp "${ex}" "${yml}"
        chown "${SERVICE_USER}:${SERVICE_USER}" "${yml}"
        chmod 0640 "${yml}"
    fi
done

# 6. Systemd unit
echo "==> Installing systemd unit"
install -m 0644 "${INSTALL_DIR}/config/systemd/wang-live-trading.service" \
    /etc/systemd/system/wang-live-trading.service
systemctl daemon-reload

# 7. Supervisor configs (optional — only if supervisor is installed)
if command -v supervisorctl >/dev/null 2>&1; then
    echo "==> Installing supervisor configs"
    install -d /etc/supervisor/conf.d
    for f in live_trading monitoring data_ingestion retrain_scheduler; do
        install -m 0644 "${INSTALL_DIR}/config/supervisor/${f}.conf" \
            "/etc/supervisor/conf.d/wang_${f}.conf"
    done
    supervisorctl reread || true
    supervisorctl update || true
fi

# 8. Log rotation
cat >/etc/logrotate.d/wang_trading <<'ROTATE'
/var/log/wang_trading/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su wang wang
}
ROTATE

cat <<INSTRUCTIONS

==> Deployment complete.

Manual next steps (operator):

  1. Edit /opt/wang_trading/config/live_trading.yaml — fill API keys.
  2. Edit /opt/wang_trading/config/live_trading.env — export
     WANG_ALLOW_LIVE_TRADING=yes (or the crypto/futures variant) only when
     you really intend to trade live.
  3. Run preflight:
       sudo -u ${SERVICE_USER} /opt/wang_trading/venv/bin/python \
           -m src.execution.preflight --full-check
  4. When every blocker passes and you're ready:
       sudo systemctl start wang-live-trading
       sudo journalctl -u wang-live-trading -f

  Emergency stop (graceful):
       sudo systemctl stop wang-live-trading
  Emergency flatten:
       sudo -u ${SERVICE_USER} /opt/wang_trading/venv/bin/python \
           -m src.execution.live_trading --emergency-flatten

INSTRUCTIONS
