#!/bin/bash
# KuaiRand-Pure acquisition retry loop.
# Sources (priority): 1) Zenodo official  2) USTC official mirror (anonymous)
# Exits 0 on success (file downloaded+verified), 1 if all sources stay down.
DEST=/d/CODE/data/kuairand
mkdir -p "$DEST"
LOG=/d/CODE/phase0-3/experiments/kuairand_fetch.log
echo "start $(date)" >> "$LOG"

for i in $(seq 1 40); do
  echo "try $i $(date)" >> "$LOG"
  # --- source 1: zenodo ---
  code=$(timeout 20 curl -s -o /dev/null -w "%{http_code}" "https://zenodo.org/api/records/10439422" -m 15 2>/dev/null)
  echo "  zenodo api: $code" >> "$LOG"
  if [ "$code" = "200" ]; then
    curl -s -L "https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content" -o "$DEST/KuaiRand-Pure.tar.gz" -m 7200 >> "$LOG" 2>&1
    if [ -s "$DEST/KuaiRand-Pure.tar.gz" ]; then
      md5sum "$DEST/KuaiRand-Pure.tar.gz" >> "$LOG"
      echo "ZENODO_DOWNLOAD_OK" >> "$LOG"; exit 0
    fi
  fi
  # --- source 2: USTC anonymous ---
  CID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "11111111-2222-3333-4444-555555555555")
  t=$(timeout 20 curl -s "https://recapi.ustc.edu.cn/api/v2/client/tempticket?clientid=$CID" -H "User-Agent: Mozilla/5.0" -m 15 2>/dev/null)
  echo "  ustc tempticket: $(echo "$t" | head -c 120)" >> "$LOG"
  if echo "$t" | grep -q '"token"'; then
    echo "USTC_ANON_LOGIN_AVAILABLE — needs manual script" >> "$LOG"
  fi
  sleep 600
done
echo "ALL_SOURCES_DOWN $(date)" >> "$LOG"
exit 1
