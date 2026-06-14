#!/usr/bin/env bash
# Smoke-test the local vLLM OpenAI-compatible server.
# Usage: ./test_vllm.sh [host:port] [model-name] [wait-seconds]
set -u

HOST="${1:-127.0.0.1:1001}"
MODEL="${2:-qwen3.5-27b}"
WAIT_SECS="${3:-900}"   # default: wait up to 15 min for server readiness
BASE="http://${HOST}"
HOSTNAME="${HOST%:*}"
PORT="${HOST##*:}"

hr() { printf '%s\n' "------------------------------------------------------------"; }

hr
echo "Target:       ${BASE}"
echo "Model:        ${MODEL}"
echo "Wait budget:  ${WAIT_SECS}s"
hr

echo "[1/4] Waiting for server to listen on ${HOST} (max ${WAIT_SECS}s)"
deadline=$(( $(date +%s) + WAIT_SECS ))
while :; do
    if (exec 3<>"/dev/tcp/${HOSTNAME}/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "  TCP port is open."
        break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "  FAIL: ${HOST} never opened within ${WAIT_SECS}s."
        echo "  Tail the log to see progress:"
        echo "    tail -f /root/.vllm-logs/${MODEL}.log"
        exit 1
    fi
    sleep 3
done

echo "  Waiting for /v1/models to respond 200 (server fully initialised)"
while :; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${BASE}/v1/models" || echo "000")
    if [ "${code}" = "200" ]; then
        echo "  /v1/models -> 200"
        break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "  FAIL: /v1/models never returned 200 (last code: ${code})."
        exit 1
    fi
    sleep 5
done

hr
echo "[2/4] GET /health"
code=$(curl -s -o /tmp/vllm_health.txt -w '%{http_code}' --max-time 10 "${BASE}/health" || echo "000")
echo "  HTTP ${code}"
[ -s /tmp/vllm_health.txt ] && sed 's/^/  | /' /tmp/vllm_health.txt

hr
echo "[3/4] GET /v1/models"
curl -sS --max-time 10 "${BASE}/v1/models" | tee /tmp/vllm_models.json
echo
if ! grep -q "\"${MODEL}\"" /tmp/vllm_models.json 2>/dev/null; then
    echo "  WARN: model id '${MODEL}' not found in /v1/models output."
fi

hr
echo "[4/4] POST /v1/chat/completions"
REQ=$(printf '{"model":"%s","messages":[{"role":"system","content":"You are a terse assistant. Do not think out loud; answer directly."},{"role":"user","content":"Reply with exactly the word: pong"}],"max_tokens":256,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' "${MODEL}")
echo "  Request: ${REQ}"
echo "  Response:"
start=$(date +%s)
http_code=$(curl -sS --max-time 300 \
    -o /tmp/vllm_chat.json \
    -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -X POST "${BASE}/v1/chat/completions" \
    -d "${REQ}")
end=$(date +%s)
elapsed=$(( end - start ))
echo "    HTTP ${http_code}  (${elapsed}s)"
sed 's/^/    | /' /tmp/vllm_chat.json
echo

hr
if [ "${http_code}" = "200" ]; then
    python3 <<'PY'
import json
with open("/tmp/vllm_chat.json") as f:
    d = json.load(f)
choice = d["choices"][0]
msg = choice["message"].get("content") or ""
reasoning = choice["message"].get("reasoning") or ""
usage = d.get("usage", {})
print(f"Assistant content:   {msg!r}")
if reasoning:
    print(f"Assistant reasoning: {reasoning!r}")
print(f"Finish reason: {choice.get('finish_reason')}")
print(f"Tokens: prompt={usage.get('prompt_tokens')} "
      f"completion={usage.get('completion_tokens')} "
      f"total={usage.get('total_tokens')}")
print("PASS")
PY
else
    echo "FAIL: chat endpoint returned HTTP ${http_code}"
    exit 1
fi
