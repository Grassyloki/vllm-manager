#!/usr/bin/env bash
# OpenAI-compatible API test suite — whiptail TUI
# Config stored next to the script as .api_tester.json (hidden dotfile).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.api_tester.json"

# ---------- colors ----------
# Minimal palette: mostly terminal default, one cool accent, muted gray for chrome.
C_RESET=$'\033[0m'
C_DIM=$'\033[2m'
C_MUTED=$'\033[38;5;244m'   # soft gray
C_ACCENT=$'\033[38;5;110m'  # soft blue-cyan
C_OK=$'\033[38;5;108m'      # muted green
C_WARN=$'\033[38;5;179m'    # muted amber
C_ERR=$'\033[38;5;174m'     # muted red

# whiptail: use terminal's own background everywhere; only the selection row
# is inverted. This removes the DOS-blue windows and bright-yellow title bars.
export NEWT_COLORS='
root=,
window=,
border=gray,
shadow=,
title=gray,
textbox=,
button=,
compactbutton=,
listbox=,
actlistbox=black,white
sellistbox=,
actsellistbox=black,white
entry=,
disentry=gray,
label=,
checkbox=,
actcheckbox=black,white
roottext=gray,
helpline=gray,
emptyscale=,
fullscale=gray,'

banner() {
    printf '\n  %sapi tester%s\n  %s──────────%s\n\n' \
        "$C_ACCENT" "$C_RESET" "$C_MUTED" "$C_RESET"
}

# ---------- dependencies ----------
check_deps() {
    local missing=()
    for cmd in whiptail jq curl bc; do
        command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
    done
    if ((${#missing[@]} > 0)); then
        printf '%sMissing commands:%s %s\n' "$C_ERR" "$C_RESET" "${missing[*]}"
        echo "  Arch:         sudo pacman -S libnewt jq curl bc"
        echo "  Debian/Ubuntu: sudo apt install whiptail jq curl bc"
        echo "  Rocky/RHEL:    sudo dnf install newt jq curl bc"
        exit 1
    fi
}

# ---------- config ----------
init_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cat >"$CONFIG_FILE" <<'EOF'
{
  "test_prompt": "what is a buffer overflow",
  "endpoints": [
    { "name": "Ollama",   "url": "http://localhost:11434/v1",  "auth": "" },
    { "name": "Lemonade", "url": "http://localhost:13305/api/v1", "auth": "" }
  ]
}
EOF
        chmod 600 "$CONFIG_FILE"
        printf '%sCreated %s%s\n' "$C_OK" "$CONFIG_FILE" "$C_RESET"
    fi
}

cfg_get() { jq -r "$1" "$CONFIG_FILE"; }

cfg_update() {
    local tmp
    tmp=$(mktemp)
    jq "$@" "$CONFIG_FILE" >"$tmp" && mv "$tmp" "$CONFIG_FILE"
}

ep_url() { cfg_get ".endpoints[$1].url"; }
ep_name() { cfg_get ".endpoints[$1].name"; }
ep_auth() { cfg_get ".endpoints[$1].auth"; }

# ---------- endpoint picker ----------
pick_endpoint() {
    while :; do
        local items=() count i name url
        count=$(cfg_get '.endpoints | length')
        for ((i = 0; i < count; i++)); do
            name=$(cfg_get ".endpoints[$i].name")
            url=$(cfg_get ".endpoints[$i].url")
            items+=("$i" "$name  —  $url")
        done
        items+=("+" "[ Add new endpoint ]")
        items+=("q" "[ Quit ]")

        local choice
        choice=$(whiptail --title "API Tester — Select Endpoint" \
            --menu "Saved endpoints:" 20 80 12 "${items[@]}" 3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            q) return 0 ;;
            +) add_endpoint ;;
            *)
                action_menu "$choice"
                [[ $? -eq 2 ]] && return 0
                ;;
        esac
    done
}

add_endpoint() {
    local name url auth
    name=$(whiptail --title "New Endpoint" --inputbox "Name:" 10 60 3>&1 1>&2 2>&3) || return
    [[ -z "$name" ]] && return
    url=$(whiptail --title "New Endpoint" \
        --inputbox "Base URL (e.g. http://localhost:11434/v1):" 10 70 3>&1 1>&2 2>&3) || return
    [[ -z "$url" ]] && return
    auth=$(whiptail --title "New Endpoint" \
        --inputbox "Auth token (leave blank if none):" 10 70 3>&1 1>&2 2>&3) || auth=""
    cfg_update --arg n "$name" --arg u "$url" --arg a "${auth:-}" \
        '.endpoints += [{name:$n, url:$u, auth:$a}]'
}

# ---------- action menu ----------
# To add a new item: append "tag|label|function_name" and define the function.
# Function receives the endpoint index as $1.
ACTIONS=(
    "status|Check endpoint status|action_status"
    "models|List available models|action_list_models"
    "test|Run test (tok/s + output)|action_run_test"
    "bulk|Bulk test (select many, summary)|action_bulk_test"
    "prompt|Edit test prompt|action_edit_prompt"
    "edit|Edit this endpoint|action_edit_endpoint"
    "delete|Delete this endpoint|action_delete_endpoint"
)

action_menu() {
    local idx="$1"
    while :; do
        local items=() entry tag label _func name url
        name=$(ep_name "$idx")
        url=$(ep_url "$idx")
        for entry in "${ACTIONS[@]}"; do
            IFS='|' read -r tag label _func <<<"$entry"
            items+=("$tag" "$label")
        done
        items+=("back" "← Back to endpoint list")
        items+=("quit" "Quit")

        local choice
        choice=$(whiptail --title "$name  —  $url" \
            --menu "Action:" 22 80 14 "${items[@]}" 3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            back) return 0 ;;
            quit) return 2 ;;
            *)
                local func=""
                for entry in "${ACTIONS[@]}"; do
                    IFS='|' read -r tag _ func <<<"$entry"
                    [[ "$tag" == "$choice" ]] && break
                done
                if [[ -n "$func" ]]; then
                    clear
                    banner
                    printf '%s[%s]%s  %s\n\n' "$C_DIM" "$name" "$C_RESET" "$url"
                    "$func" "$idx"
                    local rc=$?
                    echo
                    printf '%sPress enter to return to menu…%s' "$C_DIM" "$C_RESET"
                    read -r
                    # endpoint was deleted: bail out to endpoint picker
                    [[ "$choice" == "delete" && "$rc" -eq 0 ]] && return 0
                fi
                ;;
        esac
    done
}

# ---------- curl helper ----------
curl_auth_args() {
    local auth
    auth=$(ep_auth "$1")
    if [[ -n "$auth" ]]; then
        printf -- '-H\nAuthorization: Bearer %s\n' "$auth"
    fi
}

# ---------- actions ----------
action_status() {
    local idx="$1" base code t0 t1 elapsed tmp auth hdr=()
    base=$(ep_url "$idx")
    auth=$(ep_auth "$idx")
    [[ -n "$auth" ]] && hdr=(-H "Authorization: Bearer $auth")

    printf '%s» Status check%s  →  %s/models\n\n' "$C_ACCENT" "$C_RESET" "$base"
    tmp=$(mktemp)
    t0=$(date +%s.%N)
    code=$(curl -sS "${hdr[@]}" --max-time 10 \
        -o "$tmp" -w '%{http_code}' "$base/models" 2>/dev/null || echo "000")
    t1=$(date +%s.%N)
    elapsed=$(echo "scale=3; $t1 - $t0" | bc)

    if [[ "$code" == "200" ]]; then
        local mc
        mc=$(jq '.data | length' "$tmp" 2>/dev/null || echo "?")
        printf '%s● up%s    http %s    %ss\n' "$C_OK" "$C_RESET" "$code" "$elapsed"
        printf '%smodels visible%s  %s\n' "$C_MUTED" "$C_RESET" "$mc"
    elif [[ "$code" == "000" ]]; then
        printf '%s● unreachable%s  connection failed or timed out\n' "$C_ERR" "$C_RESET"
    else
        printf '%s● http %s%s   %ss\n\n' "$C_WARN" "$code" "$C_RESET" "$elapsed"
        head -c 800 "$tmp"
        echo
    fi
    rm -f "$tmp"
}

action_list_models() {
    local idx="$1" base resp auth hdr=()
    base=$(ep_url "$idx")
    auth=$(ep_auth "$idx")
    [[ -n "$auth" ]] && hdr=(-H "Authorization: Bearer $auth")

    printf '%s» Models%s  →  %s/models\n\n' "$C_ACCENT" "$C_RESET" "$base"
    resp=$(curl -sS "${hdr[@]}" --max-time 15 "$base/models" 2>&1) || {
        printf '%sRequest failed:%s %s\n' "$C_ERR" "$C_RESET" "$resp"
        return 1
    }

    if echo "$resp" | jq -e '.data' >/dev/null 2>&1; then
        local count
        count=$(echo "$resp" | jq '.data | length')
        printf '%s%s models%s\n' "$C_MUTED" "$count" "$C_RESET"
        while IFS= read -r m; do
            printf '  %s·%s %s\n' "$C_MUTED" "$C_RESET" "$m"
        done < <(echo "$resp" | jq -r '.data[].id')
    else
        printf '%sUnexpected response:%s\n' "$C_WARN" "$C_RESET"
        echo "$resp" | jq . 2>/dev/null || echo "$resp"
    fi
}

pick_model() {
    local idx="$1" base resp auth hdr=()
    base=$(ep_url "$idx")
    auth=$(ep_auth "$idx")
    [[ -n "$auth" ]] && hdr=(-H "Authorization: Bearer $auth")

    resp=$(curl -sS "${hdr[@]}" --max-time 15 "$base/models" 2>/dev/null) || return 1
    local ids=()
    mapfile -t ids < <(echo "$resp" | jq -r '.data[].id' 2>/dev/null)
    if ((${#ids[@]} == 0)); then
        whiptail --title "Error" --msgbox "No models returned from $base/models" 10 60
        return 1
    fi
    local items=() id
    for id in "${ids[@]}"; do
        items+=("$id" "")
    done
    whiptail --title "Select Model" --menu "Pick a model to test:" 20 70 12 \
        "${items[@]}" 3>&1 1>&2 2>&3
}

# Streams a chat completion. Populates globals:
#   LAST_TPS LAST_COUNT LAST_TTFB LAST_DURATION LAST_ERROR
# Args: idx model [show_output:1|0]
run_stream_test() {
    local idx="$1" model="$2" show="${3:-1}"
    local base prompt payload auth hdr=()
    base=$(ep_url "$idx")
    auth=$(ep_auth "$idx")
    [[ -n "$auth" ]] && hdr=(-H "Authorization: Bearer $auth")
    prompt=$(cfg_get '.test_prompt')
    payload=$(jq -nc --arg m "$model" --arg p "$prompt" \
        '{model:$m, stream:true, messages:[{role:"user", content:$p}]}')

    LAST_TPS=""; LAST_COUNT=0; LAST_TTFB=""; LAST_DURATION=""; LAST_ERROR=""
    local first_ts="" end_ts line data content req_start errfile errbuf=""
    errfile=$(mktemp)
    req_start=$(date +%s.%N)
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" != data:* ]]; then
            [[ -n "$line" ]] && errbuf+="$line"$'\n'
            continue
        fi
        data="${line#data:}"; data="${data# }"
        [[ "$data" == "[DONE]" ]] && break
        content=$(echo "$data" | jq -r '.choices[0].delta.content // empty' 2>/dev/null) || continue
        [[ -z "$content" ]] && continue
        [[ -z "$first_ts" ]] && first_ts=$(date +%s.%N)
        LAST_COUNT=$((LAST_COUNT + 1))
        [[ "$show" == "1" ]] && printf '%s' "$content"
    done < <(curl -sSN "${hdr[@]}" -X POST "$base/chat/completions" \
        -H "Content-Type: application/json" -d "$payload" 2>"$errfile")
    end_ts=$(date +%s.%N)

    if [[ -z "$first_ts" || "$LAST_COUNT" -eq 0 ]]; then
        LAST_ERROR="${errbuf:-$(cat "$errfile" 2>/dev/null)}"
        [[ -z "$LAST_ERROR" ]] && LAST_ERROR="no stream"
        rm -f "$errfile"
        return 1
    fi
    rm -f "$errfile"

    LAST_TTFB=$(echo "scale=3; $first_ts - $req_start" | bc)
    LAST_DURATION=$(echo "scale=3; $end_ts - $first_ts" | bc)
    if [[ "$(echo "$LAST_DURATION > 0" | bc)" == "1" ]]; then
        LAST_TPS=$(echo "scale=2; $LAST_COUNT / $LAST_DURATION" | bc)
    else
        LAST_TPS="∞"
    fi
    return 0
}

action_run_test() {
    local idx="$1" model prompt
    model=$(pick_model "$idx") || {
        printf '%scancelled%s\n' "$C_MUTED" "$C_RESET"
        return 1
    }
    prompt=$(cfg_get '.test_prompt')

    printf '%s» Test%s  model=%s%s%s\n' "$C_ACCENT" "$C_RESET" "$C_ACCENT" "$model" "$C_RESET"
    printf '%sprompt:%s %s\n\n' "$C_MUTED" "$C_RESET" "$prompt"
    printf '%s─── stream ───%s\n' "$C_MUTED" "$C_RESET"

    if run_stream_test "$idx" "$model" 1; then
        echo
        printf '%s─── end ───%s\n\n' "$C_MUTED" "$C_RESET"
        printf '%schunks / tokens%s   %d\n' "$C_MUTED" "$C_RESET" "$LAST_COUNT"
        printf '%sttft%s              %ss\n' "$C_MUTED" "$C_RESET" "$LAST_TTFB"
        printf '%sduration%s          %ss\n' "$C_MUTED" "$C_RESET" "$LAST_DURATION"
        printf '%stok/s%s             %s%s%s\n' \
            "$C_MUTED" "$C_RESET" "$C_ACCENT" "$LAST_TPS" "$C_RESET"
    else
        echo
        printf '%s─── end ───%s\n' "$C_MUTED" "$C_RESET"
        printf '%sNo stream received.%s\n' "$C_ERR" "$C_RESET"
        [[ -n "$LAST_ERROR" ]] && printf '%s%s%s\n' "$C_MUTED" "$LAST_ERROR" "$C_RESET"
        return 1
    fi
}

pick_models_multi() {
    local idx="$1" base resp auth hdr=()
    base=$(ep_url "$idx")
    auth=$(ep_auth "$idx")
    [[ -n "$auth" ]] && hdr=(-H "Authorization: Bearer $auth")
    resp=$(curl -sS "${hdr[@]}" --max-time 15 "$base/models" 2>/dev/null) || return 1
    local ids=()
    mapfile -t ids < <(echo "$resp" | jq -r '.data[].id' 2>/dev/null)
    if ((${#ids[@]} == 0)); then
        whiptail --title "Error" --msgbox "No models returned from $base/models" 10 60
        return 1
    fi
    local items=() id
    for id in "${ids[@]}"; do
        items+=("$id" "" "OFF")
    done
    whiptail --title "Select Models  (space = toggle, enter = start)" \
        --checklist "Pick models to bulk-test:" 20 74 12 \
        "${items[@]}" 3>&1 1>&2 2>&3
}

action_bulk_test() {
    local idx="$1" raw
    raw=$(pick_models_multi "$idx") || {
        printf '%scancelled%s\n' "$C_MUTED" "$C_RESET"
        return 1
    }
    local models=()
    eval "models=($raw)"
    if ((${#models[@]} == 0)); then
        printf '%sno models selected%s\n' "$C_MUTED" "$C_RESET"
        return 1
    fi

    local prompt; prompt=$(cfg_get '.test_prompt')
    printf '%s» Bulk test%s  %d model(s)\n' "$C_ACCENT" "$C_RESET" "${#models[@]}"
    printf '%sprompt:%s %s\n\n' "$C_MUTED" "$C_RESET" "$prompt"

    local results_tmp; results_tmp=$(mktemp)
    local m i=0 total="${#models[@]}" short_err
    for m in "${models[@]}"; do
        i=$((i + 1))
        printf '%s[%d/%d]%s %-44s ' "$C_MUTED" "$i" "$total" "$C_RESET" "$m"
        if run_stream_test "$idx" "$m" 0; then
            printf '%s%8s tok/s%s  %s(%d tok, %ss)%s\n' \
                "$C_ACCENT" "$LAST_TPS" "$C_RESET" \
                "$C_MUTED" "$LAST_COUNT" "$LAST_DURATION" "$C_RESET"
            printf 'ok\t%s\t%s\t%s\t%s\n' "$LAST_TPS" "$m" "$LAST_COUNT" "$LAST_DURATION" >>"$results_tmp"
        else
            short_err="${LAST_ERROR%%$'\n'*}"
            printf '%sfailed%s  %s%s%s\n' "$C_ERR" "$C_RESET" "$C_MUTED" "${short_err:0:40}" "$C_RESET"
            printf 'fail\t0\t%s\t-\t-\n' "$m" >>"$results_tmp"
        fi
    done

    echo
    printf '%s─── summary ───%s\n\n' "$C_MUTED" "$C_RESET"
    printf '  %-44s  %10s  %8s  %8s\n' "model" "tok/s" "tokens" "sec"
    local rule; rule=$(printf '─%.0s' $(seq 1 78))
    printf '  %s%s%s\n' "$C_MUTED" "$rule" "$C_RESET"

    # sort alphabetically by model name (case-insensitive)
    sort -t$'\t' -k3,3f "$results_tmp" | while IFS=$'\t' read -r st tps mname cnt dur; do
        if [[ "$st" == "ok" ]]; then
            printf '  %-44s  %s%10s%s  %8s  %8s\n' \
                "$mname" "$C_ACCENT" "$tps" "$C_RESET" "$cnt" "$dur"
        else
            printf '  %s%-44s%s  %10s  %8s  %8s\n' \
                "$C_ERR" "$mname" "$C_RESET" "fail" "-" "-"
        fi
    done
    rm -f "$results_tmp"
}

action_edit_prompt() {
    local current new
    current=$(cfg_get '.test_prompt')
    new=$(whiptail --title "Edit Test Prompt" \
        --inputbox "Prompt sent by 'Run test':" 12 78 "$current" 3>&1 1>&2 2>&3) || return
    cfg_update --arg p "$new" '.test_prompt = $p'
    printf '%sSaved.%s  new prompt: %s\n' "$C_OK" "$C_RESET" "$new"
}

action_edit_endpoint() {
    local idx="$1" name url auth
    name=$(ep_name "$idx")
    url=$(ep_url "$idx")
    auth=$(ep_auth "$idx")
    name=$(whiptail --title "Edit Endpoint" --inputbox "Name:" 10 60 "$name" 3>&1 1>&2 2>&3) || return
    url=$(whiptail --title "Edit Endpoint" --inputbox "Base URL:" 10 70 "$url" 3>&1 1>&2 2>&3) || return
    auth=$(whiptail --title "Edit Endpoint" --inputbox "Auth token:" 10 70 "$auth" 3>&1 1>&2 2>&3) || auth=""
    cfg_update --argjson i "$idx" --arg n "$name" --arg u "$url" --arg a "$auth" \
        '.endpoints[$i] = {name:$n, url:$u, auth:$a}'
    printf '%sSaved.%s\n' "$C_OK" "$C_RESET"
}

action_delete_endpoint() {
    local idx="$1" name
    name=$(ep_name "$idx")
    whiptail --title "Delete" --yesno "Delete endpoint '$name'?" 10 60 || return 1
    cfg_update --argjson i "$idx" 'del(.endpoints[$i])'
    printf '%sDeleted.%s\n' "$C_OK" "$C_RESET"
    return 0
}

# ---------- main ----------
main() {
    check_deps
    init_config
    clear
    banner
    printf '%sconfig:%s %s\n\n' "$C_DIM" "$C_RESET" "$CONFIG_FILE"
    pick_endpoint
    clear
    printf '%sBye.%s\n' "$C_ACCENT" "$C_RESET"
}

main "$@"
