#!/usr/bin/env bash
set -euo pipefail

# Scan files that Git would publish. Only file names are reported so the check
# itself does not copy a detected identifier into logs.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

declare -a text_files=()
declare -a notebooks=()
while IFS= read -r -d '' path; do
    case "$path" in
        scripts/check_public_release.sh)
            ;;
        *.ipynb)
            notebooks+=("$path")
            ;;
        *)
            if [[ ! -s "$path" ]] || grep -Iq . "$path"; then
                text_files+=("$path")
            fi
            ;;
    esac
done < <(git ls-files -co --exclude-standard -z)

failed=0

scan_text() {
    local label=$1
    local pattern=$2
    local matches
    if ((${#text_files[@]} == 0)); then
        return
    fi
    matches=$(rg -l -i -e "$pattern" -- "${text_files[@]}" 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        printf '[privacy] %s:\n%s\n' "$label" "$matches" >&2
        failed=1
    fi
}

scan_notebooks() {
    local pattern=$1
    local notebook
    if ! command -v jq >/dev/null 2>&1; then
        printf '[privacy] jq is required to inspect notebook text safely.\n' >&2
        failed=1
        return
    fi
    for notebook in "${notebooks[@]}"; do
        if jq -r '
            paths(strings) as $path
            | select(all($path[];
                . != "image/png"
                and . != "image/jpeg"
                and . != "application/pdf"))
            | getpath($path)
        ' "$notebook" 2>/dev/null | rg -i -e "$pattern" >/dev/null; then
            printf '[privacy] notebook text or metadata: %s\n' "$notebook" >&2
            failed=1
        fi
    done
}

path_pattern='/(srv/(scratch|project|data)|home|Users)/[^[:space:]"'\'']+|[A-Za-z]:\\Users\\[^[:space:]"'\'']+'
email_pattern='[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}'
user_id_pattern='(^|[^[:alnum:]_])z[0-9]{7}([^[:alnum:]_]|$)'
institution_pattern='(([[:alnum:]-]+\.)+(edu\.au|internal)|([[:alnum:]-]+\.){2,}local)([^[:alnum:].-]|$)|University of [[:upper:]]|[[:upper:]][[:alpha:]& -]+ University'
scheduler_pattern='^[[:space:]]*#(PBS[[:space:]]+-(P|q|M)|SBATCH[[:space:]].*(--account|--partition|--mail-user))|(^|[[:space:]])(-q|--partition)[[:space:]]+["'\'']?[[:alnum:]]'
credential_pattern='(api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd)[[:space:]]*[:=][[:space:]]*["'\'']?[^$[:space:]"'\'']+|-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----'

scan_text "absolute user or organisation path" "$path_pattern"
scan_text "email address" "$email_pattern"
scan_text "local user identifier" "$user_id_pattern"
scan_text "institution or internal domain" "$institution_pattern"
scan_text "hard-coded scheduler account, queue, or email" "$scheduler_pattern"
scan_text "credential-like assignment or private key" "$credential_pattern"

notebook_pattern="$path_pattern|$email_pattern|$user_id_pattern|$institution_pattern|$credential_pattern"
if [[ -n "${PRIVATE_MARKERS_REGEX:-}" ]]; then
    scan_text "project-specific private marker" "$PRIVATE_MARKERS_REGEX"
    notebook_pattern+="|$PRIVATE_MARKERS_REGEX"
    if git ls-files -co --exclude-standard | rg -i -e "$PRIVATE_MARKERS_REGEX" >/dev/null; then
        printf '[privacy] project-specific private marker in a file name.\n' >&2
        failed=1
    fi
fi
scan_notebooks "$notebook_pattern"

if ((failed)); then
    printf '[privacy] public-release check failed.\n' >&2
    exit 1
fi

printf '[privacy] public-release check passed (%d text files, %d notebooks).\n' \
    "${#text_files[@]}" "${#notebooks[@]}"
