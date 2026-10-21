#!/usr/bin/env bash
# PATH shim for command tracing (plan B1.2). Not run directly: install_shims.sh makes one
# symlink per sub-tool name pointing here; the symlink name selects the real binary.
#
# For every call it logs one JSON record (argv, cwd, env subset, start/end time, exit code,
# files touched) to "$BISCUIT_TRACE_DIR/<start>-<pid>-<tool>.json", then exits with the real
# tool's exit status. stdout/stderr are captured to sibling .out/.err files and replayed.
#
# Environment:
#   BISCUIT_TRACE_DIR        directory for records (required)
#   BISCUIT_SHIM_REAL_PATH   colon list searched for the real binary
#                            (default /opt/biscuit/bin:/opt/cmorph/bin)
#   BISCUIT_SHIM_ENV         space list of env vars to record
#   BISCUIT_SHIM_CAPTURE     1 (default) capture stdout/stderr, 0 pass through untouched
set -u

name=$(basename "$0")
trace_dir=${BISCUIT_TRACE_DIR:-}
real_path=${BISCUIT_SHIM_REAL_PATH:-/opt/biscuit/bin:/opt/cmorph/bin}
env_list=${BISCUIT_SHIM_ENV:-"OMP_NUM_THREADS MCR_CACHE_ROOT TMP DISPLAY FREESURFER_HOME SUBJECTS_DIR BISCUIT_ROOT"}
capture=${BISCUIT_SHIM_CAPTURE:-1}

real=""
old_ifs=$IFS
IFS=:
for d in $real_path; do
    if [ -x "$d/$name" ] && [ ! -d "$d/$name" ]; then
        real="$d/$name"
        break
    fi
done
IFS=$old_ifs
if [ -z "$real" ]; then
    echo "trace_shim: no executable '$name' in $real_path" >&2
    exit 127
fi
if [ -z "$trace_dir" ]; then
    exec "$real" "$@"
fi
mkdir -p "$trace_dir"

if stat -c %s / >/dev/null 2>&1; then
    stat_fmt() { stat -c '%s	%Y' "$1"; } # size, mtime (GNU)
else
    stat_fmt() { stat -f '%z	%m' "$1"; } # BSD
fi

now() {
    local t
    t=$(date +%s.%N)
    case $t in *N) t=$(date +%s) ;; esac
    printf '%s' "$t"
}

jesc() { # JSON string body (no quotes) of $1
    printf '%s' "$1" | tr '\t\r' '  ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' |
        awk '{ printf "%s%s", (NR > 1 ? "\\n" : ""), $0 }'
}

jarr() { # JSON array of the lines on stdin
    local first=1 line
    printf '['
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        [ $first -eq 1 ] || printf ','
        first=0
        printf '"%s"' "$(jesc "$line")"
    done
    printf ']'
}

jarr_args() { # JSON array of the arguments (newlines and empty strings preserved)
    local first=1 a
    printf '['
    for a in "$@"; do
        [ $first -eq 1 ] || printf ','
        first=0
        printf '"%s"' "$(jesc "$a")"
    done
    printf ']'
}

absolute() { case $1 in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac }

snapshot() { # "path<TAB>size<TAB>mtime" for cwd entries, argument paths and prefix globs
    local p a f
    {
        for p in "$PWD"/* "$PWD"/.[!.]*; do
            [ -e "$p" ] && printf '%s\n' "$p"
        done
        for a in "$@"; do
            case $a in -* | '') continue ;; esac
            [ ${#a} -lt 512 ] || continue
            p=$(absolute "$a")
            [ -e "$p" ] && printf '%s\n' "$p"
            if [ -d "$p" ]; then
                for f in "$p"/*; do [ -e "$f" ] && printf '%s\n' "$f"; done
            fi
            for f in "$p"*; do [ -e "$f" ] && printf '%s\n' "$f"; done
        done
    } | sort -u | while IFS= read -r p; do
        printf '%s\t%s\n' "$p" "$(stat_fmt "$p")"
    done
}

tmp=$(mktemp -d "${TMPDIR:-/tmp}/shim.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
snapshot "$@" >"$tmp/pre"
inputs=""
for a in "$@"; do
    case $a in -* | '') continue ;; esac
    [ ${#a} -lt 512 ] && [ -e "$(absolute "$a")" ] && inputs="$inputs$(absolute "$a")
"
done

t0=$(now)
id="${t0}-$$-${name}"
if [ "$capture" = 1 ]; then
    "$real" "$@" >"$trace_dir/$id.out" 2>"$trace_dir/$id.err"
    rc=$?
    cat "$trace_dir/$id.out"
    cat "$trace_dir/$id.err" >&2
else
    "$real" "$@"
    rc=$?
fi
t1=$(now)

snapshot "$@" >"$tmp/post"
created=$(awk -F'\t' 'FILENAME == ARGV[1] { pre[$1] = $2 "\t" $3; next } !($1 in pre) { print $1 }' "$tmp/pre" "$tmp/post")
modified=$(awk -F'\t' 'FILENAME == ARGV[1] { pre[$1] = $2 "\t" $3; next } ($1 in pre) && pre[$1] != $2 "\t" $3 { print $1 }' "$tmp/pre" "$tmp/post")
deleted=$(awk -F'\t' 'FILENAME == ARGV[1] { post[$1] = 1; next } !($1 in post) { print $1 }' "$tmp/post" "$tmp/pre")

envjson=""
for v in $env_list; do
    if [ -n "${!v+x}" ]; then
        [ -z "$envjson" ] || envjson="$envjson,"
        envjson="$envjson\"$v\":\"$(jesc "${!v}")\""
    fi
done

{
    printf '{"tool":"%s","real":"%s","pid":%s,"ppid":%s,"cwd":"%s",' \
        "$(jesc "$name")" "$(jesc "$real")" "$$" "$PPID" "$(jesc "$PWD")"
    printf '"argv":%s,' "$(jarr_args "$@")"
    printf '"start":%s,"end":%s,"exit":%s,' "$t0" "$t1" "$rc"
    printf '"env":{%s},' "$envjson"
    printf '"inputs":%s,"created":%s,"modified":%s,"deleted":%s' \
        "$(printf '%s' "$inputs" | jarr)" "$(printf '%s\n' "$created" | jarr)" \
        "$(printf '%s\n' "$modified" | jarr)" "$(printf '%s\n' "$deleted" | jarr)"
    if [ "$capture" = 1 ]; then
        printf ',"stdout":"%s.out","stderr":"%s.err"' "$id" "$id"
    fi
    printf '}\n'
} >"$trace_dir/$id.json"
exit "$rc"
