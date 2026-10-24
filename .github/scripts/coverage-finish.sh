#!/usr/bin/env bash
# Turn coverage data into reports and optionally upload them to Codecov.
#
# Shared by the GitHub Actions workflows and the ECMWF HPC batch jobs.
# Run it from the repository root. It is report only: it ALWAYS exits 0,
# whatever fails inside (every failure is printed as a "WARNING:" line).
# It deliberately does not use `set -e`.

# Remember whether the caller had tracing on, then turn it off before anything
# touches the Codecov token. Tracing is restored only after the upload.
_cf_xtrace=0
case $- in *x*) _cf_xtrace=1 ;; esac
set +x

_cf_usage() {
    cat <<'EOF'
Usage: coverage-finish.sh [--flag NAME] [--sha SHA] [--summary FILE] [--upload] [--rcfile PATH]

Combine coverage data, write coverage.xml and htmlcov/, and print a markdown
coverage table. Run it from the repository root. It always exits 0.

  --flag NAME     Codecov flag (e.g. pr-cpu, nightly-gpu, distributed); also used in the summary heading
  --sha SHA       commit the coverage belongs to (passed to Codecov)
  --summary FILE  append the markdown table to FILE (e.g. "$GITHUB_STEP_SUMMARY")
  --upload        upload coverage.xml to Codecov when the CODECOV_TOKEN env var is non-empty
  --rcfile PATH   coverage configuration file (default: .coveragerc)
  -h, --help      show this help

Environment:
  CODECOV_TOKEN    Codecov upload token (never printed)
  COVERAGE_PYTHON  Python used to run coverage (default: python from the active environment)
EOF
}

_cf_warn() {
    echo "WARNING: coverage-finish: $*" >&2
}

_cf_flag=""
_cf_sha=""
_cf_summary=""
_cf_upload=0
_cf_rcfile=".coveragerc"

while [ $# -gt 0 ]; do
    case "$1" in
        -h | --help)
            _cf_usage
            exit 0
            ;;
        --upload)
            _cf_upload=1
            shift
            ;;
        --flag | --sha | --summary | --rcfile)
            if [ $# -lt 2 ]; then
                _cf_warn "option $1 needs a value; ignored"
                shift
                continue
            fi
            case "$1" in
                --flag) _cf_flag="$2" ;;
                --sha) _cf_sha="$2" ;;
                --summary) _cf_summary="$2" ;;
                --rcfile) _cf_rcfile="$2" ;;
            esac
            shift 2
            ;;
        *)
            _cf_warn "unknown argument '$1'; ignored"
            shift
            ;;
    esac
done

# Python that runs coverage.
_cf_py="${COVERAGE_PYTHON:-}"
if [ -z "$_cf_py" ]; then
    if command -v python >/dev/null 2>&1; then
        _cf_py=python
    else
        _cf_py=python3
    fi
fi

_cf_rc=()
if [ -f "$_cf_rcfile" ]; then
    _cf_rc=(--rcfile "$_cf_rcfile")
else
    _cf_warn "coverage configuration '$_cf_rcfile' not found; using coverage defaults"
fi

_cf_err="$(mktemp 2>/dev/null)" || _cf_err=/dev/null

# Run one coverage subcommand, report a failure as a warning.
# Usage: _cf_run NAME ARGS...  (stdout is kept, stderr goes to a warning on failure)
_cf_coverage() {
    local name="$1"
    shift
    "$_cf_py" -m coverage "$name" "${_cf_rc[@]}" "$@" 2>"$_cf_err"
    local status=$?
    if [ "$status" -ne 0 ]; then
        _cf_warn "'coverage $name' failed (exit $status): $(tr '\n' ' ' <"$_cf_err")"
    fi
    return "$status"
}

# 1. Merge per-process data files, if any. pytest-cov already combines at the end of each run,
#    so "No data to combine" is normal. --append keeps an existing .coverage.
if ! "$_cf_py" -m coverage combine "${_cf_rc[@]}" --append --quiet >"$_cf_err" 2>&1; then
    if grep -qi "no data to combine" "$_cf_err"; then
        echo "coverage-finish: no extra data files to combine"
    else
        _cf_warn "'coverage combine' failed: $(tr '\n' ' ' <"$_cf_err")"
    fi
fi

# 2. XML (for Codecov and diff-cover) and HTML reports.
_cf_coverage xml -o coverage.xml --quiet || true
_cf_coverage html -d htmlcov --quiet || true

# 3. Markdown table on stdout and in the summary file.
_cf_md=""
if _cf_md="$(_cf_coverage report --format=markdown)" && [ -n "$_cf_md" ]; then
    printf '%s\n' "$_cf_md"
    if [ -n "$_cf_summary" ]; then
        {
            if [ -n "$_cf_flag" ]; then
                printf '## Test coverage (%s)\n\n' "$_cf_flag"
            else
                printf '## Test coverage\n\n'
            fi
            printf '%s\n\n' "$_cf_md"
        } >>"$_cf_summary" 2>/dev/null || _cf_warn "could not write to summary file '$_cf_summary'"
    fi
else
    _cf_warn "no coverage report produced"
fi

# 4. Optional upload to Codecov. Tracing stays off here: the token must not be echoed.
if [ "$_cf_upload" -eq 1 ]; then
    if [ -z "${CODECOV_TOKEN:-}" ]; then
        echo "Codecov upload skipped: no token"
    elif [ ! -f coverage.xml ]; then
        _cf_warn "Codecov upload skipped: coverage.xml was not written"
    else
        _cf_cli=""
        if command -v codecovcli >/dev/null 2>&1; then
            _cf_cli="codecovcli"
        else
            echo "coverage-finish: installing codecov-cli"
            "$_cf_py" -m pip install --quiet codecov-cli >"$_cf_err" 2>&1 ||
                _cf_warn "pip install codecov-cli failed: $(tr '\n' ' ' <"$_cf_err")"
            if command -v codecovcli >/dev/null 2>&1; then
                _cf_cli="codecovcli"
            else
                _cf_bindir="$(dirname "$(command -v "$_cf_py" 2>/dev/null)")"
                if [ -x "$_cf_bindir/codecovcli" ]; then
                    _cf_cli="$_cf_bindir/codecovcli"
                fi
            fi
        fi

        if [ -z "$_cf_cli" ]; then
            _cf_warn "codecovcli not available; Codecov upload skipped"
        else
            # codecovcli reads the token from the CODECOV_TOKEN env var; not passing it
            # with -t keeps it out of the process list on shared nodes.
            export CODECOV_TOKEN
            _cf_args=(upload-process --disable-search -f coverage.xml)
            [ -n "$_cf_flag" ] && _cf_args+=(-F "$_cf_flag")
            [ -n "$_cf_sha" ] && _cf_args+=(-C "$_cf_sha")
            _cf_out="$("$_cf_cli" "${_cf_args[@]}" 2>&1)"
            _cf_status=$?
            # Never let the token reach the log, even if codecovcli echoes it.
            printf '%s\n' "${_cf_out//"$CODECOV_TOKEN"/***}"
            if [ "$_cf_status" -ne 0 ]; then
                _cf_warn "Codecov upload failed (exit $_cf_status)"
            else
                echo "coverage-finish: Codecov upload done${_cf_flag:+ (flag $_cf_flag)}"
            fi
            unset _cf_args _cf_out
        fi
    fi
fi

[ "$_cf_err" != /dev/null ] && rm -f "$_cf_err"

# Restore the caller's tracing, now that the token is no longer used.
if [ "$_cf_xtrace" -eq 1 ]; then
    set -x
fi

exit 0
