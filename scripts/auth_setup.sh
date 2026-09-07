#!/bin/bash
# Fetch and cache an AI Gateway bearer token for Alpha-Lab.
#
# Needs Kerberos credentials entitled to read the SCV signing key. If your own
# account lacks that, run this after: suu -tr "<ticket>" <proid>
#
# Set TOKEN_FILEPATH first if the default cache dir (/var/tmp/svcacct) is owned
# by another account, and export the same value wherever the pipeline runs.

# Prefer the installed console script; fall back to the module so this works
# from a checkout whose venv predates the entry point.
if [ -n "${VIRTUAL_ENV}" ] && [ -x "${VIRTUAL_ENV}/bin/alpha-lab-prefetch-token" ]; then
    exec "${VIRTUAL_ENV}/bin/alpha-lab-prefetch-token" "$@"
fi
if command -v alpha-lab-prefetch-token >/dev/null 2>&1; then
    exec alpha-lab-prefetch-token "$@"
fi
exec "${ALPHALAB_PYTHON:-python3}" -m alpha_lab.prefetch_token "$@"
