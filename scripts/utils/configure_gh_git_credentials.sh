#!/usr/bin/env bash
# Configure this git clone to authenticate GitHub HTTPS via the gh CLI.
#
# Replit / Cursor SSH sessions often set GIT_ASKPASS=replit-git-askpass, which
# cannot supply GitHub credentials and fails pushes with:
#   fatal: could not read Username for 'https://github.com'
#
# This script only writes *repo-local* git config (never --global).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH" >&2
  exit 1
fi

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "GitHub CLI is not logged in. Run:"
  echo "  gh auth login -h github.com -p https -w"
  exit 1
fi

# Prefer gh credentials for this clone; clear prior local helpers first.
git config --local --unset-all credential.helper 2>/dev/null || true
git config --local credential.helper '!gh auth git-credential'

# Persist a local override so interactive git ignores broken Replit askpass.
# .git/ is not committed; safe for this workspace only.
ASKPASS_OVERRIDE="$ROOT/.git/git-askpass-gh.sh"
cat >"$ASKPASS_OVERRIDE" <<'EOF'
#!/usr/bin/env bash
# No-op askpass: credential.helper=!gh auth git-credential supplies tokens.
exit 0
EOF
chmod +x "$ASKPASS_OVERRIDE"
git config --local core.askPass "$ASKPASS_OVERRIDE"

echo "Configured repo-local GitHub auth:"
echo "  credential.helper=!gh auth git-credential"
echo "  core.askPass=$ASKPASS_OVERRIDE"
echo
echo "Verify with: git push / gh pr create"
