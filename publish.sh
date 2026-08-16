#!/usr/bin/env bash
#
# Publish token-meter to crates.io.
#
#   ./publish.sh                 # bump the patch version, then publish
#   ./publish.sh minor           # 0.1.4 -> 0.2.0
#   ./publish.sh major           # 0.1.4 -> 1.0.0
#   ./publish.sh 0.3.1           # publish that exact version
#   ./publish.sh --same          # republish the version already in Cargo.toml
#   ./publish.sh patch --dry-run # package + verify, upload nothing, keep the bump off disk
#   ./publish.sh --no-verify     # skip fmt/clippy/test (cargo still builds the package)
#   ./publish.sh --allow-dirty   # publish with uncommitted changes in the tree
#   ./publish.sh --yes           # no confirmation prompt
#
# The crates.io token is read from ./.env (CARGO_API_KEY=...), which is
# gitignored and must stay that way.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CRATE=token-meter
ENV_FILE=.env
FEATURES=--all-features

bump=patch
dry_run=false
verify=true
assume_yes=false
require_clean=true

for arg in "$@"; do
  case "$arg" in
    patch|minor|major) bump=$arg ;;
    --same)            bump=same ;;
    [0-9]*.[0-9]*.[0-9]*) bump=$arg ;;
    --dry-run)         dry_run=true ;;
    --allow-dirty)     require_clean=false ;;
    --no-verify)       verify=false ;;
    --yes|-y)          assume_yes=true ;;
    -h|--help)         sed -n '2,17p' "$0" | cut -c3-; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- token -------------------------------------------------------------------

[[ -f $ENV_FILE ]] || die "$ENV_FILE not found; it must contain CARGO_API_KEY=<crates.io token>"
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
[[ -n ${CARGO_API_KEY:-} ]] || die "CARGO_API_KEY is not set in $ENV_FILE"

# --- versions ----------------------------------------------------------------

current=$(sed -n '/^\[package\]/,/^\[/{s/^version *= *"\(.*\)"/\1/p;}' Cargo.toml | head -1)
[[ -n $current ]] || die "could not read the package version from Cargo.toml"

IFS=. read -r maj min pat <<<"${current%%-*}"
case "$bump" in
  same)  next=$current ;;
  patch) next="$maj.$min.$((pat + 1))" ;;
  minor) next="$maj.$((min + 1)).0" ;;
  major) next="$((maj + 1)).0.0" ;;
  *)     next=$bump ;;
esac

echo "$CRATE $current -> $next"

# crates.io refuses to overwrite a released version, so catch it before we do
# any work. A network failure here is only a warning: cargo will still refuse.
if code=$(curl -sS -o /dev/null -w '%{http_code}' \
            -H "User-Agent: $CRATE-publish (https://github.com/asale-ai/token-meter)" \
            "https://crates.io/api/v1/crates/$CRATE/$next" 2>/dev/null); then
  case "$code" in
    200) die "$CRATE $next is already published on crates.io; pick a higher version" ;;
    404) : ;; # not published yet — good
    *)   echo "warning: crates.io answered HTTP $code when checking for $next" >&2 ;;
  esac
else
  echo "warning: could not reach crates.io to check for an existing $next" >&2
fi

# --- working tree ------------------------------------------------------------

in_git=false
if git rev-parse --git-dir >/dev/null 2>&1; then
  in_git=true
  if $require_clean && [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    die "the working tree has uncommitted changes; commit them, or pass --allow-dirty"
  fi
fi

# --- checks ------------------------------------------------------------------

if $verify; then
  step "cargo fmt --check"
  # Advisory only: the crate predates any rustfmt pass, so a hard gate here
  # would block every release until someone reformats all of src/.
  cargo fmt --all -- --check >/dev/null 2>&1 || \
    echo "warning: rustfmt would reformat some files (run 'cargo fmt --all')" >&2
  step "cargo clippy $FEATURES"
  cargo clippy --all-targets $FEATURES -- -D warnings
  step "cargo test $FEATURES"
  cargo test $FEATURES
fi

# --- bump --------------------------------------------------------------------

# Restore from copies taken just before the bump rather than `git checkout`,
# so a failed publish can never discard someone else's edits.
backup_dir=
restore() {
  [[ -n $backup_dir && -d $backup_dir ]] || return 0
  cp "$backup_dir/Cargo.toml" Cargo.toml
  [[ -f $backup_dir/Cargo.lock ]] && cp "$backup_dir/Cargo.lock" Cargo.lock
  rm -rf "$backup_dir"
  backup_dir=
}
# Ctrl-C or a broken pipe must not leave a half-applied bump behind. restore()
# is a no-op once the publish has succeeded and the copy has been dropped.
trap restore EXIT INT TERM

if [[ $next != "$current" ]]; then
  step "bumping Cargo.toml to $next"
  backup_dir=$(mktemp -d)
  cp Cargo.toml "$backup_dir/"
  [[ -f Cargo.lock ]] && cp Cargo.lock "$backup_dir/"
  # Only the version under [package]; dependency versions are left alone.
  awk -v v="$next" '
    /^\[package\]/ { in_pkg = 1 }
    /^\[/ && !/^\[package\]/ { in_pkg = 0 }
    in_pkg && !done && /^version *=/ { print "version = \"" v "\""; done = 1; next }
    { print }
  ' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml
  # Keep Cargo.lock in step with the new version.
  cargo update --workspace --quiet
fi

if $dry_run; then
  step "cargo publish --dry-run"
  # --registry crates-io is mandatory: ~/.cargo/config.toml replaces the
  # crates-io source with the rsproxy mirror, and cargo will not publish to a
  # replaced source.
  cargo publish --dry-run --registry crates-io --allow-dirty $FEATURES
  if [[ $next != "$current" ]]; then
    echo
    echo "dry run: reverting the version bump"
    restore
  fi
  exit 0
fi

if ! $assume_yes; then
  printf '\nPublish %s %s to crates.io? [y/N] ' "$CRATE" "$next"
  read -r reply
  [[ $reply == [yY]* ]] || { restore; die "aborted"; }
fi

# --- publish -----------------------------------------------------------------

step "cargo publish"
if ! cargo publish --registry crates-io --token "$CARGO_API_KEY" $FEATURES --allow-dirty; then
  restore
  die "cargo publish failed; the version bump has been reverted"
fi
# Published: the bump is now permanent, so drop the rollback copy.
[[ -n $backup_dir ]] && rm -rf "$backup_dir"
backup_dir=

# --- commit and tag ----------------------------------------------------------

if $in_git && [[ $next != "$current" ]]; then
  step "committing and tagging v$next"
  git add Cargo.toml Cargo.lock
  git commit -m "release: $CRATE v$next"
  git tag -a "v$next" -m "$CRATE v$next"
  if git remote get-url origin >/dev/null 2>&1; then
    git push origin HEAD --follow-tags
  else
    echo "no origin remote; skipped push"
  fi
fi

step "published $CRATE $next"
case "$bump" in
  minor|major)
    echo "note: this is not a semver-compatible bump. Update the dependants in the"
    echo "      asale tree (asale-client/Cargo.toml, asale-server/Cargo.toml)."
    ;;
esac
echo "note: builds behind the rsproxy mirror cannot resolve $next until that index"
echo "      syncs. Wait for it before running deploy/release.sh."
