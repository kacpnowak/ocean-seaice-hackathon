#!/usr/bin/env bash
#
# Publish this kit to the public repository as ONE squashed commit per release.
#
# The development history stays private.  What participants see is a branch --
# `public` by default -- that shares no commit with `main`: the first release is
# a single root commit holding main's tracked tree, and every release after that
# appends exactly one more commit holding main's tracked tree at that moment.
# So a participant can `git pull` normally and reads a release trail, and nobody
# reads the 70-odd development commits, the branch names or the messages.
#
#   bash scripts/publish.sh "Initial public release"
#   bash scripts/publish.sh --dry-run "Add the coupling notebook"
#   bash scripts/publish.sh --yes "Fix the eval CLI"          # no prompt
#
# The commit is built with `git commit-tree` from `main^{tree}` rather than by
# checking out an orphan branch and re-adding the files.  It is the same result
# -- `git diff main public` is empty, by construction rather than by hope -- but
# it never touches HEAD, the index or the working tree, so a failure halfway
# through cannot leave your checkout in a state you then have to rescue.  It is
# also the reason nothing gitignored can ever be published: `main^{tree}` IS the
# set of tracked files and nothing else.  The guards below check the two ways
# that could still go wrong (a tracked file that matches an ignore rule, and a
# development directory that was accidentally added).
#
# It does not push.  There may not be a remote yet, and guessing one is how you
# push a repository somewhere nobody meant.  It prints the exact command.

set -euo pipefail

SOURCE_BRANCH="main"        # what gets published
REMOTE_BRANCH="main"        # what participants clone
public_branch="public"      # the local squashed branch

assume_yes=0
dry_run=0
push_remote=""
message=""

die() {
	printf 'publish: %s\n' "$1" >&2
	shift || true
	for line in "$@"; do printf '         %s\n' "$line" >&2; done
	exit 1
}

usage() {
	cat <<'EOF'
Usage: bash scripts/publish.sh [options] "release message"

Appends one squashed commit to the `public` branch, holding main's tracked tree.
The development history is never published.

Options:
  -y, --yes             do not ask for confirmation (for scripts)
  -n, --dry-run         show everything, change nothing
      --branch NAME     use NAME instead of `public` as the local squashed branch
      --push REMOTE     push to REMOTE after committing (default: print the
                        command instead, so you can read it first)
  -h, --help            this text
EOF
}

while (($#)); do
	case "$1" in
	-y | --yes) assume_yes=1 ;;
	-n | --dry-run) dry_run=1 ;;
	--branch)
		[[ ${2:-} ]] || die "--branch needs a branch name"
		public_branch="$2"
		shift
		;;
	--push)
		[[ ${2:-} ]] || die "--push needs a remote name"
		push_remote="$2"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	--)
		shift
		[[ -z $message ]] || die "give exactly one release message"
		message="${1:-}"
		break
		;;
	-*) die "unknown option: $1" "run: bash scripts/publish.sh --help" ;;
	*)
		[[ -z $message ]] || die "give exactly one release message, quoted" \
			"got a second argument: $1"
		message="$1"
		;;
	esac
	shift
done

# ---------------------------------------------------------------------------
# Where are we
# ---------------------------------------------------------------------------
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)" ||
	die "not inside a git repository"
cd "$repo_root"

# ---------------------------------------------------------------------------
# Refusals.  This script rewrites branches; every one of these is a state in
# which the thing it would publish is not the thing you think it is.
# ---------------------------------------------------------------------------
[[ -n $message ]] || {
	usage >&2
	die "no release message." \
		"Each release commit is the only thing participants read about what changed."
}

current_branch="$(git rev-parse --abbrev-ref HEAD)"
[[ $current_branch == "$SOURCE_BRANCH" ]] ||
	die "on branch '$current_branch', not '$SOURCE_BRANCH'." \
		"Only $SOURCE_BRANCH is published. Switch to it first:" \
		"  git switch $SOURCE_BRANCH"

if [[ -n "$(git status --porcelain)" ]]; then
	printf 'publish: the working tree is not clean:\n\n' >&2
	git status --short >&2
	printf '\n' >&2
	die "refusing to publish a half-finished edit." \
		"Commit it, stash it or discard it, then run this again." \
		"(Everything listed above is either uncommitted or untracked-and-not-ignored;" \
		" only committed files are published, so what you see now is not what would go.)"
fi

for state in MERGE_HEAD REBASE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG; do
	if [[ -e "$(git rev-parse --git-path "$state")" ]]; then
		die "a $state is in progress; finish or abort it first."
	fi
done

# The development workspace must never be published.  It is untracked and
# self-ignoring (.superpowers/sdd/.gitignore holds a single `*`), which is
# exactly the kind of fact that stops being true without anyone noticing, so it
# is asserted rather than assumed.
if [[ -n "$(git ls-files -- .superpowers)" ]]; then
	die ".superpowers/ has tracked files and would be published:" \
		"$(git ls-files -- .superpowers | tr '\n' ' ')" \
		"Run: git rm -r --cached .superpowers"
fi
if [[ -n "$(git ls-files --others --exclude-standard -- .superpowers)" ]]; then
	die ".superpowers/ is no longer fully ignored:" \
		"$(git ls-files --others --exclude-standard -- .superpowers | tr '\n' ' ')" \
		"It is only kept out of the publication by its own .gitignore."
fi

git rev-parse --verify --quiet "$SOURCE_BRANCH^{commit}" >/dev/null ||
	die "there is no '$SOURCE_BRANCH' branch here."

mapfile -t files < <(git ls-tree -r --name-only "$SOURCE_BRANCH")
((${#files[@]})) || die "'$SOURCE_BRANCH' has no files."

# A file can be tracked AND match an ignore rule -- if it was added before the
# rule existed, git keeps tracking it and `git status` stays quiet.  That is the
# one way something gitignored can reach the public repository, so look with
# --no-index, which is the only mode that reports it.
ignored_but_tracked="$(printf '%s\n' "${files[@]}" | git check-ignore --no-index --stdin || true)"
[[ -z $ignored_but_tracked ]] ||
	die "these files are tracked but match a .gitignore rule, and would be published:" \
		"$(printf '%s ' $ignored_but_tracked)" \
		"Run: git rm --cached <path>   (then commit) if they should not be in git."

# ---------------------------------------------------------------------------
# What would happen
# ---------------------------------------------------------------------------
if git show-ref --verify --quiet "refs/heads/$public_branch"; then
	branch_exists=1
	previous="$(git rev-parse "$public_branch")"
	published_so_far="$(git rev-list --count "$public_branch")"

	# `public` must share no commit with `main`.  If it ever does, something has
	# merged or rebased the two and the next push would carry the development
	# history out.  (`rev-list public ^main` drops every commit main can reach.)
	off_main="$(git rev-list --count "$public_branch" "^$SOURCE_BRANCH")"
	[[ $off_main == "$published_so_far" ]] ||
		die "'$public_branch' shares $((published_so_far - off_main)) commit(s) with '$SOURCE_BRANCH'." \
			"That branch is no longer a squashed history and must not be pushed." \
			"Inspect it:  git log --oneline $public_branch"

	[[ "$(git rev-parse "$public_branch^{tree}")" != "$(git rev-parse "$SOURCE_BRANCH^{tree}")" ]] ||
		die "'$public_branch' already holds exactly this tree; there is nothing new to publish." \
			"Commit something on $SOURCE_BRANCH first."
else
	branch_exists=0
	previous=""
	published_so_far=0
fi

total_bytes="$(git ls-tree -r -l "$SOURCE_BRANCH" | awk '{s += $4} END {print s + 0}')"
human_size="$(awk -v b="$total_bytes" 'BEGIN {
	split("B KiB MiB GiB", u, " "); i = 1
	while (b >= 1024 && i < 4) { b /= 1024; i++ }
	printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i]
}')"

printf '\n'
printf 'Publishing %s (%s) as release %d on branch "%s"\n' \
	"$SOURCE_BRANCH" "$(git rev-parse --short "$SOURCE_BRANCH")" \
	"$((published_so_far + 1))" "$public_branch"
if ((branch_exists)); then
	printf 'Previous release: %s  %s\n' \
		"$(git rev-parse --short "$public_branch")" \
		"$(git log -1 --format=%s "$public_branch")"
else
	printf 'This is the FIRST release: a new root commit, no parent, no history behind it.\n'
fi
printf '\n%d files, %s:\n\n' "${#files[@]}" "$human_size"
printf '  %s\n' "${files[@]}"
printf '\nCommit message:\n\n'
printf '  %s\n' "$message"
printf '\n'
if ((dry_run)); then printf -- '--dry-run: nothing below this line will happen.\n\n'; fi

if ((dry_run)); then
	:
elif ((assume_yes)); then
	printf 'Continuing (--yes).\n\n'
else
	[[ -t 0 ]] || die "not a terminal, and --yes was not given; refusing to publish unasked."
	read -r -p 'Publish this? [y/N] ' reply
	[[ $reply == [yY] || $reply == [yY][eE][sS] ]] || die "cancelled; nothing changed."
	printf '\n'
fi

# ---------------------------------------------------------------------------
# Do it.  Two plumbing commands and no checkout.
# ---------------------------------------------------------------------------
tree="$(git rev-parse "$SOURCE_BRANCH^{tree}")"
if ((dry_run)); then
	printf 'Would commit tree %s on "%s" and leave the push to you.\n' \
		"$(git rev-parse --short "$tree")" "$public_branch"
else
	parents=()
	if ((branch_exists)); then parents=(-p "$previous"); fi
	commit="$(git commit-tree "$tree" ${parents[@]+"${parents[@]}"} -m "$message")"
	if ((branch_exists)); then
		git update-ref "refs/heads/$public_branch" "$commit" "$previous"
	else
		git update-ref "refs/heads/$public_branch" "$commit" ""
	fi

	# Verify what was actually written, not what was intended.
	[[ -z "$(git diff --stat "$SOURCE_BRANCH" "$public_branch")" ]] ||
		die "BUG: the published tree differs from $SOURCE_BRANCH. Nothing has been pushed."
	total="$(git rev-list --count "$public_branch")"
	[[ "$total" == "$((published_so_far + 1))" ]] ||
		die "BUG: '$public_branch' has $total commits, expected $((published_so_far + 1))."

	printf 'Committed %s on "%s" -- %s now holds %d squashed commit(s):\n\n' \
		"$(git rev-parse --short "$commit")" "$public_branch" "$public_branch" "$total"
	git log --oneline "$public_branch" | sed 's/^/  /'
	printf '\n'
fi

# ---------------------------------------------------------------------------
# Pushing is a separate decision.
# ---------------------------------------------------------------------------
if [[ -n $push_remote ]] && ((!dry_run)); then
	git remote get-url "$push_remote" >/dev/null 2>&1 ||
		die "no remote called '$push_remote'."
	printf 'Pushing to %s (%s)...\n' "$push_remote" "$(git remote get-url "$push_remote")"
	git push "$push_remote" "$public_branch:$REMOTE_BRANCH"
	printf '\nPushed. Participants see %s as "%s".\n' "$REMOTE_BRANCH" "$REMOTE_BRANCH"
	exit 0
fi

printf 'Nothing has been pushed. To publish it, run:\n\n'
mapfile -t remotes < <(git remote)
if ((${#remotes[@]} == 0)); then
	printf '  git remote add origin git@github.com:<owner>/<repo>.git   # once it exists\n'
	printf '  git push -u origin %s:%s\n' "$public_branch" "$REMOTE_BRANCH"
else
	for remote in "${remotes[@]}"; do
		printf '  git push %s %s:%s        # %s\n' \
			"$remote" "$public_branch" "$REMOTE_BRANCH" "$(git remote get-url "$remote")"
	done
fi
printf '\nThat pushes ONLY "%s". "%s" is never pushed by this script.\n' \
	"$public_branch" "$SOURCE_BRANCH"
