#!/bin/bash

# 1. Update .gitignore
GITIGNORE=".gitignore"
if [ ! -f "$GITIGNORE" ]; then
    touch "$GITIGNORE"
fi

# List of patterns to exclude
IGNORES=(
    "dataset/"
    "outputs/"
    "local/"
    "wandb/"
    "*.mp4"
    "*.jpg"
    "*.png"
    "__pycache__/"
)

echo "🔍 Checking .gitignore..."
for pattern in "${IGNORES[@]}"; do
    if ! grep -Fxq "$pattern" "$GITIGNORE"; then
        echo "$pattern" >> "$GITIGNORE"
        echo "✅ Added '$pattern' to .gitignore"
    fi
done

# 2. Config Identity
git config user.email "viswanathan.vsn@gmail.com"
git config user.name "viswavsn81"

# 3. Set New Remote
NEW_REMOTE="https://github.com/viswavsn81/pyru_robot_pickwire_poc"

echo "🔄 Switching remote to: $NEW_REMOTE"
if git remote | grep -q "^origin$"; then
    git remote remove origin
fi
git remote add origin "$NEW_REMOTE"

# 4. Add & Commit
echo "📦 Adding files..."
git add .
echo "💾 Committing..."
git commit -m "Auto-backup to POC repo: $(date)"

# 5. Fix Branch & Push
TARGET_BRANCH="Fix-calibration"
echo "🔄 Ensuring '$TARGET_BRANCH' tracks current state..."
# Force target branch to match current HEAD
git checkout -B "$TARGET_BRANCH"

echo "🚀 Pushing to GitHub (Force Pushing)..."
git push -u origin "$TARGET_BRANCH" --force

echo "✅ Done! Code pushed to branch: $TARGET_BRANCH"
