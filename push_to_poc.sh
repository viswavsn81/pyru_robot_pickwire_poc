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

# 2. Config Identity (Ensuring logic works)
git config user.email "viswanathan.vsn@gmail.com"
git config user.name "viswavsn81"

# 3. Set New Remote
NEW_REMOTE="https://github.com/viswavsn81/pyru_robot_pickwire_poc"

echo "🔄 Switching remote to: $NEW_REMOTE"

# Remove existing origin if it exists to clean start
if git remote | grep -q "^origin$"; then
    git remote remove origin
fi

# Add new origin
git remote add origin "$NEW_REMOTE"

# 4. Add & Commit
echo "📦 Adding files..."
git add .
echo "💾 Committing..."
git commit -m "Auto-backup to POC repo: $(date)"

# 5. Push
echo "🚀 Pushing to GitHub (Force Pushing to overwrite)..."
# Using -f to ensure local state replaces remote content for this "Sync"
git branch -M main
git push -u origin main --force

echo "✅ Done! Code pushed to $NEW_REMOTE"
