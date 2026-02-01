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
    "__pycache__/"
    "*.mp4"
    "*.jpg"
    "*.png"
)

echo "🔍 Checking .gitignore..."
for pattern in "${IGNORES[@]}"; do
    if ! grep -Fxq "$pattern" "$GITIGNORE"; then
        echo "$pattern" >> "$GITIGNORE"
        echo "✅ Added '$pattern' to .gitignore"
    fi
done

# 2. Config Identity (Using provided details)
git config user.email "viswanathan.vsn@gmail.com"
git config user.name "viswavsn81"

# 3. Remote Setup
TARGET_REMOTE="https://github.com/viswavsn81/pyru_robot_arm.git"

# Check if origin exists
if git remote | grep -q "^origin$"; then
    # Origin exists, update URL just in case
    git remote set-url origin "$TARGET_REMOTE"
else
    # Origin doesn't exist, add it
    git remote add origin "$TARGET_REMOTE"
fi

echo "✅ Remote origin configured: $TARGET_REMOTE"

# 4. Add & Commit
echo "📦 Adding files..."
git add .
echo "💾 Committing..."
git commit -m "Auto-update: $(date)"

# 5. Push
echo "🚀 Pushing to GitHub..."
git push -u origin main

echo "✅ Done!"
