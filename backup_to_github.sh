#!/bin/bash

# Configuration
REPO_URL="https://github.com/viswavsn81/pyru_robot_arm.git"
EMAIL="viswanathan.vsn@gmail.com"
NAME="viswavsn81"

echo "🚀 Starting Backup Script..."

# 1. Configure Git Identity
echo "🔧 Configuring Git Identity..."
git config --global user.email "$EMAIL"
git config --global user.name "$NAME"

# 2. Fix Remotes
echo "🔗 Fixing Remote Origin..."
# Remove origin if it exists to avoid conflicts
if git remote | grep -q "^origin$"; then
    echo "   Removing existing origin..."
    git remote remove origin
fi

# Add the new origin
echo "   Adding new origin: $REPO_URL"
git remote add origin "$REPO_URL"

# 3. Perform Backup
echo "📦 Staging files..."
git add .

echo "📝 Committing..."
git commit -m "Auto-backup via script"

echo "☁️ Pushing to GitHub (Force)..."
# Using --force to overwrite if history diverged (as requested)
git push -u origin main --force

echo "✅ Backup Complete!"
