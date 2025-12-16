#!/bin/bash
# Quick setup script for GitHub Token

echo "=== GitHub Token Setup for NNAST ==="
echo ""

# Check if token is already set
if [ -n "$GITHUB_TOKEN" ]; then
    echo "✓ GITHUB_TOKEN is already set"
    echo "  Current value: ${GITHUB_TOKEN:0:10}..."
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

echo ""
echo "To get a GitHub Personal Access Token:"
echo "1. Go to: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Select scopes: public_repo (or repo for private repos)"
echo "4. Copy the generated token"
echo ""

read -p "Enter your GitHub token: " token

if [ -z "$token" ]; then
    echo "Error: Token cannot be empty"
    exit 1
fi

# Ask user preference
echo ""
echo "How would you like to save the token?"
echo "1) Add to ~/.zshrc (persistent, system-wide)"
echo "2) Create .env file in project (project-specific)"
read -p "Choose (1 or 2): " choice

case $choice in
    1)
        # Add to ~/.zshrc
        if grep -q "GITHUB_TOKEN" ~/.zshrc 2>/dev/null; then
            echo "Updating existing GITHUB_TOKEN in ~/.zshrc..."
            sed -i '' "s/export GITHUB_TOKEN=.*/export GITHUB_TOKEN=$token/" ~/.zshrc
        else
            echo "Adding GITHUB_TOKEN to ~/.zshrc..."
            echo "" >> ~/.zshrc
            echo "# GitHub API Token for NNAST" >> ~/.zshrc
            echo "export GITHUB_TOKEN=$token" >> ~/.zshrc
        fi
        echo "✓ Token added to ~/.zshrc"
        echo "  Run 'source ~/.zshrc' or open a new terminal to apply changes"
        ;;
    2)
        # Create .env file
        project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        env_file="$project_root/.env"
        
        if [ -f "$env_file" ] && grep -q "GITHUB_TOKEN" "$env_file" 2>/dev/null; then
            echo "Updating existing GITHUB_TOKEN in .env..."
            sed -i '' "s/GITHUB_TOKEN=.*/GITHUB_TOKEN=$token/" "$env_file"
        else
            echo "Creating .env file..."
            echo "# GitHub API Token" >> "$env_file"
            echo "# Get your token from: https://github.com/settings/tokens" >> "$env_file"
            echo "GITHUB_TOKEN=$token" >> "$env_file"
        fi
        echo "✓ Token saved to .env file"
        echo "  Note: .env file is already in .gitignore"
        ;;
    *)
        echo "Invalid choice. Setup cancelled."
        exit 1
        ;;
esac

echo ""
echo "Setup complete!"
echo ""
echo "To verify, run:"
echo "  python3 -c \"import os; print('Token set:', bool(os.getenv('GITHUB_TOKEN')))\""

