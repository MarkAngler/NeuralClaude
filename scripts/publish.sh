#!/bin/bash

# NeuralClaude NPM Publishing Script
# Usage: ./scripts/publish.sh [patch|minor|major]

set -e

# Check if version type is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [patch|minor|major]"
    echo "Example: $0 patch"
    exit 1
fi

VERSION_TYPE=$1

# Validate version type
if [[ "$VERSION_TYPE" != "patch" && "$VERSION_TYPE" != "minor" && "$VERSION_TYPE" != "major" ]]; then
    echo "Error: Version type must be 'patch', 'minor', or 'major'"
    exit 1
fi

# Navigate to the neural-llm-memory directory
cd "$(dirname "$0")/../neural-llm-memory"

# Check if we're on the main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Warning: You're not on the main branch (current: $CURRENT_BRANCH)"
    read -p "Do you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Ensure we have the latest code
echo "Pulling latest changes..."
git pull origin main

# Build the project
echo "Building the project..."
cargo build --release

# Run tests
echo "Running tests..."
cargo test

# Copy the binary to the bin directory
echo "Preparing binary for distribution..."
mkdir -p bin
cp ../target/release/mcp_server bin/neuralclaude
chmod +x bin/neuralclaude

# Bump the version
echo "Bumping version ($VERSION_TYPE)..."
CURRENT_VERSION=$(node -p "require('./package.json').version")
npm version $VERSION_TYPE --no-git-tag-version

NEW_VERSION=$(node -p "require('./package.json').version")
echo "Version bumped from $CURRENT_VERSION to $NEW_VERSION"

# Commit the version change
git add package.json
git commit -m "chore: bump version to $NEW_VERSION"

# Create a git tag
git tag "v$NEW_VERSION"

# Publish to npm
echo "Publishing to npm..."
npm publish

# Push the commit and tag
echo "Pushing changes to git..."
git push origin main
git push origin "v$NEW_VERSION"

echo ""
echo "✅ Successfully published neuralclaude v$NEW_VERSION to npm!"
echo ""
echo "Post-publish checklist:"
echo "- [ ] Update release notes on GitHub"
echo "- [ ] Test installation: npx neuralclaude@$NEW_VERSION"
echo "- [ ] Update documentation if needed"