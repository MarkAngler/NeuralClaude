#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Check if version argument is provided
if [ $# -eq 0 ]; then
    print_error "No version provided"
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.2.0"
    exit 1
fi

NEW_VERSION=$1

# Validate version format (basic check)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format. Use semantic versioning (e.g., 0.2.0)"
    exit 1
fi

print_info "Preparing to release version $NEW_VERSION"

# Get current versions
CURRENT_CARGO_VERSION=$(grep '^version = ' Cargo.toml | head -1 | cut -d '"' -f 2)
CURRENT_NPM_VERSION=$(grep '"version":' package.json | cut -d '"' -f 4)

print_info "Current Cargo version: $CURRENT_CARGO_VERSION"
print_info "Current npm version: $CURRENT_NPM_VERSION"

# Update Cargo.toml version
print_step "Updating Cargo.toml version to $NEW_VERSION"
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml
rm Cargo.toml.bak

# Update package.json version
print_step "Updating package.json version to $NEW_VERSION"
sed -i.bak "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" package.json
rm package.json.bak

# Clean previous builds
print_step "Cleaning previous builds"
cargo clean
rm -rf bin/neuralclaude bin/neuralclaude-debug

# Build release binary
print_step "Building release binary"
cargo build --release --bin mcp_server_simple

# Create bin directory if it doesn't exist
mkdir -p bin

# Copy and rename the binary
print_step "Copying binary to bin/neuralclaude"
cp target/release/mcp_server_simple bin/neuralclaude

# Ensure binary is executable
chmod +x bin/neuralclaude

# Optionally build debug version
if [ "${BUILD_DEBUG:-false}" == "true" ]; then
    print_step "Building debug binary"
    cargo build --bin mcp_server_simple
    cp target/debug/mcp_server_simple bin/neuralclaude-debug
    chmod +x bin/neuralclaude-debug
fi

# Run tests
print_step "Running tests"
cargo test

# Check if we're logged into npm
if ! npm whoami >/dev/null 2>&1; then
    print_error "Not logged into npm. Please run 'npm login' first"
    exit 1
fi

# Confirm before publishing
print_info "Ready to publish version $NEW_VERSION to npm"
read -p "Do you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Publishing cancelled"
    exit 1
fi

# Publish to npm
print_step "Publishing to npm"
npm publish

# Tag the release in git
print_step "Creating git tag v$NEW_VERSION"
git add Cargo.toml package.json
git commit -m "Release v$NEW_VERSION

- Bump version to $NEW_VERSION
- Build and publish to npm

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

print_info "Release complete! 🎉"
print_info "Version $NEW_VERSION has been published to npm"
print_info "Don't forget to push the tag: git push origin v$NEW_VERSION"