#!/bin/bash

echo "Installing NeuralClaude npm package..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js >= 14.0.0"
    exit 1
fi

# Install npm dependencies
echo "Installing npm dependencies..."
npm install

# Build the native module with napi feature
echo "Building native module..."
npm run build

echo "Build complete! You can now use the package with:"
echo "  const { NeuralMemorySystem } = require('./index.js');"
echo ""
echo "Try running the example:"
echo "  node example.js"