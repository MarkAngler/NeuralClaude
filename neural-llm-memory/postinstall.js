#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const binaryPath = path.join(__dirname, 'bin', 'neuralclaude');

// Ensure binary is executable
try {
  fs.chmodSync(binaryPath, '755');
  console.log('✓ NeuralClaude MCP server installed successfully');
  console.log('  Run with: neuralclaude');
} catch (error) {
  console.error('Warning: Could not set executable permissions on binary');
}