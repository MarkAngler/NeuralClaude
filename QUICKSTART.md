# NeuralClaude Quick Start Guide

## Installation via npm

To install the NeuralClaude package from npm:

```bash
npm install neuralclaude
```

## Building from Source

If you want to build from source or contribute to the project:

### Prerequisites

1. **Rust** - Install from [rustup.rs](https://rustup.rs/)
2. **Node.js** >= 14.0.0 - Install from [nodejs.org](https://nodejs.org/)

### Build Steps

1. Clone the repository:
```bash
git clone https://github.com/markangler/NeuralClaude.git
cd NeuralClaude/neural-llm-memory
```

2. Install dependencies and build:
```bash
./install.sh
```

Or manually:
```bash
npm install
npm run build
```

3. Test the installation:
```bash
node example.js
```

## Usage Example

```javascript
const { NeuralMemorySystem } = require('neuralclaude');

async function demo() {
  // Create memory system
  const memory = new NeuralMemorySystem({
    dimensions: 768,
    capacity: 1000
  });

  // Store information
  await memory.store('key1', 'Important information');
  
  // Search for similar content
  const results = await memory.search('information', 5);
  console.log(results);
}

demo();
```

## Publishing to npm (Maintainers Only)

1. Update version in `package.json`
2. Create and push a git tag:
```bash
git tag v0.1.0
git push origin v0.1.0
```
3. The GitHub Action will automatically build and publish to npm

## Development

- `npm run build` - Build the native module
- `npm run test` - Run Rust tests
- `npm run test-js` - Run JavaScript example
- `npm run clean` - Clean build artifacts

## Troubleshooting

### Build Errors

If you encounter build errors:

1. Ensure Rust is up to date:
```bash
rustup update
```

2. Clear the build cache:
```bash
npm run clean
npm install
npm run build
```

### Platform-Specific Issues

- **Linux**: May need to install build essentials: `sudo apt-get install build-essential`
- **macOS**: Ensure Xcode Command Line Tools are installed: `xcode-select --install`
- **Windows**: Use Visual Studio Build Tools or full Visual Studio installation

## License

MIT