const binary = require('node-gyp-build')(__dirname);

// Re-export all functionality from the native module
module.exports = binary;

// Add convenience methods if needed
module.exports.createMemorySystem = function(config) {
  return new binary.NeuralMemorySystem(config);
};

module.exports.VERSION = require('./package.json').version;