const { NeuralMemorySystem } = require('./index.js');

async function main() {
  console.log('Creating NeuralClaude memory system...');
  
  // Initialize the memory system
  const memory = new NeuralMemorySystem({
    dimensions: 768,
    capacity: 1000,
    threshold: 0.7,
    persistPath: './example_memory.db'
  });

  console.log('\n1. Storing memories...');
  
  // Store some memories
  await memory.store('fact_1', 'The Earth orbits around the Sun', {
    category: 'astronomy',
    type: 'fact'
  });

  await memory.store('fact_2', 'Water boils at 100 degrees Celsius at sea level', {
    category: 'physics',
    type: 'fact'
  });

  await memory.store('fact_3', 'The Sun is a star at the center of our solar system', {
    category: 'astronomy',
    type: 'fact'
  });

  await memory.store('preference_1', 'User prefers dark mode for coding', {
    category: 'preferences',
    type: 'user_setting'
  });

  console.log('✓ Stored 4 memories');

  console.log('\n2. Retrieving a specific memory...');
  const retrieved = await memory.retrieve('fact_1');
  console.log('Retrieved:', retrieved);

  console.log('\n3. Searching for similar memories...');
  const searchResults = await memory.search('solar system astronomy', 3);
  console.log('Search results:');
  searchResults.forEach(result => {
    console.log(`  - ${result.key} (similarity: ${result.similarity.toFixed(3)})`);
    console.log(`    Content: ${result.content}`);
  });

  console.log('\n4. Getting memory statistics...');
  const stats = await memory.getStats();
  console.log('Stats:', stats);

  console.log('\n5. Listing all keys...');
  const keys = await memory.listKeys();
  console.log('All keys:', keys);

  console.log('\n6. Exporting memories to file...');
  await memory.exportToFile('./exported_memories.json');
  console.log('✓ Exported to exported_memories.json');

  console.log('\n7. Deleting a memory...');
  const deleted = await memory.delete('preference_1');
  console.log(`Deleted preference_1: ${deleted}`);

  console.log('\n8. Final statistics...');
  const finalStats = await memory.getStats();
  console.log('Final stats:', finalStats);

  console.log('\nExample completed!');
}

main().catch(console.error);