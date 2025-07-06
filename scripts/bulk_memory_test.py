#!/usr/bin/env python3
"""
Bulk memory testing script to trigger evolution and measure improvements
"""
import json
import random
import time
from datetime import datetime

# Knowledge domains for diverse memory storage
DOMAINS = {
    "programming": {
        "python": [
            "List comprehensions are more efficient than traditional loops for simple transformations",
            "The Global Interpreter Lock (GIL) prevents true parallel execution of threads in CPython",
            "Decorators are functions that modify other functions, enabling aspect-oriented programming",
            "Context managers (with statement) ensure proper resource cleanup using __enter__ and __exit__",
            "Generator expressions save memory by yielding values one at a time instead of creating lists",
            "The walrus operator := allows assignment within expressions, introduced in Python 3.8",
            "f-strings provide faster string formatting than .format() or % formatting",
            "Type hints improve code readability and enable static type checking with tools like mypy",
            "asyncio enables concurrent execution of coroutines for I/O-bound operations",
            "dataclasses reduce boilerplate code for creating classes that primarily store data"
        ],
        "rust": [
            "Ownership system prevents data races at compile time without garbage collection",
            "Borrowing rules: one mutable reference OR multiple immutable references, never both",
            "Zero-cost abstractions mean high-level code compiles to efficient machine code",
            "Pattern matching with match expressions provides exhaustive case handling",
            "Traits define shared behavior similar to interfaces but with more flexibility",
            "Lifetimes explicitly track how long references are valid to prevent dangling pointers",
            "Result<T, E> and Option<T> types make error handling explicit and composable",
            "Cargo package manager handles dependencies, building, testing, and documentation",
            "unsafe blocks allow low-level operations while maintaining safety boundaries",
            "macro_rules! enables powerful metaprogramming and code generation"
        ],
        "javascript": [
            "Promises handle asynchronous operations with then/catch chaining",
            "async/await syntax provides cleaner asynchronous code than callback pyramids",
            "Arrow functions have lexical this binding and concise syntax",
            "Destructuring assignment extracts values from arrays and objects elegantly",
            "Template literals support multi-line strings and expression interpolation",
            "The spread operator ... expands iterables and copies objects shallowly",
            "Optional chaining ?. safely accesses nested properties without errors",
            "ES modules provide standardized import/export syntax for code organization",
            "WeakMap and WeakSet allow garbage collection of unused keys",
            "Proxy objects intercept and customize object operations dynamically"
        ]
    },
    "science": {
        "physics": [
            "Quantum entanglement creates correlated states between particles regardless of distance",
            "The uncertainty principle limits simultaneous knowledge of position and momentum",
            "General relativity describes gravity as curvature of spacetime, not a force",
            "Conservation laws arise from symmetries according to Noether's theorem",
            "Wave-particle duality means matter exhibits both wave and particle properties",
            "The second law of thermodynamics states entropy always increases in isolated systems",
            "Special relativity shows time dilation occurs at high velocities",
            "Quantum tunneling allows particles to pass through classically forbidden barriers",
            "The standard model describes fundamental particles and three of four forces",
            "Dark matter and dark energy comprise 95% of the universe's content"
        ],
        "biology": [
            "DNA replication is semiconservative, preserving one original strand",
            "Natural selection drives evolution through differential reproductive success",
            "Mitochondria generate ATP through oxidative phosphorylation",
            "Neurons communicate via electrical and chemical synapses",
            "CRISPR-Cas9 enables precise genome editing using bacterial defense mechanisms",
            "Photosynthesis converts light energy into chemical energy in chloroplasts",
            "Protein folding determines function through 3D structure",
            "Stem cells can differentiate into multiple cell types",
            "The immune system recognizes self from non-self using MHC molecules",
            "Epigenetics involves heritable changes without DNA sequence alterations"
        ]
    },
    "history": {
        "ancient": [
            "The Rosetta Stone enabled decipherment of Egyptian hieroglyphics",
            "The Roman Republic fell due to civil wars and concentration of power",
            "The Silk Road connected East and West through trade networks",
            "Greek philosophy laid foundations for Western scientific thought",
            "The Bronze Age collapse around 1200 BCE reshaped Mediterranean civilizations",
            "Hammurabi's Code was one of the earliest written legal systems",
            "The Agricultural Revolution enabled permanent settlements and population growth",
            "Ancient China's Four Great Inventions transformed global technology",
            "The Library of Alexandria attempted to collect all human knowledge",
            "Mesopotamian cuneiform was among the first writing systems"
        ],
        "modern": [
            "The Industrial Revolution transformed economies from agricultural to manufacturing",
            "World War I's trench warfare led to unprecedented casualties",
            "The Cold War shaped global politics through ideological competition",
            "The Internet revolution democratized information access worldwide",
            "Decolonization movements reshaped the post-WWII global order",
            "The Space Race drove technological innovation and national prestige",
            "The Great Depression highlighted vulnerabilities in capitalist systems",
            "The Berlin Wall's fall symbolized the end of European division",
            "The Manhattan Project ushered in the nuclear age",
            "The Civil Rights Movement challenged systemic discrimination"
        ]
    }
}

def generate_memory_key(domain, subdomain, index):
    """Generate a structured key for memory storage"""
    return f"test/{domain}/{subdomain}/fact_{index}"

def store_memories():
    """Store diverse memories across domains"""
    stored = []
    
    for domain, subdomains in DOMAINS.items():
        for subdomain, facts in subdomains.items():
            for i, fact in enumerate(facts):
                key = generate_memory_key(domain, subdomain, i)
                stored.append({
                    "key": key,
                    "content": fact,
                    "domain": domain,
                    "subdomain": subdomain
                })
    
    return stored

def search_memories():
    """Perform diverse searches across stored memories"""
    searches = [
        # Programming searches
        {"query": "python memory efficiency", "expected_domain": "programming"},
        {"query": "rust ownership borrowing", "expected_domain": "programming"},
        {"query": "javascript asynchronous programming", "expected_domain": "programming"},
        {"query": "functional programming concepts", "expected_domain": "programming"},
        {"query": "type safety static checking", "expected_domain": "programming"},
        
        # Science searches
        {"query": "quantum mechanics principles", "expected_domain": "science"},
        {"query": "evolution natural selection", "expected_domain": "science"},
        {"query": "energy conservation thermodynamics", "expected_domain": "science"},
        {"query": "cellular biology DNA", "expected_domain": "science"},
        {"query": "particle physics forces", "expected_domain": "science"},
        
        # History searches
        {"query": "ancient civilizations writing", "expected_domain": "history"},
        {"query": "industrial revolution technology", "expected_domain": "history"},
        {"query": "world war global conflict", "expected_domain": "history"},
        {"query": "trade routes silk road", "expected_domain": "history"},
        {"query": "social movements rights", "expected_domain": "history"},
        
        # Cross-domain searches
        {"query": "innovation and discovery", "expected_domain": "mixed"},
        {"query": "fundamental principles laws", "expected_domain": "mixed"},
        {"query": "communication systems", "expected_domain": "mixed"},
        {"query": "efficiency optimization", "expected_domain": "mixed"},
        {"query": "historical technological advances", "expected_domain": "mixed"}
    ]
    
    return searches

def calculate_relevance_score(query, result_content, expected_domain):
    """Calculate a relevance score for search results"""
    # Simple scoring based on keyword matches and domain relevance
    query_words = set(query.lower().split())
    content_words = set(result_content.lower().split())
    
    # Word overlap score
    overlap = len(query_words & content_words)
    word_score = overlap / len(query_words) if query_words else 0
    
    # Domain match bonus
    domain_bonus = 0.3 if expected_domain in result_content.lower() else 0
    
    # Length penalty (prefer concise relevant results)
    length_penalty = min(1.0, 100 / len(result_content)) * 0.2
    
    return min(1.0, word_score + domain_bonus + length_penalty)

if __name__ == "__main__":
    print("🚀 Starting bulk memory test to trigger evolution...")
    print(f"📅 Test started at: {datetime.now()}")
    
    # Store all memories
    print("\n📝 Storing memories across domains...")
    memories = store_memories()
    print(f"✅ Prepared {len(memories)} memories for storage")
    
    # Output as JSON for the next step
    with open("/tmp/bulk_memories.json", "w") as f:
        json.dump({
            "memories": memories,
            "searches": search_memories()
        }, f, indent=2)
    
    print("\n💾 Memory data saved to /tmp/bulk_memories.json")
    print("🎯 Ready for bulk operations to trigger evolution!")