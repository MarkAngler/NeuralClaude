use neural_llm_memory::emotional::EmotionalState;
use neural_llm_memory::emotional_types::{
    Emotion, AffectiveWeighting, EmotionalRegulation,
    SomaticMarker, EmpathySimulator, EmotionalIntelligence, EmotionalMemory
};
use neural_llm_memory::memory::MemoryBank;
use neural_llm_memory::emotional_integration::{
    EmotionallyAwareMemory, EmotionalDecisionMaker, EmotionalAdaptiveMemory
};
// use neural_llm_memory::adaptive::AdaptiveMemoryModule;
use ndarray::Array1;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

fn main() {
    println!("🎭 Neural Memory with Emotional Intelligence Demo\n");
    
    // Initialize systems
    let memory_bank = Arc::new(RwLock::new(MemoryBank::new(1000, 100)));
    let emotional_memory = Arc::new(EmotionallyAwareMemory::new(memory_bank.clone()));
    // let adaptive_module = Arc::new(AdaptiveMemoryModule::new(768, 1000));
    
    // Demo 1: Emotional State Creation and Blending
    println!("📊 Demo 1: Emotional States and Blending");
    println!("{}", "=".repeat(50));
    
    let joy = EmotionalState::from_emotion(Emotion::Joy, 0.8);
    println!("Pure Joy: valence={:.2}, arousal={:.2}", joy.valence, joy.arousal);
    
    let surprise_fear = EmotionalState::blend(vec![
        (Emotion::Surprise, 0.6),
        (Emotion::Fear, 0.4),
    ]);
    println!("Surprise + Fear blend: valence={:.2}, arousal={:.2}", 
             surprise_fear.valence, surprise_fear.arousal);
    
    // Demo 2: Affective Memory Storage
    println!("\n📝 Demo 2: Storing Memories with Emotional Context");
    println!("{}", "=".repeat(50));
    
    // Store a joyful memory
    let happy_content = Array1::from_vec(vec![0.8, 0.9, 0.7, 0.85, 0.95]);
    emotional_memory.store_with_emotion(
        "birthday_party".to_string(),
        happy_content,
        EmotionalState::from_emotion(Emotion::Joy, 0.9)
    ).unwrap();
    println!("✅ Stored joyful memory: 'birthday_party'");
    
    // Store a fearful memory
    let scary_content = Array1::from_vec(vec![0.2, 0.1, 0.3, 0.15, 0.25]);
    emotional_memory.store_with_emotion(
        "near_accident".to_string(),
        scary_content,
        EmotionalState::from_emotion(Emotion::Fear, 0.8)
    ).unwrap();
    println!("✅ Stored fearful memory: 'near_accident'");
    
    // Store a neutral memory
    let neutral_content = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    emotional_memory.store_with_emotion(
        "grocery_shopping".to_string(),
        neutral_content,
        EmotionalState::neutral()
    ).unwrap();
    println!("✅ Stored neutral memory: 'grocery_shopping'");
    
    // Demo 3: Mood-Congruent Retrieval
    println!("\n🔍 Demo 3: Mood-Congruent Memory Retrieval");
    println!("{}", "=".repeat(50));
    
    let query = Array1::from_vec(vec![0.6, 0.6, 0.6, 0.6, 0.6]);
    
    // Retrieve in happy mood
    let happy_mood = EmotionalState::from_emotion(Emotion::Joy, 0.7);
    let happy_results = emotional_memory.retrieve_mood_congruent(&query, &happy_mood, 2);
    println!("\nRetrieval in HAPPY mood:");
    for (key, score) in &happy_results {
        println!("  - {}: {:.3}", key, score);
    }
    
    // Retrieve in fearful mood
    let fearful_mood = EmotionalState::from_emotion(Emotion::Fear, 0.6);
    let fear_results = emotional_memory.retrieve_mood_congruent(&query, &fearful_mood, 2);
    println!("\nRetrieval in FEARFUL mood:");
    for (key, score) in &fear_results {
        println!("  - {}: {:.3}", key, score);
    }
    
    // Demo 4: Somatic Markers (Gut Feelings)
    println!("\n🎯 Demo 4: Somatic Markers and Intuition");
    println!("{}", "=".repeat(50));
    
    let risky_pattern = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.9]);
    let safe_pattern = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    
    // Check intuition before learning
    println!("Intuition before learning:");
    match emotional_memory.get_intuition(&risky_pattern) {
        Some(feeling) => println!("  Risky pattern: valence={:.2}", feeling.valence),
        None => println!("  Risky pattern: No gut feeling yet"),
    }
    
    // Simulate learning from bad experience
    let mut somatic = SomaticMarker::new();
    somatic.learn_association(
        risky_pattern.to_vec(),
        EmotionalState::from_emotion(Emotion::Fear, 0.9)
    );
    
    // Check intuition after learning
    let similar_risky = Array1::from_vec(vec![0.85, 0.15, 0.8, 0.2, 0.85]);
    match somatic.get_gut_feeling(&similar_risky.to_vec()) {
        Some(feeling) => println!("  Similar risky pattern after learning: valence={:.2} (negative = warning!)", 
                                feeling.valence),
        None => println!("  No gut feeling"),
    }
    
    // Demo 5: Emotional Regulation
    println!("\n🎛️ Demo 5: Emotional Regulation");
    println!("{}", "=".repeat(50));
    
    let mut regulation = EmotionalRegulation::new();
    
    // Simulate emotional journey
    let emotions = vec![
        ("Baseline", EmotionalState::neutral()),
        ("Sudden anger", EmotionalState::from_emotion(Emotion::Anger, 0.9)),
        ("After regulation", EmotionalState::neutral()), // Will be updated
    ];
    
    for (label, state) in emotions.iter().take(2) {
        println!("{}: valence={:.2}, arousal={:.2}", label, state.valence, state.arousal);
        regulation.update_state(state.clone());
        thread::sleep(Duration::from_millis(100));
    }
    
    // Apply regulation
    regulation.regulate();
    let regulated = regulation.current_state.read().unwrap().clone();
    println!("After regulation: valence={:.2}, arousal={:.2} (less extreme)", 
             regulated.valence, regulated.arousal);
    
    // Demo 6: Empathy Simulation
    println!("\n💝 Demo 6: Empathy and Emotional Contagion");
    println!("{}", "=".repeat(50));
    
    let mut empathy_sim = EmpathySimulator::new();
    
    // Model another agent's emotional state
    let context = vec![0.3, 0.7, 0.4]; // Some context features
    empathy_sim.other_models.insert(
        "friend_alice".to_string(),
        EmotionalState::from_emotion(Emotion::Sadness, 0.7)
    );
    
    let alice_state = empathy_sim.simulate_other("friend_alice", &context);
    println!("Simulated Alice's emotion: valence={:.2}, arousal={:.2}", 
             alice_state.valence, alice_state.arousal);
    
    // Apply emotional contagion
    let mut my_state = EmotionalState::from_emotion(Emotion::Joy, 0.8);
    println!("\nBefore contagion: My valence={:.2}", my_state.valence);
    
    empathy_sim.emotional_contagion(&mut my_state, &alice_state);
    println!("After contagion: My valence={:.2} (less joyful due to friend's sadness)", 
             my_state.valence);
    
    // Demo 7: Emotional Decision Making
    println!("\n🤔 Demo 7: Emotion-Influenced Decision Making");
    println!("{}", "=".repeat(50));
    
    let mut decision_maker = EmotionalDecisionMaker::new(emotional_memory.clone());
    
    // Define options with features
    let options = vec![
        ("safe_investment".to_string(), Array1::from_vec(vec![0.6, 0.7, 0.8, 0.6, 0.7])),
        ("risky_venture".to_string(), Array1::from_vec(vec![0.9, 0.3, 0.2, 0.9, 0.1])),
        ("balanced_approach".to_string(), Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5])),
    ];
    
    // Make decision in different emotional states
    let confident_mood = EmotionalState::from_emotion(Emotion::Pride, 0.7);
    let anxious_mood = EmotionalState::from_emotion(Emotion::Anxiety, 0.8);
    
    let confident_choice = decision_maker.decide_with_emotion(options.clone(), &confident_mood);
    println!("Decision when CONFIDENT: {}", confident_choice);
    
    let anxious_choice = decision_maker.decide_with_emotion(options.clone(), &anxious_mood);
    println!("Decision when ANXIOUS: {}", anxious_choice);
    
    // Demo 8: Emotional Intelligence Metrics
    println!("\n📈 Demo 8: Emotional Intelligence Assessment");
    println!("{}", "=".repeat(50));
    
    let eq = EmotionalIntelligence::new();
    println!("Initial EQ Score: {:.2}", eq.eq_score());
    println!("Components:");
    println!("  - Self-awareness: {:.2}", eq.self_awareness);
    println!("  - Self-regulation: {:.2}", eq.self_regulation);
    println!("  - Social awareness: {:.2}", eq.social_awareness);
    println!("  - Relationship skills: {:.2}", eq.relationship_skill);
    
    // Demo 9: Integrated Emotional-Adaptive Memory
    println!("\n🧠 Demo 9: Emotional-Adaptive Memory Integration");
    println!("{}", "=".repeat(50));
    
    let mut integrated = EmotionalAdaptiveMemory::new(emotional_memory.clone());
    
    // Store with both systems
    let important_memory = Array1::from_vec(vec![0.7, 0.8, 0.9, 0.8, 0.7]);
    let proud_moment = EmotionalState::from_emotion(Emotion::Pride, 0.85);
    
    integrated.store_adaptive_emotional(
        "achievement".to_string(),
        important_memory,
        proud_moment
    ).unwrap();
    
    // Retrieve with holistic approach
    let search_query = Array1::from_vec(vec![0.75, 0.75, 0.85, 0.75, 0.75]);
    let current_mood = EmotionalState::from_emotion(Emotion::Hope, 0.6);
    
    let results = integrated.retrieve_holistic(&search_query, &current_mood, 3);
    println!("\nHolistic retrieval results:");
    for (key, score) in results {
        println!("  - {}: {:.3}", key, score);
    }
    
    // Demo 10: Emotional Pattern Detection
    println!("\n🔄 Demo 10: Emotional Pattern Analysis");
    println!("{}", "=".repeat(50));
    
    let mut pattern_regulation = EmotionalRegulation::new();
    
    // Simulate mood swings
    let mood_sequence = vec![
        EmotionalState::from_emotion(Emotion::Joy, 0.9),
        EmotionalState::from_emotion(Emotion::Anger, 0.8),
        EmotionalState::from_emotion(Emotion::Sadness, 0.7),
        EmotionalState::from_emotion(Emotion::Joy, 0.8),
        EmotionalState::from_emotion(Emotion::Fear, 0.6),
        EmotionalState::from_emotion(Emotion::Relief, 0.7),
        EmotionalState::from_emotion(Emotion::Anxiety, 0.8),
        EmotionalState::from_emotion(Emotion::Sadness, 0.6),
        EmotionalState::from_emotion(Emotion::Anger, 0.7),
        EmotionalState::from_emotion(Emotion::Joy, 0.5),
    ];
    
    for state in mood_sequence {
        pattern_regulation.update_state(state);
    }
    
    let patterns = pattern_regulation.detect_patterns();
    println!("Detected emotional patterns:");
    for pattern in patterns {
        println!("  - {}", pattern);
    }
    
    println!("\n✨ Demo Complete!");
    println!("\nKey Insights:");
    println!("1. Emotions significantly influence memory storage and retrieval");
    println!("2. Mood-congruent retrieval shows how current emotions bias recall");
    println!("3. Somatic markers provide 'gut feelings' based on past experiences");
    println!("4. Emotional regulation helps maintain stability");
    println!("5. Empathy and contagion create social emotional dynamics");
    println!("6. Decision-making is influenced by emotional states");
    println!("7. Emotional intelligence can be measured and improved");
    println!("8. Integration with adaptive memory creates holistic intelligence");
}