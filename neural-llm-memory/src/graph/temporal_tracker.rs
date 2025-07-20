//! Temporal tracking for node access patterns

use crate::graph::core::NodeId;
use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use std::sync::Arc;

/// Tracks temporal access patterns for nodes
pub struct TemporalTracker {
    /// Recent node accesses with timestamps
    access_log: Arc<DashMap<NodeId, Vec<DateTime<Utc>>>>,
    /// Window for considering nodes "recent"
    recency_window: Duration,
    /// Minimum accesses to be considered active
    min_activity_threshold: usize,
}

impl TemporalTracker {
    pub fn new(recency_window_hours: i64) -> Self {
        Self {
            access_log: Arc::new(DashMap::new()),
            recency_window: Duration::hours(recency_window_hours),
            min_activity_threshold: 2,
        }
    }
    
    /// Record an access to a node
    pub fn record_access(&self, node_id: &NodeId) {
        let mut accesses = self.access_log.entry(node_id.clone())
            .or_insert_with(Vec::new);
        accesses.push(Utc::now());
        
        // Prune old accesses
        let cutoff = Utc::now() - self.recency_window;
        accesses.retain(|&time| time > cutoff);
    }
    
    /// Record multiple accesses
    pub fn record_batch_access(&self, node_ids: &[NodeId]) {
        let now = Utc::now();
        let cutoff = now - self.recency_window;
        
        for node_id in node_ids {
            let mut accesses = self.access_log.entry(node_id.clone())
                .or_insert_with(Vec::new);
            accesses.push(now);
            accesses.retain(|&time| time > cutoff);
        }
    }
    
    /// Get nodes that have been accessed recently
    pub fn get_active_nodes(&self, min_accesses: usize) -> Vec<NodeId> {
        self.access_log.iter()
            .filter(|entry| entry.value().len() >= min_accesses)
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// Get access frequency for a specific node
    pub fn get_access_frequency(&self, node_id: &NodeId) -> usize {
        self.access_log.get(node_id)
            .map(|entry| entry.value().len())
            .unwrap_or(0)
    }
    
    /// Get access sequences for pattern mining
    pub fn get_access_sequences(&self) -> Result<Vec<AccessSequence>> {
        let mut sequences = Vec::new();
        
        // Collect all access events with timestamps
        let mut all_events: Vec<(DateTime<Utc>, NodeId)> = Vec::new();
        
        for entry in self.access_log.iter() {
            let node_id = entry.key().clone();
            for &timestamp in entry.value().iter() {
                all_events.push((timestamp, node_id.clone()));
            }
        }
        
        // Sort by timestamp
        all_events.sort_by_key(|(timestamp, _)| *timestamp);
        
        // Group into sequences based on time proximity
        let sequence_gap = Duration::minutes(5);
        let mut current_sequence = Vec::new();
        let mut last_time = None;
        
        for (timestamp, node_id) in all_events {
            if let Some(prev_time) = last_time {
                if timestamp - prev_time > sequence_gap {
                    // Start new sequence
                    if current_sequence.len() >= 2 {
                        sequences.push(AccessSequence {
                            nodes: current_sequence.iter().map(|(_, id): &(DateTime<Utc>, NodeId)| id.clone()).collect(),
                            start_time: current_sequence.first()
                                .map(|(t, _)| *t)
                                .unwrap_or(timestamp),
                            end_time: timestamp,
                        });
                    }
                    current_sequence.clear();
                }
            }
            
            current_sequence.push((timestamp, node_id));
            last_time = Some(timestamp);
        }
        
        // Add final sequence
        if current_sequence.len() >= 2 {
            let start = current_sequence.first().map(|(t, _)| *t).unwrap();
            let end = current_sequence.last().map(|(t, _)| *t).unwrap();
            sequences.push(AccessSequence {
                nodes: current_sequence.into_iter().map(|(_, id)| id).collect(),
                start_time: start,
                end_time: end,
            });
        }
        
        Ok(sequences)
    }
    
    /// Get temporal co-occurrence patterns
    pub fn get_cooccurrence_patterns(&self, window_minutes: i64) -> Vec<CooccurrencePattern> {
        let window = Duration::minutes(window_minutes);
        let mut patterns = Vec::new();
        
        // Find nodes that frequently occur together
        let node_list: Vec<NodeId> = self.access_log.iter()
            .map(|entry| entry.key().clone())
            .collect();
        
        for i in 0..node_list.len() {
            for j in i+1..node_list.len() {
                let node_a = &node_list[i];
                let node_b = &node_list[j];
                
                let cooccurrences = self.count_cooccurrences(node_a, node_b, window);
                
                if cooccurrences >= self.min_activity_threshold {
                    patterns.push(CooccurrencePattern {
                        node_a: node_a.clone(),
                        node_b: node_b.clone(),
                        count: cooccurrences,
                        window_minutes: window_minutes,
                    });
                }
            }
        }
        
        // Sort by count descending
        patterns.sort_by_key(|p| std::cmp::Reverse(p.count));
        
        patterns
    }
    
    /// Count how often two nodes co-occur within a time window
    fn count_cooccurrences(&self, node_a: &NodeId, node_b: &NodeId, window: Duration) -> usize {
        let accesses_a = match self.access_log.get(node_a) {
            Some(entry) => entry.value().clone(),
            None => return 0,
        };
        
        let accesses_b = match self.access_log.get(node_b) {
            Some(entry) => entry.value().clone(),
            None => return 0,
        };
        
        let mut count = 0;
        
        // For each access to node_a, check if node_b was accessed within the window
        for &time_a in &accesses_a {
            for &time_b in &accesses_b {
                let time_diff = (time_a - time_b).abs();
                if time_diff <= window {
                    count += 1;
                    break; // Count each access_a only once
                }
            }
        }
        
        count
    }
    
    /// Clear old entries beyond the recency window
    pub fn prune_old_entries(&self) {
        let cutoff = Utc::now() - self.recency_window;
        
        // Remove nodes with no recent accesses
        let nodes_to_remove: Vec<NodeId> = self.access_log.iter()
            .filter(|entry| {
                entry.value().iter().all(|&time| time <= cutoff)
            })
            .map(|entry| entry.key().clone())
            .collect();
        
        for node_id in nodes_to_remove {
            self.access_log.remove(&node_id);
        }
    }
    
    /// Get statistics about temporal patterns
    pub fn get_stats(&self) -> TemporalStats {
        let total_nodes = self.access_log.len();
        let total_accesses: usize = self.access_log.iter()
            .map(|entry| entry.value().len())
            .sum();
        
        let active_nodes = self.get_active_nodes(self.min_activity_threshold).len();
        
        TemporalStats {
            total_tracked_nodes: total_nodes,
            total_accesses,
            active_nodes,
            window_hours: self.recency_window.num_hours() as usize,
        }
    }
}

/// Represents a sequence of node accesses
#[derive(Debug, Clone)]
pub struct AccessSequence {
    pub nodes: Vec<NodeId>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

/// Represents a co-occurrence pattern between two nodes
#[derive(Debug, Clone)]
pub struct CooccurrencePattern {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub count: usize,
    pub window_minutes: i64,
}

/// Statistics about temporal tracking
#[derive(Debug, Clone)]
pub struct TemporalStats {
    pub total_tracked_nodes: usize,
    pub total_accesses: usize,
    pub active_nodes: usize,
    pub window_hours: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporal_tracking() {
        let tracker = TemporalTracker::new(24);
        
        // Record some accesses
        let node1 = NodeId::from("node1");
        let node2 = NodeId::from("node2");
        
        tracker.record_access(&node1);
        tracker.record_access(&node1);
        tracker.record_access(&node2);
        
        // Check frequencies
        assert_eq!(tracker.get_access_frequency(&node1), 2);
        assert_eq!(tracker.get_access_frequency(&node2), 1);
        
        // Check active nodes
        let active = tracker.get_active_nodes(2);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0], node1);
    }
    
    #[test]
    fn test_cooccurrence_detection() {
        let tracker = TemporalTracker::new(24);
        
        let node1 = NodeId::from("node1");
        let node2 = NodeId::from("node2");
        let node3 = NodeId::from("node3");
        
        // Record accesses that should co-occur
        tracker.record_batch_access(&[node1.clone(), node2.clone()]);
        std::thread::sleep(std::time::Duration::from_millis(100));
        tracker.record_batch_access(&[node1.clone(), node2.clone()]);
        
        // Record separate access
        std::thread::sleep(std::time::Duration::from_secs(10));
        tracker.record_access(&node3);
        
        // Check co-occurrence patterns
        let patterns = tracker.get_cooccurrence_patterns(5);
        assert!(!patterns.is_empty());
        
        // Should find co-occurrence between node1 and node2
        let pattern = patterns.iter()
            .find(|p| (p.node_a == node1 && p.node_b == node2) || 
                      (p.node_a == node2 && p.node_b == node1));
        assert!(pattern.is_some());
    }
}