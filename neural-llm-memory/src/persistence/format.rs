//! Persistence format definitions and versioning

use serde::{Serialize, Deserialize};

/// Persistence format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceFormat {
    /// Binary format using bincode (fastest, smallest)
    Binary,
    /// JSON format (human-readable, larger)
    Json,
    /// Compressed binary format (smallest size)
    Compressed,
}

impl PersistenceFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            PersistenceFormat::Binary => "bin",
            PersistenceFormat::Json => "json",
            PersistenceFormat::Compressed => "gz",
        }
    }
    
    /// Get format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "bin" => Some(PersistenceFormat::Binary),
            "json" => Some(PersistenceFormat::Json),
            "gz" => Some(PersistenceFormat::Compressed),
            _ => None,
        }
    }
    
    /// Get human-readable format name
    pub fn name(&self) -> &'static str {
        match self {
            PersistenceFormat::Binary => "Binary",
            PersistenceFormat::Json => "JSON",
            PersistenceFormat::Compressed => "Compressed Binary",
        }
    }
}

/// Format version for compatibility checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl FormatVersion {
    /// Current format version
    pub fn current() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
    
    /// Create a new version
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self { major, minor, patch }
    }
    
    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Major version must match
        if self.major != other.major {
            return false;
        }
        
        // Can read older minor versions
        if self.minor < other.minor {
            return false;
        }
        
        true
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// File header for quick format detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHeader {
    /// Magic bytes for format identification
    pub magic: [u8; 8],
    
    /// Format version
    pub version: FormatVersion,
    
    /// Format type
    pub format_type: String,
    
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Checksum of the data (optional)
    pub checksum: Option<u64>,
}

impl FileHeader {
    /// Magic bytes for neural network files
    pub const MAGIC: [u8; 8] = *b"NEURALNW";
    
    /// Create a new file header
    pub fn new(format_type: &str) -> Self {
        Self {
            magic: Self::MAGIC,
            version: FormatVersion::current(),
            format_type: format_type.to_string(),
            created_at: chrono::Utc::now(),
            checksum: None,
        }
    }
    
    /// Validate magic bytes
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err("Invalid magic bytes".to_string());
        }
        
        if !FormatVersion::current().is_compatible_with(&self.version) {
            return Err(format!(
                "Incompatible version: {} (current: {})",
                self.version,
                FormatVersion::current()
            ));
        }
        
        Ok(())
    }
}

/// Statistics about a saved file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStats {
    /// File size in bytes
    pub size_bytes: u64,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Total parameters
    pub total_parameters: usize,
    
    /// Compression ratio (if applicable)
    pub compression_ratio: Option<f32>,
    
    /// Save duration in milliseconds
    pub save_duration_ms: u64,
}

impl FileStats {
    /// Get human-readable size
    pub fn size_human(&self) -> String {
        let size = self.size_bytes as f64;
        if size < 1024.0 {
            format!("{} B", size)
        } else if size < 1024.0 * 1024.0 {
            format!("{:.2} KB", size / 1024.0)
        } else if size < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.2} MB", size / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", size / (1024.0 * 1024.0 * 1024.0))
        }
    }
    
    /// Get parameters per megabyte
    pub fn params_per_mb(&self) -> f32 {
        let mb = self.size_bytes as f32 / (1024.0 * 1024.0);
        if mb > 0.0 {
            self.total_parameters as f32 / mb
        } else {
            0.0
        }
    }
}

