use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{kimi_k2, Tokenizer, TOKENIZER_POOL};

// ---------------------------------------------------------------------------
// String constants for apply_chat_template (original path)
// ---------------------------------------------------------------------------
const IM_END: &str = "<|im_end|>";
const IM_USER: &str = "<|im_user|>";
const IM_ASSISTANT: &str = "<|im_assistant|>";
const IM_SYSTEM: &str = "<|im_system|>";
const IM_MIDDLE: &str = "<|im_middle|>";
const TOOL_CALLS_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const TOOL_CALLS_SECTION_END: &str = "<|tool_calls_section_end|>";
const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";
const TOOL_CALL_ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";

// ---------------------------------------------------------------------------
// Special token IDs (from kimi_k2_special.json) for direct emission
// ---------------------------------------------------------------------------
const IM_END_ID: u32 = 163586;
const IM_USER_ID: u32 = 163587;
const IM_ASSISTANT_ID: u32 = 163588;
const IM_SYSTEM_ID: u32 = 163594;
const TOOL_CALLS_SECTION_BEGIN_ID: u32 = 163595;
const TOOL_CALLS_SECTION_END_ID: u32 = 163596;
const TOOL_CALL_BEGIN_ID: u32 = 163597;
const TOOL_CALL_ARGUMENT_BEGIN_ID: u32 = 163598;
const TOOL_CALL_END_ID: u32 = 163599;
const IM_MIDDLE_ID: u32 = 163601;

const DEFAULT_SYSTEM_PROMPT: &str = "You are Kimi, an AI assistant created by Moonshot AI.";

// Threshold for switching to parallel encoding of uncached messages
const PARALLEL_THRESHOLD: usize = 4;

// ---------------------------------------------------------------------------
// Encoding statistics & configuration
// ---------------------------------------------------------------------------

/// Statistics collected during `ChatEncoder::encode_messages` calls.
/// Tracks cache utilisation and parallelism decisions.
#[derive(Debug, Clone, Default)]
pub struct EncodeStats {
    pub total_messages: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub parallel_batches: u64,
    pub sequential_batches: u64,
}

impl EncodeStats {
    /// Cache hit rate as a fraction in [0, 1]. Returns 0.0 when no messages have been processed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }
}

/// Fine-grained feature toggles for `ChatEncoder`.
/// All flags default to `true`, matching the fully-optimised path.
#[derive(Debug, Clone)]
pub struct ChatEncoderConfig {
    pub enable_cache: bool,
    pub enable_parallel: bool,
    pub enable_buffer_reuse: bool,
}

impl Default for ChatEncoderConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            enable_parallel: true,
            enable_buffer_reuse: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCallInput>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCallInput {
    pub id: Option<String>,
    pub function: FunctionCallInput,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FunctionCallInput {
    pub name: Option<String>,
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeMessagesError {
    UnsupportedRole(String),
}

impl Display for EncodeMessagesError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRole(role) => write!(f, "unsupported role: {role}"),
        }
    }
}

impl std::error::Error for EncodeMessagesError {}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Original path: apply_chat_template → encode (baseline)
// ---------------------------------------------------------------------------

pub fn apply_chat_template(
    messages: &[Message],
    tools: Option<&[Value]>,
    add_generation_prompt: bool,
) -> Result<String, EncodeMessagesError> {
    let mut prompt = String::new();

    if let Some(tools) = tools {
        prompt.push_str(IM_SYSTEM);
        prompt.push_str("tool_declare");
        prompt.push_str(IM_MIDDLE);
        let tools_json = serde_json::to_string(tools)
            .expect("serializing tool definitions should never fail");
        prompt.push_str(&tools_json);
        prompt.push_str(IM_END);
    }

    let has_system_first = messages.first().is_some_and(|m| m.role == "system");

    for (index, message) in messages.iter().enumerate() {
        if index == 0 && !has_system_first {
            prompt.push_str(IM_SYSTEM);
            prompt.push_str("system");
            prompt.push_str(IM_MIDDLE);
            prompt.push_str(DEFAULT_SYSTEM_PROMPT);
            prompt.push_str(IM_END);
        }

        let role_name = message
            .name
            .as_deref()
            .unwrap_or(message.role.as_str());

        match message.role.as_str() {
            "user" => {
                prompt.push_str(IM_USER);
                prompt.push_str(role_name);
                prompt.push_str(IM_MIDDLE);
            }
            "assistant" => {
                prompt.push_str(IM_ASSISTANT);
                prompt.push_str(role_name);
                prompt.push_str(IM_MIDDLE);
            }
            "system" | "tool" => {
                prompt.push_str(IM_SYSTEM);
                prompt.push_str(role_name);
                prompt.push_str(IM_MIDDLE);
            }
            other => return Err(EncodeMessagesError::UnsupportedRole(other.to_string())),
        }

        if message.role == "assistant" && message.tool_calls.is_some() {
            if let Some(content) = &message.content {
                prompt.push_str(content);
            }
            prompt.push_str(TOOL_CALLS_SECTION_BEGIN);
            if let Some(tool_calls) = &message.tool_calls {
                for tc in tool_calls {
                    let id = tc.id.as_deref().unwrap_or("");
                    prompt.push_str(TOOL_CALL_BEGIN);
                    prompt.push_str(id);
                    prompt.push_str(TOOL_CALL_ARGUMENT_BEGIN);
                    match &tc.function.arguments {
                        Some(Value::String(s)) => prompt.push_str(s),
                        Some(v) => {
                            let json = serde_json::to_string(v)
                                .expect("serializing arguments should never fail");
                            prompt.push_str(&json);
                        }
                        None => {}
                    }
                    prompt.push_str(TOOL_CALL_END);
                }
            }
            prompt.push_str(TOOL_CALLS_SECTION_END);
        } else if message.role == "tool" {
            let tool_call_id = message.tool_call_id.as_deref().unwrap_or("");
            prompt.push_str(&format!("## Return of {tool_call_id}\n"));
            if let Some(content) = &message.content {
                prompt.push_str(content);
            }
        } else if let Some(content) = &message.content {
            prompt.push_str(content);
        }

        prompt.push_str(IM_END);
    }

    if add_generation_prompt {
        prompt.push_str(IM_ASSISTANT);
        prompt.push_str("assistant");
        prompt.push_str(IM_MIDDLE);
    }

    Ok(prompt)
}

pub fn tokenize_messages(
    messages: &[Message],
    tools: Option<&[Value]>,
    add_generation_prompt: bool,
) -> Result<(Vec<u32>, String), EncodeMessagesError> {
    let prompt = apply_chat_template(messages, tools, add_generation_prompt)?;
    let tokens = kimi_k2().encode(&prompt, None);
    Ok((tokens, prompt))
}

// ---------------------------------------------------------------------------
// Step 1: Direct token emission (eliminates Aho-Corasick + string building)
// ---------------------------------------------------------------------------

/// Encode a single message to tokens, including its role header and trailing `<|im_end|>`.
/// This is the building block used by both `tokenize_messages_direct` and `ChatEncoder`.
///
/// NOTE: assumes message content does not contain special token strings. If it does,
/// those strings will be BPE-encoded as regular text rather than mapped to special IDs.
fn encode_single_message(
    message: &Message,
    tokenizer: &Tokenizer,
    tokens: &mut Vec<u32>,
) -> Result<(), EncodeMessagesError> {
    let role_name = message.name.as_deref().unwrap_or(message.role.as_str());

    match message.role.as_str() {
        "user" => tokens.push(IM_USER_ID),
        "assistant" => tokens.push(IM_ASSISTANT_ID),
        "system" | "tool" => tokens.push(IM_SYSTEM_ID),
        other => return Err(EncodeMessagesError::UnsupportedRole(other.to_string())),
    }

    tokenizer.encode_text_only_into(role_name, tokens);
    tokens.push(IM_MIDDLE_ID);

    if message.role == "assistant" && message.tool_calls.is_some() {
        if let Some(content) = &message.content {
            tokenizer.encode_text_only_into(content.as_str(), tokens);
        }
        tokens.push(TOOL_CALLS_SECTION_BEGIN_ID);
        if let Some(tool_calls) = &message.tool_calls {
            for tc in tool_calls {
                let id = tc.id.as_deref().unwrap_or("");
                tokens.push(TOOL_CALL_BEGIN_ID);
                tokenizer.encode_text_only_into(id, tokens);
                tokens.push(TOOL_CALL_ARGUMENT_BEGIN_ID);
                match &tc.function.arguments {
                    Some(Value::String(s)) => {
                        tokenizer.encode_text_only_into(s.as_str(), tokens);
                    }
                    Some(v) => {
                        let json = serde_json::to_string(v)
                            .expect("serializing arguments should never fail");
                        tokenizer.encode_text_only_into(json.as_str(), tokens);
                    }
                    None => {}
                }
                tokens.push(TOOL_CALL_END_ID);
            }
        }
        tokens.push(TOOL_CALLS_SECTION_END_ID);
    } else if message.role == "tool" {
        // Tool header + content must be encoded as a single text segment
        // to match pretokenizer boundaries of the original pipeline.
        let tool_call_id = message.tool_call_id.as_deref().unwrap_or("");
        let mut tool_text = format!("## Return of {tool_call_id}\n");
        if let Some(content) = &message.content {
            tool_text.push_str(content);
        }
        tokenizer.encode_text_only_into(tool_text.as_str(), tokens);
    } else if let Some(content) = &message.content {
        tokenizer.encode_text_only_into(content.as_str(), tokens);
    }

    tokens.push(IM_END_ID);
    Ok(())
}

/// Like `encode_single_message` but returns a new Vec (for parallel / cached use).
fn encode_single_message_vec(
    message: &Message,
    tokenizer: &Tokenizer,
) -> Result<Vec<u32>, EncodeMessagesError> {
    let mut tokens = Vec::new();
    encode_single_message(message, tokenizer, &mut tokens)?;
    Ok(tokens)
}

/// Emit the tools-declaration prefix tokens.
fn encode_tools_prefix(tools: &[Value], tokenizer: &Tokenizer, tokens: &mut Vec<u32>) {
    tokens.push(IM_SYSTEM_ID);
    tokenizer.encode_text_only_into("tool_declare", tokens);
    tokens.push(IM_MIDDLE_ID);
    let tools_json =
        serde_json::to_string(tools).expect("serializing tool definitions should never fail");
    tokenizer.encode_text_only_into(tools_json.as_str(), tokens);
    tokens.push(IM_END_ID);
}

/// Emit the default system prompt tokens.
fn encode_default_system_prompt(tokenizer: &Tokenizer, tokens: &mut Vec<u32>) {
    tokens.push(IM_SYSTEM_ID);
    tokenizer.encode_text_only_into("system", tokens);
    tokens.push(IM_MIDDLE_ID);
    tokenizer.encode_text_only_into(DEFAULT_SYSTEM_PROMPT, tokens);
    tokens.push(IM_END_ID);
}

/// Emit the generation prompt suffix tokens.
fn encode_generation_prompt(tokenizer: &Tokenizer, tokens: &mut Vec<u32>) {
    tokens.push(IM_ASSISTANT_ID);
    tokenizer.encode_text_only_into("assistant", tokens);
    tokens.push(IM_MIDDLE_ID);
}

/// Tokenize messages by emitting special token IDs directly, calling
/// `encode_text_only` only on message content. This skips the Aho-Corasick
/// scan and avoids building the full template string.
pub fn tokenize_messages_direct(
    messages: &[Message],
    tools: Option<&[Value]>,
    add_generation_prompt: bool,
) -> Result<Vec<u32>, EncodeMessagesError> {
    let tokenizer = kimi_k2();
    let mut tokens = Vec::new();

    if let Some(tools) = tools {
        encode_tools_prefix(tools, tokenizer, &mut tokens);
    }

    let has_system_first = messages.first().is_some_and(|m| m.role == "system");

    for (index, message) in messages.iter().enumerate() {
        if index == 0 && !has_system_first {
            encode_default_system_prompt(tokenizer, &mut tokens);
        }
        encode_single_message(message, tokenizer, &mut tokens)?;
    }

    if add_generation_prompt {
        encode_generation_prompt(tokenizer, &mut tokens);
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Steps 2-4: ChatEncoder with message caching, parallel encoding, buffer reuse
// ---------------------------------------------------------------------------

/// Compute a u64 hash of all message fields that affect token output.
fn hash_message(message: &Message) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    message.role.hash(&mut hasher);
    message.content.hash(&mut hasher);
    message.name.hash(&mut hasher);
    message.tool_call_id.hash(&mut hasher);
    if let Some(tool_calls) = &message.tool_calls {
        tool_calls.len().hash(&mut hasher);
        for tc in tool_calls {
            tc.id.hash(&mut hasher);
            tc.function.name.hash(&mut hasher);
            if let Some(args) = &tc.function.arguments {
                // Value doesn't impl Hash; serialize for hashing
                let json = serde_json::to_string(args).unwrap_or_default();
                json.hash(&mut hasher);
            } else {
                0u8.hash(&mut hasher);
            }
        }
    } else {
        0u8.hash(&mut hasher);
    }
    hasher.finish()
}

/// A stateful chat encoder that caches per-message token sequences across
/// calls. Designed for multi-turn conversations where most prior messages
/// are unchanged between turns.
///
/// Usage:
/// ```ignore
/// let mut encoder = ChatEncoder::new();
/// // First turn
/// let tokens1 = encoder.encode_messages(&messages1, None, true)?;
/// // Second turn (appends new messages, prior ones are cached)
/// let tokens2 = encoder.encode_messages(&messages2, None, true)?;
/// ```
pub struct ChatEncoder {
    /// Per-message token cache keyed by content hash.
    cache: HashMap<u64, Vec<u32>>,
    /// Reusable output buffer (Step 4: avoids reallocation across calls).
    token_buffer: Vec<u32>,
    /// Feature toggles.
    config: ChatEncoderConfig,
    /// Cumulative encoding statistics.
    stats: EncodeStats,
}

impl ChatEncoder {
    pub fn new() -> Self {
        Self::new_with_config(ChatEncoderConfig::default())
    }

    pub fn new_with_config(config: ChatEncoderConfig) -> Self {
        Self {
            cache: HashMap::new(),
            token_buffer: Vec::new(),
            config,
            stats: EncodeStats::default(),
        }
    }

    /// Returns the number of cached message entries.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Clear the message cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Returns a snapshot of the cumulative statistics.
    pub fn stats(&self) -> &EncodeStats {
        &self.stats
    }

    /// Resets the cumulative statistics to zero.
    pub fn reset_stats(&mut self) {
        self.stats = EncodeStats::default();
    }

    /// Encode a message sequence, returning a borrowed slice of the internal buffer.
    ///
    /// This is the zero-copy path: the returned slice points directly into
    /// `self.token_buffer`, avoiding a ~300KB clone for 75K-token conversations.
    ///
    /// The borrow checker ensures the caller cannot call any `&mut self` method
    /// while holding the returned reference.
    pub fn encode_messages_ref(
        &mut self,
        messages: &[Message],
        tools: Option<&[Value]>,
        add_generation_prompt: bool,
    ) -> Result<&[u32], EncodeMessagesError> {
        let tokenizer = kimi_k2();

        // Step 4: buffer reuse — reuse capacity or allocate fresh
        if self.config.enable_buffer_reuse {
            self.token_buffer.clear();
        } else {
            self.token_buffer = Vec::new();
        }

        // Tools prefix (not cached — typically small and may vary)
        if let Some(tools) = tools {
            encode_tools_prefix(tools, tokenizer, &mut self.token_buffer);
        }

        let has_system_first = messages.first().is_some_and(|m| m.role == "system");

        // Step 2: compute hashes and identify uncached messages
        let hashes: Vec<u64> = messages.iter().map(|m| hash_message(m)).collect();
        let uncached: Vec<usize> = if self.config.enable_cache {
            hashes
                .iter()
                .enumerate()
                .filter(|(_, h)| !self.cache.contains_key(h))
                .map(|(i, _)| i)
                .collect()
        } else {
            // Cache disabled — treat every message as uncached
            (0..messages.len()).collect()
        };

        // Track stats
        let msg_count = messages.len() as u64;
        self.stats.total_messages += msg_count;
        let hits = msg_count - uncached.len() as u64;
        self.stats.cache_hits += hits;
        self.stats.cache_misses += uncached.len() as u64;

        // Step 3: encode uncached messages (parallel when beneficial)
        if !uncached.is_empty() {
            let use_parallel =
                self.config.enable_parallel && uncached.len() >= PARALLEL_THRESHOLD;

            if use_parallel {
                self.stats.parallel_batches += 1;
            } else {
                self.stats.sequential_batches += 1;
            }

            let new_entries: Vec<(u64, Result<Vec<u32>, EncodeMessagesError>)> = if use_parallel {
                TOKENIZER_POOL.install(|| {
                    uncached
                        .par_iter()
                        .map(|&i| {
                            let h = hashes[i];
                            let result = encode_single_message_vec(&messages[i], tokenizer);
                            (h, result)
                        })
                        .collect()
                })
            } else {
                uncached
                    .iter()
                    .map(|&i| {
                        let h = hashes[i];
                        let result = encode_single_message_vec(&messages[i], tokenizer);
                        (h, result)
                    })
                    .collect()
            };

            for (h, result) in new_entries {
                let tokens = result?;
                self.cache.insert(h, tokens);
            }
        }

        // Assemble output from cache
        for (index, _message) in messages.iter().enumerate() {
            if index == 0 && !has_system_first {
                encode_default_system_prompt(tokenizer, &mut self.token_buffer);
            }
            let h = hashes[index];
            if let Some(cached) = self.cache.get(&h) {
                self.token_buffer.extend_from_slice(cached);
            }
        }

        if add_generation_prompt {
            encode_generation_prompt(tokenizer, &mut self.token_buffer);
        }

        // When cache is disabled, clear it after assembly so it doesn't persist
        if !self.config.enable_cache {
            self.cache.clear();
        }

        Ok(&self.token_buffer)
    }

    /// Convenience wrapper that returns an owned `Vec<u32>`.
    ///
    /// Use [`encode_messages_ref`] in performance-critical paths where you
    /// can consume the slice immediately (e.g. the FFI borrowed path).
    pub fn encode_messages(
        &mut self,
        messages: &[Message],
        tools: Option<&[Value]>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, EncodeMessagesError> {
        self.encode_messages_ref(messages, tools, add_generation_prompt)
            .map(|s| s.to_vec())
    }
}

impl Default for ChatEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chat() {
        let messages = vec![Message::new("user", "hello")];
        let prompt = apply_chat_template(&messages, None, true).expect("should succeed");
        assert_eq!(
            prompt,
            concat!(
                "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>",
                "<|im_user|>user<|im_middle|>hello<|im_end|>",
                "<|im_assistant|>assistant<|im_middle|>",
            )
        );
    }

    #[test]
    fn test_system_message() {
        let messages = vec![
            Message::new("system", "You are a helpful assistant."),
            Message::new("user", "hello"),
        ];
        let prompt = apply_chat_template(&messages, None, true).expect("should succeed");
        assert!(prompt.starts_with("<|im_system|>system<|im_middle|>You are a helpful assistant.<|im_end|>"));
        assert!(!prompt.contains(DEFAULT_SYSTEM_PROMPT));
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "Hi there!"),
            Message::new("user", "how are you?"),
        ];
        let prompt = apply_chat_template(&messages, None, true).expect("should succeed");
        assert!(prompt.contains("<|im_user|>user<|im_middle|>hello<|im_end|>"));
        assert!(prompt.contains("<|im_assistant|>assistant<|im_middle|>Hi there!<|im_end|>"));
        assert!(prompt.contains("<|im_user|>user<|im_middle|>how are you?<|im_end|>"));
        assert!(prompt.ends_with("<|im_assistant|>assistant<|im_middle|>"));
    }

    #[test]
    fn test_tool_calls() {
        let messages = vec![Message {
            role: "assistant".to_string(),
            tool_calls: Some(vec![ToolCallInput {
                id: Some("call_123".to_string()),
                function: FunctionCallInput {
                    name: Some("search".to_string()),
                    arguments: Some(Value::String("{\"query\":\"hello\"}".to_string())),
                },
            }]),
            ..Message::default()
        }];
        let prompt = apply_chat_template(&messages, None, false).expect("should succeed");
        assert!(prompt.contains(TOOL_CALLS_SECTION_BEGIN));
        assert!(prompt.contains(TOOL_CALLS_SECTION_END));
        assert!(prompt.contains("<|tool_call_begin|>call_123<|tool_call_argument_begin|>"));
        assert!(prompt.contains("{\"query\":\"hello\"}"));
    }

    #[test]
    fn test_no_generation_prompt() {
        let messages = vec![Message::new("user", "hello")];
        let prompt = apply_chat_template(&messages, None, false).expect("should succeed");
        assert!(!prompt.ends_with("<|im_assistant|>assistant<|im_middle|>"));
        assert!(prompt.ends_with("<|im_end|>"));
    }

    // -----------------------------------------------------------------------
    // Direct emission equivalence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_direct_basic_chat() {
        let messages = vec![Message::new("user", "hello")];
        let (baseline, _) = tokenize_messages(&messages, None, true).unwrap();
        let direct = tokenize_messages_direct(&messages, None, true).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline for basic chat");
    }

    #[test]
    fn test_direct_system_message() {
        let messages = vec![
            Message::new("system", "You are a helpful assistant."),
            Message::new("user", "hello"),
        ];
        let (baseline, _) = tokenize_messages(&messages, None, true).unwrap();
        let direct = tokenize_messages_direct(&messages, None, true).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline with system message");
    }

    #[test]
    fn test_direct_multi_turn() {
        let messages = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "Hi there! How can I help you today?"),
            Message::new("user", "What is the capital of France?"),
            Message::new("assistant", "The capital of France is Paris."),
            Message::new("user", "Thanks!"),
        ];
        let (baseline, _) = tokenize_messages(&messages, None, true).unwrap();
        let direct = tokenize_messages_direct(&messages, None, true).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline for multi-turn");
    }

    #[test]
    fn test_direct_tool_calls() {
        let messages = vec![
            Message::new("user", "Search for cats"),
            Message {
                role: "assistant".to_string(),
                content: Some("Let me search for that.".to_string()),
                tool_calls: Some(vec![ToolCallInput {
                    id: Some("call_abc".to_string()),
                    function: FunctionCallInput {
                        name: Some("web_search".to_string()),
                        arguments: Some(Value::String("{\"query\":\"cats\"}".to_string())),
                    },
                }]),
                ..Message::default()
            },
        ];
        let (baseline, _) = tokenize_messages(&messages, None, false).unwrap();
        let direct = tokenize_messages_direct(&messages, None, false).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline with tool calls");
    }

    #[test]
    fn test_direct_tool_result() {
        let messages = vec![
            Message::new("user", "Search for cats"),
            Message {
                role: "tool".to_string(),
                content: Some("Found 42 results about cats.".to_string()),
                tool_call_id: Some("call_abc".to_string()),
                ..Message::default()
            },
        ];
        let (baseline, _) = tokenize_messages(&messages, None, true).unwrap();
        let direct = tokenize_messages_direct(&messages, None, true).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline with tool result");
    }

    #[test]
    fn test_direct_with_tools() {
        let tools: Vec<Value> = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })];
        let messages = vec![Message::new("user", "What's the weather in Paris?")];
        let (baseline, _) = tokenize_messages(&messages, Some(&tools), true).unwrap();
        let direct = tokenize_messages_direct(&messages, Some(&tools), true).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline with tools");
    }

    #[test]
    fn test_direct_no_generation_prompt() {
        let messages = vec![Message::new("user", "hello")];
        let (baseline, _) = tokenize_messages(&messages, None, false).unwrap();
        let direct = tokenize_messages_direct(&messages, None, false).unwrap();
        assert_eq!(baseline, direct, "direct emission must match baseline without generation prompt");
    }

    // -----------------------------------------------------------------------
    // ChatEncoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_encoder_basic() {
        let mut encoder = ChatEncoder::new();
        let messages = vec![Message::new("user", "hello")];
        let result = encoder.encode_messages(&messages, None, true).unwrap();
        let direct = tokenize_messages_direct(&messages, None, true).unwrap();
        assert_eq!(result, direct, "ChatEncoder must match direct emission");
        assert_eq!(encoder.cache_len(), 1);
    }

    #[test]
    fn test_chat_encoder_multi_turn_caching() {
        let mut encoder = ChatEncoder::new();

        // Turn 1
        let messages1 = vec![
            Message::new("user", "hello"),
        ];
        let result1 = encoder.encode_messages(&messages1, None, true).unwrap();
        assert_eq!(encoder.cache_len(), 1);

        // Turn 2: same first message + new messages
        let messages2 = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "Hi there!"),
            Message::new("user", "how are you?"),
        ];
        let result2 = encoder.encode_messages(&messages2, None, true).unwrap();
        assert_eq!(encoder.cache_len(), 3);

        // Verify turn 2 matches direct emission
        let direct2 = tokenize_messages_direct(&messages2, None, true).unwrap();
        assert_eq!(result2, direct2, "ChatEncoder multi-turn must match direct");

        // Turn 3: same messages + one new
        let messages3 = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "Hi there!"),
            Message::new("user", "how are you?"),
            Message::new("assistant", "I'm doing great!"),
            Message::new("user", "What's 2+2?"),
        ];
        let result3 = encoder.encode_messages(&messages3, None, true).unwrap();
        assert_eq!(encoder.cache_len(), 5);

        let direct3 = tokenize_messages_direct(&messages3, None, true).unwrap();
        assert_eq!(result3, direct3, "ChatEncoder turn 3 must match direct");

        // Verify result1 matches
        let direct1 = tokenize_messages_direct(&messages1, None, true).unwrap();
        assert_eq!(result1, direct1);
    }

    #[test]
    fn test_chat_encoder_with_tools() {
        let mut encoder = ChatEncoder::new();
        let tools: Vec<Value> = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web"
            }
        })];
        let messages = vec![Message::new("user", "search for cats")];
        let result = encoder.encode_messages(&messages, Some(&tools), true).unwrap();
        let direct = tokenize_messages_direct(&messages, Some(&tools), true).unwrap();
        assert_eq!(result, direct);
    }

    #[test]
    fn test_chat_encoder_clear_cache() {
        let mut encoder = ChatEncoder::new();
        let messages = vec![Message::new("user", "hello")];
        let _ = encoder.encode_messages(&messages, None, true).unwrap();
        assert_eq!(encoder.cache_len(), 1);
        encoder.clear_cache();
        assert_eq!(encoder.cache_len(), 0);
    }

    // -----------------------------------------------------------------------
    // ChatEncoderConfig + EncodeStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_basic() {
        let mut encoder = ChatEncoder::new();
        assert_eq!(encoder.stats().total_messages, 0);
        assert_eq!(encoder.stats().hit_rate(), 0.0);

        let msgs = vec![Message::new("user", "hello")];
        let _ = encoder.encode_messages(&msgs, None, true).unwrap();

        assert_eq!(encoder.stats().total_messages, 1);
        assert_eq!(encoder.stats().cache_misses, 1);
        assert_eq!(encoder.stats().cache_hits, 0);
        assert_eq!(encoder.stats().hit_rate(), 0.0);

        // Second call — same message should be a cache hit
        let _ = encoder.encode_messages(&msgs, None, true).unwrap();
        assert_eq!(encoder.stats().total_messages, 2);
        assert_eq!(encoder.stats().cache_hits, 1);
        assert_eq!(encoder.stats().cache_misses, 1);
        assert!((encoder.stats().hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_multi_turn() {
        let mut encoder = ChatEncoder::new();

        let msgs1 = vec![Message::new("user", "hello")];
        let _ = encoder.encode_messages(&msgs1, None, true).unwrap();
        // 1 miss
        assert_eq!(encoder.stats().cache_misses, 1);

        let msgs2 = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "Hi!"),
            Message::new("user", "how are you?"),
        ];
        let _ = encoder.encode_messages(&msgs2, None, true).unwrap();
        // 1 hit (hello) + 2 misses (Hi!, how are you?)
        assert_eq!(encoder.stats().cache_hits, 1);
        assert_eq!(encoder.stats().cache_misses, 3); // 1 + 2
        assert_eq!(encoder.stats().total_messages, 4); // 1 + 3
    }

    #[test]
    fn test_stats_reset() {
        let mut encoder = ChatEncoder::new();
        let msgs = vec![Message::new("user", "hello")];
        let _ = encoder.encode_messages(&msgs, None, true).unwrap();
        assert!(encoder.stats().total_messages > 0);

        encoder.reset_stats();
        assert_eq!(encoder.stats().total_messages, 0);
        assert_eq!(encoder.stats().cache_hits, 0);
        assert_eq!(encoder.stats().cache_misses, 0);
        assert_eq!(encoder.stats().hit_rate(), 0.0);
    }

    #[test]
    fn test_config_cache_disabled() {
        let config = ChatEncoderConfig {
            enable_cache: false,
            enable_parallel: true,
            enable_buffer_reuse: true,
        };
        let mut encoder = ChatEncoder::new_with_config(config);
        let msgs = vec![Message::new("user", "hello")];

        let r1 = encoder.encode_messages(&msgs, None, true).unwrap();
        let r2 = encoder.encode_messages(&msgs, None, true).unwrap();

        // Results must still be correct
        let direct = tokenize_messages_direct(&msgs, None, true).unwrap();
        assert_eq!(r1, direct);
        assert_eq!(r2, direct);

        // Cache should be empty after each call (cleared when disabled)
        assert_eq!(encoder.cache_len(), 0);

        // Stats should show all misses (no caching)
        assert_eq!(encoder.stats().cache_hits, 0);
        assert_eq!(encoder.stats().cache_misses, 2);
    }

    #[test]
    fn test_config_parallel_disabled() {
        let config = ChatEncoderConfig {
            enable_cache: true,
            enable_parallel: false,
            enable_buffer_reuse: true,
        };
        let mut encoder = ChatEncoder::new_with_config(config);

        // Use enough messages to normally trigger parallel (>= PARALLEL_THRESHOLD)
        let msgs = vec![
            Message::new("user", "msg1"),
            Message::new("assistant", "msg2"),
            Message::new("user", "msg3"),
            Message::new("assistant", "msg4"),
            Message::new("user", "msg5"),
        ];

        let result = encoder.encode_messages(&msgs, None, true).unwrap();
        let direct = tokenize_messages_direct(&msgs, None, true).unwrap();
        assert_eq!(result, direct);

        // Should have recorded sequential, not parallel
        assert_eq!(encoder.stats().parallel_batches, 0);
        assert_eq!(encoder.stats().sequential_batches, 1);
    }

    #[test]
    fn test_config_buffer_reuse_disabled() {
        let config = ChatEncoderConfig {
            enable_cache: true,
            enable_parallel: true,
            enable_buffer_reuse: false,
        };
        let mut encoder = ChatEncoder::new_with_config(config);
        let msgs = vec![Message::new("user", "hello")];

        let r1 = encoder.encode_messages(&msgs, None, true).unwrap();
        let r2 = encoder.encode_messages(&msgs, None, true).unwrap();

        let direct = tokenize_messages_direct(&msgs, None, true).unwrap();
        assert_eq!(r1, direct);
        assert_eq!(r2, direct);
    }

    #[test]
    fn test_config_all_disabled_matches_direct() {
        // direct-only mode: no cache, no parallel, no buffer reuse
        let config = ChatEncoderConfig {
            enable_cache: false,
            enable_parallel: false,
            enable_buffer_reuse: false,
        };
        let mut encoder = ChatEncoder::new_with_config(config);

        let msgs = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "Hi!"),
            Message::new("user", "how are you?"),
        ];

        let result = encoder.encode_messages(&msgs, None, true).unwrap();
        let direct = tokenize_messages_direct(&msgs, None, true).unwrap();
        assert_eq!(result, direct);
    }
}
