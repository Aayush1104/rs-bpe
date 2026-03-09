use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::kimi_k2;

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

const DEFAULT_SYSTEM_PROMPT: &str = "You are Kimi, an AI assistant created by Moonshot AI.";

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
}
