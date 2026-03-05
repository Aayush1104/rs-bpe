use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::deepseek_32;

const BOS_TOKEN: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
const EOS_TOKEN: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
const USER_TOKEN: &str = "<\u{ff5c}User\u{ff5c}>";
const ASSISTANT_TOKEN: &str = "<\u{ff5c}Assistant\u{ff5c}>";
const SYSTEM_TOKEN: &str = "<\u{ff5c}System\u{ff5c}>";
const TOOL_CALL_BEGIN_TOKEN: &str = "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>";
const TOOL_CALL_END_TOKEN: &str = "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>";
const TOOL_SEP_TOKEN: &str = "<\u{ff5c}tool\u{2581}sep\u{ff5c}>";
const TOOL_OUTPUT_BEGIN_TOKEN: &str = "<\u{ff5c}tool\u{2581}output\u{2581}begin\u{ff5c}>";
const TOOL_OUTPUT_END_TOKEN: &str = "<\u{ff5c}tool\u{2581}output\u{2581}end\u{ff5c}>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "\u{ff5c}DSML\u{ff5c}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub response_format: Option<Value>,
    pub tool_calls: Option<Vec<ToolCallInput>>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolDefinition {
    pub function: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCallInput {
    pub function: FunctionCallInput,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FunctionCallInput {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeMessagesError {
    UnsupportedRole(String),
    MissingDeveloperContent,
    InvalidToolMessageOrder,
    MissingAssistantToolCalls,
    ToolCallOrderOutOfRange {
        order: usize,
        available: usize,
    },
    MissingAssistantReasoning {
        index: usize,
    },
    InvalidToolArgumentsJson {
        function_name: String,
        reason: String,
    },
    InvalidToolArgumentsObject {
        function_name: String,
    },
}

impl Display for EncodeMessagesError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRole(role) => write!(f, "unsupported role: {role}"),
            Self::MissingDeveloperContent => write!(f, "developer message content must not be empty"),
            Self::InvalidToolMessageOrder => write!(f, "tool message must follow an assistant message"),
            Self::MissingAssistantToolCalls => {
                write!(f, "assistant message before tool message has no tool_calls")
            }
            Self::ToolCallOrderOutOfRange { order, available } => write!(
                f,
                "tool message order out of range: order={order}, available={available}"
            ),
            Self::MissingAssistantReasoning { index } => write!(
                f,
                "assistant message at index {index} requires reasoning_content or tool_calls in thinking mode"
            ),
            Self::InvalidToolArgumentsJson {
                function_name,
                reason,
            } => write!(
                f,
                "invalid tool call arguments JSON for function '{function_name}': {reason}"
            ),
            Self::InvalidToolArgumentsObject { function_name } => write!(
                f,
                "tool call arguments for function '{function_name}' must decode to a JSON object"
            ),
        }
    }
}

impl std::error::Error for EncodeMessagesError {}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct ParsedMessage {
    role: String,
    content: String,
    tools: Vec<ToolDefinition>,
    response_format: Option<Value>,
    tool_calls: Vec<ToolCall>,
    reasoning_content: Option<String>,
}

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
    thinking_mode: ThinkingMode,
    context: Option<&[Message]>,
    drop_thinking: bool,
    add_default_bos_token: bool,
) -> Result<String, EncodeMessagesError> {
    let mut all_messages: Vec<ParsedMessage> = Vec::new();

    if add_default_bos_token {
        all_messages.push(ParsedMessage {
            role: "bos".to_string(),
            content: String::new(),
            tools: Vec::new(),
            response_format: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        });
    }

    if let Some(context_messages) = context {
        all_messages.extend(process_messages(context_messages, drop_thinking));
    }

    all_messages.extend(process_messages(messages, drop_thinking));

    let last_user_idx = find_last_user_index(&all_messages)
        .map(|idx| idx as isize)
        .unwrap_or(-1);

    let mut prompt = String::new();
    for (index, message) in all_messages.iter().enumerate() {
        prompt.push_str(&render_message(
            index,
            message,
            &all_messages,
            thinking_mode,
            last_user_idx,
        )?);
    }

    Ok(prompt)
}

pub fn tokenize_messages(
    messages: &[Message],
    thinking_mode: ThinkingMode,
    context: Option<&[Message]>,
    drop_thinking: bool,
    add_default_bos_token: bool,
) -> Result<(Vec<u32>, String), EncodeMessagesError> {
    let prompt = apply_chat_template(
        messages,
        thinking_mode,
        context,
        drop_thinking,
        add_default_bos_token,
    )?;
    let tokens = deepseek_32().encode(&prompt, None);
    Ok((tokens, prompt))
}

fn process_messages(messages: &[Message], drop_thinking: bool) -> Vec<ParsedMessage> {
    let available_roles = ["user", "assistant", "system", "developer", "tool", "bos"];
    let mut parsed_messages = Vec::new();

    for (index, msg) in messages.iter().enumerate() {
        if available_roles.contains(&msg.role.as_str()) {
            parsed_messages.push(ParsedMessage {
                role: msg.role.clone(),
                content: msg.content.clone().unwrap_or_default(),
                tools: msg.tools.clone().unwrap_or_default(),
                response_format: msg.response_format.clone(),
                tool_calls: normalize_tool_calls(msg.tool_calls.as_deref().unwrap_or(&[])),
                reasoning_content: msg.reasoning_content.clone(),
            });
            continue;
        }

        if msg.role == "assistant"
            && index + 1 < messages.len()
            && messages[index + 1].role == "tool"
        {
            let mut tool_calls = normalize_tool_calls(msg.tool_calls.as_deref().unwrap_or(&[]));

            if tool_calls.is_empty() {
                let next_message = &messages[index + 1];
                tool_calls.push(ToolCall {
                    name: "unknown".to_string(),
                    arguments: next_message.content.clone().unwrap_or_default(),
                });
            }

            parsed_messages.push(ParsedMessage {
                role: "assistant".to_string(),
                content: msg.content.clone().unwrap_or_default(),
                tools: Vec::new(),
                response_format: None,
                tool_calls,
                reasoning_content: msg.reasoning_content.clone(),
            });
            continue;
        }

        parsed_messages.push(ParsedMessage {
            role: "user".to_string(),
            content: msg.content.clone().unwrap_or_default(),
            tools: Vec::new(),
            response_format: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        });
    }

    if drop_thinking {
        drop_thinking_messages(&mut parsed_messages);
    }

    parsed_messages
}

fn normalize_tool_calls(tool_calls: &[ToolCallInput]) -> Vec<ToolCall> {
    tool_calls
        .iter()
        .map(|tool_call| ToolCall {
            name: tool_call
                .function
                .name
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            arguments: tool_call
                .function
                .arguments
                .clone()
                .unwrap_or_else(|| "{}".to_string()),
        })
        .collect()
}

fn render_message(
    index: usize,
    message: &ParsedMessage,
    all_messages: &[ParsedMessage],
    thinking_mode: ThinkingMode,
    last_user_idx: isize,
) -> Result<String, EncodeMessagesError> {
    let role = message.role.as_str();
    let mut prompt = String::new();

    match role {
        "bos" => {
            prompt.push_str(BOS_TOKEN);
            prompt.push_str(&message.content);
        }
        "tool" => {
            let mut prev_assistant_idx = index as isize - 1;
            while prev_assistant_idx >= 0
                && all_messages[prev_assistant_idx as usize].role == "tool"
            {
                prev_assistant_idx -= 1;
            }
            if prev_assistant_idx < 0
                || all_messages[prev_assistant_idx as usize].role != "assistant"
            {
                return Err(EncodeMessagesError::InvalidToolMessageOrder);
            }

            let assistant_tool_calls = &all_messages[prev_assistant_idx as usize].tool_calls;
            if assistant_tool_calls.is_empty() {
                return Err(EncodeMessagesError::MissingAssistantToolCalls);
            }

            let tool_call_order = index - prev_assistant_idx as usize;
            if assistant_tool_calls.len() < tool_call_order {
                return Err(EncodeMessagesError::ToolCallOrderOutOfRange {
                    order: tool_call_order,
                    available: assistant_tool_calls.len(),
                });
            }

            let tool_call = &assistant_tool_calls[tool_call_order - 1];
            let tool_call_id = format!("{}{}", tool_call.name, short_hash(&tool_call.arguments));

            if tool_call_order == 1 {
                prompt.push_str("\n\n<function_results>");
            }

            prompt.push_str(&format!(
                "{tool_output_begin}name={name}{tool_sep}{tool_call_id}\n{content}{tool_output_end}",
                tool_output_begin = TOOL_OUTPUT_BEGIN_TOKEN,
                name = tool_call.name,
                tool_sep = TOOL_SEP_TOKEN,
                content = message.content,
                tool_output_end = TOOL_OUTPUT_END_TOKEN,
            ));

            if tool_call_order == assistant_tool_calls.len() {
                prompt.push_str("\n</function_results>");
                if (index as isize) >= last_user_idx && thinking_mode == ThinkingMode::Thinking {
                    prompt.push_str("\n\n");
                    prompt.push_str(THINKING_START_TOKEN);
                } else {
                    prompt.push_str("\n\n");
                    prompt.push_str(THINKING_END_TOKEN);
                }
            }
        }
        "assistant" => {
            let tool_calls = if message.tool_calls.is_empty() {
                String::new()
            } else {
                let mut parts = Vec::with_capacity(message.tool_calls.len());
                for tool_call in &message.tool_calls {
                    let tool_call_id =
                        format!("{}{}", tool_call.name, short_hash(&tool_call.arguments));
                    parts.push(format!(
                        "{tool_call_begin}{name}{tool_sep}{tool_call_id}\n<{dsml}function={name}>\n{arguments}\n</{dsml}function>{tool_call_end}",
                        tool_call_begin = TOOL_CALL_BEGIN_TOKEN,
                        name = tool_call.name,
                        tool_sep = TOOL_SEP_TOKEN,
                        tool_call_id = tool_call_id,
                        dsml = DSML_TOKEN,
                        arguments = encode_arguments_to_dsml(&tool_call.arguments, &tool_call.name)?,
                        tool_call_end = TOOL_CALL_END_TOKEN,
                    ));
                }
                format!(
                    "\n\n<function_calls>\n{}\n</function_calls>",
                    parts.join("\n")
                )
            };

            let summary_content = message.content.as_str();
            let mut thinking_part = THINKING_END_TOKEN.to_string();
            if thinking_mode == ThinkingMode::Thinking && (index as isize) > last_user_idx {
                if message
                    .reasoning_content
                    .as_deref()
                    .is_none_or(str::is_empty)
                    && message.tool_calls.is_empty()
                {
                    return Err(EncodeMessagesError::MissingAssistantReasoning { index });
                }

                thinking_part = format!(
                    "{}{}",
                    message.reasoning_content.as_deref().unwrap_or_default(),
                    THINKING_END_TOKEN
                );
            }

            prompt.push_str(&format!(
                "{assistant}<think>{reasoning}{tool_calls}{content}{eos}",
                assistant = ASSISTANT_TOKEN,
                reasoning = thinking_part,
                tool_calls = tool_calls,
                content = summary_content,
                eos = EOS_TOKEN,
            ));
        }
        "system" => {
            prompt.push_str(&format!(
                "{system}{content}",
                system = SYSTEM_TOKEN,
                content = message.content,
            ));

            if !message.tools.is_empty() {
                let tools = message
                    .tools
                    .iter()
                    .map(|tool| tool.function.clone())
                    .collect::<Vec<_>>();
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(&tools));
            }

            if let Some(response_format) = &message.response_format {
                prompt.push_str("\n\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n");
                prompt.push_str("As an example, for the schema ");
                prompt.push_str("{\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n");
                prompt.push_str("the object ");
                prompt.push_str("{\"foo\": [\"bar\", \"baz\"]}");
                prompt.push_str(" is a well-formatted instance of the schema.\n");
                prompt.push_str("The object ");
                prompt.push_str("{\"properties\": {\"foo\": [\"bar\", \"baz\"]}}");
                prompt.push_str(" is not well-formatted.\n\n");
                prompt.push_str("Here is the output schema:\n```\n");
                prompt.push_str(&to_json(response_format));
                prompt.push_str("\n```\n");
            }
        }
        "developer" => {
            if message.content.is_empty() {
                return Err(EncodeMessagesError::MissingDeveloperContent);
            }

            let mut content_developer = String::new();

            if !message.tools.is_empty() {
                let tools = message
                    .tools
                    .iter()
                    .map(|tool| tool.function.clone())
                    .collect::<Vec<_>>();
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(&tools));
            }

            if let Some(response_format) = &message.response_format {
                content_developer.push_str("\n\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n");
                content_developer.push_str("As an example, for the schema ");
                content_developer.push_str("{\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n");
                content_developer.push_str("the object ");
                content_developer.push_str("{\"foo\": [\"bar\", \"baz\"]}");
                content_developer.push_str(" is a well-formatted instance of the schema.\n");
                content_developer.push_str("The object ");
                content_developer.push_str("{\"properties\": {\"foo\": [\"bar\", \"baz\"]}}");
                content_developer.push_str(" is not well-formatted.\n\n");
                content_developer.push_str("Here is the output schema:\n```\n");
                content_developer.push_str(&to_json(response_format));
                content_developer.push_str("\n```\n");
            }

            content_developer.push_str("\n\n# The user's message is:\n");
            content_developer.push_str(&message.content);

            prompt.push_str(&format!(
                "{user}{content}",
                user = USER_TOKEN,
                content = content_developer,
            ));

            if index as isize == last_user_idx && index + 1 == all_messages.len() {
                prompt.push_str(ASSISTANT_TOKEN);
                if thinking_mode == ThinkingMode::Thinking {
                    prompt.push_str(THINKING_START_TOKEN);
                } else {
                    prompt.push_str(THINKING_END_TOKEN);
                }
            }
        }
        "user" => {
            prompt.push_str(&format!(
                "{user}{content}",
                user = USER_TOKEN,
                content = message.content,
            ));

            if index as isize == last_user_idx && index + 1 == all_messages.len() {
                prompt.push_str(ASSISTANT_TOKEN);
                if thinking_mode == ThinkingMode::Thinking {
                    prompt.push_str(THINKING_START_TOKEN);
                } else {
                    prompt.push_str(THINKING_END_TOKEN);
                }
            }
        }
        _ => return Err(EncodeMessagesError::UnsupportedRole(role.to_string())),
    }

    Ok(prompt)
}

fn render_tools(tools: &[Value]) -> String {
    let tool_schemas = tools.iter().map(to_json).collect::<Vec<_>>().join("\n");
    format!(
        "## Tools\n\nTo call a tool, please use this format:\n\
<{dsml_token}function_calls>\n\
<{dsml_token}function=example_function_name>\n\
<{dsml_token}parameter=example_parameter_name type=example_parameter_type>\n\
example_value\n\
</{dsml_token}parameter>\n\
</{dsml_token}function>\n\
</{dsml_token}function_calls>\n\n\
**IMPORTANT!**\n\
- Function calls MUST follow the specified format.\n\
- Required parameters MUST be specified.\n\
- Put the entire function call reply in one line.\n\
- The values in `<{dsml_token}parameter=xxx>` should be XML escaped.\n\
- If no function call is needed, please directly provide a semantic long answer.\n\
- If there are no function calls available, answer all user requests conversationally.\n\
\n\
Here are the available tools:\n\n\
{tool_schemas}",
        dsml_token = DSML_TOKEN,
        tool_schemas = tool_schemas
    )
}

fn find_last_user_index(messages: &[ParsedMessage]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(idx, msg)| (msg.role == "user" || msg.role == "developer").then_some(idx))
}

fn drop_thinking_messages(messages: &mut Vec<ParsedMessage>) {
    let mut assistant_start_idx: Option<usize> = None;
    let mut index = 0usize;

    while index < messages.len() {
        let msg = &messages[index];
        let has_reasoning = msg
            .reasoning_content
            .as_deref()
            .is_some_and(|reasoning| !reasoning.is_empty());
        let has_content = !msg.content.is_empty();

        if msg.role == "assistant" && has_reasoning && !has_content {
            if assistant_start_idx.is_none() {
                assistant_start_idx = Some(index);
            }
        } else if msg.role == "assistant" && has_content {
            assistant_start_idx = None;
        } else if msg.role == "user" && assistant_start_idx.is_some() {
            let start = assistant_start_idx.expect("checked is_some");
            messages.drain(start..index);
            assistant_start_idx = None;
            index = start;
            continue;
        }

        if index + 1 == messages.len() {
            if let Some(start) = assistant_start_idx {
                messages.drain(start..);
            }
            break;
        }

        index += 1;
    }
}

fn encode_arguments_to_dsml(
    arguments: &str,
    function_name: &str,
) -> Result<String, EncodeMessagesError> {
    let parsed: Value = serde_json::from_str(arguments).map_err(|err| {
        EncodeMessagesError::InvalidToolArgumentsJson {
            function_name: function_name.to_string(),
            reason: err.to_string(),
        }
    })?;

    let object =
        parsed
            .as_object()
            .ok_or_else(|| EncodeMessagesError::InvalidToolArgumentsObject {
                function_name: function_name.to_string(),
            })?;

    let mut dsml_params = Vec::with_capacity(object.len());
    for (key, value) in object {
        let value_type = if value.is_string() { "string" } else { "json" };
        let value_str = value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| to_json(value));
        dsml_params.push(format!(
            "<parameter={key} type={value_type}>{value_str}</parameter>"
        ));
    }

    Ok(dsml_params.join("\n"))
}

fn to_json(value: &Value) -> String {
    serde_json::to_string(value).expect("serializing serde_json::Value should never fail")
}

fn short_hash(text: &str) -> String {
    let digest = md5_digest(text.as_bytes());
    let mut hex = String::with_capacity(8);
    for byte in digest.iter().take(4) {
        hex.push(hex_char((byte >> 4) & 0x0f));
        hex.push(hex_char(byte & 0x0f));
    }
    hex
}

fn hex_char(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => unreachable!("nibble must be in range 0..=15"),
    }
}

fn md5_digest(input: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let bit_len = (input.len() as u64) * 8;
    let mut msg = input.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let start = i * 4;
            *word = u32::from_le_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }

        let mut a = a0;
        let mut b = b0;
        let mut c = c0;
        let mut d = d0;

        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };

            let tmp = d;
            d = c;
            c = b;
            let sum = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(S[i]));
            a = tmp;
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut digest = [0u8; 16];
    digest[0..4].copy_from_slice(&a0.to_le_bytes());
    digest[4..8].copy_from_slice(&b0.to_le_bytes());
    digest[8..12].copy_from_slice(&c0.to_le_bytes());
    digest[12..16].copy_from_slice(&d0.to_le_bytes());
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_chat_template_chat_mode() {
        let messages = vec![Message::new("user", "hello")];
        let prompt = apply_chat_template(&messages, ThinkingMode::Chat, None, true, true)
            .expect("apply_chat_template should succeed");

        assert_eq!(
            prompt,
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>hello<\u{ff5c}Assistant\u{ff5c}></think>"
        );
    }

    #[test]
    fn test_apply_chat_template_thinking_mode() {
        let messages = vec![Message::new("user", "hello")];
        let prompt = apply_chat_template(&messages, ThinkingMode::Thinking, None, true, true)
            .expect("apply_chat_template should succeed");

        assert_eq!(
            prompt,
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>hello<\u{ff5c}Assistant\u{ff5c}><think>"
        );
    }

    #[test]
    fn test_drop_thinking_messages_before_user() {
        let messages = vec![
            Message {
                role: "assistant".to_string(),
                reasoning_content: Some("draft reasoning".to_string()),
                ..Message::default()
            },
            Message::new("user", "question"),
        ];

        let prompt = apply_chat_template(&messages, ThinkingMode::Thinking, None, true, true)
            .expect("apply_chat_template should succeed");

        assert!(!prompt.contains("draft reasoning"));
        assert!(prompt.contains("<\u{ff5c}User\u{ff5c}>question"));
    }

    #[test]
    fn test_tool_call_and_output_share_tool_call_id() {
        let messages = vec![
            Message {
                role: "assistant".to_string(),
                tool_calls: Some(vec![ToolCallInput {
                    function: FunctionCallInput {
                        name: Some("search".to_string()),
                        arguments: Some("{\"query\":\"hello\"}".to_string()),
                    },
                }]),
                ..Message::default()
            },
            Message::new("tool", "world"),
        ];

        let prompt = apply_chat_template(&messages, ThinkingMode::Chat, None, true, true)
            .expect("apply_chat_template should succeed");
        let tool_call_id = format!("search{}", short_hash("{\"query\":\"hello\"}"));

        assert_eq!(prompt.matches(&tool_call_id).count(), 2);
        assert!(prompt.contains("<function_calls>"));
        assert!(prompt.contains("<function_results>"));
    }

    #[test]
    fn test_md5_short_hash_matches_python() {
        assert_eq!(short_hash("{}"), "99914b93");
        assert_eq!(short_hash("{\"a\":1}"), "bb6cb5c6");
    }
}
