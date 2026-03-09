use once_cell::sync::Lazy;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::Once;

// Global static instances of tokenizers
static CL100K_TOKENIZER: Lazy<Mutex<Option<&'static ::bpe_openai::Tokenizer>>> =
    Lazy::new(|| Mutex::new(None));
static O200K_TOKENIZER: Lazy<Mutex<Option<&'static ::bpe_openai::Tokenizer>>> =
    Lazy::new(|| Mutex::new(None));
static DEEPSEEK_TOKENIZER: Lazy<Mutex<Option<&'static ::bpe_openai::Tokenizer>>> =
    Lazy::new(|| Mutex::new(None));
static DEEPSEEK_32_TOKENIZER: Lazy<Mutex<Option<&'static ::bpe_openai::Tokenizer>>> =
    Lazy::new(|| Mutex::new(None));
static KIMI_K2_TOKENIZER: Lazy<Mutex<Option<&'static ::bpe_openai::Tokenizer>>> =
    Lazy::new(|| Mutex::new(None));
static CL100K_INIT: Once = Once::new();
static O200K_INIT: Once = Once::new();
static DEEPSEEK_INIT: Once = Once::new();
static DEEPSEEK_32_INIT: Once = Once::new();
static KIMI_K2_INIT: Once = Once::new();

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(JsonValue::Bool(v));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(JsonValue::Number(v.into()));
    }
    if let Ok(v) = value.extract::<u64>() {
        return Ok(JsonValue::Number(v.into()));
    }
    if let Ok(v) = value.extract::<f64>() {
        let number = JsonNumber::from_f64(v)
            .ok_or_else(|| PyValueError::new_err("invalid floating-point value in JSON payload"))?;
        return Ok(JsonValue::Number(number));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(JsonValue::String(v));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(out));
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(out));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut out = JsonMap::new();
        for (key, item) in dict.iter() {
            let key = key.extract::<String>().map_err(|_| {
                PyTypeError::new_err("JSON object keys must be strings for apply_chat_template")
            })?;
            out.insert(key, py_to_json(&item)?);
        }
        return Ok(JsonValue::Object(out));
    }

    Err(PyTypeError::new_err(
        "unsupported Python value type for apply_chat_template",
    ))
}

fn dict_get_optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    value
        .extract::<String>()
        .map(Some)
        .map_err(|_| PyTypeError::new_err(format!("'{key}' must be a string or None")))
}

fn parse_function_call(
    value: &Bound<'_, PyAny>,
) -> PyResult<::bpe_openai::deepseek_v32::FunctionCallInput> {
    let function_dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("'function' must be a dict"))?;

    Ok(::bpe_openai::deepseek_v32::FunctionCallInput {
        name: dict_get_optional_string(function_dict, "name")?,
        arguments: dict_get_optional_string(function_dict, "arguments")?,
    })
}

fn parse_tool_calls(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<::bpe_openai::deepseek_v32::ToolCallInput>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("'tool_calls' must be a list"))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each item in 'tool_calls' must be a dict"))?;
        let function = match dict.get_item("function")? {
            Some(function_value) => parse_function_call(&function_value)?,
            None => ::bpe_openai::deepseek_v32::FunctionCallInput::default(),
        };
        out.push(::bpe_openai::deepseek_v32::ToolCallInput { function });
    }

    Ok(out)
}

fn parse_tools(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<::bpe_openai::deepseek_v32::ToolDefinition>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("'tools' must be a list"))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each item in 'tools' must be a dict"))?;
        let function = match dict.get_item("function")? {
            Some(function_value) => py_to_json(&function_value)?,
            None => JsonValue::Null,
        };
        out.push(::bpe_openai::deepseek_v32::ToolDefinition { function });
    }
    Ok(out)
}

fn parse_messages(value: &Bound<'_, PyAny>) -> PyResult<Vec<::bpe_openai::deepseek_v32::Message>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("messages/context must be a list of dict"))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each message must be a dict"))?;

        let role = dict_get_optional_string(dict, "role")?.unwrap_or_default();
        let content = dict_get_optional_string(dict, "content")?;
        let reasoning_content = dict_get_optional_string(dict, "reasoning_content")?;

        let tools = match dict.get_item("tools")? {
            Some(v) if !v.is_none() => Some(parse_tools(&v)?),
            _ => None,
        };
        let response_format = match dict.get_item("response_format")? {
            Some(v) if !v.is_none() => Some(py_to_json(&v)?),
            _ => None,
        };
        let tool_calls = match dict.get_item("tool_calls")? {
            Some(v) if !v.is_none() => Some(parse_tool_calls(&v)?),
            _ => None,
        };

        out.push(::bpe_openai::deepseek_v32::Message {
            role,
            content,
            tools,
            response_format,
            tool_calls,
            reasoning_content,
        });
    }

    Ok(out)
}

fn parse_kimi_messages(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<::bpe_openai::kimi_k2::Message>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("messages must be a list of dict"))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each message must be a dict"))?;

        let role = dict_get_optional_string(dict, "role")?.unwrap_or_default();
        let content = dict_get_optional_string(dict, "content")?;
        let name = dict_get_optional_string(dict, "name")?;
        let tool_call_id = dict_get_optional_string(dict, "tool_call_id")?;

        let tool_calls = match dict.get_item("tool_calls")? {
            Some(v) if !v.is_none() => Some(parse_kimi_tool_calls(&v)?),
            _ => None,
        };

        out.push(::bpe_openai::kimi_k2::Message {
            role,
            content,
            name,
            tool_calls,
            tool_call_id,
        });
    }

    Ok(out)
}

fn parse_kimi_tool_calls(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<::bpe_openai::kimi_k2::ToolCallInput>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("'tool_calls' must be a list"))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each item in 'tool_calls' must be a dict"))?;

        let id = dict_get_optional_string(dict, "id")?;
        let function = match dict.get_item("function")? {
            Some(function_value) => {
                let func_dict = function_value
                    .downcast::<PyDict>()
                    .map_err(|_| PyTypeError::new_err("'function' must be a dict"))?;
                let name = dict_get_optional_string(func_dict, "name")?;
                let arguments = match func_dict.get_item("arguments")? {
                    Some(v) if !v.is_none() => Some(py_to_json(&v)?),
                    _ => None,
                };
                ::bpe_openai::kimi_k2::FunctionCallInput { name, arguments }
            }
            None => ::bpe_openai::kimi_k2::FunctionCallInput::default(),
        };
        out.push(::bpe_openai::kimi_k2::ToolCallInput { id, function });
    }

    Ok(out)
}

fn parse_kimi_tools(value: &Bound<'_, PyAny>) -> PyResult<Vec<serde_json::Value>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("'tools' must be a list"))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        out.push(py_to_json(&item)?);
    }
    Ok(out)
}

/// Python wrapper for ParallelOptions
#[pyclass]
#[derive(Clone)]
struct ParallelOptions {
    inner: ::bpe_openai::ParallelOptions,
}

#[pymethods]
impl ParallelOptions {
    #[new]
    #[pyo3(signature = (min_batch_size = None, chunk_size = None, max_threads = None, use_thread_pool = None))]
    fn new(
        min_batch_size: Option<usize>,
        chunk_size: Option<usize>,
        max_threads: Option<usize>,
        use_thread_pool: Option<bool>,
    ) -> Self {
        let mut options = ::bpe_openai::ParallelOptions::default();

        if let Some(min_batch_size) = min_batch_size {
            options.min_batch_size = min_batch_size;
        }

        if let Some(chunk_size) = chunk_size {
            options.chunk_size = chunk_size;
        }

        if let Some(max_threads) = max_threads {
            options.max_threads = max_threads;
        }

        if let Some(use_thread_pool) = use_thread_pool {
            options.use_thread_pool = use_thread_pool;
        }

        Self { inner: options }
    }

    #[getter]
    fn min_batch_size(&self) -> usize {
        self.inner.min_batch_size
    }

    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }

    #[getter]
    fn max_threads(&self) -> usize {
        self.inner.max_threads
    }

    #[getter]
    fn use_thread_pool(&self) -> bool {
        self.inner.use_thread_pool
    }
}

#[pyclass]
struct Tokenizer(&'static ::bpe_openai::Tokenizer);

#[pymethods]
impl Tokenizer {
    fn count(&self, input: &str) -> usize {
        self.0.count(input)
    }

    fn count_till_limit(&self, input: Cow<str>, limit: usize) -> Option<usize> {
        self.0.count_till_limit(input.as_ref(), limit)
    }

    #[pyo3(signature = (input, allowed_special = None))]
    fn encode(&self, input: Cow<str>, allowed_special: Option<Vec<String>>) -> Vec<u32> {
        let allowed_special =
            allowed_special.map(|items| items.into_iter().collect::<HashSet<String>>());
        let allowed_special_refs = allowed_special.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.as_str())
                .collect::<HashSet<&str>>()
        });
        self.0.encode(input.as_ref(), allowed_special_refs.as_ref())
    }

    #[pyo3(signature = (texts, allowed_special = None))]
    fn encode_batch(
        &self,
        texts: Vec<String>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<(Vec<Vec<u32>>, usize, f64)> {
        let str_texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let allowed_special =
            allowed_special.map(|items| items.into_iter().collect::<HashSet<String>>());
        let allowed_special_refs = allowed_special.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.as_str())
                .collect::<HashSet<&str>>()
        });
        let result = self
            .0
            .encode_batch(&str_texts, allowed_special_refs.as_ref());
        Ok((result.tokens, result.total_tokens, result.time_taken))
    }

    #[pyo3(signature = (texts, options = None, allowed_special = None))]
    fn encode_batch_parallel(
        &self,
        texts: Vec<String>,
        options: Option<ParallelOptions>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<(Vec<Vec<u32>>, usize, f64, usize)> {
        let str_texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let rust_options = options.map(|opts| opts.inner);
        let allowed_special =
            allowed_special.map(|items| items.into_iter().collect::<HashSet<String>>());
        let allowed_special_refs = allowed_special.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.as_str())
                .collect::<HashSet<&str>>()
        });
        let tokens =
            self.0
                .encode_batch_parallel(&str_texts, rust_options, allowed_special_refs.as_ref());
        let total_tokens = tokens.iter().map(|t| t.len()).sum();

        // Backward compatibility values
        let time_taken = 0.0;
        let threads_used = num_cpus::get();

        Ok((tokens, total_tokens, time_taken, threads_used))
    }

    #[pyo3(signature = (input, chunk_size = 64, options = None, allowed_special = None))]
    fn encode_split_chunks_parallel(
        &self,
        input: Cow<str>,
        chunk_size: usize,
        options: Option<ParallelOptions>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<(Vec<Vec<u32>>, usize, f64, usize)> {
        let rust_options = options.map(|opts| opts.inner);
        let allowed_special =
            allowed_special.map(|items| items.into_iter().collect::<HashSet<String>>());
        let allowed_special_refs = allowed_special.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.as_str())
                .collect::<HashSet<&str>>()
        });
        let tokens = self.0.encode_split_chunks_parallel(
            input.as_ref(),
            chunk_size,
            rust_options,
            allowed_special_refs.as_ref(),
        );
        let total_tokens = tokens.iter().map(|t| t.len()).sum();

        // Backward compatibility values
        let time_taken = 0.0;
        let threads_used = num_cpus::get();

        Ok((tokens, total_tokens, time_taken, threads_used))
    }

    fn decode(&self, tokens: Vec<u32>) -> Option<String> {
        self.0.decode(&tokens)
    }

    fn decode_batch(&self, batch_tokens: Vec<Vec<u32>>) -> Vec<Option<String>> {
        self.0.decode_batch(&batch_tokens)
    }

    fn decode_batch_parallel(
        &self,
        batch_tokens: Vec<Vec<u32>>,
        options: Option<ParallelOptions>,
    ) -> Vec<Option<String>> {
        let rust_options = options.map(|opts| opts.inner);
        self.0.decode_batch_parallel(&batch_tokens, rust_options)
    }

    #[pyo3(signature = (input, allowed_special = None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        input: Cow<str>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let allowed_special =
            allowed_special.map(|items| items.into_iter().collect::<HashSet<String>>());
        let allowed_special_refs = allowed_special.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.as_str())
                .collect::<HashSet<&str>>()
        });
        let pieces = self
            .0
            .split_with_special(input.as_ref(), allowed_special_refs.as_ref());
        PyList::new(py, pieces)
    }

    #[pyo3(signature = (input, chunk_size = 64, allowed_special = None))]
    fn split_chunks<'py>(
        &self,
        py: Python<'py>,
        input: Cow<str>,
        chunk_size: usize,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let allowed_special =
            allowed_special.map(|items| items.into_iter().collect::<HashSet<String>>());
        let allowed_special_refs = allowed_special.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.as_str())
                .collect::<HashSet<&str>>()
        });
        let chunks = self
            .0
            .split_chunks(input.as_ref(), chunk_size, allowed_special_refs.as_ref());
        PyList::new(py, chunks)
    }

    #[pyo3(signature = (messages, thinking_mode = "chat", context = None, drop_thinking = true, add_default_bos_token = true, tools = None, add_generation_prompt = true))]
    fn apply_chat_template(
        &self,
        messages: &Bound<'_, PyAny>,
        thinking_mode: &str,
        context: Option<&Bound<'_, PyAny>>,
        drop_thinking: bool,
        add_default_bos_token: bool,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<String> {
        if std::ptr::eq(self.0, ::bpe_openai::kimi_k2()) {
            let rust_messages = parse_kimi_messages(messages)?;
            let rust_tools = match tools {
                Some(t) if !t.is_none() => Some(parse_kimi_tools(t)?),
                _ => None,
            };
            return ::bpe_openai::kimi_k2::apply_chat_template(
                &rust_messages,
                rust_tools.as_deref(),
                add_generation_prompt,
            )
            .map_err(|err| PyValueError::new_err(err.to_string()));
        }

        if !std::ptr::eq(self.0, ::bpe_openai::deepseek_32()) {
            return Err(PyNotImplementedError::new_err(
                "Tokenizer.apply_chat_template is only supported for deepseek_32() and kimi_k2()",
            ));
        }

        let thinking_mode = match thinking_mode.to_ascii_lowercase().as_str() {
            "chat" => ::bpe_openai::deepseek_v32::ThinkingMode::Chat,
            "thinking" => ::bpe_openai::deepseek_v32::ThinkingMode::Thinking,
            _ => {
                return Err(PyValueError::new_err(
                    "thinking_mode must be either 'chat' or 'thinking'",
                ))
            }
        };

        let rust_messages = parse_messages(messages)?;
        let rust_context = match context {
            Some(ctx) if !ctx.is_none() => Some(parse_messages(ctx)?),
            _ => None,
        };

        ::bpe_openai::deepseek_v32::apply_chat_template(
            &rust_messages,
            thinking_mode,
            rust_context.as_deref(),
            drop_thinking,
            add_default_bos_token,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn special_tokens(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let special_tokens = match self.0.special_tokens() {
            Some(tokens) => tokens,
            None => return Ok(None),
        };

        let dict = PyDict::new(py);
        for (token, id) in special_tokens.iter() {
            dict.set_item(token, *id)?;
        }
        Ok(Some(dict.into()))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.0.bpe.num_tokens()
    }
}

/// BPE tokenizer interface
#[pymodule]
fn bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_class::<ParallelOptions>()?;
    m.add_function(wrap_pyfunction!(cl100k_base, m)?)?;
    m.add_function(wrap_pyfunction!(o200k_base, m)?)?;
    m.add_function(wrap_pyfunction!(deepseek_base, m)?)?;
    m.add_function(wrap_pyfunction!(deepseek_32, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_cl100k, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_o200k, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_deepseek, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_deepseek_32, m)?)?;
    m.add_function(wrap_pyfunction!(kimi_k2, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_kimi_k2, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    Ok(())
}

#[pyfunction]
fn cl100k_base() -> PyResult<Tokenizer> {
    CL100K_INIT.call_once(|| {
        let mut tokenizer = CL100K_TOKENIZER.lock().unwrap();
        *tokenizer = Some(::bpe_openai::cl100k_base());
    });

    let tokenizer_opt = CL100K_TOKENIZER.lock().unwrap();
    Ok(Tokenizer(tokenizer_opt.as_ref().unwrap()))
}

#[pyfunction]
fn o200k_base() -> PyResult<Tokenizer> {
    O200K_INIT.call_once(|| {
        let mut tokenizer = O200K_TOKENIZER.lock().unwrap();
        *tokenizer = Some(::bpe_openai::o200k_base());
    });

    let tokenizer_opt = O200K_TOKENIZER.lock().unwrap();
    Ok(Tokenizer(tokenizer_opt.as_ref().unwrap()))
}

#[pyfunction]
fn deepseek_base() -> PyResult<Tokenizer> {
    DEEPSEEK_INIT.call_once(|| {
        let mut tokenizer = DEEPSEEK_TOKENIZER.lock().unwrap();
        *tokenizer = Some(::bpe_openai::deepseek_base());
    });

    let tokenizer_opt = DEEPSEEK_TOKENIZER.lock().unwrap();
    Ok(Tokenizer(tokenizer_opt.as_ref().unwrap()))
}

#[pyfunction]
fn deepseek_32() -> PyResult<Tokenizer> {
    DEEPSEEK_32_INIT.call_once(|| {
        let mut tokenizer = DEEPSEEK_32_TOKENIZER.lock().unwrap();
        *tokenizer = Some(::bpe_openai::deepseek_32());
    });

    let tokenizer_opt = DEEPSEEK_32_TOKENIZER.lock().unwrap();
    Ok(Tokenizer(tokenizer_opt.as_ref().unwrap()))
}

#[pyfunction]
fn is_cached_cl100k() -> PyResult<bool> {
    let tokenizer = CL100K_TOKENIZER.lock().unwrap();
    Ok(tokenizer.is_some())
}

#[pyfunction]
fn is_cached_o200k() -> PyResult<bool> {
    let tokenizer = O200K_TOKENIZER.lock().unwrap();
    Ok(tokenizer.is_some())
}

#[pyfunction]
fn is_cached_deepseek() -> PyResult<bool> {
    let tokenizer = DEEPSEEK_TOKENIZER.lock().unwrap();
    Ok(tokenizer.is_some())
}

#[pyfunction]
fn is_cached_deepseek_32() -> PyResult<bool> {
    let tokenizer = DEEPSEEK_32_TOKENIZER.lock().unwrap();
    Ok(tokenizer.is_some())
}

#[pyfunction]
fn kimi_k2() -> PyResult<Tokenizer> {
    KIMI_K2_INIT.call_once(|| {
        let mut tokenizer = KIMI_K2_TOKENIZER.lock().unwrap();
        *tokenizer = Some(::bpe_openai::kimi_k2());
    });

    let tokenizer_opt = KIMI_K2_TOKENIZER.lock().unwrap();
    Ok(Tokenizer(tokenizer_opt.as_ref().unwrap()))
}

#[pyfunction]
fn is_cached_kimi_k2() -> PyResult<bool> {
    let tokenizer = KIMI_K2_TOKENIZER.lock().unwrap();
    Ok(tokenizer.is_some())
}

#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    Ok(rayon::current_num_threads())
}
