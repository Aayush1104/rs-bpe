use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
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
static CL100K_INIT: Once = Once::new();
static O200K_INIT: Once = Once::new();
static DEEPSEEK_INIT: Once = Once::new();

/// Python wrapper for ParallelOptions
#[pyclass]
#[derive(Clone)]
struct ParallelOptions {
    inner: ::bpe_openai::ParallelOptions,
}

#[pymethods]
impl ParallelOptions {
    #[new]
    fn new(
        min_batch_size: Option<usize>,
        chunk_size: Option<usize>,
        max_threads: Option<usize>,
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
    m.add_function(wrap_pyfunction!(is_cached_cl100k, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_o200k, m)?)?;
    m.add_function(wrap_pyfunction!(is_cached_deepseek, m)?)?;
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
fn get_num_threads() -> PyResult<usize> {
    Ok(rayon::current_num_threads())
}
