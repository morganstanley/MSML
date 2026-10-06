use std::{path::PathBuf, sync::Mutex};

use hiriluk::{
    ChopOptions, ChopOutput, ChopProfile as RustChopProfile, ChopResult, SourceError,
    StreamTokenizer, TokenizerError,
};
use numpy::IntoPyArray;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyAny,
};

/// Opt-in statistics from the most recently completed chop operation.
#[pyclass(module = "hiriluk._hiriluk", frozen, skip_from_py_object)]
#[derive(Clone)]
struct ChopProfile {
    #[pyo3(get)]
    total_tokens: usize,
    #[pyo3(get)]
    ttft_seconds: Option<f64>,
    #[pyo3(get)]
    elapsed_seconds: f64,
}

impl From<RustChopProfile> for ChopProfile {
    fn from(profile: RustChopProfile) -> Self {
        Self {
            total_tokens: profile.total_tokens,
            ttft_seconds: profile.time_to_first_token.map(|value| value.as_secs_f64()),
            elapsed_seconds: profile.elapsed.as_secs_f64(),
        }
    }
}

enum ChopInput {
    Text(String),
    File(PathBuf),
}

#[pyclass(module = "hiriluk._hiriluk", frozen)]
struct Chopper {
    name: &'static str,
    gigatoken: bool,
    profile: bool,
    tokenizer: StreamTokenizer,
    last_profile: Mutex<Option<ChopProfile>>,
}

impl Chopper {
    fn clear_profile(&self) -> PyResult<()> {
        *self
            .last_profile
            .lock()
            .map_err(|_| PyRuntimeError::new_err("chop profile lock was poisoned"))? = None;
        Ok(())
    }

    fn publish_profile(&self, profile: Option<RustChopProfile>) -> PyResult<()> {
        if let Some(profile) = profile {
            *self
                .last_profile
                .lock()
                .map_err(|_| PyRuntimeError::new_err("chop profile lock was poisoned"))? =
                Some(profile.into());
        }
        Ok(())
    }

    fn run_chop(
        &self,
        py: Python<'_>,
        input: ChopInput,
        gigatoken: bool,
        output: Option<&str>,
        dump: Option<PathBuf>,
    ) -> PyResult<ChopResult> {
        self.clear_profile()?;
        let options = ChopOptions::new(gigatoken, output, dump, self.profile).map_err(py_error)?;
        let result = py.detach(|| match input {
            ChopInput::Text(text) => self.tokenizer.tokenize_string(text, options),
            ChopInput::File(path) => self.tokenizer.tokenize_file(path, options),
        });
        result.map_err(py_error)
    }

    fn python_output(&self, py: Python<'_>, result: ChopResult) -> PyResult<Py<PyAny>> {
        self.publish_profile(result.profile)?;
        match result.output {
            ChopOutput::Array(tokens) => Ok(tokens.into_pyarray(py).into_any().unbind()),
            ChopOutput::Written { tokens } => Ok(tokens.into_pyobject(py)?.into_any().unbind()),
        }
    }
}

#[pymethods]
impl Chopper {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn gigatoken(&self) -> bool {
        self.gigatoken
    }

    #[getter]
    fn profile(&self) -> bool {
        self.profile
    }

    #[getter]
    fn last_profile(&self, py: Python<'_>) -> PyResult<Option<Py<ChopProfile>>> {
        let profile = self
            .last_profile
            .lock()
            .map_err(|_| PyRuntimeError::new_err("chop profile lock was poisoned"))?
            .clone();
        profile.map(|profile| Py::new(py, profile)).transpose()
    }

    #[pyo3(signature = (text, *, gigatoken=None, output=None, dump=None))]
    fn chop(
        &self,
        py: Python<'_>,
        text: &str,
        gigatoken: Option<bool>,
        output: Option<String>,
        dump: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        let result = self.run_chop(
            py,
            ChopInput::Text(text.to_owned()),
            gigatoken.unwrap_or(self.gigatoken),
            output.as_deref(),
            dump,
        )?;
        self.python_output(py, result)
    }

    #[pyo3(signature = (file_path, *, gigatoken=None, output=None, dump=None))]
    fn chop_file(
        &self,
        py: Python<'_>,
        file_path: PathBuf,
        gigatoken: Option<bool>,
        output: Option<String>,
        dump: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        let result = self.run_chop(
            py,
            ChopInput::File(file_path),
            gigatoken.unwrap_or(self.gigatoken),
            output.as_deref(),
            dump,
        )?;
        self.python_output(py, result)
    }

    #[pyo3(signature = (iterator, *, gigatoken=None, output=None, dump=None))]
    fn chop_stream(
        &self,
        iterator: &Bound<'_, PyAny>,
        gigatoken: Option<bool>,
        output: Option<String>,
        dump: Option<PathBuf>,
    ) -> PyResult<()> {
        let _ = (iterator, gigatoken, output, dump);
        Err(PyNotImplementedError::new_err(
            "chop_stream() is not implemented yet",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Chopper(name={:?}, gigatoken={}, profile={})",
            self.name, self.gigatoken, self.profile
        )
    }
}

fn py_error(error: TokenizerError) -> PyErr {
    match error {
        TokenizerError::Source(SourceError::Llama31pDisabled) => PyNotImplementedError::new_err(
            "llama31p is disabled because its merge dictionary requires properization",
        ),
        TokenizerError::Source(SourceError::UnknownModel(name)) => PyValueError::new_err(format!(
            "unknown encoding {name:?}; expected one of {}",
            StreamTokenizer::encoding_names().join(", ")
        )),
        TokenizerError::InvalidOutput(message) => PyValueError::new_err(message),
        TokenizerError::IteratorNotImplemented => PyNotImplementedError::new_err(
            "output='iterator' is not implemented yet; use output=None, 'array', 'json', or 'compact'",
        ),
        TokenizerError::InvalidPath { role } => {
            PyValueError::new_err(format!("{role} path is not valid UTF-8"))
        }
        TokenizerError::Io(error) => error.into(),
        error => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyfunction]
#[pyo3(signature = (encoding_name, *, gigatoken=false, profile=false))]
fn get_chopper(
    py: Python<'_>,
    encoding_name: &str,
    gigatoken: bool,
    profile: bool,
) -> PyResult<Chopper> {
    let tokenizer = py
        .detach(|| {
            let tokenizer = StreamTokenizer::new_for_mode(encoding_name, gigatoken)?;
            tokenizer.prepare(gigatoken)?;
            Ok::<_, TokenizerError>(tokenizer)
        })
        .map_err(py_error)?;
    let name = tokenizer.name();
    Ok(Chopper {
        name,
        gigatoken,
        profile,
        tokenizer,
        last_profile: Mutex::new(None),
    })
}

#[pyfunction]
fn list_encoding_names() -> Vec<&'static str> {
    StreamTokenizer::encoding_names().to_vec()
}

#[pymodule]
fn _hiriluk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Chopper>()?;
    module.add_class::<ChopProfile>()?;
    module.add_function(wrap_pyfunction!(get_chopper, module)?)?;
    module.add_function(wrap_pyfunction!(list_encoding_names, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
