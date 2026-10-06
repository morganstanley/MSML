//! High-level named tokenizer API.
//!
//! This module owns model resolution, process-wide immutable model caching,
//! DFA/SIMD engine construction, output selection, and recovery after failed
//! operations. Language bindings should only translate their native values to
//! these types and translate the result back.

use std::{
    collections::HashMap,
    error::Error,
    io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::Duration,
};

use thiserror::Error;

use crate::{
    Cache, StreamEngine, StreamStats,
    benchmark_utils::{
        DEFAULT_MODELS, SourceError, Target, canonical_model_name, load_target_for_mode,
    },
};

static TARGETS: LazyLock<Mutex<HashMap<&'static str, Arc<Target>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Error returned by the high-level tokenizer API.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// The requested tokenizer name is unknown or intentionally disabled.
    #[error(transparent)]
    Source(#[from] SourceError),
    /// Loading or compiling a registered tokenizer failed.
    #[error("could not load tokenizer {name:?}: {message}")]
    ModelLoad { name: String, message: String },
    /// The selected output mode and destination are incompatible.
    #[error("{0}")]
    InvalidOutput(String),
    /// Explicit iterator output is reserved but not implemented.
    #[error(
        "output='iterator' is not implemented yet; use output=None, 'array', 'json', or 'compact'"
    )]
    IteratorNotImplemented,
    /// A path cannot be passed to the underlying UTF-8 path API.
    #[error("{role} path is not valid UTF-8")]
    InvalidPath { role: &'static str },
    /// File access or output failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The selected tokenizer engine could not be initialized.
    #[error("could not initialize tokenizer: {0}")]
    Initialization(String),
    /// Tokenization failed after initialization.
    #[error("tokenization failed: {0}")]
    Tokenization(String),
    /// A process-global tokenizer lock was poisoned by a prior panic.
    #[error("tokenizer state lock was poisoned")]
    State,
}

/// Where a chop operation sends its token IDs.
pub enum OutputMode {
    /// Return one contiguous packed `Vec<u32>`.
    Array,
    /// Stream a human-readable JSON integer array to this path.
    Json(PathBuf),
    /// Stream headerless little-endian `u32` IDs to this path.
    Compact(PathBuf),
}

impl OutputMode {
    fn destination(&self) -> Option<&Path> {
        match self {
            Self::Json(path) | Self::Compact(path) => Some(path),
            Self::Array => None,
        }
    }
}

/// Per-call behavior for [`StreamTokenizer::tokenize_string`] and
/// [`StreamTokenizer::tokenize_file`].
pub struct ChopOptions {
    gigatoken: bool,
    profile: bool,
    output: OutputMode,
}

impl ChopOptions {
    /// Validate the language-neutral output arguments used by public bindings.
    pub fn new(
        gigatoken: bool,
        output: Option<&str>,
        dump: Option<PathBuf>,
        profile: bool,
    ) -> Result<Self, TokenizerError> {
        let output = match (output, dump) {
            (None, None) | (Some("array"), None) => OutputMode::Array,
            (None, Some(_)) => {
                return Err(TokenizerError::InvalidOutput(
                    "dump requires output='json' or output='compact'".to_owned(),
                ));
            }
            (Some("array"), Some(_)) => {
                return Err(TokenizerError::InvalidOutput(
                    "output='array' cannot be combined with dump".to_owned(),
                ));
            }
            (Some("iterator"), None) => {
                return Err(TokenizerError::IteratorNotImplemented);
            }
            (Some("iterator"), Some(_)) => {
                return Err(TokenizerError::InvalidOutput(
                    "output='iterator' cannot be combined with dump".to_owned(),
                ));
            }
            (Some("json"), Some(path)) => OutputMode::Json(path),
            (Some("compact"), Some(path)) => OutputMode::Compact(path),
            (Some("json" | "compact"), None) => {
                return Err(TokenizerError::InvalidOutput(
                    "output='json' and output='compact' require dump".to_owned(),
                ));
            }
            (Some(other), _) => {
                return Err(TokenizerError::InvalidOutput(format!(
                    "unknown output {other:?}; expected None, 'array', 'iterator', 'json', or 'compact'"
                )));
            }
        };
        Ok(Self {
            gigatoken,
            profile,
            output,
        })
    }

    /// Construct options directly from a typed output mode.
    pub fn with_output(gigatoken: bool, output: OutputMode, profile: bool) -> Self {
        Self {
            gigatoken,
            profile,
            output,
        }
    }
}

/// Token output returned by a completed chop operation.
pub enum ChopOutput {
    /// One contiguous packed token buffer.
    Array(Vec<u32>),
    /// Tokens were written directly by Rust.
    Written { tokens: usize },
}

/// Opt-in statistics for one chop operation.
pub struct ChopProfile {
    /// Number of explicit token IDs produced by the operation.
    pub total_tokens: usize,
    /// Time from entering the Rust chop operation until its first token ID was
    /// emitted. Empty inputs have no first token and therefore return `None`.
    pub time_to_first_token: Option<Duration>,
    /// Total wall time spent in the Rust chop operation.
    pub elapsed: Duration,
}

impl From<StreamStats> for ChopProfile {
    fn from(stats: StreamStats) -> Self {
        Self {
            total_tokens: stats.total_tokens,
            time_to_first_token: stats.time_to_first_token,
            elapsed: stats.total,
        }
    }
}

/// The output and optional profile for one chop operation.
pub struct ChopResult {
    pub output: ChopOutput,
    /// Present only when profiling was requested.
    pub profile: Option<ChopProfile>,
}

#[derive(Default)]
struct EngineSlots {
    dfa: Option<StreamEngine>,
    gigatoken: Option<StreamEngine>,
}

impl EngineSlots {
    fn slot_mut(&mut self, gigatoken: bool) -> &mut Option<StreamEngine> {
        if gigatoken {
            &mut self.gigatoken
        } else {
            &mut self.dfa
        }
    }

    fn get_or_create(
        &mut self,
        target: &Target,
        gigatoken: bool,
    ) -> Result<&mut StreamEngine, TokenizerError> {
        let slot = self.slot_mut(gigatoken);
        if slot.is_none() {
            *slot = Some(build_engine(target, gigatoken)?);
        }
        Ok(slot.as_mut().expect("tokenizer slot was initialized"))
    }

    fn invalidate(&mut self, gigatoken: bool) {
        *self.slot_mut(gigatoken) = None;
    }
}

/// A reusable tokenizer selected by one registered encoding/model name.
///
/// Model lookup, source-cache loading, dictionary compilation, and all
/// DFA/SIMD/cache construction are owned by the Rust library.
pub struct StreamTokenizer {
    name: &'static str,
    target: Arc<Target>,
    engines: Mutex<EngineSlots>,
}

impl StreamTokenizer {
    #[inline]
    fn effective_gigatoken(&self, requested: bool) -> bool {
        requested && self.target.fast_path_allowed
    }

    /// Load a registered encoding/model by name.
    ///
    /// Aliases such as `r50k_base` are canonicalized. Immutable compiled model
    /// data is shared process-wide; mutable streaming/cache state belongs to
    /// this tokenizer.
    pub fn new(name: &str) -> Result<Self, TokenizerError> {
        Self::new_for_mode(name, false)
    }

    /// Load a registered tokenizer and optimize its first engine preparation
    /// for the requested execution mode.
    ///
    /// This does not restrict later per-call mode changes. A DFA-first target
    /// reconstructs the fast pair table once, on demand, from its compact
    /// runtime arena if SIMD is requested later.
    pub fn new_for_mode(name: &str, gigatoken: bool) -> Result<Self, TokenizerError> {
        let (name, target) = cached_target(name, gigatoken)?;
        Ok(Self {
            name,
            target,
            engines: Mutex::new(EngineSlots::default()),
        })
    }

    /// Names accepted as primary public tokenizer names.
    pub fn encoding_names() -> &'static [&'static str] {
        DEFAULT_MODELS
    }

    /// The canonical registered name selected by the constructor.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Initialize one execution mode without processing input.
    ///
    /// Bindings can call this in their constructor so initialization remains
    /// outside a later encoding timer.
    pub fn prepare(&self, gigatoken: bool) -> Result<(), TokenizerError> {
        let gigatoken = self.effective_gigatoken(gigatoken);
        self.engines
            .lock()
            .map_err(|_| TokenizerError::State)?
            .get_or_create(self.target.as_ref(), gigatoken)
            .map(|_| ())
    }

    /// Tokenize one UTF-8 string into an array or file.
    pub fn tokenize_string(
        &self,
        text: String,
        options: ChopOptions,
    ) -> Result<ChopResult, TokenizerError> {
        let ChopOptions {
            gigatoken: requested_gigatoken,
            profile,
            output,
        } = options;
        let gigatoken = self.effective_gigatoken(requested_gigatoken);

        let result = {
            let mut engines = self.engines.lock().map_err(|_| TokenizerError::State)?;
            let engine = engines.get_or_create(self.target.as_ref(), gigatoken)?;
            run_tokenize_string(engine, &text, output, profile)
        };
        if result.is_err() {
            // I/O or UTF-8 failures can leave a stateful stream mid-document.
            self.engines
                .lock()
                .map_err(|_| TokenizerError::State)?
                .invalidate(gigatoken);
        }
        result
    }

    /// Tokenize one UTF-8 file.
    pub fn tokenize_file(
        &self,
        input: PathBuf,
        options: ChopOptions,
    ) -> Result<ChopResult, TokenizerError> {
        if let Some(output) = options.output.destination() {
            reject_same_input_output(&input, output)?;
        }

        let ChopOptions {
            gigatoken: requested_gigatoken,
            profile,
            output,
        } = options;
        let gigatoken = self.effective_gigatoken(requested_gigatoken);

        let input = path_string(&input, "input")?;
        let result = {
            let mut engines = self.engines.lock().map_err(|_| TokenizerError::State)?;
            let engine = engines.get_or_create(self.target.as_ref(), gigatoken)?;
            run_tokenize_file(engine, input, output, profile)
        };
        if result.is_err() {
            self.engines
                .lock()
                .map_err(|_| TokenizerError::State)?
                .invalidate(gigatoken);
        }
        result
    }
}

fn cached_target(
    name: &str,
    retain_fast_source: bool,
) -> Result<(&'static str, Arc<Target>), TokenizerError> {
    let name = canonical_model_name(name)?;
    if let Some(target) = TARGETS.lock().map_err(|_| TokenizerError::State)?.get(name) {
        return Ok((name, Arc::clone(target)));
    }

    // Loading may read disk or network caches, so never hold the registry lock
    // across it. A racing duplicate is dropped after the winner is installed.
    let loaded = Arc::new(
        load_target_for_mode(name, retain_fast_source).map_err(|error| {
            TokenizerError::ModelLoad {
                name: name.to_owned(),
                message: error.to_string(),
            }
        })?,
    );
    let mut targets = TARGETS.lock().map_err(|_| TokenizerError::State)?;
    let target = Arc::clone(targets.entry(name).or_insert(loaded));
    Ok((name, target))
}

fn build_engine(target: &Target, gigatoken: bool) -> Result<StreamEngine, TokenizerError> {
    let gigatoken = gigatoken && target.fast_path_allowed;
    let result = if gigatoken {
        StreamEngine::with_segmenter_config_runtime(
            target.enc,
            target.segmenter.clone(),
            target.rank_to_id.clone(),
            target.normalizer,
            target.presplit,
            target.ignore_merges,
            Arc::clone(&target.vocab),
            Arc::clone(&target.tokenizer),
            Cache::Full,
        )
        .map(|mut engine| {
            engine.install_fast_bpe(target.fast_bpe_tables());
            engine
        })
    } else {
        target.discard_fast_bpe_source();
        StreamEngine::with_segmenter_config_runtime_dfa(
            target.enc,
            target.segmenter.clone(),
            target.rank_to_id.clone(),
            target.normalizer,
            target.presplit,
            target.ignore_merges,
            Arc::clone(&target.vocab),
            Arc::clone(&target.tokenizer),
            Cache::Lookup,
        )
    };
    result.map_err(|error| TokenizerError::Initialization(error.to_string()))
}

fn run_tokenize_string(
    engine: &mut StreamEngine,
    text: &str,
    output: OutputMode,
    profile: bool,
) -> Result<ChopResult, TokenizerError> {
    if profile {
        let (output, stats) = match output {
            OutputMode::Array => {
                let (tokens, stats) = engine
                    .tokenize_string_to_vec_profiled(text)
                    .map_err(operation_error)?;
                (ChopOutput::Array(tokens), stats)
            }
            OutputMode::Json(path) => {
                let path = path_string(&path, "dump")?;
                let stats = engine
                    .tokenize_string_json_profiled(text, path)
                    .map_err(operation_error)?;
                (
                    ChopOutput::Written {
                        tokens: stats.total_tokens,
                    },
                    stats,
                )
            }
            OutputMode::Compact(path) => {
                let path = path_string(&path, "dump")?;
                let stats = engine
                    .tokenize_string_u32_le_profiled(text, path)
                    .map_err(operation_error)?;
                (
                    ChopOutput::Written {
                        tokens: stats.total_tokens,
                    },
                    stats,
                )
            }
        };
        return Ok(ChopResult {
            output,
            profile: Some(stats.into()),
        });
    }

    let output = match output {
        OutputMode::Array => ChopOutput::Array(
            engine
                .tokenize_string_to_vec_unprofiled(text)
                .map_err(operation_error)?,
        ),
        OutputMode::Json(path) => ChopOutput::Written {
            tokens: engine
                .tokenize_string_json_unprofiled(text, path_string(&path, "dump")?)
                .map_err(operation_error)?,
        },
        OutputMode::Compact(path) => ChopOutput::Written {
            tokens: engine
                .tokenize_string_u32_le_unprofiled(text, path_string(&path, "dump")?)
                .map_err(operation_error)?,
        },
    };
    Ok(ChopResult {
        output,
        profile: None,
    })
}

fn run_tokenize_file(
    engine: &mut StreamEngine,
    input: &str,
    output: OutputMode,
    profile: bool,
) -> Result<ChopResult, TokenizerError> {
    if profile {
        let (output, stats) = match output {
            OutputMode::Array => {
                let (tokens, stats) = engine
                    .tokenize_file_to_vec_profiled(input)
                    .map_err(operation_error)?;
                (ChopOutput::Array(tokens), stats)
            }
            OutputMode::Json(path) => {
                let stats = engine
                    .tokenize_file_json_profiled(input, path_string(&path, "dump")?)
                    .map_err(operation_error)?;
                (
                    ChopOutput::Written {
                        tokens: stats.total_tokens,
                    },
                    stats,
                )
            }
            OutputMode::Compact(path) => {
                let stats = engine
                    .tokenize_file_u32_le_profiled(input, path_string(&path, "dump")?)
                    .map_err(operation_error)?;
                (
                    ChopOutput::Written {
                        tokens: stats.total_tokens,
                    },
                    stats,
                )
            }
        };
        return Ok(ChopResult {
            output,
            profile: Some(stats.into()),
        });
    }

    let output = match output {
        OutputMode::Array => ChopOutput::Array(
            engine
                .tokenize_file_to_vec_unprofiled(input)
                .map_err(operation_error)?,
        ),
        OutputMode::Json(path) => ChopOutput::Written {
            tokens: engine
                .tokenize_file_json_unprofiled(input, path_string(&path, "dump")?)
                .map_err(operation_error)?,
        },
        OutputMode::Compact(path) => ChopOutput::Written {
            tokens: engine
                .tokenize_file_u32_le_unprofiled(input, path_string(&path, "dump")?)
                .map_err(operation_error)?,
        },
    };
    Ok(ChopResult {
        output,
        profile: None,
    })
}

fn operation_error(error: Box<dyn Error>) -> TokenizerError {
    match error.downcast::<io::Error>() {
        Ok(error) => TokenizerError::Io(*error),
        Err(error) => TokenizerError::Tokenization(error.to_string()),
    }
}

fn path_string<'a>(path: &'a Path, role: &'static str) -> Result<&'a str, TokenizerError> {
    path.to_str().ok_or(TokenizerError::InvalidPath { role })
}

fn reject_same_input_output(input: &Path, output: &Path) -> Result<(), TokenizerError> {
    if input == output
        || same_file_identity(input, output)
        || matches!(
            (normalized_path(input), normalized_path(output)),
            (Ok(input), Ok(output)) if input == output
        )
    {
        return Err(TokenizerError::InvalidOutput(
            "input path and dump path must be different".to_owned(),
        ));
    }
    Ok(())
}

fn normalized_path(path: &Path) -> io::Result<PathBuf> {
    if path.exists() {
        return std::fs::canonicalize(path);
    }
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(std::fs::canonicalize(parent)?.join(name))
}

#[cfg(unix)]
fn same_file_identity(left: &Path, right: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    matches!(
        (std::fs::metadata(left), std::fs::metadata(right)),
        (Ok(left), Ok(right)) if left.dev() == right.dev() && left.ino() == right.ino()
    )
}

#[cfg(not(unix))]
fn same_file_identity(_left: &Path, _right: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_arguments_are_validated_in_core() {
        assert!(matches!(
            ChopOptions::new(false, None, None, false).unwrap().output,
            OutputMode::Array
        ));
        assert!(matches!(
            ChopOptions::new(false, Some("array"), None, false)
                .unwrap()
                .output,
            OutputMode::Array
        ));
        assert!(matches!(
            ChopOptions::new(false, Some("iterator"), None, false),
            Err(TokenizerError::IteratorNotImplemented)
        ));
        assert!(matches!(
            ChopOptions::new(false, None, Some(PathBuf::from("x")), false),
            Err(TokenizerError::InvalidOutput(_))
        ));
        assert!(matches!(
            ChopOptions::new(false, Some("iterator"), Some(PathBuf::from("x")), false),
            Err(TokenizerError::InvalidOutput(_))
        ));
        assert!(matches!(
            ChopOptions::new(false, Some("array"), Some(PathBuf::from("x")), false),
            Err(TokenizerError::InvalidOutput(_))
        ));
        assert!(matches!(
            ChopOptions::new(false, Some("json"), None, false),
            Err(TokenizerError::InvalidOutput(_))
        ));
        assert!(matches!(
            ChopOptions::new(false, Some("other"), None, false),
            Err(TokenizerError::InvalidOutput(_))
        ));
    }

    #[test]
    fn aliases_and_disabled_models_are_resolved_before_loading() {
        assert_eq!(canonical_model_name("r50k_base").unwrap(), "r50k");
        assert!(matches!(
            StreamTokenizer::new("llama31p"),
            Err(TokenizerError::Source(SourceError::Llama31pDisabled))
        ));
        assert!(matches!(
            StreamTokenizer::new("not-a-model"),
            Err(TokenizerError::Source(SourceError::UnknownModel(_)))
        ));
    }

    /// End-to-end through the public API: `p50k_base`'s single-gap output
    /// relabel (see `stream_tokenizer::detect_affine_relabel`) must still
    /// produce tiktoken's real ranks, not the internal gap-compacted ids,
    /// when driven by the fast/SIMD engine's batched streaming path.
    ///
    /// Ignored by default: loads real tokenizer data through tiktoken's
    /// data-gym cache (network on a cold cache).
    #[test]
    #[ignore = "requires the tiktoken data-gym cache (network on a cold cache)"]
    fn real_p50k_streaming_output_uses_tiktoken_ranks_not_compact_ids() {
        let tokenizer = StreamTokenizer::new("p50k").expect("load p50k");
        let code =
            "def f(x):\n    if x:\n        return x\n    return 0\n\nclass Foo:\n        pass\n";
        let options = ChopOptions::with_output(true, OutputMode::Array, false);
        let result = tokenizer
            .tokenize_string(code.to_owned(), options)
            .expect("tokenize p50k code sample");
        let ChopOutput::Array(tokens) = result.output else {
            panic!("array output mode must return ChopOutput::Array");
        };
        // Real tiktoken ranks, not internal compact ids: `50258` and `50262`
        // are the 4- and 8-space indentation runs, each one higher than its
        // compact id because both sit above the reserved `50256`.
        assert_eq!(
            tokens,
            [
                4299, 277, 7, 87, 2599, 198, 50_258, 611, 2124, 25, 198, 50_262, 1441, 2124, 198,
                50_258, 1441, 657, 198, 198, 4871, 36080, 25, 198, 50_262, 1208, 198
            ]
        );
    }
}
