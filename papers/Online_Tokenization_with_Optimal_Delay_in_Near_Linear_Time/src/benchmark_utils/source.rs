//! Resolve tokenizer source files without keeping generated dictionaries in the
//! repository.
//!
//! Hugging Face models use the normal Hub snapshot cache, shared with
//! `huggingface_hub`, Transformers, and Tokenizers. OpenAI encodings use the
//! same URL-keyed cache layout as Python `tiktoken`. Only the upstream
//! `.tiktoken`/`tokenizer.json` file is cached; compiled MTC dictionaries remain
//! in process memory.

use std::{
    env,
    ffi::{OsStr, OsString},
    fmt::{self, Write as _},
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use hf_hub::{Cache as HfCache, Repo, RepoType, api::sync::ApiBuilder};
use sha1::{Digest as _, Sha1};
use sha2::Sha256;
use thiserror::Error;

use crate::Encoding;

const TOKENIZER_JSON: &str = "tokenizer.json";
const DEFAULT_HF_REVISION: &str = "main";
const TIKTOKEN_DEFAULT_CACHE: &str = "data-gym-cache";

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The format of a local tokenizer source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFormat {
    /// OpenAI's base64-token/rank text format.
    Tiktoken,
    /// A Hugging Face `tokenizer.json`.
    HuggingFaceJson,
}

/// Canonical metadata for one OpenAI/tiktoken encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenAiSource {
    pub name: &'static str,
    pub encoding: Encoding,
    pub url: &'static str,
    pub sha256: &'static str,
}

/// A tokenizer file in a Hugging Face model repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HuggingFaceSource {
    pub name: String,
    pub repo_id: String,
    pub revision: String,
    pub filename: String,
}

impl HuggingFaceSource {
    /// A `tokenizer.json` from the repository's `main` revision.
    pub fn tokenizer_json(name: impl Into<String>, repo_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            repo_id: repo_id.into(),
            revision: DEFAULT_HF_REVISION.to_owned(),
            filename: TOKENIZER_JSON.to_owned(),
        }
    }

    /// Select an explicit Hub revision (branch, tag, or commit).
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }
}

/// A source understood by the runtime model loader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSource {
    OpenAi(OpenAiSource),
    HuggingFace(HuggingFaceSource),
    Local { path: PathBuf, format: SourceFormat },
}

impl ModelSource {
    pub fn format(&self) -> SourceFormat {
        match self {
            Self::OpenAi(_) => SourceFormat::Tiktoken,
            Self::HuggingFace(_) => SourceFormat::HuggingFaceJson,
            Self::Local { format, .. } => *format,
        }
    }

    pub fn encoding(&self) -> Option<Encoding> {
        match self {
            Self::OpenAi(source) => Some(source.encoding),
            Self::HuggingFace(_) | Self::Local { .. } => None,
        }
    }
}

/// Source bytes plus their cache/local path when one exists.
#[derive(Clone, Debug)]
pub struct LoadedSource {
    pub source: ModelSource,
    pub bytes: Vec<u8>,
    /// `None` only when tiktoken caching was explicitly disabled or the
    /// best-effort default cache was not writable.
    pub path: Option<PathBuf>,
}

#[derive(Error)]
pub enum SourceError {
    #[error("unknown encoding/tokenizer {0:?}")]
    UnknownModel(String),

    #[error(
        "llama31p is disabled: its merge dictionary requires properization, \
         which is not performed by the runtime loader"
    )]
    Llama31pDisabled,

    #[error("Hugging Face cache path in {variable} is empty")]
    EmptyHfCachePath { variable: &'static str },

    #[error(
        "Hugging Face offline mode is enabled and {repo_id}@{revision}/{filename} \
         is not in the shared cache at {cache_dir}"
    )]
    HfOfflineMiss {
        repo_id: String,
        revision: String,
        filename: String,
        cache_dir: PathBuf,
    },

    #[error("failed to resolve Hugging Face file {repo_id}@{revision}/{filename}: {message}")]
    HuggingFace {
        repo_id: String,
        revision: String,
        filename: String,
        message: String,
    },

    #[error("failed to download {url}: {message}")]
    Download { url: String, message: String },

    #[error(
        "SHA-256 mismatch for {location}: expected {expected}, got {actual}; \
         refusing to use the tokenizer data"
    )]
    HashMismatch {
        location: String,
        expected: String,
        actual: String,
    },

    #[error("failed to read tokenizer source {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to update tokenizer cache {path}: {source}")]
    CacheWrite {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

impl fmt::Debug for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

const R50K: OpenAiSource = OpenAiSource {
    name: "r50k_base",
    encoding: Encoding::R50k,
    url: "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    sha256: "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
};

const P50K: OpenAiSource = OpenAiSource {
    name: "p50k_base",
    encoding: Encoding::P50k,
    url: "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    sha256: "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
};

const CL100K: OpenAiSource = OpenAiSource {
    name: "cl100k_base",
    encoding: Encoding::Cl100k,
    url: "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    sha256: "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
};

const O200K: OpenAiSource = OpenAiSource {
    name: "o200k_base",
    encoding: Encoding::O200k,
    url: "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    sha256: "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
};

fn hf_registry_source(
    name: &str,
    env_var: &'static str,
    default_repo: &'static str,
) -> ModelSource {
    let repo_id = env::var(env_var).unwrap_or_else(|_| default_repo.to_owned());
    ModelSource::HuggingFace(HuggingFaceSource::tokenizer_json(name, repo_id))
}

/// Resolve a benchmark/profile name to its upstream source.
///
/// The existing `MTC_HF_<NAME>` overrides are retained. The short and canonical
/// names for native tiktoken encodings are both accepted.
pub fn canonical_model_name(name: &str) -> Result<&'static str, SourceError> {
    match name {
        "r50k" | "r50k_base" => Ok("r50k"),
        "p50k" | "p50k_base" => Ok("p50k"),
        "cl100k" | "cl100k_base" => Ok("cl100k"),
        "o200k" | "o200k_base" => Ok("o200k"),
        "gpt2" => Ok("gpt2"),
        "roberta" => Ok("roberta"),
        "starcoder" => Ok("starcoder"),
        "mistral" => Ok("mistral"),
        "qwen" | "qwen3" => Ok("qwen"),
        "gptoss" => Ok("gptoss"),
        "llama4" => Ok("llama4"),
        "llama31p" => Err(SourceError::Llama31pDisabled),
        other => Err(SourceError::UnknownModel(other.to_owned())),
    }
}

pub fn source_for(name: &str) -> Result<ModelSource, SourceError> {
    let source = match canonical_model_name(name)? {
        "r50k" => ModelSource::OpenAi(R50K),
        "p50k" => ModelSource::OpenAi(P50K),
        "cl100k" => ModelSource::OpenAi(CL100K),
        "o200k" => ModelSource::OpenAi(O200K),
        "gpt2" => hf_registry_source("gpt2", "MTC_HF_GPT2", "gpt2"),
        "roberta" => hf_registry_source("roberta", "MTC_HF_ROBERTA", "roberta-base"),
        "starcoder" => hf_registry_source("starcoder", "MTC_HF_STARCODER", "bigcode/starcoder2-3b"),
        "mistral" => hf_registry_source(
            "mistral",
            "MTC_HF_MISTRAL",
            "mistralai/Mistral-Nemo-Base-2407",
        ),
        "qwen" | "qwen3" => hf_registry_source("qwen", "MTC_HF_QWEN", "Qwen/Qwen3-8B"),
        "gptoss" => hf_registry_source("gptoss", "MTC_HF_GPTOSS", "openai/gpt-oss-20b"),
        "llama4" => hf_registry_source(
            "llama4",
            "MTC_HF_LLAMA4",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        ),
        _ => unreachable!("canonical model registry and source registry diverged"),
    };
    Ok(source)
}

/// Construct a local Hugging Face JSON source, useful for a future
/// `Tokenizer.from_file(...)` binding.
pub fn local_hf_tokenizer(path: impl Into<PathBuf>) -> ModelSource {
    ModelSource::Local {
        path: path.into(),
        format: SourceFormat::HuggingFaceJson,
    }
}

/// Construct a local OpenAI `.tiktoken` source.
pub fn local_tiktoken(path: impl Into<PathBuf>) -> ModelSource {
    ModelSource::Local {
        path: path.into(),
        format: SourceFormat::Tiktoken,
    }
}

/// Resolve and read a named tokenizer source.
pub fn load_source(name: &str) -> Result<LoadedSource, SourceError> {
    resolve_source(source_for(name)?)
}

/// Resolve and read an explicit source.
pub fn resolve_source(source: ModelSource) -> Result<LoadedSource, SourceError> {
    match &source {
        ModelSource::OpenAi(spec) => load_openai(*spec, source.clone()),
        ModelSource::HuggingFace(spec) => {
            let path = resolve_hf_source(spec)?;
            let bytes = read_path(&path)?;
            Ok(LoadedSource {
                source,
                bytes,
                path: Some(path),
            })
        }
        ModelSource::Local { path, .. } => {
            let bytes = read_path(path)?;
            Ok(LoadedSource {
                source: source.clone(),
                bytes,
                path: Some(path.clone()),
            })
        }
    }
}

/// Resolve an arbitrary Hub tokenizer for a future `from_pretrained` API.
///
/// Offline loads and immutable commit revisions are cache-first. Mutable
/// branches and tags are resolved against the Hub when online so a stale
/// `refs/<revision>` entry is not treated as permanently current.
pub fn resolve_hf_tokenizer(repo_id: &str, revision: Option<&str>) -> Result<PathBuf, SourceError> {
    let source = HuggingFaceSource::tokenizer_json(repo_id, repo_id)
        .with_revision(revision.unwrap_or(DEFAULT_HF_REVISION));
    resolve_hf_source(&source)
}

fn read_path(path: &Path) -> Result<Vec<u8>, SourceError> {
    fs::read(path).map_err(|source| SourceError::Read {
        path: path.to_owned(),
        source,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TiktokenCache {
    Disabled,
    Directory { path: PathBuf, user_specified: bool },
}

fn tiktoken_cache_from(
    get_env: impl Fn(&str) -> Option<OsString>,
    temp_dir: &Path,
) -> TiktokenCache {
    for variable in ["TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"] {
        if let Some(value) = get_env(variable) {
            if value.is_empty() {
                return TiktokenCache::Disabled;
            }
            return TiktokenCache::Directory {
                path: PathBuf::from(value),
                user_specified: true,
            };
        }
    }
    TiktokenCache::Directory {
        path: temp_dir.join(TIKTOKEN_DEFAULT_CACHE),
        user_specified: false,
    }
}

fn tiktoken_cache() -> TiktokenCache {
    tiktoken_cache_from(|name| env::var_os(name), &env::temp_dir())
}

fn load_openai(spec: OpenAiSource, source: ModelSource) -> Result<LoadedSource, SourceError> {
    load_openai_with_cache(spec, source, tiktoken_cache())
}

fn load_openai_with_cache(
    spec: OpenAiSource,
    source: ModelSource,
    cache: TiktokenCache,
) -> Result<LoadedSource, SourceError> {
    let cache_target = match &cache {
        TiktokenCache::Disabled => None,
        TiktokenCache::Directory { path, .. } => Some(path.join(tiktoken_cache_key(spec.url))),
    };

    if let Some(path) = &cache_target {
        match fs::read(path) {
            Ok(bytes) if verify_sha256(&bytes, spec.sha256).is_ok() => {
                return Ok(LoadedSource {
                    source,
                    bytes,
                    path: Some(path.clone()),
                });
            }
            Ok(_) => {
                // Match tiktoken: a corrupt cache entry is discarded and fetched
                // again. Ignore NotFound in case another process won the race.
                if let Err(error) = fs::remove_file(path)
                    && error.kind() != io::ErrorKind::NotFound
                {
                    let user_specified = matches!(
                        cache,
                        TiktokenCache::Directory {
                            user_specified: true,
                            ..
                        }
                    );
                    if user_specified {
                        return Err(SourceError::CacheWrite {
                            path: path.clone(),
                            source: error,
                        });
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(SourceError::Read {
                    path: path.clone(),
                    source: error,
                });
            }
        }
    }

    let bytes = download(spec.url)?;
    verify_sha256(&bytes, spec.sha256)?;

    let mut resolved_path = None;
    if let (Some(path), TiktokenCache::Directory { user_specified, .. }) = (&cache_target, &cache) {
        match atomic_cache_write(path, &bytes) {
            Ok(()) => resolved_path = Some(path.clone()),
            Err(source_error) if *user_specified => {
                return Err(SourceError::CacheWrite {
                    path: path.clone(),
                    source: source_error,
                });
            }
            Err(_) => {
                // The default temp cache is best-effort in Python tiktoken too.
            }
        }
    }

    Ok(LoadedSource {
        source,
        bytes,
        path: resolved_path,
    })
}

fn download(url: &str) -> Result<Vec<u8>, SourceError> {
    let response = ureq::get(url)
        .header(
            "User-Agent",
            concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .map_err(|error| SourceError::Download {
            url: url.to_owned(),
            message: error.to_string(),
        })?;
    let mut bytes = Vec::new();
    let (_, body) = response.into_parts();
    body
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|error| SourceError::Download {
            url: url.to_owned(),
            message: error.to_string(),
        })?;
    Ok(bytes)
}

fn atomic_cache_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "cache path has no parent"))?;
    fs::create_dir_all(parent)?;

    let mut last_collision = None;
    for _ in 0..16 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(
            ".{}.{}.{}.tmp",
            path.file_name()
                .and_then(OsStr::to_str)
                .unwrap_or("tokenizer"),
            std::process::id(),
            sequence
        );
        let temp_path = parent.join(temp_name);
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                last_collision = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };

        let result = (|| {
            file.write_all(bytes)?;
            file.flush()?;
            drop(file);
            fs::rename(&temp_path, path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        return result;
    }
    Err(last_collision.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a temporary cache file",
        )
    }))
}

fn tiktoken_cache_key(url: &str) -> String {
    hex_digest(Sha1::digest(url.as_bytes()).as_slice())
}

fn verify_sha256(bytes: &[u8], expected: &str) -> Result<(), SourceError> {
    let actual = hex_digest(<Sha256 as sha2::Digest>::digest(bytes).as_slice());
    if actual == expected {
        Ok(())
    } else {
        Err(SourceError::HashMismatch {
            location: "tokenizer source".to_owned(),
            expected: expected.to_owned(),
            actual,
        })
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn hf_cache_dir_from(get_env: impl Fn(&str) -> Option<OsString>) -> Result<PathBuf, SourceError> {
    for variable in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(value) = get_env(variable) {
            if value.is_empty() {
                return Err(SourceError::EmptyHfCachePath { variable });
            }
            return Ok(PathBuf::from(value));
        }
    }
    Ok(HfCache::from_env().path().clone())
}

fn hf_cache_dir() -> Result<PathBuf, SourceError> {
    hf_cache_dir_from(|name| env::var_os(name))
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.to_ascii_uppercase().as_str(),
                "1" | "ON" | "YES" | "TRUE"
            )
        })
        .unwrap_or(false)
}

fn is_hf_commit_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn cached_hf_path(cache_dir: &Path, source: &HuggingFaceSource) -> Option<PathBuf> {
    let repo = Repo::with_revision(
        source.repo_id.clone(),
        RepoType::Model,
        source.revision.clone(),
    );
    let cache = HfCache::new(cache_dir.to_owned());
    if let Some(path) = cache.repo(repo.clone()).get(&source.filename) {
        return Some(path);
    }

    // Python's hub client may have a commit snapshot without a `refs/<commit>`
    // indirection when callers explicitly requested a commit SHA.
    let direct_snapshot = cache_dir
        .join(repo.folder_name())
        .join("snapshots")
        .join(&source.revision)
        .join(&source.filename);
    direct_snapshot.exists().then_some(direct_snapshot)
}

fn resolve_hf_source(source: &HuggingFaceSource) -> Result<PathBuf, SourceError> {
    let cache_dir = hf_cache_dir()?;
    resolve_hf_source_with(
        source,
        cache_dir,
        env_flag("HF_HUB_OFFLINE"),
        hf_token_override(),
    )
}

/// `None` means to retain `hf-hub`'s normal `$HF_HOME/token` behavior.
/// `Some(None)` explicitly disables that default because a custom
/// `HF_TOKEN_PATH` was present but empty/unreadable.
fn hf_token_override() -> Option<Option<String>> {
    hf_token_override_from(|name| env::var_os(name), |path| fs::read_to_string(path))
}

fn hf_token_override_from(
    get_env: impl Fn(&str) -> Option<OsString>,
    read_token: impl Fn(&Path) -> io::Result<String>,
) -> Option<Option<String>> {
    if let Some(value) = get_env("HF_TOKEN") {
        if let Some(token) = value.into_string().ok().and_then(nonempty_trimmed) {
            return Some(Some(token));
        }
    }
    let path = get_env("HF_TOKEN_PATH")?;
    Some(read_token(Path::new(&path)).ok().and_then(nonempty_trimmed))
}

fn nonempty_trimmed(value: String) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn resolve_hf_source_with(
    source: &HuggingFaceSource,
    cache_dir: PathBuf,
    offline: bool,
    token_override: Option<Option<String>>,
) -> Result<PathBuf, SourceError> {
    let cached = cached_hf_path(&cache_dir, source);
    if offline {
        if let Some(path) = cached {
            return Ok(path);
        }
        return Err(SourceError::HfOfflineMiss {
            repo_id: source.repo_id.clone(),
            revision: source.revision.clone(),
            filename: source.filename.clone(),
            cache_dir,
        });
    }

    // A full commit SHA identifies immutable content, so no network request is
    // needed once its snapshot exists. Branches and tags are mutable and must
    // be resolved online before selecting a cached snapshot.
    if is_hf_commit_revision(&source.revision)
        && let Some(path) = cached
    {
        return Ok(path);
    }

    let repo = Repo::with_revision(
        source.repo_id.clone(),
        RepoType::Model,
        source.revision.clone(),
    );
    let mut builder = ApiBuilder::from_env()
        .with_cache_dir(cache_dir.clone())
        .with_progress(false);
    if let Some(token) = token_override {
        builder = builder.with_token(token);
    }
    let api = builder.build().map_err(|error| SourceError::HuggingFace {
        repo_id: source.repo_id.clone(),
        revision: source.revision.clone(),
        filename: source.filename.clone(),
        message: error.to_string(),
    })?;

    let api_repo = api.repo(repo.clone());
    if !is_hf_commit_revision(&source.revision) {
        let info = api_repo.info().map_err(|error| SourceError::HuggingFace {
            repo_id: source.repo_id.clone(),
            revision: source.revision.clone(),
            filename: source.filename.clone(),
            message: error.to_string(),
        })?;
        let resolved_source = HuggingFaceSource {
            revision: info.sha.clone(),
            ..source.clone()
        };
        if let Some(path) = cached_hf_path(&cache_dir, &resolved_source) {
            // Keep the shared branch/tag ref current for other HF clients. The
            // exact snapshot is already usable even when a read-only cache
            // prevents this best-effort metadata update.
            let _ = HfCache::new(cache_dir).repo(repo).create_ref(&info.sha);
            return Ok(path);
        }
    }

    // `get` is cache-first even for mutable refs in hf-hub 0.4, so use
    // `download` after resolving online. This fetch also updates refs/<revision>.
    api_repo
        .download(&source.filename)
        .map_err(|error| SourceError::HuggingFace {
            repo_id: source.repo_id.clone(),
            revision: source.revision.clone(),
            filename: source.filename.clone(),
            message: error.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn env_map<'a>(entries: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |key| {
            entries
                .iter()
                .find_map(|(candidate, value)| (*candidate == key).then(|| OsString::from(value)))
        }
    }

    #[test]
    fn registry_has_canonical_openai_sources_and_aliases() {
        let ModelSource::OpenAi(short) = source_for("cl100k").unwrap() else {
            panic!("expected OpenAI source");
        };
        let ModelSource::OpenAi(long) = source_for("cl100k_base").unwrap() else {
            panic!("expected OpenAI source");
        };
        assert_eq!(short, long);
        assert_eq!(short.encoding, Encoding::Cl100k);
        assert_eq!(
            short.sha256,
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
        );
        assert!(matches!(
            source_for("gpt2"),
            Ok(ModelSource::HuggingFace(_))
        ));
        assert!(matches!(
            source_for("llama31p"),
            Err(SourceError::Llama31pDisabled)
        ));
    }

    #[test]
    fn tiktoken_cache_key_matches_python_tiktoken() {
        assert_eq!(
            tiktoken_cache_key(R50K.url),
            "0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4"
        );
        assert_eq!(
            tiktoken_cache_key(O200K.url),
            "fb374d419588a4632f3f557e76b4b70aebbca790"
        );
    }

    #[test]
    fn tiktoken_cache_environment_precedence_and_disable_match_python() {
        let temporary = Path::new("/tmp/example");
        assert_eq!(
            tiktoken_cache_from(
                env_map(&[
                    ("TIKTOKEN_CACHE_DIR", "/preferred"),
                    ("DATA_GYM_CACHE_DIR", "/legacy"),
                ]),
                temporary,
            ),
            TiktokenCache::Directory {
                path: PathBuf::from("/preferred"),
                user_specified: true,
            }
        );
        assert_eq!(
            tiktoken_cache_from(env_map(&[("TIKTOKEN_CACHE_DIR", "")]), temporary),
            TiktokenCache::Disabled
        );
        assert_eq!(
            tiktoken_cache_from(env_map(&[]), temporary),
            TiktokenCache::Directory {
                path: temporary.join("data-gym-cache"),
                user_specified: false,
            }
        );
    }

    #[test]
    fn valid_cached_openai_file_never_needs_network() {
        let dir = tempdir().unwrap();
        let bytes = b"YQ== 0\n";
        let hash = hex_digest(<Sha256 as sha2::Digest>::digest(bytes).as_slice());
        let spec = OpenAiSource {
            name: "test",
            encoding: Encoding::R50k,
            url: "https://invalid.example.test/test.tiktoken",
            sha256: Box::leak(hash.into_boxed_str()),
        };
        let path = dir.path().join(tiktoken_cache_key(spec.url));
        fs::write(&path, bytes).unwrap();
        let loaded = load_openai_with_cache(
            spec,
            ModelSource::OpenAi(spec),
            TiktokenCache::Directory {
                path: dir.path().to_owned(),
                user_specified: true,
            },
        )
        .unwrap();
        assert_eq!(loaded.bytes, bytes);
        assert_eq!(loaded.path.as_deref(), Some(path.as_path()));
    }

    #[test]
    fn hf_cache_lookup_uses_standard_snapshot_layout_without_network() {
        let dir = tempdir().unwrap();
        let source = HuggingFaceSource::tokenizer_json("fixture", "org/model");
        let repo = Repo::with_revision(
            source.repo_id.clone(),
            RepoType::Model,
            source.revision.clone(),
        );
        let repo_dir = dir.path().join(repo.folder_name());
        let snapshot = repo_dir.join("snapshots").join("abc123");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::write(repo_dir.join("refs").join("main"), "abc123").unwrap();
        fs::write(snapshot.join(TOKENIZER_JSON), "{}").unwrap();

        assert_eq!(
            cached_hf_path(dir.path(), &source),
            Some(snapshot.join(TOKENIZER_JSON))
        );
    }

    #[test]
    fn hf_offline_mode_uses_cache_without_resolving_mutable_revision() {
        let dir = tempdir().unwrap();
        let source = HuggingFaceSource::tokenizer_json("fixture", "org/model");
        let repo = Repo::with_revision(
            source.repo_id.clone(),
            RepoType::Model,
            source.revision.clone(),
        );
        let repo_dir = dir.path().join(repo.folder_name());
        let snapshot = repo_dir.join("snapshots").join("cached-commit");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::write(repo_dir.join("refs").join("main"), "cached-commit").unwrap();
        fs::write(snapshot.join(TOKENIZER_JSON), "{}").unwrap();

        assert_eq!(
            resolve_hf_source_with(&source, dir.path().to_owned(), true, Some(None)).unwrap(),
            snapshot.join(TOKENIZER_JSON)
        );
    }

    #[test]
    fn hf_offline_mode_reports_cache_miss_before_building_api() {
        let dir = tempdir().unwrap();
        let source = HuggingFaceSource::tokenizer_json("fixture", "invalid.example/model");
        let error =
            resolve_hf_source_with(&source, dir.path().to_owned(), true, Some(None)).unwrap_err();
        assert!(matches!(error, SourceError::HfOfflineMiss { .. }));
        assert!(error.to_string().contains("offline mode is enabled"));
    }

    #[test]
    fn only_full_commit_sha_is_treated_as_immutable() {
        assert!(is_hf_commit_revision(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(is_hf_commit_revision(
            "0123456789ABCDEF0123456789ABCDEF01234567"
        ));
        assert!(!is_hf_commit_revision("main"));
        assert!(!is_hf_commit_revision("v1.0"));
        assert!(!is_hf_commit_revision("0123456789abcdef"));
        assert!(!is_hf_commit_revision(
            "g123456789abcdef0123456789abcdef01234567"
        ));
    }

    #[test]
    fn hf_token_environment_precedes_standard_token_path() {
        let token = hf_token_override_from(
            env_map(&[
                ("HF_TOKEN", "  environment-secret\n"),
                ("HF_TOKEN_PATH", "/ignored/token"),
            ]),
            |_| panic!("HF_TOKEN_PATH must not be read when HF_TOKEN is set"),
        );
        assert_eq!(token, Some(Some("environment-secret".to_owned())));

        let empty_environment = hf_token_override_from(
            env_map(&[("HF_TOKEN", " "), ("HF_TOKEN_PATH", "/custom/token")]),
            |path| {
                assert_eq!(path, Path::new("/custom/token"));
                Ok("file-secret".to_owned())
            },
        );
        assert_eq!(empty_environment, Some(Some("file-secret".to_owned())));
    }

    #[test]
    fn hf_token_path_is_read_and_trimmed_without_entering_errors() {
        let token =
            hf_token_override_from(env_map(&[("HF_TOKEN_PATH", "/custom/token")]), |path| {
                assert_eq!(path, Path::new("/custom/token"));
                Ok("  file-secret\r\n".to_owned())
            });
        assert_eq!(token, Some(Some("file-secret".to_owned())));

        let missing =
            hf_token_override_from(env_map(&[("HF_TOKEN_PATH", "/custom/missing")]), |_| {
                Err(io::Error::new(io::ErrorKind::NotFound, "not found"))
            });
        assert_eq!(missing, Some(None));
    }

    #[test]
    fn hf_cache_environment_uses_modern_then_legacy_names() {
        assert_eq!(
            hf_cache_dir_from(env_map(&[
                ("HF_HUB_CACHE", "/modern"),
                ("HUGGINGFACE_HUB_CACHE", "/legacy"),
            ]))
            .unwrap(),
            PathBuf::from("/modern")
        );
        assert_eq!(
            hf_cache_dir_from(env_map(&[("HUGGINGFACE_HUB_CACHE", "/legacy")])).unwrap(),
            PathBuf::from("/legacy")
        );
    }
}
