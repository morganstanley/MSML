"""Embedding production and disk-backed vector storage."""

import hashlib
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from math import ceil
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
from cachetools import LRUCache, cached
from numpy.typing import DTypeLike, NDArray
from openai import APIConnectionError, APIStatusError

from alpha_lab import token_metrics
from alpha_lab.providers import get_openai_client
from alpha_lab.utils import RetryTimer, truncate_text


def _is_transient_embedding_error(error: Exception) -> bool:
    """Whether a failed provider call may succeed when repeated unchanged."""
    if isinstance(error, APIConnectionError):
        return True
    return isinstance(error, APIStatusError) and (
        error.status_code in (408, 409, 429) or error.status_code >= 500
    )


class EmbeddingError(RuntimeError):
    """Raised when producing an embedding fails (e.g. a provider/transport error).

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` handlers (the
    memory tools) still surface it as a graceful error rather than a crash.
    """


class EmbeddingModel:
    """OpenAI embedding model wrapper producing (optionally) L2-normalized vectors."""

    def __init__(
        self,
        name: Literal["text-embedding-3-large", "text-embedding-3-small"] = "text-embedding-3-large",
        dtype: DTypeLike = np.float32,
        normalize: bool = True,
    ):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be np.float32 or np.float64")

        self.name = name
        self.normalize = normalize
        self.dtype = dtype

    @cached_property
    def _client(self) -> Any:
        """The sync OpenAI client, built once with the SDK's own retries disabled.

        :meth:`EmbeddingStore.embed` owns the retry policy, so leaving the SDK's
        enabled would stack a second round of attempts and backoff underneath it.
        Caching the client is safe because the on-prem client re-injects its
        bearer token on every request.
        """
        return get_openai_client().with_options(max_retries=0)

    def close(self) -> None:
        """Release the cached client, if one was ever built.

        Never constructs a client just to close one, and clears the cache
        so a later use lazily rebuilds rather than hitting a closed client.
        """
        client = self.__dict__.pop("_client", None)
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def __call__(self, input: str) -> NDArray[np.floating]:
        """Embed a single string.

        Args:
            input: Text to embed.

        Returns:
            The embedding vector, L2-normalized when ``normalize`` is set.

        Raises:
            EmbeddingError: If the service returns a vector that is not finite,
                or one whose norm is too small to divide by.
        """
        result = self._embed(self._client, self.name, input)
        vector = np.asarray(result, copy=True, dtype=self.dtype)
        if self.normalize:
            # A zero norm is left unscaled; a denormal one overflows the division
            # to infinity, which the finiteness check below catches.
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm

        if not np.isfinite(vector).all():
            raise EmbeddingError("Embedding vector contains non-finite values")

        return vector

    @staticmethod
    @cached(LRUCache(16384), key=lambda client, name, input: (name, input), lock=Lock())
    def _embed(client: Any, name: str, input: str) -> NDArray[np.float64]:
        """Fetch one raw vector, shared across models by name and text.

        Cached at full precision and independent of any instance's ``dtype`` or
        ``normalize`` setting, both of which :meth:`__call__` applies afterwards
        to its own copy. The client is excluded from the key: the same model name
        yields the same vector whichever client fetched it. Usage is recorded
        here rather than in :meth:`__call__` so cache hits, which make no
        provider call, aren't double-counted.
        """
        result = client.embeddings.create(model=name, input=input)
        token_metrics.record_embedding_usage(result, model=name, system="openai")
        return np.asarray(result.data[0].embedding, dtype=np.float64)

    @property
    def dim(self) -> int:
        """Dimensionality of the model's embedding vectors."""
        return 3072 if self.name == "text-embedding-3-large" else 1536


class EmbeddingStore:
    """Embedding manager with production and disk-backed storage.

    When vectors are L2-normalized (``EmbeddingModel.normalize``, the default), the
    dot-product similarity equals cosine similarity. Rows are keyed by memory
    ``index`` in fixed-size pages.
    """

    def __init__(
        self,
        root: str | Path,
        model: str | EmbeddingModel | None = "text-embedding-3-large",
        similarity: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ] = np.matmul,
        page_limit: int = 1024,
        token_limit: int = 8192,
        retry_timer: RetryTimer | None = None,
    ) -> None:
        """Bind to the directory holding the vector pages.

        ``retry_timer`` controls the number and timing of attempts after transient
        provider failures. Context-length corrections remain immediate and
        independent of that timing policy.
        """
        if (
            isinstance(token_limit, bool)
            or not isinstance(token_limit, int)
            or token_limit <= 0
        ):
            raise ValueError("token_limit must be a positive integer")

        self.root = Path(root)
        self.model = (
            model
            if isinstance(model, EmbeddingModel)
            else EmbeddingModel()
            if model is None
            else EmbeddingModel(model)
        )
        self.similarity = similarity
        self.page_limit = page_limit
        self.token_limit = token_limit
        self.retry_timer = RetryTimer() if retry_timer is None else retry_timer

    def close(self) -> None:
        """Release the model's client; the store owns its model's lifecycle."""
        self.model.close()

    def embed(self, text: str) -> NDArray[np.floating]:
        """Embed ``text``, fitted to ``token_limit``.

        Length is fit exactly via tiktoken when available, estimated otherwise;
        context-length failures are retried immediately with a corrective shrink.
        Transient connection and retryable HTTP failures use exponential backoff
        with jitter. Deterministic provider failures are not retried.

        Args:
            text: Text to embed.

        Returns:
            The embedding vector for the (possibly truncated) text.

        Raises:
            EmbeddingError: If the provider call is non-retryable or all retries fail.
        """
        text = truncate_text(text, self.token_limit)
        last_error: Exception | None = None
        for _ in self.retry_timer:
            try:
                return np.asarray(self.model(text), dtype=self.model.dtype)
            except Exception as error:
                if "context length" in str(error).lower():
                    text = text[:ceil(0.85 * len(text))]
                    try:
                        return np.asarray(self.model(text), dtype=self.model.dtype)
                    except Exception as corrected_error:
                        error = corrected_error

                if not _is_transient_embedding_error(error):
                    raise EmbeddingError(str(error)) from error

                last_error = error

        message = "Embedding request failed after all attempts"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise EmbeddingError(message) from last_error

    def compare(
        self,
        input: str | NDArray[np.floating],
        index: Sequence[int] | NDArray[np.intp] | None = None,
    ) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
        """Score ``input`` against stored vectors by dot product (equal to cosine
        similarity when the vectors are L2-normalized, as they are at production).

        Args:
            input: A query string (embedded on the fly) or a precomputed 1-D vector.
            index: Rows to score against. ``None`` scores every stored row; a
                sequence or array restricts scoring to those row indices.

        Returns:
            An ``(index, score)`` pair of aligned arrays — the row indices scored
            and their similarity scores. Both are empty when the store has no rows.

        Raises:
            ValueError: If ``input`` is not a 1-D vector.
        """
        vec = np.asarray(
            input if isinstance(input, np.ndarray) else self.embed(input),
            dtype=self.model.dtype,
        )
        if vec.ndim != 1:
            raise ValueError("input vector must be 1-D")

        # Stored rows are ``[fingerprint, *vector]``; prepend a 0 so the hash column
        # contributes ``hash * 0 == 0`` to the dot product and drops out of the score.
        vec = np.pad(vec, (1, 0))
        order = (
            np.argsort(index)
            if index is not None and (np.diff(index) < 0).any()
            else None
        )
        sort_ = index if order is None else np.take(index, order)
        parts = [self.similarity(mat, vec) for mat in self.read(sort_)]
        if not parts:  # empty store: nothing to score against
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=self.model.dtype)
        score = np.concatenate(parts)
        if index is None:  # only return scores for valid rows
            valid = ~np.isnan(score)
            index, = np.where(valid)
            score = score[valid]
        else:
            score = np.nan_to_num(score, nan=float("-inf"))
            if order is not None:
                score[order] = score

        return np.asarray(index), score

    def read(
        self, index: Sequence[int] | NDArray[np.intp] | None = None
    ) -> Iterator[NDArray[np.floating]]:
        """Yield stored vectors.

        Args:
            index: ``None`` yields every page in row order; an ascending array
                yields the selected rows, page by page.

        Yields:
            Vector blocks — a full page or the selected rows per page.

        Raises:
            ValueError: If an index array is not in ascending order.
        """
        if index is None:
            for path in sorted(self.root.glob("*.npy"), key=lambda p: int(p.stem)):
                yield np.load(path)
            return

        if (np.diff(index) < 0).any():
            raise ValueError("index must be in ascending order")

        page_indices, row_indices = np.divmod(index, self.page_limit)
        for page_index in np.unique(page_indices):
            rows = row_indices[page_indices == page_index]
            page = self._load_page(page_index)
            yield page[rows]

    def write(
        self, index: int, fingerprint: float, vector: NDArray[np.floating]
    ) -> None:
        """Persist ``vector`` at row ``index``, tagged with ``fingerprint`` in column 0.

        ``fingerprint`` identifies the content the vector was produced from, so a
        rebuild can tell a present-but-stale vector from a current one. The stored
        row is ``[fingerprint, *vector]`` (width ``dim + 1``).

        Args:
            index: Row index to write.
            fingerprint: Content fingerprint stored in column 0.
            vector: The (``dim``-wide) vector to store at that row.
        """
        vector = np.asarray(vector, dtype=self.model.dtype).reshape(-1)
        if vector.shape != (self.model.dim,):
            raise ValueError(f"vector must have shape {(self.model.dim,)}, got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("vector must contain only finite values")
        row = np.empty(self.model.dim + 1, dtype=self.model.dtype)
        row[0] = float(fingerprint)
        row[1:] = vector

        self.root.mkdir(parents=True, exist_ok=True)
        page_index, row_index = divmod(index, self.page_limit)

        page = self._load_page(page_index, mode="r+", create=True)
        page[row_index] = row
        page.flush()
        del page

    def get(self, index: int) -> NDArray[np.floating] | None:
        """Return the stored ``[fingerprint, *vector]`` row for ``index``.

        Args:
            index: Row index to read.

        Returns:
            The row (column 0 is the fingerprint, ``1:`` the vector), or ``None``
            when the page is absent or the row is non-finite (an unwritten slot is
            all-NaN).
        """
        try:
            block = next(self.read(np.asarray([index], dtype=np.intp)))
        except (FileNotFoundError, StopIteration):
            return None
        if block.shape[0] == 0:
            return None
        row = block[0]
        return None if not np.isfinite(row).all() else row.copy()

    def get_fingerprint(self, input: str) -> float:
        """Return a stable content fingerprint for ``input``."""
        hash_key = hashlib.sha256(input.encode("utf-8")).digest()
        precision = np.finfo(self.model.dtype).nmant + 1
        return float(int.from_bytes(hash_key, "big") % (2**precision))

    def _page_path(self, index: int) -> Path:
        return self.root / f"{index}.npy"

    def _load_page(
        self, index: int, *, mode: str = "r", create: bool = False
    ) -> NDArray[np.floating]:
        path = self._page_path(index)
        if not path.exists():
            if not create:
                msg = f"page {index} does not exist"
                raise FileNotFoundError(msg)

            np.save(
                path,
                np.full(
                    (self.page_limit, self.model.dim + 1),
                    np.nan,
                    dtype=self.model.dtype,
                ),
            )
        return np.load(path, mmap_mode=mode)
