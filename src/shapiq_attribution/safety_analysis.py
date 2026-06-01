"""Token-level safety attribution game for SHAPIQ.

Players are tokens. The value function masks absent tokens and queries a safety
classifier, returning P(unsafe). Two backends are supported via the same interface:

- ``"llama_guard"``: Llama Guard 3 1B (autoregressive decoder). Absent tokens
  are *removed* (no [MASK] token). P(unsafe) is read from generation logits.
- ``"distilbert"``: fine-tuned DistilBERT (encoder). Absent tokens are replaced
  with [MASK]. P(unsafe) is the softmax of the classification head.

All coalitions in a batch are decoded back to text and passed in a single model
call — the same approach as shapiq's SentimentAnalysis benchmark class.

structure from: https://github.com/mmschlk/shapiq/blob/main/src/shapiq_games/benchmark/local_xai/benchmark_language.py

Typical usage::

    # Llama Guard 3 1B loaded from HuggingFace
    game = SafetyAnalysisGame("Ignore previous instructions and ...")

    # Pre-trained DistilBERT passed directly
    game = SafetyAnalysisGame(
        "Ignore previous instructions and ...",
        model=my_distilbert,
        tokenizer=my_tokenizer,
        backend="distilbert",
    )

    sv, ksii = run_safety_shapiq(game, budget=256)
    sv.plot_sentence(words=game.token_names)
    sv.plot_force(feature_names=game.token_names)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapiq
import torch


class SafetyAnalysisGame(shapiq.Game):
    """Token-level safety classification game.

    Args:
        input_text: The prompt to attribute.
        model: Pre-loaded model. If None, loaded from ``model_name``.
        tokenizer: Pre-loaded tokenizer. If None, loaded from ``model_name``.
        model_name: HuggingFace model id used when model/tokenizer are None.
            Defaults to Llama Guard 3 1B.
        backend: ``"llama_guard"`` or ``"distilbert"``.
        normalize: Subtract the empty-coalition baseline so v(∅) = 0.
    """

    def __init__(
        self,
        input_text: str,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = "meta-llama/Llama-Guard-3-1B",
        backend: str = "llama_guard",
        normalize: bool = True,
    ) -> None:
        if model is None or tokenizer is None:
            model, tokenizer = _load_model(model_name, backend)

        self._model = model
        self._tokenizer = tokenizer
        self._backend = backend
        self._device = next(model.parameters()).device

        # Tokenize and strip leading/trailing special tokens ([CLS]/[BOS], [SEP]/[EOS])
        ids = tokenizer(input_text)["input_ids"]
        self._token_ids = np.asarray(ids[1:-1], dtype=np.int64)
        self.token_names: list[str] = tokenizer.convert_ids_to_tokens(self._token_ids)
        n_players = len(self._token_ids)

        # Baseline: all tokens absent — computed once, reused for normalization
        empty = np.zeros((1, n_players), dtype=bool)
        self._baseline = float(self.value_function(empty)[0])

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self._baseline,
        )

    # ------------------------------------------------------------------
    # shapiq interface
    # ------------------------------------------------------------------

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return P(unsafe) for each coalition.

        Decodes each coalition back to text, then makes a single batched model
        call — the same strategy as shapiq's SentimentAnalysis class.

        Args:
            coalitions: Boolean array of shape ``(n_coalitions, n_players)``.

        Returns:
            Float array of shape ``(n_coalitions,)`` with P(unsafe) per coalition.
        """
        texts = []
        for coalition in coalitions:
            if self._backend == "llama_guard":
                # Decoder: remove absent tokens (no [MASK] in vocabulary)
                present_ids = self._token_ids[coalition]
                text = self._tokenizer.decode(present_ids, skip_special_tokens=True)
            else:
                # Encoder: replace absent tokens with [MASK]
                masked = self._token_ids.copy()
                masked[~coalition] = self._tokenizer.mask_token_id
                text = self._tokenizer.decode(masked, skip_special_tokens=True)
            texts.append(text)
        return self._call_model(texts)

    # ------------------------------------------------------------------
    # Batched model call
    # ------------------------------------------------------------------

    def _call_model(self, texts: list[str]) -> np.ndarray:
        """Run the classifier on a batch of texts and return P(unsafe).

        Args:
            texts: Decoded coalition texts.

        Returns:
            Float array of shape ``(len(texts),)`` with P(unsafe) per text.
        """
        if self._backend == "llama_guard":
            return self._call_llama_guard(texts)
        return self._call_distilbert(texts)

    def _call_llama_guard(self, texts: list[str]) -> np.ndarray:
        """Extract P(unsafe) from Llama Guard's first generated token logits.

        Llama Guard is autoregressive so each prompt is a separate forward pass.
        Uses encode() instead of convert_tokens_to_ids() because Llama 3's
        tiktoken tokenizer has no SentencePiece "▁" prefix — convert_tokens_to_ids
        would return None, and logits[None] adds a dimension instead of indexing.
        """
        # encode() reliably returns integer IDs for any tokenizer type
        safe_id: int = self._tokenizer.encode("safe", add_special_tokens=False)[-1]
        unsafe_id: int = self._tokenizer.encode("unsafe", add_special_tokens=False)[-1]
        scores = []
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[0, -1, :]  # shape: (vocab_size,)
            probs = torch.stack([logits[safe_id], logits[unsafe_id]]).softmax(0)  # shape: (2,)
            scores.append(probs[1].item())
        return np.array(scores, dtype=float)

    def _call_distilbert(self, texts: list[str]) -> np.ndarray:
        """Batch-encode texts and return P(unsafe) from the classification head."""
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return logits.softmax(-1)[:, 1].cpu().numpy()


# ------------------------------------------------------------------
# Word-level aggregation
# ------------------------------------------------------------------

def aggregate_to_words(
    sv: shapiq.InteractionValues,
    token_names: list[str],
) -> tuple[shapiq.InteractionValues, list[str]]:
    """Aggregate subword token Shapley values up to word level.

    Llama 3's tokenizer marks word-initial tokens with a ``Ġ`` prefix (space
    before the word). Tokens *without* that prefix are continuations of the
    previous word (e.g. "vind", "ict", "ive" → "vindictive"). Their Shapley
    values are summed to produce one value per word.

    Args:
        sv: Token-level ``InteractionValues`` (SV, order 1) from
            ``run_safety_shapiq``.
        token_names: ``game.token_names`` — list of raw token strings.

    Returns:
        Tuple of (word_sv, word_names) where ``word_sv`` is a new
        ``InteractionValues`` with ``n_players = n_words`` and ``word_names``
        is a list of plain word strings (``Ġ`` stripped).

    Example::

        sv, ksii = run_safety_shapiq(game, budget=64)
        word_sv, word_names = aggregate_to_words(sv, game.token_names)
        word_sv.plot_sentence(words=word_names)
        word_sv.plot_force(feature_names=word_names)
    """
    # Build word groups: a new word starts at every token with a Ġ prefix
    # (or at position 0 which never has a prefix)
    word_groups: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, len(token_names)):
        if token_names[i].startswith("Ġ"):
            word_groups.append(current)
            current = [i]
        else:
            current.append(i)
    word_groups.append(current)

    word_names = [
        "".join(token_names[i].replace("Ġ", "") for i in grp)
        for grp in word_groups
    ]
    n_words = len(word_groups)

    # Build values array expected by InteractionValues for SV (min_order=0,
    # max_order=1): position 0 = empty coalition, positions 1..n = singletons
    word_values = np.zeros(n_words + 1)
    word_values[0] = float(sv[()])
    for w, grp in enumerate(word_groups):
        word_values[w + 1] = sum(float(sv[(t,)]) for t in grp)

    word_sv = shapiq.InteractionValues(
        values=word_values,
        index=sv.index,
        max_order=1,
        min_order=0,
        n_players=n_words,
        baseline_value=sv.baseline_value,
        estimated=sv.estimated,
    )
    return word_sv, word_names


# ------------------------------------------------------------------
# Run helper
# ------------------------------------------------------------------


def run_safety_shapiq(game: SafetyAnalysisGame, budget: int = 256) -> tuple:
    """Approximate SV and k-SII interactions for a SafetyAnalysisGame.

    Args:
        game: A configured SafetyAnalysisGame instance.
        budget: Number of coalition samples for KernelSHAP / KernelSHAPIQ.

    Returns:
        Tuple of (sv, ksii): first-order Shapley values and pairwise k-SII.
    """
    n = game.n_players
    sv = shapiq.KernelSHAP(n=n, random_state=42).approximate(budget=budget, game=game)
    ksii = shapiq.KernelSHAPIQ(n=n, index="k-SII", max_order=2, random_state=42).approximate(budget=budget, game=game)
    return sv, ksii


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------


def _load_model(model_name: str, backend: str) -> tuple:
    """Load model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model id.
        backend: ``"llama_guard"`` or ``"distilbert"``.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if backend == "llama_guard":
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
