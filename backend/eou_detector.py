"""
End-of-Utterance (EOU) detection using LiveKit's turn detector ONNX model.

Loads the model directly via ONNX Runtime (bypasses LiveKit Agents framework).
Determines if the user is done speaking based on transcript + conversation context.

The model predicts whether <|im_end|> should appear next — so the final
<|im_end|> must be stripped from the input. The multilingual model (v0.4.1-intl)
also expects NFKC-normalized, lowercased, punctuation-stripped text.

~400MB RAM, ~25ms inference on CPU.
"""

import logging
import os
import re
import sys
import unicodedata

import numpy as np

log = logging.getLogger('eou_detector')
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter('[EOU] %(message)s'))
    log.addHandler(_h)
    log.propagate = False

EOU_ENABLED = os.environ.get('EOU_ENABLED', 'false').lower() == 'true'
EOU_THRESHOLD = float(os.environ.get('EOU_THRESHOLD', '0.5'))

_MAX_HISTORY_TOKENS = 128
_MAX_HISTORY_TURNS = 6

_session = None
_tokenizer = None


def _load_model():
    """Load the ONNX model and tokenizer directly from HuggingFace."""
    global _session, _tokenizer
    if _session is not None:
        return

    if not EOU_ENABLED:
        return

    try:
        os.environ.setdefault('ORT_LOG_LEVEL', 'ERROR')
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from transformers import AutoTokenizer

        repo = 'livekit/turn-detector'
        revision = 'v0.4.1-intl'

        log.info('Downloading EOU model...')
        model_path = hf_hub_download(
            repo, 'model_q8.onnx', subfolder='onnx', revision=revision
        )

        log.info('Loading ONNX session...')
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'], sess_options=opts
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            repo, revision=revision, truncation_side='left'
        )
        log.info(f'EOU model loaded (threshold: {EOU_THRESHOLD})')

    except Exception as e:
        log.error(f'Failed to load EOU model: {e}')


def _normalize_text(text):
    """Normalize text for the multilingual model: NFKC, lowercase, strip punctuation."""
    if not text:
        return ''
    text = unicodedata.normalize('NFKC', text.lower())
    # Remove punctuation except apostrophes and hyphens
    text = ''.join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith('P') and ch not in ("'", "-"))
    )
    return re.sub(r'\s+', ' ', text).strip()


def is_end_of_utterance(transcript, conversation_history=None):
    """
    Check if the transcript represents a complete utterance.

    Returns True if the user appears done speaking (or EOU is disabled/unavailable).
    """
    if not EOU_ENABLED:
        return True

    _load_model()
    if _session is None or _tokenizer is None:
        return True

    try:
        # Build chat-style input from conversation history + current transcript.
        # Merge adjacent same-role messages (matches LiveKit SDK behavior).
        raw_messages = []
        if conversation_history:
            for msg in conversation_history[-_MAX_HISTORY_TURNS:]:
                role = msg.get('role', 'user')
                if role not in ('user', 'assistant'):
                    continue
                content = msg.get('content', '')
                raw_messages.append({'role': role, 'content': content})
        raw_messages.append({'role': 'user', 'content': transcript})

        # Normalize text and merge adjacent same-role turns
        merged = []
        last = None
        for m in raw_messages:
            content = _normalize_text(m['content'])
            if not content:
                continue
            if last and last['role'] == m['role']:
                last['content'] += f' {content}'
            else:
                entry = {'role': m['role'], 'content': content}
                merged.append(entry)
                last = entry

        # Use the tokenizer's built-in chat template, then strip the final
        # <|im_end|>. The model's job is to predict WHETHER <|im_end|> should
        # appear next — if we include it, we're asking the wrong question.
        text = _tokenizer.apply_chat_template(
            merged,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )
        ix = text.rfind('<|im_end|>')
        if ix != -1:
            text = text[:ix]

        # Tokenize with truncation (left-side, preserves most recent context)
        inputs = _tokenizer(
            text,
            add_special_tokens=False,
            return_tensors='np',
            max_length=_MAX_HISTORY_TOKENS,
            truncation=True,
        )

        # Run inference — model only accepts input_ids (no attention_mask)
        outputs = _session.run(
            None, {'input_ids': inputs['input_ids'].astype('int64')}
        )
        eou_prob = float(outputs[0].flatten()[-1])

        is_eou = eou_prob >= EOU_THRESHOLD
        log.info(
            f'EOU prob: {eou_prob:.4f} (threshold: {EOU_THRESHOLD}) '
            f'-> {"END" if is_eou else "CONTINUE"} | "{transcript[:60]}"'
        )
        return is_eou

    except Exception as e:
        log.error(f'EOU prediction error: {e}')
        return True


def preload():
    """Pre-load the model (call during startup)."""
    if EOU_ENABLED:
        _load_model()
