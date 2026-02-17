"""
End-of-Utterance (EOU) detection using LiveKit's turn detector ONNX model.

Loads the model directly via ONNX Runtime (bypasses LiveKit Agents framework).
Determines if the user is done speaking based on transcript + conversation context.

~400MB RAM, ~25ms inference on CPU.
"""

import logging
import os
import sys

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

        _tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        log.info(f'EOU model loaded (threshold: {EOU_THRESHOLD})')

    except Exception as e:
        log.error(f'Failed to load EOU model: {e}')


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
        # Build chat-style input from conversation history + current transcript
        messages = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                messages.append({'role': role, 'content': content})
        messages.append({'role': 'user', 'content': transcript})

        # Format as ChatML tokens (what the model was trained on)
        text = ''.join(
            f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n'
            for m in messages
        )

        inputs = _tokenizer(text, return_tensors='np')
        outputs = _session.run(None, {'input_ids': inputs['input_ids']})

        # Output is per-token EOU probability; last token is what we want
        probs = outputs[0][0]  # shape: (seq_len,)
        eou_prob = float(probs[-1])

        is_eou = eou_prob >= EOU_THRESHOLD
        log.info(
            f'EOU prob: {eou_prob:.3f} (threshold: {EOU_THRESHOLD}) '
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
