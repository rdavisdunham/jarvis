"""Summarize content-free jarvis.latency log lines; no provider calls or database access.

Usage: python scripts/summarize_latency.py < worker.log
Counts describe events/attempts, not unique successful requests. Precommit events do
not prove a save; live_append_sent is not audible playback or browser paint.
"""
import json
import math
import statistics
import sys
from collections import defaultdict


def summarize(lines):
    stages = defaultdict(lambda: {'events': 0, 'errors': 0, 'metrics': defaultdict(list)})
    for line in lines:
        if 'latency ' not in line:
            continue
        try:
            event = json.loads(line.split('latency ', 1)[1])
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict) or not isinstance(event.get('stage'), str):
            continue
        stage = stages[event['stage']]
        stage['events'] += 1
        stage['errors'] += event.get('outcome') in {'error', 'failed', 'partial', 'cancelled', 'expired'}
        for metric in ('duration_ms', 'queue_ms', 'dispatch_ms'):
            value = event.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                stage['metrics'][metric].append(value)
    result = {}
    for name, stage in stages.items():
        metrics = {}
        for metric, values in stage['metrics'].items():
            ordered = sorted(values)
            metrics[metric] = {'n': len(values), 'median': statistics.median(values),
                               'p95': ordered[math.ceil(len(values)*.95)-1]}
        result[name] = {**stage, 'metrics': metrics}
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(sys.stdin), indent=2))
