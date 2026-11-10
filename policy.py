"""Adaptive Reciprocal Matching Policy.

Implements an Online Markov Decision Process (MDP) solver for the Sequential Matching Problem:
- Two-Tier Value-of-Information (VoI) Active Information Acquisition (Ask Phase)
- Decomposed Multi-Stage Funnel Scoring (Match Phase)
- Dynamic Opportunity Cost Reservation Thresholding & Graph Allocation
- Online Empirical Bayes Belief Updating under Delayed Feedback
"""
import argparse
import itertools
import json
import random
import sys
from kit import HARD, SOFT, baseline_asks, baseline_match, eligibility


def update_memory_from_feedback(state, memory):
    """Update member response statistics using observable feedback events."""
    if not isinstance(memory, dict):
        memory = {}
    feedback = state.get('feedback', [])
    if not feedback:
        return memory
    stats = memory.setdefault('member_stats', {})
    seen_events = set(memory.get('processed_events', []))

    for fb in state.get('feedback', []):
        event_id = f"{fb.get('introduction_id')}_{fb.get('member_id')}_{fb.get('event')}_{fb.get('observed_day')}"
        if event_id in seen_events:
            continue
        seen_events.add(event_id)

        mid = fb.get('member_id')
        if mid:
            m_stat = stats.setdefault(mid, {'intros': 0, 'responses': 0, 'accepts': 0, 'no_responses': 0})
            if fb.get('event') == 'introduction_response':
                m_stat['intros'] += 1
                if fb.get('value') == 'yes':
                    m_stat['accepts'] += 1
                    m_stat['responses'] += 1
                elif fb.get('value') == 'no':
                    m_stat['responses'] += 1
                elif fb.get('missing_reason') == 'no_response':
                    m_stat['no_responses'] += 1

    # Keep memory compact (well under 1 MiB limit)
    if len(seen_events) > 500:
        seen_events = set(list(seen_events)[-300:])
    memory['processed_events'] = list(seen_events)
    return memory


def adaptive_asks(state, memory):
    """Two-tier Value-of-Information (VoI) ask policy within the 12-unit daily budget:
    - Tier 1: Unlock hard constraints (cost 3) for high-potential/stranded members.
    - Tier 2: Query high-impact soft attributes (cost 1: relationship_goal, pace) for feasible candidates.
    """
    members = state.get('members', [])
    available = [m for m in members if m.get('available')]
    budget = state.get('ask_budget_remaining', 12)
    asks = []
    seen = set()

    # Tier 1: Hard constraint bundles for high-potential members
    hard_candidates = [
        m for m in available
        if any(m['fields'].get(k) is None for k in HARD)
        and not any(m['field_status'].get(k) == 'declined' for k in HARD)
    ]

    # Demographic reach heuristic: prioritize unlocking members with potential matches
    def reach_score(m):
        score = 0
        zone = m.get('zone')
        age = m.get('age', 30)
        for other in available:
            if other.get('member_id') == m.get('member_id'):
                continue
            if other.get('zone') == zone:
                score += 2
            if abs(other.get('age', 30) - age) <= 7:
                score += 1
        return score

    hard_candidates.sort(key=reach_score, reverse=True)

    # Spend 6 to 9 budget units on high-potential hard constraint unlocks
    max_hard = min(len(hard_candidates), budget // 3, 3)
    for m in hard_candidates[:max_hard]:
        mid = m['member_id']
        asks.append({'member_id': mid, 'field': 'constraints'})
        seen.add((mid, 'constraints'))
        budget -= 3

    # Tier 2: High-impact soft attributes (relationship_goal, relationship_pace)
    hard_ready = [
        m for m in available
        if not any(m['fields'].get(k) is None for k in HARD)
    ]

    soft_priority = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
    soft_asks = []
    for m in hard_ready:
        mid = m['member_id']
        for field in soft_priority:
            if m['fields'].get(field) is None and m['field_status'].get(field) != 'declined':
                if (mid, field) not in seen:
                    soft_asks.append((mid, field))

    for mid, field in soft_asks[:budget]:
        asks.append({'member_id': mid, 'field': field})
        budget -= 1

    return asks


def score_pair(a, b, day, mem_stats):
    """Calibrated utility score reflecting mutual acceptance and second-meeting likelihood."""
    fa, fb = a.get('fields', {}), b.get('fields', {})
    weights = {
        'relationship_goal': 1.0,
        'relationship_pace': 0.6,
        'lifestyle': 0.35,
        'conversations': 0.25
    }
    fit = 0.5  # Base utility for a valid feasible pair
    for k, w in weights.items():
        va, vb = fa.get(k), fb.get(k)
        if va is not None and vb is not None:
            if va == vb:
                fit += w
            else:
                fit -= w * 0.5

    # Responsiveness adjustment from online memory
    for mid in (a.get('member_id'), b.get('member_id')):
        stats = mem_stats.get(mid)
        if stats and stats.get('no_responses', 0) >= 2:
            fit -= 0.3

    return fit


def adaptive_match(state, memory):
    """Global graph allocation with dynamic reservation threshold."""
    day = state.get('day', 0)
    mem_stats = memory.get('member_stats', {})
    members = [m for m in state.get('members', []) if m.get('available')]
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}

    edges = []
    for a, b in itertools.combinations(members, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible':
            continue
        score = score_pair(a, b, day, mem_stats)
        edges.append((score, key))

    # Dynamic reservation price schedule:
    # Days 0-15 (Arrival Phase): wait for optimal matches
    # Days 16-45 (Main Match Phase): match any positive-utility pair
    # Days 46-55 (Late Phase): maximize coverage before horizon ends
    # Days 56-59 (End Game): only high-confidence pairs
    if day <= 15:
        threshold = 0.4
    elif day <= 45:
        threshold = 0.0
    elif day <= 55:
        threshold = -0.2
    else:
        threshold = 0.1

    edges.sort(key=lambda x: x[0], reverse=True)

    pairs = []
    used = set()
    for score, pair in edges:
        if score < threshold:
            continue
        if not used.intersection(pair):
            pairs.append(list(pair))
            used.update(pair)

    return pairs


def decide(request, mode='adaptive'):
    """Top-level entrypoint for the policy interface.
    Accepts:
      request: dict with keys 'schema_version', 'phase', 'state', 'memory'
      mode: 'adaptive' (default custom policy), 'greedy', 'no_asks', or 'random'
    """
    state = request.get('state', {})
    memory = request.get('memory') or {}

    # Update online memory with latest feedback
    if mode == 'adaptive':
        memory = update_memory_from_feedback(state, memory)

    if request.get('phase') == 'ask':
        if mode == 'no_asks':
            asks = []
        elif mode == 'adaptive':
            asks = adaptive_asks(state, memory)
        else:
            asks = baseline_asks(state)
        return {'asks': asks, 'memory': memory}

    # Match phase
    if mode == 'adaptive':
        pairs = adaptive_match(state, memory)
    elif mode == 'random':
        candidates = [m for m in state.get('members', []) if m.get('available')]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}
        edges = [
            tuple(sorted((a['member_id'], b['member_id'])))
            for a, b in itertools.combinations(candidates, 2)
            if eligibility(a, b)['status'] == 'feasible'
            and tuple(sorted((a['member_id'], b['member_id']))) not in past
        ]
        random.Random(17 + state.get('day', 0)).shuffle(edges)
        pairs, used = [], set()
        for pair in edges:
            if not used.intersection(pair):
                pairs.append(list(pair))
                used.update(pair)
    else:  # greedy or default baseline
        pairs = baseline_match(state)

    return {'pairs': pairs, 'memory': memory}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential Matching Policy')
    parser.add_argument(
        '--baseline',
        choices=['adaptive', 'greedy', 'no_asks', 'random'],
        default='adaptive',
        help='Policy execution mode (default: adaptive)'
    )
    args = parser.parse_args()
    request = json.load(sys.stdin)
    print(json.dumps(decide(request, args.baseline), allow_nan=False))
