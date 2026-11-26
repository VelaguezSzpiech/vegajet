# EVICT prefix-depth controller

`core/evict_depth.hpp` exposes the small, replayable controller used by speculative decode loops:

```cpp
EvictDepth depth(max_depth);
uint32_t k = depth.choose(available_draft, unique_experts);  // advisory, [0, min(available_draft, max_depth)]
// ... target verifies the first k drafts ...
depth.observe(attempted, accepted);                          // target result, accepted in [0, attempted]
```

The controller only limits how many draft tokens are verified. It never supplies, commits, rewrites or ranks a
token. The target verifier is the sole authority for the accepted prefix and for the correction/bonus token that
follows it.

## Inputs

* `available_draft` - draft tokens currently offered by the drafter.
* `unique_experts` - the exact number of distinct `(layer, routed-expert)` keys read back from the last
  **completed** target verifier pass. `core/route_union.hpp` produces it from the existing rank-0 top-ID rows with
  a bounded 64x256-bit set; repeated ids collapse and disjoint ids accumulate. It is a lagged cost-feedback
  signal, not a prediction of the candidate prefix.
* `0` means unknown (no routing pressure): cold start, a deferred or unavailable readback, a zero-token pass, a
  capture that did not complete all expected layers for `M > 16`, or a value already consumed by the one-shot
  `RouteUnionFeedback`. A caller must pass `0` in those cases. `top_k * layers`, a uniform-routing expectation,
  and any other metadata are not measurements and must not be substituted.
* `attempted` / `accepted` - the target verifier's result for the previous proposal. `accepted` is the prefix the
  target accepted, not a drafter score or a predicted acceptance.

## Policy

* Cold start: no observations yet, so the prior `P(accept) = 0.75` sets a nonzero proposal. Drafting is allowed;
  the target still decides.
* Verified history: each observation is a geometric chain; per-position acceptance uses the standard
  `accepted / (accepted + 1)` estimate (`1.0` on full acceptance), smoothed with an EWMA.
* Depth: the largest `k <= max_depth` whose chance of being reached is at least `min_survival` (default 5%).
* Expert pressure: the acceptance estimate is divided by
  `max(1, unique_experts / (experts_per_depth * min(available_draft, max_depth)))`. A denser measured routed-expert
  union therefore shortens the prefix without changing the target decision.
* First-token rejection: `accepted == 0` with `attempted > 0`. Two consecutive all-reject verifies switch the
  controller to target-only proposals.
* Probing: target-only steps are cheap and make progress, but they cannot reveal a regime change. After
  `probe_after` target-only proposals the controller offers a one-token probe. The target may reject it (cooldown
  restarts) or accept it (the depth estimate climbs again). This is what lets a prose-to-code transition recover.

## Depth zero is a real target step

`choose() == 0` means "do not verify drafts this step", not "do not decode". The caller issues the ordinary
single-token target decode, consumes one real step (unless the request already ended) and emits the target token.
A zero proposal must never stall a request, and it must never be turned into a controller-generated token.

## Replay trace (Claude contract)

Keep these columns separate in every replay row:

| column | source |
|---|---|
| proposed depth | `choose()` return value |
| unique experts | `choose()` argument: measured count, or `0` = unknown |
| unique source | `measured` (exact `RouteUnion` dedup of a completed pass) or `unknown` |
| unique tokens | token count of the completed pass the measurement covers; `0` when unknown |
| verified prefix | target verifier, `0..proposed` |
| correction/target token | target token after the accepted prefix (bonus token at depth > 0, the only token at depth 0) |
| emitted tokens | target-accepted drafts, then the correction token |

Required replay cases:

1. **Cold start** - no observations: a positive proposal, target still authoritative.
2. **Rejection at token one** - `observe(k, 0)` repeated: depth falls to zero, every zero row emits exactly one
   target token, and a probe appears after the configured cooldown.
3. **Full acceptance** - `observe(k, k)` repeated: proposal reaches `min(available_draft, max_depth)`.
4. **Code/prose changes** - a high-acceptance phase, a rejection phase, and a recovery phase. Depth must fall in
   the rejection phase, target-only rows must still make progress, and probes must recover after the phase change.

For every row: nonzero depth emits `verified_prefix + 1` tokens; depth zero emits exactly one token; no row before
EOS emits zero tokens. The controller's proposal is advisory and may be shorter than the available draft.

## Integration

`OrnithModel::draft_step()` consumes `EvictDepth` and returns an empty draft graph at depth zero; the caller's
ordinary `decode_step(token, 1)` is the target-only path. `OrnithModel::accept_draft()` forwards the target's
accepted prefix to `observe()`. The model configures `experts_per_depth` from its routing width
(`n_expert_used * n_trunk`) so the expert signal is in `(layer, expert)` units, and exposes
`note_route_ids(ids, layers, tokens, top_k, layer_stride_ids)` / `note_route_union(count, tokens)` for the backend
to publish the exact union of a completed target pass. `draft_step()` consumes that signal once; if no new pass
was published, the next proposal sees `0` (unknown), not a stale or estimated count. The legacy `MoeKGovernor`
remains observed for compatibility but no longer chooses the draft length.

The DFlash2 GPU bench path (`run_greedy_dflash2` in `models/ornith/ornith_gpu.cpp`) also uses this controller when
`VEGAJET_EVICT_ADAPT=1`. It asks `step_batch(..., capture_routes=true)` to copy each layer's existing rank-0
top-ID rows into a pinned 24 KiB staging region of the 4 MiB I/O buffer with the existing `kfd_memcpy_u32` kernel.
After `wait_idle()` the host dedups the completed pass with `RouteUnion` and publishes the exact count through
`RouteUnionFeedback`. The capture is bounded to `M <= 16` and covers both the per-layer loop and the fused `M=1`
super loop (its descriptor points at the same rank-0 top-ID buffer); if fewer than 48 layer copies were enqueued,
the feedback is explicitly unknown. The measured union is consumed by the next proposal, and
`accept_draft`-equivalent feedback is `observe(dk, acc)` after the target comparison. At `k_cur == 0` that loop
runs `step_batch({gen.back()}, ...)` and commits the target token, which is the required depth-zero progress; its
1-token capture is recorded like any other completed pass.
`VEGAJET_EVICT_TRACE=1` (with `VEGAJET_EVICT_ADAPT=1`) prints one `[EVICT pos=...] proposed=... unique=...
unique_source=... unique_tokens=... verified=... correction=... emitted=...` row per cycle, keeping the replay
columns and the measurement provenance separate. `VEGAJET_EVICT_ROUTE_IDS=0` disables only the capture (the
controller keeps running, and the signal remains unknown); it exists to isolate the copy/dedup overhead in an
A/B and does not change the target's authority.
