# 21. Interactive, Reasoning, and Agentic Systems

A voice assistant listens while the user speaks, begins responding, calls a
tool, and stops mid-sentence when the user interrupts. There is no clean request
followed by a clean response. Input and output overlap inside a long-lived
session, the "prompt" is rewritten under the model's feet as speech recognition
revises itself, and the most important latency in the product is not time-to-
first-token but time-to-silence after the user starts talking over the answer.
The user steers constantly — interrupting, backchanneling ("mm-hm"), pausing,
resuming — and the system must treat each of those as a scheduled event, not
as noise to filter out.

Interactive inference is defined by deadlines, interruption, and suspended
work, not merely low average latency. Voice and video systems expose stale
output immediately. Reasoning models expose long, heavy-tailed generations;
agentic systems add tool waits during which a request is alive but may not
deserve accelerator memory. Every mechanism from Parts I through IV survives,
but the scheduler must now understand more than a stream of decode tokens.

## Build the latency budget backward

Suppose the product allows 700 milliseconds from the end of a user's phrase to
the beginning of audible speech. That budget includes network transport, speech
recognition, endpoint detection, language-model queueing and prefill, first
tokens, text-to-speech startup, and audio buffering.

**A live conversation streams through several models and transports.**

```blockdiag
flowchart LR
    A["Audio input"] --> R["Streaming ASR"]
    R --> L["Language model"]
    V["Video or events"] --> E["Incremental encoder"]
    E --> L
    L --> T["Tools and retrieval"]
    T --> L
    L --> S["Streaming TTS"]
    S --> P["Playback buffer"]
```


Improving language TTFT from 250 to 200 milliseconds helps. It does not save a
system that spends 600 milliseconds detecting the end of speech. This is why
the budget is built *backward* — start from the user-visible interval, subtract
stage by stage, and let the residuals name the priority — instead of forward,
where each team optimizes its own component and the sum quietly misses.

Assign a budget to every stage and measure it at boundaries visible to the
user, with one owner per stage. Audio generated but waiting in a playback
buffer has not reached the user. A video frame completed after its display
deadline may be useless even if throughput is high. Reserve an explicit margin
rather than allocating to 100 percent — the stages' tails are correlated (a
CPU-spiking host delays ASR and TTS together), so a budget assembled from
median stage times fails exactly when everything is slow at once.

Against this chapter's declared stage costs, the allocation looks like:

| Stage | Budget | Owner | Characteristic failure |
| --- | --- | --- | --- |
| endpointing | 180 ms | ASR service | waits too long; fires too early |
| LLM first token | 220 ms | inference fleet | queueing eats the margin |
| TTS first audio | 140 ms | synthesis pool | cold start, chunk cadence |
| transport + buffering | 90 ms | client and edge | jitter, underrun |
| margin | 70 ms | — | correlated tails consume it |

The table's discipline is that each row's owner cannot spend another row's
budget: when the LLM stage wants 300 ms, the negotiation is explicit and the
product owner sees which user-visible interval degrades.

## Stream input as well as output

Ordinary generation receives a complete prompt. A real-time system can process
partial audio, frames, or events as they arrive. An incremental encoder updates
state without replaying the entire input.

The session may contain several coordinated models:

```text
microphone -> speech recognition -> language model -> speech synthesis
                                      |
                                      +-> tools and retrieval
camera ----> vision encoder ----------+
```

Each arrow can stream. The streams also run on different clocks — audio at
tens of chunks per second, camera events a few times per second, tool results
arriving whenever the tool finishes — so a single global deadline is wrong;
every stream carries its own age bound and the session reconciles them at the
points where they meet. Buffers and queues need maximum ages, not only maximum
sizes. Old audio or frames may be dropped because processing them would delay
more current information.

Bidirectional transports such as WebSocket or streaming RPC carry events in
both directions. The protocol should define ordering, heartbeats, flow control,
reconnection, and which side owns the session. Reconnection is the hardest of
these, and it has exactly two honest shapes: *resume*, where the client replays
its last received sequence number per stream and the server continues from
committed state (requiring the server to retain unacknowledged events, and
requiring the session state that the next section discusses to have an owner);
or *restart*, where the session re-initializes visibly — new turn generation,
fresh endpointing, and a transcript rebuilt from history. A protocol that
silently mixes the two produces duplicated or missing audio with no way to
diagnose which happened.
Transport keepalive is not the same as model-worker health — a socket can be
open while the engine behind it has wedged, which is why Chapter 17's health
semantics apply per component, not just per connection.

### Queues bound age, not size

The reason age bounds replace size bounds deserves the arithmetic. Media
consumes at real-time by definition: one second of audio takes one second to
play. Suppose a transient stall leaves 500 ms of audio queued. If the system
diligently plays *everything*, the backlog drains at exactly real-time — every
subsequent sample, including everything recorded *after* the stall, is now
delayed by 500 ms forever. No amount of later throughput fixes it, because the
pipe's width is fixed by the clock, not by capacity. The only exit is
dropping: when a chunk's age exceeds its deadline, discard it and resync to
live — trading a gap the user hears once for a lag the user hears forever.
This inverts the reliability instinct built into every other queue in this
book. TCP-style guaranteed delivery is precisely wrong for live media; the
right contract is bounded-staleness delivery, where the queue's job is to
decide *what to skip*, not what to hold. The same logic sizes the playback
buffer: deep enough to hide jitter, shallow enough that an interrupt command
issued now reaches silence quickly — buffer depth is interruption latency
waiting to happen.

## Speech is a chain of clocks

Automatic speech recognition is not merely a file-to-text model placed before
an LLM. A streaming recognizer emits partial hypotheses that may be revised as
more audio arrives. Endpoint detection decides when a phrase is complete. If
the endpoint waits too long, the language model starts late; if it fires too
early, the prompt is incomplete.

Partial hypotheses create a hidden coupling back into Part III: every token
the LLM prefills against a *stale* hypothesis is work that a revision
invalidates. If the recognizer revises "book a table for two" into "book a
table for twelve," the KV blocks computed past the revision point are wrong —
Chapter 16's prefix reuse, running in miniature, once per hypothesis update.
A well-built pipeline therefore gates LLM starts on hypothesis stability (or
prefills only the stable prefix), accepting some start latency to avoid paying
repeated invalidated prefills.

Google's work on
[joint endpointing and decoding](https://research.google/pubs/joint-endpointing-and-decoding-with-end-to-end-models/)
frames this as a quality-latency trade rather than a preprocessing detail.

Record three ASR times separately: audio arrival to first partial text, end of
speech to stable transcript, and the age of audio at every emitted hypothesis.
The last one catches a system that produces frequent updates from an
ever-growing backlog. The distinction is visible in G's timeline: the partial
arriving at 0.8 s describing speech spoken since 0.6 s has an audio age of
about 200 ms — live, useful. A partial arriving at 3.0 s that still describes
audio from 1.0 s has an age of two seconds: the recognizer looks productive
(three updates!) while actually narrating the past. Downstream stages that
trust recency will act on stale text; the age metric is what makes the backlog
visible before that happens.

Text-to-speech has an equally important split. Time to first audio measures
startup, while real-time factor compares synthesis duration with audio
duration:

```text
real-time factor = synthesis time / generated audio duration
```

A factor below one is necessary for sustained streaming, but not sufficient
for a good conversation. The first chunk can still arrive late, chunk
boundaries can click, or the playback buffer can grow until speech no longer
matches the current turn. Measure first-audio latency, chunk cadence, underruns,
buffer depth, and interruption-to-silence.

Suppose ASR stabilization takes 180 ms, LLM first token takes 220 ms, TTS first
audio takes 140 ms, and network plus buffering uses 90 ms. The total is 630 ms.
A 20 percent faster decoder saves 44 ms, while better endpointing that saves
100 ms has more than twice the product impact. Stage budgets turn an attractive
kernel result into the correct system priority — and note where the 180 ms of
endpointing sits: it is *waiting*, not compute, which is why it yields to
better decisions rather than better hardware.

End-to-end streaming models change the component boundaries but not the need
for explicit latency and quality measures. Meta's
[SeamlessStreaming research](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
is one example of evaluating streaming speech generation with latency,
robustness, and perceptual criteria together.

## Interruption is a first-class transition

When the user begins speaking over the assistant, the service may stop audio
playback, cancel text-to-speech, stop future language scheduling, and decide how
much of the generated but unheard text belongs in conversation history. Even
*detecting* the interruption is a pipeline decision: a voice-activity detector
watching the input while TTS plays must distinguish a real barge-in from the
assistant's own voice arriving through the microphone — echo cancellation does
most of the work on device, and where it cannot, deployments fall back to
half-duplex politeness or require the new speech to exceed an energy threshold
for some duration. Every one of those mitigations adds detection latency to
the interruption-to-silence budget before any cancellation even starts.

**Interruption advances the session generation and fences late work.**

```blockdiag
flowchart LR
    G7["Turn generation 7"] --> O["Text and audio in flight"]
    I["User interruption"] --> G8["Advance to generation 8"]
    G8 --> C["Cancel scheduling, synthesis, and playback"]
    O --> X{"Event generation current?"}
    X -->|No| D["Discard late event"]
    X -->|Yes| P["Deliver event"]
```

| User-visible interval | Begins | Ends | Primary owner |
| --- | --- | --- | --- |
| end-of-turn to text | stable endpoint | first useful token | ASR and LLM path |
| end-of-turn to audio | stable endpoint | audible first sample | full speech pipeline |
| interruption to silence | new speech detected | playback stopped | session controller |
| stream freshness | media arrival | processing or drop | deadline-aware queues |


Those actions cannot happen atomically across several services. Use a session
generation number or turn ID. Events from an old generation are ignored after
the interruption advances the session.

Cancellation should flow upstream quickly, but components must also clean up
late completions. A tool call may finish after the turn has been abandoned. Its
result should not silently enter a new turn.

### Fencing with generations

The generation counter is the chapter's central data structure, and it is
worth spelling out the contract each component signs. On advancing from
generation 7 to 8, the session controller bumps the number *first*; every
other action follows from components observing it. Language-model scheduling
checks the generation before admitting the next step's work. The TTS service
checks before synthesizing each chunk. Playback checks before rendering each
buffer. Tool results carry the generation they were requested under, and the
history-commit rule consults it: only text actually heard — delivered to
playback under the then-current generation — enters the visible conversation;
generated-but-unheard text is recorded as diagnostic state, never silently
treated as spoken.

Readers have met this pattern twice before. It is Chapter 17's membership
epoch (a stale health report self-invalidates) and Chapter 20's policy version
(a trajectory stamped with an old version cannot contaminate a new round).
Real-time interruption adds only one twist: the fence must be checked at
*consumption* time, not just dispatch time, because the consumer — the user's
ear — is the one component that cannot be rolled back. Five hundred
milliseconds of stale audio played after an interrupt is a worse failure than
five hundred milliseconds of compute wasted generating it, which is why the
interruption-to-silence budget in G's worked timeline (100 ms) is the tightest
number in the chapter.

The cancellation order follows from each consumer's granularity. Bumping the
generation is microseconds and must go first — it is the only step that makes
every later race benign. Stopping playback is next and is bounded by buffer
depth, which is why the age-vs-size dive called depth "interruption latency
waiting to happen." Cancelling TTS takes effect at chunk boundaries; the
in-flight chunk is wasted but fenced. Cancelling language-model scheduling
takes effect at *engine-step* boundaries — at the Atlas decode cadence, up to
one step's worth of batch compute continues before the check runs, which is
the worst-case wasted compute and is bounded and acceptable. Tool calls are
the long pole: they may run for seconds, cannot be revoked, and are handled
purely by fencing at completion — their results become diagnostics if the
generation moved. The pattern: cancel where granularity is fine, fence where
it is not, and always bump the fence first.

## Session state needs an owner

A live session can own audio buffers, encoder state, conversation tokens, KV
blocks, parser state, tool calls, and output already sent but not yet played.

Keeping all state on one worker simplifies consistency and creates affinity.
Worker loss then ends or reconstructs the session. Externalizing selected state
supports migration at the cost of serialization and latency.

Classify state by recovery value. Conversation text is compact and easy to
store. KV state is larger but saves prefill — at the Atlas constant of
320 KiB per token, ten minutes of dense conversation accumulates KV that
dwarfs its own transcript by orders of magnitude, which is why externalizing
KV wholesale is rarely worth it and checkpointing *prefix boundaries* often
is. A causal video model's recurrent state may be both large and expensive to
reconstruct. Checkpoint frequency should follow the product's recovery
requirement: how many seconds of conversation is a worker loss allowed to
cost?

Migration, when chosen, inherits Chapter 17's sticky-session escape rules —
affinity is a preference evaluated per turn, never a requirement the user's
latency pays for.

Checkpoint frequency has its own arithmetic. If the product tolerates losing R
seconds of conversation, checkpointing roughly every R/2 bounds the expected
loss near R/4 to R/2 while amortizing the write. The cost per checkpoint is
state size over write bandwidth — trivial for the transcript's kilobytes,
material for recurrent media state — so the checkpoint *cadence* should differ
per state class: conversation text at turn boundaries, KV prefix anchors at
stable endpoints only, media state only if the recovery contract demands it.
Checkpointing everything at one cadence either wastes bandwidth on cheap state
or under-protects expensive state. In-flight tool calls and parser state need
an owner decision too: if a worker dies holding a pending tool invocation, the
recovering session must either re-issue it — needing a stable request ID so the
tool can suppress duplicates, per Chapter 17 — or leave it abandoned and let
the turn's fence mark its eventual completion as diagnostic. Silence in the
meantime must be bounded by a timeout, because a hung tool otherwise becomes a
session that never responds again.

## Reasoning changes the workload shape

A reasoning model may emit an internal reasoning stream before its visible
answer, use a parser to separate the two, and vary that work by orders of
magnitude across superficially similar prompts. Admission therefore needs a
budget for total generated work, not merely visible answer length. Traces must
record reasoning and answer tokens separately so that a product cannot appear
to reduce latency by hiding work from its accounting.

**An agentic turn alternates active generation with suspended work.**

```blockdiag
flowchart LR
    U["User turn"] --> R["Reasoning generation"]
    R --> P["Reasoning and tool parser"]
    P -->|Need evidence| T["Tool call"]
    T --> W["Suspended request"]
    W -->|Result and valid fence| R
    P -->|Ready| A["Visible answer"]
```

Chapter 22 owns the parser and wire contract. This chapter owns what that
contract does to scheduling: reasoning tokens consume decode slots, parser
state must survive interruption, and a disabled or shortened reasoning mode is
a quality-tier decision rather than a free speed switch. The current
[SGLang reasoning-parser documentation](https://docs.sglang.ai/advanced_features/separate_reasoning.html)
shows the explicit separation, while the
[vLLM feature index](https://docs.vllm.ai/en/latest/features/)
tracks the serving controls that interact with it.

Compressing or discarding reasoning history can reduce the next turn's prefill
and KV footprint, but it changes model input and may change answer quality.
Treat the summary algorithm and version as cache identity, evaluate the quality
trade, and apply Chapter 26's retention rules to both the original reasoning
and its summary.

## Tool gaps create suspended requests

While a tool runs, retaining all KV blocks buys a fast resume and charges scarce
memory to idle wall time. Releasing them saves capacity and later pays a prefill
or restore cost. Make that choice from expected tool latency, state size,
available cache tiers, and the turn's remaining deadline. A short database read
may justify retention; a human-approval step usually does not.

Suspension needs a stable request identity, an expiry time, and a generation
fence. When the tool returns, the router must find the session owner or restore
its state, then verify that the turn is still current before resuming. Tool
execution also needs its own idempotency key: retrying a generation is safe only
when it cannot repeat an external action. Late results become diagnostic data,
not new model input.

## Graceful degradation is a scheduler policy

During overload, a real-time service cannot let queues grow indefinitely.
Possible responses include lowering video resolution or frame rate, using a
smaller model, shortening speculative lookahead, skipping optional tools,
reducing generated speech, or rejecting a new session.

Choose degradation in product terms, and order the ladder before overload
arrives. Dropping every other frame may preserve a conversation; allowing two
seconds of stale audio may destroy it — perceptually, staleness is worse than
poverty. A defensible ladder spends cheap quality first:

| Tier | Action | User-visible effect | Reversible? |
| --- | --- | --- | --- |
| 1 | shorten speculative lookahead | slightly slower reactions | instantly |
| 2 | skip optional tools and enrichment | plainer answers | per turn |
| 3 | reduce speech rate or verbosity | terser assistant | per turn |
| 4 | downshift video resolution or frame rate | softer video | seconds |
| 5 | reject new sessions; protect established ones | busy signal for some users | on capacity |

Each tier is a *scheduler policy* with its own trigger metric — queue age,
deadline-miss rate, session count — not an operator's manual intervention.
Tier 1 refunds slack immediately, as Chapter 19's sessions do; tier 5 is last
because it converts overload into unavailability. The scheduler needs deadlines
and quality tiers to execute
any of this mechanically rather than as ad-hoc throttles.

Fairness also changes. A long-lived session should not own a permanent batch
slot while silent. An active speaker may deserve temporary priority. Use quotas
so one tenant's continuous sessions cannot exclude new users.

## Observe a conversation, not isolated calls

Per-request TTFT does not capture a voice conversation. Measure end-of-turn to
first audio, interruption-to-silence, gap and overlap duration, stale event
rate, tool delay, session recovery, and perceptual quality. Gap and overlap
deserve definition because they come from the same annotation and point in
opposite directions: *gap* is silence between the user finishing and the
assistant starting (or between turns) — awkwardness the user feels as
hesitation; *overlap* is speech during assistant audio — either barge-in (fine,
the interruption machinery exists for it) or the assistant still talking over
a finished user turn (never fine). Overlap rate that survives the fence is a
direct quality regression; gap percentiles are where "it feels slow"
complaints live even when every stage met its budget.

Trace events with session ID, turn generation, component timestamp, and clock
source. Distributed clocks are imperfect, so include monotonic stage durations
and propagate trace context. Session recovery is itself a measured event with
two distinct shapes — resume onto a warm worker versus restart from history —
and users experience them differently, so track recovery rate and recovery
latency per shape rather than blended. The stale-event rate deserves emphasis because it
is a leading indicator: events discarded by the generation fence are the
system reporting, quantitatively, how often cancellation lost the race — a
rate that climbs before users start complaining about assistants that talk
over them.

## Worked example: a late audio chunk

One coherent reading of G's timeline: the user stops speaking at 2.4 seconds;
partials had arrived at 0.8, 1.5, and 2.2. Endpointing stabilizes at 2.6, the
LLM routes and prefills by 2.8, emits a tool call at 3.0, a holding phrase
covers the tool wait, and response audio begins at 3.8 — an end-of-turn-to-
first-audio span of 1.4 s, decomposed in G as 200 ms endpointing, 200 ms
language startup, 600 ms tool time overlapped with the holding phrase, and
200 ms TTS plus buffer. The holding phrase is itself a mechanism worth naming:
a short spoken response ("One moment —") synthesized immediately so the user
hears acknowledgment while the tool runs, at the cost of a second, tiny TTS
job racing the real one through the same synthesis pool. At 5.1 the user interrupts, advancing the turn
generation from 7 to 8; playback becomes silent by 5.18 — the 80 ms inside
the 100 ms interruption budget. A generation-7 audio chunk arrives at 5.4 and
is discarded by the fence, exactly as designed.

Two budgets governed different halves of those ten seconds, and they fail
differently. Missing the 1.4-second response budget feels slow; missing the
100-ms silence budget feels broken — the assistant talking over the user is
the canonical real-time failure. The counterfactual shows why the fence earns
its complexity: without it, the chunk arriving at 5.4 plays after silence was
restored at 5.18, exposing the user to 220 milliseconds or more of stale
speech resuming mid-word — precisely the failure users describe as "it wouldn't
stop talking." The generation number makes distributed cancellation coherent
across models that cannot share a lock; every event additionally carries a
session ID, stream sequence number, and deadline, so ordering, duplication,
and staleness are all independently detectable. And the discard at 5.4 is the
system *working*: generated-but-unheard text was never committed as something
the user heard.

## Practice: specify a ten-second protocol trace

Draw partial ASR, endpointing, LLM prefill, tool execution, TTS, playback, and
interruption on one timeline. Budget end-of-turn to first audio and interruption
to silence separately. Delay the tool and deliver a stale audio event after
cancellation.

Write ordering, generation, backpressure, history-commit, and cleanup rules.
The complete worked timeline is in
[Appendix G](../appendices/g-worked-solutions.md#21-ten-second-conversation).

Parts I through IV have focused on execution mechanisms. Part V turns them into
a production contract: APIs, experiments, operations, economics, and security.
