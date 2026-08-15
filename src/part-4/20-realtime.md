# 20. Real-Time and Interactive Systems

A voice assistant listens while the user speaks, begins responding, calls a
tool, and stops mid-sentence when the user interrupts. There is no clean request
followed by a clean response. Input and output overlap inside a long-lived
session.

Real-time inference is defined by deadlines and interruption, not merely low
average latency.

## Build the latency budget backward

Suppose the product allows 700 milliseconds from the end of a user's phrase to
the beginning of audible speech. That budget includes network transport, speech
recognition, endpoint detection, language-model queueing and prefill, first
tokens, text-to-speech startup, and audio buffering.

Improving language TTFT from 250 to 200 milliseconds helps. It does not save a
system that spends 600 milliseconds detecting the end of speech.

Assign a budget to every stage and measure it at boundaries visible to the
user. Audio generated but waiting in a playback buffer has not reached the
user. A video frame completed after its display deadline may be useless even if
throughput is high.

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

Each arrow can stream. Buffers and queues need maximum ages, not only maximum
sizes. Old audio or frames may be dropped because processing them would delay
more current information.

Bidirectional transports such as WebSocket or streaming RPC carry events in
both directions. The protocol should define ordering, heartbeats, flow control,
reconnection, and which side owns the session. Transport keepalive is not the
same as model-worker health.

## Speech is a chain of clocks

Automatic speech recognition is not merely a file-to-text model placed before
an LLM. A streaming recognizer emits partial hypotheses that may be revised as
more audio arrives. Endpoint detection decides when a phrase is complete. If
the endpoint waits too long, the language model starts late; if it fires too
early, the prompt is incomplete.

Google's work on
[joint endpointing and decoding](https://research.google/pubs/joint-endpointing-and-decoding-with-end-to-end-models/)
frames this as a quality-latency trade rather than a preprocessing detail.

Record three ASR times separately: audio arrival to first partial text, end of
speech to stable transcript, and the age of audio at every emitted hypothesis.
The last one catches a system that produces frequent updates from an
ever-growing backlog.

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
kernel result into the correct system priority.

End-to-end streaming models change the component boundaries but not the need
for explicit latency and quality measures. Meta's
[SeamlessStreaming research](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
is one example of evaluating streaming speech generation with latency,
robustness, and perceptual criteria together.

## Interruption is a first-class transition

When the user begins speaking over the assistant, the service may stop audio
playback, cancel text-to-speech, stop future language scheduling, and decide how
much of the generated but unheard text belongs in conversation history.

Those actions cannot happen atomically across several services. Use a session
generation number or turn ID. Events from an old generation are ignored after
the interruption advances the session.

Cancellation should flow upstream quickly, but components must also clean up
late completions. A tool call may finish after the turn has been abandoned. Its
result should not silently enter a new turn.

## Session state needs an owner

A live session can own audio buffers, encoder state, conversation tokens, KV
blocks, parser state, tool calls, and output already sent but not yet played.

Keeping all state on one worker simplifies consistency and creates affinity.
Worker loss then ends or reconstructs the session. Externalizing selected state
supports migration at the cost of serialization and latency.

Classify state by recovery value. Conversation text is compact and easy to
store. KV state is larger but saves prefill. A causal video model's recurrent
state may be both large and expensive to reconstruct. Checkpoint frequency
should follow the product's recovery requirement.

## Graceful degradation is a scheduler policy

During overload, a real-time service cannot let queues grow indefinitely.
Possible responses include lowering video resolution or frame rate, using a
smaller model, shortening speculative lookahead, skipping optional tools,
reducing generated speech, or rejecting a new session.

Choose degradation in product terms. Dropping every other frame may preserve a
conversation; allowing two seconds of stale audio may destroy it. The scheduler
needs deadlines and quality tiers to make the right trade.

Fairness also changes. A long-lived session should not own a permanent batch
slot while silent. An active speaker may deserve temporary priority. Use quotas
so one tenant's continuous sessions cannot exclude new users.

## Observe a conversation, not isolated calls

Per-request TTFT does not capture a voice conversation. Measure end-of-turn to
first audio, interruption-to-silence, gap and overlap duration, stale event
rate, tool delay, session recovery, and perceptual quality.

Trace events with session ID, turn generation, component timestamp, and clock
source. Distributed clocks are imperfect, so include monotonic stage durations
and propagate trace context.

## Worked example: a late audio chunk

The user stops speaking at 2.4 seconds. Endpointing stabilizes at 2.6, the LLM
emits a tool call at 3.0, a holding phrase covers the tool wait, and response
audio begins at 3.8. At 5.1 the user interrupts, advancing the turn generation
from 7 to 8; playback becomes silent by 5.18. A generation-7 audio chunk arrives
at 5.4 and is discarded.

The generation number makes distributed cancellation coherent. Every event also
needs a session ID, stream sequence number, and deadline. Generated but unheard
text is not silently committed as something the user heard.

## Practice: specify a ten-second protocol trace

Draw partial ASR, endpointing, LLM prefill, tool execution, TTS, playback, and
interruption on one timeline. Budget end-of-turn to first audio and interruption
to silence separately. Delay the tool and deliver a stale audio event after
cancellation.

Write ordering, generation, backpressure, history-commit, and cleanup rules.
The complete worked timeline is in
[Appendix G](../appendices/g-worked-solutions.md#20-ten-second-conversation).

Parts I through IV have focused on execution mechanisms. Part V turns them into
a production contract: APIs, experiments, operations, economics, and security.
