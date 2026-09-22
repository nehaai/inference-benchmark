"""
Frozen prompt set for the inference benchmark.

Covers a 2x2 grid so both bottlenecks show up in the data:
- input length  -> affects TTFT (prefill, compute-bound)
- output length -> affects total latency (decode, bandwidth-bound)

Do not edit these once you start running the real benchmark --
every provider needs to see the exact same prompts for the
comparison to be fair.
"""

LONG_ARTICLE = """
Artificial intelligence inference is the process of running a trained
model to generate outputs from new inputs, as opposed to training,
which is the process of learning the model's parameters from data.
Inference systems must balance latency, throughput, and cost, and the
techniques used to do this -- batching, caching, quantization,
parallelism -- all involve trade-offs between these three goals. As
models have grown larger and context windows longer, the engineering
challenges around serving them efficiently have become a discipline
in their own right, distinct from the research work of designing and
training the models themselves. Understanding where time is spent in
a single forward pass, how memory bandwidth constrains decoding speed,
and how scheduling decisions affect tail latency under load are all
necessary skills for anyone deploying these systems in production
rather than just prototyping with them locally.

A useful mental model for inference is to split a single request into
two distinct phases with very different performance characteristics.
The first phase, prefill, happens when the model processes the entire
input prompt at once. Because every token in the prompt can attend to
every other token in parallel, this phase makes heavy use of matrix
multiplication and is generally compute bound, meaning the bottleneck
is how many floating point operations the accelerator can perform per
second. The second phase, decode, happens one token at a time. Each
new token depends on everything generated before it, so the model
cannot parallelize across output tokens the way it can across input
tokens. Decode is instead bandwidth bound, meaning the bottleneck is
how quickly the accelerator can move the model's weights and the
growing key-value cache between memory and compute units for every
single incremental step. This asymmetry between phases is the root
cause of most of the interesting engineering problems in inference
serving. A request with a long prompt and a short expected answer
will spend most of its wall clock time in prefill, while a request
with a short prompt and a long expected answer will spend most of
its time in decode, and a system tuned for one pattern will often
perform poorly on the other.

Batching is one of the primary levers operators have for improving
throughput without proportionally increasing latency. Naive batching
groups a fixed set of requests together and processes them in lock
step, but this wastes capacity whenever requests in the batch finish
generating at different lengths, since the whole batch is often held
until the longest request completes. Continuous batching addresses
this by allowing the scheduler to add new requests and remove
completed ones at every decoding step, rather than waiting for an
entire batch to finish before admitting new work. This keeps the
accelerator closer to fully utilized and reduces the average waiting
time for tail requests, at the cost of a more complex scheduler that
must interleave prefill and decode work fairly. A poorly designed
scheduler can allow a single very long prefill to stall many
in-flight decode requests, which is why techniques like chunked
prefill exist: rather than running an entire long prompt through in
one uninterrupted burst, the scheduler breaks it into smaller pieces
and interleaves them with pending decode steps from other requests,
smoothing out latency for everyone sharing the same accelerator.

Memory is the other central constraint in serving large models
efficiently. Every request in flight requires storing a key-value
cache, sometimes shortened to KV cache, which holds intermediate
attention state for every token processed so far so that the model
does not need to recompute attention over the entire prior sequence
at every new decoding step. The size of this cache grows linearly
with sequence length and with the number of concurrent requests,
and on many accelerators it can consume more memory than the model's
own weights once context lengths and batch sizes grow large. This
is why prefix caching has become a common optimization: when many
requests share an identical prefix, such as a common system prompt
or a repeated set of instructions, the KV cache for that shared
portion can be computed once and reused across requests rather than
recomputed from scratch every time. For workloads with a lot of
prompt repetition, such as a customer support agent that reuses the
same instructions across thousands of conversations, prefix caching
can meaningfully reduce both latency and cost, since the compute
bound prefill phase for the shared portion is effectively skipped
on every request after the first.

Quantization is a complementary technique that reduces the
precision used to store and compute with model weights, commonly
moving from sixteen bit floating point down to eight bit integer or
four bit formats. Because decode is bandwidth bound, reducing the
number of bytes that need to move between memory and compute for
every token directly increases how many tokens per second a system
can produce, independent of any change in raw compute capability.
The tradeoff is a potential loss in output quality, and the size of
that loss depends heavily on which quantization scheme is used and
how carefully it is calibrated against the specific model. Some
quantization methods target only the weights while leaving
activations at higher precision, while more aggressive methods
quantize both, and the right choice for a given deployment usually
requires empirical testing on the specific downstream task rather
than trusting published benchmarks from a different domain.

Parallelism strategies address a different problem: models that are
too large to fit on a single accelerator at all. Tensor parallelism
splits individual matrix operations across multiple devices, so
that a single forward pass is computed cooperatively rather than on
one chip, which reduces latency for a single request but requires
fast interconnects between devices since intermediate results must
be exchanged at every layer. Pipeline parallelism instead splits the
model by layer, assigning different layers to different devices and
passing activations along a pipeline, which can improve throughput
for many concurrent requests but tends to increase latency for any
single request because of the sequential handoffs between stages.
Expert parallelism, relevant for mixture of experts architectures,
routes different tokens to different subsets of the model's
parameters that live on different devices, which can increase
effective model capacity without a proportional increase in compute
per token, but introduces its own load balancing challenges when
certain experts are used far more often than others.

At the systems level, operators must also decide between running
their own inference infrastructure, commonly described as bring
your own compute, or relying on a managed inference provider that
handles hardware provisioning, scaling, and much of the low level
optimization work. Running your own infrastructure offers more
control over cost at scale and the ability to apply very specific
optimizations for a known workload, but requires meaningful
investment in systems engineering, capacity planning, and ongoing
operational maintenance. Managed providers trade some of that
control for convenience, faster time to production, and access to
optimizations the provider has already built, though pricing and
available models vary considerably between providers and workloads
that are unusual in their input or output length distribution may
not always be served optimally by a general purpose managed system
tuned for typical traffic patterns.

Finally, no discussion of inference systems is complete without
acknowledging that the metrics used to evaluate them are easy to
report misleadingly. Average latency numbers can hide severe tail
behavior that only shows up under real concurrent load, throughput
figures measured at low concurrency say little about how a system
degrades as traffic scales, and cost per token comparisons that
ignore quality differences between models are not comparing like
with like. A metric increasingly favored by practitioners is
goodput, defined as the fraction of requests that meet a specified
service level objective for both latency and quality, since it
forces an explicit tradeoff to be made rather than optimizing a
single number in isolation. Any serious evaluation of an inference
system, whether a managed provider or a self hosted deployment,
should be conducted under realistic concurrency and realistic
prompt distributions, because performance characteristics measured
in isolation frequently fail to predict performance under the
conditions that actually matter in production.
""".strip()

PROMPTS = [
    # short input, short output
    {
        "id": "short_short_1",
        "category": "short_input_short_output",
        "prompt": "What's the capital of France?",
    },
    {
        "id": "short_short_2",
        "category": "short_input_short_output",
        "prompt": "Define recursion in one sentence.",
    },
    # short input, long output
    {
        "id": "short_long_1",
        "category": "short_input_long_output",
        "prompt": "Write a 400-word short story about a lighthouse keeper who receives a strange radio signal.",
    },
    {
        "id": "short_long_2",
        "category": "short_input_long_output",
        "prompt": "Explain how continuous batching works in LLM inference, in about 400 words.",
    },
    # long input, short output
    {
        "id": "long_short_1",
        "category": "long_input_short_output",
        "prompt": f"{LONG_ARTICLE}\n\nSummarize the above in one sentence.",
    },
    {
        "id": "long_short_2",
        "category": "long_input_short_output",
        "prompt": f"{LONG_ARTICLE}\n\nWhat is the main tension described in the passage above? Answer in one sentence.",
    },
    # long input, long output
    {
        "id": "long_long_1",
        "category": "long_input_long_output",
        "prompt": f"{LONG_ARTICLE}\n\nWrite a detailed 400-word analysis of the trade-offs described above, with examples.",
    },
    {
        "id": "long_long_2",
        "category": "long_input_long_output",
        "prompt": f"{LONG_ARTICLE}\n\nRewrite the passage above as a 400-word explainer aimed at someone with no technical background.",
    },
]

if __name__ == "__main__":
    # quick sanity check when you run this file directly
    for p in PROMPTS:
        print(f"{p['id']:15s} {p['category']:25s} len={len(p['prompt'])} chars")