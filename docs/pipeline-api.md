# Pipeline API

`Pipeline` is the low-level tensor and stage execution API. Use `WorldModel`
for raw text, image, video, and action generation.

## Load a package

```python
from onnx_world_model import Pipeline

pipeline = Pipeline("output/cosmos3-edge")

print(pipeline.profile)
print(pipeline.inputs)
print(pipeline.outputs)
print(pipeline.stages)
print(pipeline.metadata)
```

All component sessions are loaded eagerly. `Pipeline` owns immutable model
sessions and can be shared; each `PipelineSession` owns one request or
trajectory's state.

## Concurrency limits

A `Pipeline` can cap how many executions run through it at once. This is
*admission scheduling*: it decides how many executions are inside the runtime
and in what order queued ones enter. It is **not** batching — nothing is
merged, split, reordered, or preempted.

```python
pipeline = Pipeline(
    "output/cosmos3-edge",
    max_concurrent_executions=4,
    max_concurrent_by_stage_kind={"iterative": 2},
)

print(pipeline.max_concurrent_executions)        # 4
print(dict(pipeline.max_concurrent_by_stage_kind))  # {'iterative': 2}
```

`WorldModel` takes the same two keyword arguments and forwards them.
`OnnxModel` and `LatentDynamicsModel` do not: they are single graphs, not
pipelines.

- `0`, and any stage kind left out of the mapping, means unlimited. That is
  the default, and it is exactly what every earlier release did.
- An execution is admitted only when there is room under both the non-zero
  global cap and the non-zero cap for its own stage kind.
- Keys must name a stage kind the runtime executes: `single_pass`,
  `autoregressive`, `iterative`, `state_transition`, `composite`, or
  `on_demand`. An unknown or empty key raises `WorldModelError` when the
  pipeline is constructed, rather than silently never applying. A count must
  be a plain non-negative `int`; `True` raises `TypeError` and `-1` raises
  `ValueError`.

**What counts as one execution.** Exactly four calls are executions, and each
takes exactly one permit for its whole duration:

| Call | Permits |
|---|---|
| `run_stage()` | one, held for the whole stage |
| `step_stage()` | one, held for that pass |
| `StageRun.step()` | one, held only during the call |
| `StageRun.finish()` with steps left to drain | one, held for the whole drain |
| `begin_stage()` | none — it resolves a plan and executes nothing |
| `StageRun.finish()` on a completed run | none — it returns the cached result |
| An idle `StageRun` between `step()` calls | none |
| `close()`, `request_cancellation()`, `outputs`, `state()`, `snapshot()`, `restore()`, `fork()`, the checkpoint methods | none |

The rules the API guarantees:

- **Shared by the pipeline, not by the session.** The cap belongs to the
  `Pipeline`. Every session it creates, every `fork()` of those sessions, and
  every `StageRun` they produce compete for the same permits. Two separately
  constructed `Pipeline` objects — even over the same package directory — get
  independent caps.
- **Fair within a kind, non-blocking across kinds.** Queued requests are
  admitted oldest first. A stage kind that is at its own cap is skipped rather
  than blocking the head of the queue, so a different, eligible kind still
  enters; within one kind, no later request passes an earlier one.
- **Work conserving.** A freed permit immediately admits the oldest eligible
  waiter. Nothing waits while capacity is idle.
- **Queued work is still cancellable.** A request waiting for a permit keeps
  observing its `cancellation` token and `timeout`. Cancelling it raises
  `CancelledError`, an expired deadline raises `DeadlineExceededError`, and in
  both cases the stage never started, so nothing was applied and the queue
  position is released.
- **Permits are never leaked.** A failed component, a cancelled step, and an
  exception on the way out all return the permit, because it is released by
  scope exit rather than by a success path.
- **Admission comes before the session.** A request is admitted before it
  takes the session lock, so a queued call can still find the session busy
  with another `StageRun` once it is let in and raise `WorldModelError` then.
  That ordering is deliberate: checking the session before waiting would only
  read staler state.

**Not included: batching.** There is no continuous batching, dynamic batching,
request merging or splitting, priority, or preemption. Adding batching would
need machinery this runtime does not have — request compatibility keys, tensor
concatenation and result splitting across lane boundaries, per-lane recurrent
state and RNG streams, and a KV-cache manager that can admit and evict lanes
in the middle of a stage.

The C++ equivalent is `PipelineSchedulingOptions`, passed to the `Pipeline`
constructor or as the third argument of `Pipeline::Load`. Both are defaulted,
so existing code keeps compiling and keeps its unlimited behavior; the
`Pipeline` layout changed, so C++ consumers must be recompiled.

## Scheduling stats

`Pipeline.scheduling_stats` reads the admission scheduler right now. It is
operational observability for admission only — how much is admitted and how
much is waiting. It reports no timings and no throughput, and it is not a
profiler.

```python
pipeline = Pipeline("output/cosmos3-edge", max_concurrent_executions=4)

stats = pipeline.scheduling_stats
print(stats.active_executions)                      # 0
print(stats.queued_executions)                      # 0
print(stats.active_by_stage_kind["iterative"])      # 0
print(stats.queued_by_stage_kind["autoregressive"])  # 0
```

`PipelineSchedulingStats` is a frozen dataclass, and both mappings are
read-only views:

| Field | Meaning |
|---|---|
| `active_executions` | executions holding a permit right now |
| `queued_executions` | executions waiting for a permit right now |
| `active_by_stage_kind` | the active count broken down by stage kind |
| `queued_by_stage_kind` | the waiting count broken down by stage kind |

- **Every stage kind is always present.** Both mappings contain all six
  executable kinds — `single_pass`, `autoregressive`, `iterative`,
  `state_transition`, `composite`, and `on_demand` — so a kind can be read
  without testing for its key. A kind with nothing happening maps to `0`.
- **These are permit counts, not execution counts.** A stage kind capped by
  neither the global limit nor its own is admitted without taking a permit at
  all, so an unlimited pipeline reports zeros no matter how much work is
  inside it. Under a global limit every admitted execution is counted, in the
  total and against its own stage kind, whether or not that kind has a cap of
  its own.
- **It is a value, not a live view.** The whole reading is taken at one moment
  under the scheduler's own lock, so its fields agree with each other, and it
  never changes afterwards. Read it again to see a newer moment. Any field may
  already be stale by the time it is inspected, because other threads keep
  running.
- **Reading changes nothing.** It admits nothing, queues nothing, takes no
  session lock, and cannot block an execution.
- **Copies share one scheduler,** so copies of a pipeline report the same
  numbers, and every session and `StageRun` created from any of them is
  counted.

The C++ equivalent is `Pipeline::scheduling_stats()`, which returns the same
fields as a `PipelineSchedulingStats` value:

```cpp
const PipelineSchedulingStats stats = pipeline.scheduling_stats();
if (stats.queued_executions > stats.active_executions) {
  // Admission, not the model, is the bottleneck right now.
}
```

## Telemetry

Telemetry is opt-in and observability only: it changes what is measured and
never what is executed, in what order, or with what result. It is off by
default, and off means the pipeline holds no collector at all rather than one
that ignores everything, so an unmeasured pipeline pays one pointer test per
instrumentation site. Counters are described here; the optional per-run ONNX
Runtime node traces layered on top of them are in
[ONNX Runtime node traces](#onnx-runtime-node-traces).

```python
from onnx_world_model import Pipeline

pipeline = Pipeline("output/cosmos3-edge", enable_telemetry=True)
session = pipeline.create_session()
session.run_stage("reasoner_prompt", {"input_ids": input_ids})

telemetry = pipeline.telemetry_snapshot
component = telemetry.components["reasoner_embedding"]
print(component.successful_calls)                      # 1
print(component.input_bytes, component.output_bytes)   # exact byte totals
print(component.total_duration_ns)                     # nanoseconds

stage = telemetry.stages["reasoner_prompt"]
print(stage.successful_executions, stage.steps, stage.completions)  # 1 1 1

pipeline.reset_telemetry()          # counters restart, epoch advances
print(pipeline.telemetry_snapshot.epoch)  # 2
```

`WorldModel` forwards the same option and exposes the same two members:

```python
from onnx_world_model import WorldModel

model = WorldModel.from_pretrained("output/cosmos3-edge", enable_telemetry=True)
result = model.text.generate("Describe this scene.")
print(result.timings)                            # per-request wall clock
print(model.telemetry_snapshot.stages["reasoner_decode"].steps)  # tokens decoded
model.reset_telemetry()
```

The two are different measurements on purpose: `timings` is what one request
took, while `telemetry_snapshot` is what the runtime underneath did.

### What a reading contains

`PipelineTelemetrySnapshot` is a frozen dataclass; its three mappings are
read-only views and `traces` is a tuple.

| Field | Meaning |
|---|---|
| `enabled` | whether this pipeline collects at all |
| `epoch` | which collection epoch these counters belong to |
| `components` | `PipelineComponentStats` per manifest component name |
| `stages` | `PipelineStageStats` per manifest stage name |
| `admission_by_stage_kind` | `PipelineAdmissionStats` per stage kind |
| `transfers` | one pipeline-wide `PipelineTransferStats` |
| `traces` | this epoch's kept `PipelineTraceRecord` values; concurrent order is unspecified |
| `dropped_traces` | records past `max_trace_records`, whose files were still written |
| `failed_traces` | kept records whose trace file could not be identified; dropped calls are not scanned |

`PipelineComponentStats` — one call is one invocation of that component's
session from inside a stage execution:

| Field | Meaning |
|---|---|
| `successful_calls` | calls that returned outputs |
| `failed_calls` | calls that raised anything but a cancellation or deadline |
| `cancelled_calls` | calls that raised `CancelledError` |
| `deadline_exceeded_calls` | calls that raised `DeadlineExceededError` |
| `total_duration_ns` | nanoseconds inside the call, every attempt |
| `max_duration_ns` | the longest single call, in nanoseconds |
| `input_bytes` | bytes presented to the call, every attempt |
| `output_bytes` | bytes returned by a successful call |

`PipelineStageStats`:

| Field | Meaning |
|---|---|
| `successful_executions` | executions that finished |
| `failed_executions` | executions that raised anything but a cancellation or deadline |
| `cancelled_executions` | executions that raised `CancelledError` |
| `deadline_exceeded_executions` | executions that raised `DeadlineExceededError` |
| `steps` | non-terminal stage steps that completed |
| `completions` | terminal `completed` events |
| `total_execution_duration_ns` | nanoseconds executing, after admission |
| `max_execution_duration_ns` | the longest single execution, in nanoseconds |

`PipelineAdmissionStats`:

| Field | Meaning |
|---|---|
| `queued_acquisitions` | acquisitions that had to wait for a permit |
| `admitted_acquisitions` | acquisitions that received a permit |
| `cancelled_while_queued` | queued acquisitions released by a cancel |
| `deadline_while_queued` | queued acquisitions released by a deadline |
| `total_wait_ns` | nanoseconds spent in the queue, every outcome |
| `max_wait_ns` | the longest single queue wait, in nanoseconds |

`PipelineTransferStats`:

| Field | Meaning |
|---|---|
| `device_to_host_copies` | materializations the runtime performed |
| `device_to_host_bytes` | bytes those materializations moved |
| `component_input_bytes_device_resident` | presented bytes that still lived on a device |
| `component_input_bytes_host` | presented bytes that already lived on the host |

### How it counts

- **Durations are nanoseconds of wall-clock time,** and byte totals are exact
  tensor byte sizes. A duration covers every attempt whatever its outcome; a
  maximum never exceeds its own total.
- **An execution is one admission lease scope:** one `run_stage`, one
  `step_stage`, one `StageRun.step`, or one `StageRun.finish` that still has
  work to drain. `begin_stage`, an idle handle, and a `finish` that returns an
  already-completed run's cached outputs are not executions and change nothing.
- **Steps and completions measure stage progress, not API calls.** One
  `run_stage` of a three-step iterative stage is one execution, three steps,
  and exactly one completion, and stepping that same stage explicitly is four
  executions — the fourth produces the terminal event — with the same three
  steps and the same one completion. A direct `step_stage` bypasses the event
  state machine, so it counts one step and never a completion.
- **Outcomes are classified by error code, never by message,** so a
  cancellation and a deadline are their own counters rather than failures, and
  the four counters in each group are mutually exclusive.
- **Stage duration starts after the permit is granted,** so queue wait is
  reported once, by the admission counters, instead of being folded into the
  stage.
- **An unlimited stage kind records no admission at all.** It is admitted
  without taking a permit, so an unconfigured pipeline reports zeros there
  while its executions are still measured. An acquisition granted at once is
  admitted with a zero wait and is not counted as queued; only one that had to
  wait is. A grant that raced a cancellation counts once, as admitted.
- **An execution stopped while it was still queued never became an
  execution.** It is reported by `admission_by_stage_kind` and does not appear
  in `stages`.
- **A reading is a detached value,** but deliberately not one atomic instant:
  each counter is copied individually from a running system, because the
  alternative is a lock on the execution path. Read it again for a newer
  moment.
- **`reset_telemetry()` starts a new epoch rather than editing counters.** An
  execution already running finishes into the epoch it started in, so its
  remaining counts land in the previous epoch and are absent from the new one.
- **Copies share one collector,** so copies of a pipeline report the same
  counters, and every session, forked session, and `StageRun` created from any
  of them is counted. A separately constructed pipeline gets its own.
- **A disabled pipeline reads as `enabled=False`,** epoch `0`, and three empty
  mappings — nothing was collected, which is different from nothing having
  happened. An enabled reading always carries every manifest component, every
  manifest stage, and all six stage kinds, so a key is read without testing for
  it.

### What telemetry does not report

- **Node timings inside a trace.** A trace file is discovered and pointed at,
  never parsed: see [ONNX Runtime node traces](#onnx-runtime-node-traces)
  below. Parsing one on the execution path would cost more than the work it
  measures.
- **Execution-provider peak or current memory.** Only the provider knows it,
  and this milestone deliberately defers it rather than estimating it.
- **Host-to-device bytes.** Uploads happen inside ONNX Runtime when it binds an
  input, so this runtime cannot measure them exactly and refuses to guess.
  `component_input_bytes_device_resident` and `component_input_bytes_host` are
  presentation, not transfer: they say where each component's inputs already
  lived, and a tensor presented twice is counted twice.
- **Copies ONNX Runtime makes below the session.** `device_to_host_copies`
  counts what the session materializes itself. When ONNX Runtime stages a
  foreign device buffer to the host while binding an input, that copy happens
  inside the backend and is not counted here.
- **Aggregation.** There is no sampling, no window, no percentile, and no
  metrics export. A snapshot is a set of cumulative counters for one epoch
  plus a bounded list of trace records; anything above that belongs to the
  caller.

The C++ equivalent is `Pipeline::telemetry_snapshot()` and
`Pipeline::ResetTelemetry()`:

```cpp
PipelineTelemetryOptions telemetry;
telemetry.enabled = true;
Pipeline pipeline = Pipeline::Load("output/cosmos3-edge", {}, {}, {}, telemetry);

const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
const PipelineStageStats& stage = snapshot.stages.at("reasoner_decode");
if (stage.steps > 0) {
  const std::uint64_t average_ns =
      stage.total_execution_duration_ns / stage.steps;
  // Nanoseconds per decoded token, this epoch.
}
pipeline.ResetTelemetry();
```

## ONNX Runtime node traces

Node-level traces are a second, narrower opt-in on top of telemetry. With a
`telemetry_trace_directory` set, every component call additionally asks ONNX
Runtime to profile *that one call* and write its trace there. Profiling is per
run, never per session: nothing is rebuilt, a session that is also serving an
untraced call is unaffected, and no `SessionOptions` are changed.

```python
from onnx_world_model import Pipeline

pipeline = Pipeline(
    "output/cosmos3-edge",
    enable_telemetry=True,            # required: tracing is layered on top
    telemetry_trace_directory="traces",
    max_trace_records=256,            # in-memory records, not files
)
session = pipeline.create_session()
session.run_stage("reasoner_prompt", {"input_ids": input_ids})

for record in pipeline.telemetry_snapshot.traces:
    print(record.component, record.outcome, record.size_bytes, record.path)
```

`WorldModel` forwards both arguments unchanged.

Each traced call publishes exactly one `PipelineTraceRecord`:

| Field | Meaning |
|---|---|
| `epoch` | the collection epoch this call recorded into |
| `trace_id` | process-wide identifier, monotonic across collectors and epochs, in start order |
| `component` | the manifest component whose call this was |
| `path` | the trace file as a `pathlib.Path`, or `None` when `profiling_failed` |
| `outcome` | `success`, `failure`, `cancelled`, or `deadline_exceeded` |
| `duration_ns` | wall-clock nanoseconds of the component call |
| `size_bytes` | size of that file, `0` when `profiling_failed` |
| `profiling_failed` | whether no usable trace file could be identified |

### How tracing behaves

- **Configuration is checked while the pipeline loads.** A trace directory
  without `enable_telemetry` raises `WorldModelError`, a `max_trace_records`
  that is not an `int` greater than zero is rejected, and a path that exists
  as something other than a directory fails there too. A directory that does
  not exist yet is created. None of this can become a per-run failure.
- **Each call gets a unique prefix** containing its component, epoch, a
  process-lifetime nonce, process-wide trace ID, and pid inside the configured
  directory. The component name is reduced to ASCII letters, digits, dot,
  underscore, and hyphen. Independent pipelines sharing a directory,
  concurrent calls, epochs, and process lifetimes therefore never collide.
- **ONNX Runtime names the file, not this runtime.** It appends its own local
  timestamp and `.json` to the prefix, so the file is found by listing the
  directory for that prefix rather than by predicting a name.
- **A trace is discovered, never parsed.** The runtime records the path and
  the size and stops there; reading node timings is the caller's job, because
  parsing a trace on the execution path would cost more than the work it
  measures.
- **Profiling never changes the call.** Zero matching files, several matching
  files, an empty file, or a filesystem error is a *profiling* failure: it
  increments `failed_traces`, records an empty C++ path (`None` in Python), and leaves the model call
  and its outcome exactly as they were. A call that failed or was cancelled is
  still recorded, with its own outcome, because that is usually the
  interesting trace.
- **Concurrent record order is unspecified.** File discovery happens outside
  the record mutex; sorting by process-wide `trace_id` recovers call start
  order.
- **The cap bounds memory, not profiling.** Once `max_trace_records` records
  are kept or reserved, later calls still profile and still write files but
  skip directory discovery; `dropped_traces` counts those calls. A profiling
  failure among dropped calls is intentionally unknown and is not included in
  `failed_traces`.
- **`reset_telemetry()` clears records and deletes nothing.** Trace files stay
  exactly where ONNX Runtime wrote them; this runtime never removes one, so
  disk usage is the caller's to manage.
- **An early ONNX Runtime failure can leave an empty file.** ONNX Runtime
  creates the file when the run starts, so a run that fails before writing any
  event leaves a zero-byte file, which is reported as a profiling failure.
- **ONNX Runtime's own limits still apply.** A profiled session stops
  recording after roughly one million events, and a provider that internally
  replays a graph — a CUDA graph capture retry, for instance — can put more
  than one execution of the same node in one file. Both are ONNX Runtime
  behaviours; this runtime reports the file it was given.

The same surface exists on the generic model API, where the prefix is supplied
per call and the result is still just the outputs:

```python
from onnx_world_model import OnnxModel

model = OnnxModel("model.onnx")
outputs = model.run(inputs, profile_prefix="traces/encoder")
# traces/encoder_2026-01-31_12-00-00_123.json
```

The C++ surface is the same two options and the same per-call prefix:

```cpp
PipelineTelemetryOptions telemetry;
telemetry.enabled = true;
telemetry.trace_directory = "traces";
telemetry.max_trace_records = 256;
Pipeline pipeline = Pipeline::Load("output/cosmos3-edge", {}, {}, {}, telemetry);

for (const PipelineTraceRecord& record : pipeline.telemetry_snapshot().traces) {
  if (!record.profiling_failed) {
    // record.path is the file ONNX Runtime wrote for one component call.
  }
}

// The generic model API takes the same prefix per call.
Model model = Model::Load("model.onnx");
NamedTensors outputs = model.Run(
    inputs, ModelRunOptions{.profile_file_prefix = "traces/encoder"});
```

## Execution providers

Provider names are case-insensitive and may use short names such as `cuda`,
`dml`, `qnn`, and `cpu`, or ONNX Runtime names such as
`CUDAExecutionProvider`. CPU must be last because it is the fallback provider.

```python
from onnx_world_model import Pipeline, available_execution_providers

print(available_execution_providers())

pipeline = Pipeline(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    provider_options={
        "cuda": {
            "device_id": 0,
            "gpu_mem_limit": 24 * 1024**3,
            "use_tf32": False,
        },
        "cpu": {"use_arena": True},
    },
)

print(pipeline.execution_providers)
```

The requested order is intersected with each component's manifest preferences.
An unavailable provider is skipped only when a later fallback is available.
The loaded ONNX Runtime build must contain the requested provider.

Device-resident component handoff is opt-in. Register the EP's shared library
once before constructing a pipeline, then enable `device_outputs`:

```python
from onnx_world_model import Pipeline, register_execution_provider_library

register_execution_provider_library(
    "CUDAExecutionProvider",
    "/path/to/onnxruntime_providers_cuda.dll",
)
pipeline = Pipeline(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    device_outputs=True,
)
```

Direct connections, recurrent state, public outputs, and shape-only views
retain their device buffers. Host-authored programs and numeric transforms
materialize each device source once at the transform boundary. Python return
values are always independent NumPy arrays.

## Component placement

Every option above is pipeline-wide. `component_placement` overrides them for
one component at a time, at load time, while that component's ONNX Runtime
session is still being built.

```python
from onnx_world_model import ComponentPlacementSpec, Pipeline

pipeline = Pipeline(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    provider_options={"cuda": {"gpu_mem_limit": 24 * 1024**3}},
    component_placement={
        "world_model": ComponentPlacementSpec(
            providers=["cuda", "cpu"],
            # device_id is a native ONNX Runtime provider option, not an
            # argument this runtime invents a second spelling for.
            provider_options={"cuda": {"device_id": 1}},
        ),
        "vae_decoder": {
            "providers": ["cpu"],
            "graph_optimization": "basic",
            "intra_op_threads": 4,
        },
    },
)
```

A `ComponentPlacementSpec` and an equivalent plain mapping are interchangeable.
A field left out — or set to `None` — inherits the pipeline-wide value rather
than overriding it.

| Field | Effect |
|---|---|
| `providers` | This component's provider order, most preferred first |
| `provider_options` | Merged over the global options per provider and per key, component wins |
| `graph_optimization` | Replaces the global level for this component |
| `intra_op_threads` | Replaces the global count for this component |
| `inter_op_threads` | Replaces the global count for this component |

The ONNX Runtime library path, log severity, and `device_outputs` stay
pipeline-wide: they are process- or package-level policy rather than
per-component placement.

**How providers are chosen.** For each component, in order:

1. the component's own `providers`, if it supplied any;
2. otherwise the pipeline-wide `providers`, if any were given;
3. otherwise that component's `preferred_execution_providers` from the
   manifest.

Whichever list is chosen is then filtered by the component's manifest
preferences — which is exactly what every earlier release did — unless
`allow_unpreferred_providers=True` **and** that component supplied its own
list. The flag never widens a component that inherited the global order or its
own preferences, so turning it on cannot quietly move a component nobody named.
An explicitly named CPU provider is always kept, because ONNX Runtime refuses
to build a session with no fallback for a node the preferred provider does not
implement.

**How provider options are chosen.** Global options are a pipeline-wide
default, so options for a provider a component does not run on are simply not
that component's business and are dropped. A component's own options are a
statement about that component, so the same mismatch is an error:

```python
Pipeline(
    "output/cosmos3-edge",
    component_placement={
        "vae_decoder": {"providers": ["cpu"], "provider_options": {"cuda": {}}}
    },
)  # WorldModelError: 'cuda' is not one of the providers it runs on
```

Everything decidable from the manifest is rejected before a single component
model file is opened: an empty or unknown component name, a provider repeated
in one list, and a negative thread count all raise `WorldModelError`. Argument
shape is checked earlier still — a provider list that is a bare string, a
non-integer or `bool` thread count, and an unknown placement key raise
`TypeError` or `ValueError`. Whether a provider exists and whether this ONNX
Runtime build supports it stays a single decision made by the backend.

`WorldModel` takes the same two keyword arguments and forwards them.
`OnnxModel` and `LatentDynamicsModel` do not: they are single graphs, not
pipelines.

The C++ equivalent is `PipelinePlacementOptions`, the final defaulted argument
of `Pipeline::Load` and `PipelinePackage::Load`. It is deliberately **not** a
parameter of `Pipeline(PipelinePackage, scheduling)`: that package's sessions
already exist, so accepting placement there could only be a silent no-op.

**Not included.** Placement decides how a session is built and nothing else.
There is no warm-up, no lazy or deferred component loading, no offload or
eviction, and no peer-to-peer device-to-device transfer.

## Transfer plan

`Pipeline.transfer_plan` reports what every manifest connection would have to
do to move its tensor, given where ONNX Runtime actually placed the two ports.

```python
plan = pipeline.transfer_plan

print(plan.device_outputs_enabled)
for transfer in plan.transfers:
    print(transfer.source, "->", transfer.target, transfer.kind, transfer.reason)
```

`PipelineTransferPlan` and `PipelineTransfer` are frozen dataclasses. There is
exactly one transfer per connection, in manifest order, recurrent edges
included.

| Kind | Meaning |
|---|---|
| `direct` | Same known device, nothing in between: the producer's buffer can be handed over |
| `upload` | Host to a non-CPU device |
| `download` | A non-CPU device to the host |
| `host_staged` | Two different non-CPU devices, staged through the host because there is no peer-to-peer path |
| `host_transform` | A transform this runtime evaluates on the host sits between the ports |
| `unknown` | At least one endpoint's device is unreported, so nothing may be assumed |

- **Classification is conservative.** `direct` — and the
  `direct_bind_eligible` flag that goes with it — is claimed only when both
  devices are known, identical, and nothing sits between them. Every other
  kind carries a one-sentence `reason`; `direct` carries an empty one.
- **`unknown` outranks everything.** A backend that reports no placement for a
  port makes that connection `unknown` even if it also has a host transform.
- **`reshape` is conservative too.** A reshape only relabels axes, but binding
  a device buffer requires the original shape to equal the shape of the view
  wrapped around it, so a reshape counts as `direct` only when the declared
  source, target, and any explicit transform shape are identical and static.
  Every other reshape is reported as `host_transform`.
- **This is the configured plan, not the effective one.**
  `device_outputs_enabled` mirrors the `device_outputs` option. When it is
  false, every component output is bound to the host regardless of what the
  plan says, so the effective behavior is CPU-bound even where the plan reports
  `upload`, `download`, or `host_staged`. The plan is deliberately not
  rewritten in that case, because it answers "how is this package placed",
  which is what a caller needs before turning device outputs on.
- **Nothing executes from it.** The plan is inspection only in this milestone.
  Reading it takes no lock, changes nothing, and is computed once while the
  package loads.

Each port's placement is also visible on its own, through the model metadata
of any graph loaded with `OnnxModel`:

```python
from onnx_world_model import OnnxModel

model = OnnxModel("output/cosmos3-edge/world_model/model.onnx")
for spec in model.metadata.outputs:
    print(spec.name, spec.device)   # DeviceSpec(type='cuda', id=0) or None
```

`TensorSpec.device` is `None` when the backend reports no placement. It is
runtime placement rather than part of the graph signature, so it never takes
part in tensor validation.

The C++ equivalents are `Pipeline::transfer_plan()` and
`PipelinePackage::transfer_plan()`, which return the same
`PipelineTransferPlan` value, and `TensorSpec::device`, a
`std::optional<TensorDevice>`.

## Precision report

`Pipeline.precision_report` answers one honest question: what does this
runtime actually know about each component's numeric precision?

```python
for entry in pipeline.precision_report:
    print(entry.component, entry.declared_parameter_dtype)   # 'float16' or None
    for port in entry.graph_inputs:
        print("  in ", port.name, port.dtype)
    for port in entry.graph_outputs:
        print("  out", port.name, port.dtype)
    for port in entry.state_inputs + entry.state_outputs:
        print("  state", port.name, port.dtype)
    print("  providers", entry.execution_providers)
```

`ComponentPrecisionReport` and `PrecisionPort` are frozen dataclasses and the
report is an immutable tuple with one entry per manifest component, in manifest
order. The C++ equivalents are `Pipeline::precision_report()` and
`PipelinePackage::precision_report()`, which return
`std::vector<ComponentPrecisionReport>` with a
`std::optional<DataType> declared_parameter_data_type`.

| Field | Where it comes from | What it means |
|---|---|---|
| `declared_parameter_dtype` | The manifest's `parameter_dtype` | The exporter's **unverified claim** about parameter storage |
| `graph_inputs`, `graph_outputs` | The loaded ONNX Runtime session | The real port types of the graph, in graph order |
| `state_inputs`, `state_outputs` | Manifest states, typed from the session | The `component.port` endpoints that carry `PipelineState` |
| `execution_providers` | Runtime session configuration | Providers registered after availability filtering; not per-node assignment |

- **Reading it changes nothing.** The report is computed on demand from the
  manifest and the already-loaded session metadata. It opens no file, calls no
  extra ONNX Runtime API, takes no lock, and returns a detached value.
- **It enforces nothing.** The authoritative precision checks are the ones that
  already exist: manifest port types are validated against the live graph while
  the package loads, and connection and recurrent-state types are validated by
  the manifest checks. This report adds no rule and rejects no package.
- **Ports are activations, parameters are weights.** The declared parameter
  dtype is never compared with a port dtype, because they describe different
  things. A component that declares `int8` parameters while exposing `float32`
  activations and `int64` token ids is normal, not a finding.
- **A non-floating declaration is legal.** `parameter_dtype` is not restricted
  to floating types; `int8` is exactly how a quantized-weight claim is spelled
  today.
- **State ports are qualified.** They are named `component.port`, listed in
  manifest state order, and de-duplicated on first occurrence, so two states
  that share a carrier port report it once.
- **An empty provider list means empty.** A custom in-process backend that
  reports no providers is reported with none, never credited with CPU.
- **Provider order is not graph partition data.** A registered provider can
  claim zero nodes and still appear in `execution_providers`; the loaded
  Session API does not expose final per-node assignment through this report.

### Where `parameter_dtype` comes from

`parameter_dtype` is per component and optional in the schema, with one
exception: a manifest that declares an **executable profile** must declare a
parameter dtype for every component, exactly as it must declare a semantic name
for every input and execution-provider hints for every component. A
profile-less manifest may omit it, and the report then says `None` rather than
guessing a default.

### What this report cannot tell you

**It cannot tell you whether a component is quantized.** MatMulNBits, a QDQ
pair, QLinear operators, a vendor-specific blocked format, and plain float
weights all produce exactly the same report. Two limits cause this, and both
are real today:

1. **ONNX Runtime session visibility.** This runtime reaches ONNX Runtime only
   through the `Session` public API, which exposes graph inputs and outputs.
   It does not expose ordinary initializers or nodes, so no weight tensor and
   no quantized operator is observable from a loaded session. Inspecting them
   would require either an ONNX model parser in this repository or an ORT graph
   or compile API that exposes initializers — a dependency and a scope decision
   that has not been taken.
2. **No exported provenance.** The current Mobius checkout emits no tracked
   quantization provenance in `pipeline.json`. There is no field that records a
   quantization scheme, a bit width, a group size, symmetry, or calibration, so
   there is nothing for the runtime to read, and inventing one here would be a
   claim rather than a report.

Two further gaps are worth naming, so they are not mistaken for findings:

- **The dtype vocabulary is the classic one.** Manifest dtypes parse to
  `FLOAT`, `FLOAT16`, `BFLOAT16`, `DOUBLE`, `INT64`, `INT32`, `INT16`, `INT8`,
  `UINT64`, `UINT32`, `UINT16`, `UINT8`, and `BOOL`. There is no `INT4`,
  `UINT4`, `FP8`, or `FP4` spelling, so a package using one of those formats
  cannot declare it at all today, and `parameter_dtype` will name the container
  type instead.
- **KV-cache precision is not declared.** A state's precision is whatever its
  carrier ports report; the manifest has no `kv_cache_dtype`, so nothing states
  the intended cache precision independently of the ports.

### Follow-up: a coordination contract, not runtime fields

The next step is an exporter change first and a runtime change second. It is
recorded here as a contract so that neither side invents the other's data:

1. **Mobius emits provenance.** `pipeline.json` gains a per-component
   `weight_quantization` record — scheme, bit width, group size, symmetry, and
   calibration provenance — and a `kv_cache_dtype` for the states it exports.
   Until that exists there is nothing to validate against.
2. **Then the runtime validates declarations.** With provenance in the
   manifest, the loader can check the declaration against the exported record
   and check state port dtypes against the declared cache precision, and this
   report can carry those fields instead of only the declaration.
3. **Initializer verification is a separate decision.** Confirming that the
   weights on disk match any declaration requires an ONNX parser in this
   repository or an ONNX Runtime graph or compile API that exposes
   initializers. It is not implied by step 2 and is not planned here.

**Numeric parity is the exporter's test, not the runtime's.** Whether a
quantized vision tower still produces acceptable pixels belongs in Mobius
golden tests, which own the reference outputs and the calibration data. Runtime
CI validates that a package loads, that its ports and providers are what the
manifest says, and that stages execute; when Mobius exports tolerances and
provenance, runtime CI can consume them, but it must not invent them.

## Input contract

Callers provide tokenized, normalized, and packed tensors using semantic or
qualified port names. Inspect package metadata instead of hard-coding one
checkpoint's shapes.

```python
session = pipeline.create_session()

reasoner_inputs = {
    "text.token_ids": reasoner_input_ids,
    "vision.pixel_values": pixel_values,
}

session.run_stage("reasoner_prompt", reasoner_inputs)
reasoning = session.run_stage(
    "reasoner_decode",
    options={"max_tokens": 64, "seed": 1234},
)
session.release_stage("reasoner_decode")
```

World generation uses packed initial state:

```python
world_inputs = {
    "text.token_ids": generator_input_ids,
    "diffusion.initial_vision_latent": packed_vision_noise,
    "diffusion.initial_action_latent": packed_action_noise,
}

world = session.run_stage(
    "world_generation",
    world_inputs,
    options={
        "mode": "action",
        "action_domain": "droid_lerobot",
        "num_inference_steps": 50,
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)
session.release_stage("world_generation")

video = session.run_stage(
    "decode_video",
    options={
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)
session.release_stage("decode_video")
```

## Conditioned, guided generation contract

Image-to-video generation is described entirely by the manifest. An iterative
stage that supports it declares the capabilities
`classifier_free_guidance` and `conditioned_diffusion` and these options:

```json
{
  "name": "world_generation",
  "kind": "iterative",
  "capabilities": ["loop_carried_state", "classifier_free_guidance",
                   "conditioned_diffusion"],
  "options": {
    "guidance": {
      "kind": "classifier_free",
      "conditioning_input": "generator.input_ids",
      "scale_option": "guidance_scale",
      "default_scale": 1.0,
      "combine": "unconditional + scale * (conditional - unconditional)"
    },
    "conditioning": {
      "vision": {
        "encoder_stage": "encode_video",
        "encoder_input": "video_encoder.sample",
        "encoder_output": "video_encoder.latent",
        "state": "vision_state",
        "conditioned_latent_frames_option": "vision_conditioned_latent_frames",
        "default_conditioned_latent_frames": [],
        "packing": {
          "spatial_patch_size": 2,
          "temporal_patch_size": 1,
          "input_layout": "BCTHW",
          "output_layout": "NC",
          "channel_order": "patch_height_patch_width_channel"
        }
      }
    }
  }
}
```

`guidance` accepts two optional fields: `unconditional_input` names the input
the host uses for the unconditional value, and `outputs` lists the guided
predictions. Without them the runtime accepts `unconditional:<port>` (also
`unconditional:<semantic>` and `unconditional:<alias>`) and guides every
scheduler-driven recurrent output of the stage. `combine` is validated, not
interpreted: only the formula above is supported. `scale_option` is resolved
from the run options first, then from the selected scheduler mode override,
then from `default_scale`; a scale of 1 runs one pass, any other scale requires
an unconditional value.

The two capabilities are part of the contract in both directions: a stage that
declares `guidance` must advertise `classifier_free_guidance`, a stage that
declares `conditioning` must advertise `conditioned_diffusion`, and a stage
that advertises either without the matching option is rejected. The runtime
publishes everything it implements, and a `required_capabilities` entry is
honored when the runtime implements it or when a component or stage in the same
manifest declares that it provides it; only names nothing can satisfy fail to
load:

```python
from onnx_world_model import supported_pipeline_capabilities

print(supported_pipeline_capabilities())
# ('action_domain_program', 'attention_mask_program', 'classifier_free_guidance',
#  'conditioned_diffusion', 'iterative_scheduler', 'loop_carried_state', ...)
```

The C++ equivalent is `PipelineManifest::SupportedCapabilities()`.

`conditioning.<modality>.packing` must describe the same packing the runtime's
video unpatchify inverts (`BCTHW` -> `NC` with
`patch_height_patch_width_channel` channels and `temporal_patch_size` 1);
anything else is rejected at load time, as is a `spatial_patch_size` that
disagrees with `metadata.packing.latent_patch_size`. `encoder_output` must also
be a public pipeline output so the host can read the encoded latent back from
the encoder stage.

`conditioning.<modality>` may add an optional `preprocessing` block describing
how conditioning frames reach the encoder:

```json
"preprocessing": {
  "resize": "stretch_to_target",
  "resample": "bilinear",
  "convert_rgb": true,
  "rescale_factor": 0.00392156862745098,
  "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
}
```

`resample` names the filter (`bilinear`, `bicubic`, `nearest`, or `lanczos`);
`resize` may name either the filter or the only supported strategy,
`stretch_to_target`, which scales frames straight to the requested output
geometry. `rescale_factor` must be `1/255` and `convert_rgb` must be true,
since that is what the runtime does. Omitting the block keeps the default
bilinear resize into the signed `[-1, 1]` range.

Generation recipes and prompt packaging live in the manifest metadata:

```json
"metadata": {
  "generation_recipes": {
    "image_to_video": {
      "conditioning": {"modality": "image", "encoder_stage": "encode_video",
                       "conditioned_latent_frames": [0]},
      "prompt": {"positive": "json_or_text",
                 "negative_asset": "assets/negative_prompt.json",
                 "add_resolution_template": false,
                 "add_duration_template": false,
                 "use_system_prompt": false},
      "height": 480, "width": 832, "frames": 121, "fps": 24.0
    }
  },
  "generator_prompt": {
    "chat": {
      "add_generation_prompt": true,
      "add_vision_id": false,
      "enable_thinking": true
    },
    "suffix_token_ids": [151645, 151652],
    "system_prompts": {"image": "...", "video": "..."},
    "templates": {
      "duration": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
      "inverse_duration": "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS.",
      "image_resolution": "This image is of {height}x{width} resolution.",
      "inverse_image_resolution": "This image is not of {height}x{width} resolution.",
      "video_resolution": "This video is of {height}x{width} resolution.",
      "inverse_video_resolution": "This video is not of {height}x{width} resolution."
    }
  }
}
```

`generation_recipes.<mode>` supplies the default geometry, fps, prompt flags,
and conditioned latent frames for `mode`; the mode name is also the scheduler
`mode_overrides` key, which is where `num_inference_steps`, `flow_shift`,
`use_karras_sigmas`, and `guidance_scale` come from. `generator_prompt` is
optional and backward compatible: `suffix_token_ids` (or `suffix_tokens` with
token strings) are appended after the chat template, `system_prompts` selects a
per-modality system message, and `templates` supplies the metadata sentences. A
missing section means the runtime performs that step exactly as it did before,
and requesting a template or system prompt the package does not declare is an
error rather than a guess.

A recipe's `prompt` block may restate any `generator_prompt` field (`chat`,
`system_prompts`, `templates`, `suffix_token_ids`, `suffix_tokens`) to override
it for that mode only, and `"negative_default": "empty" | "asset"` selects what
an omitted `negative_prompt` means. The default is `"empty"`, matching the
reference pipeline; `"asset"` uses the shipped `negative_asset`. That asset may
hold the prompt text, a wrapper object with a `negative_prompt`/`default`/
mode-named string, or — as Cosmos3 Edge ships it — the structured JSON document
itself, which is passed through verbatim as a JSON string.

`chat.add_generation_prompt`, `chat.add_vision_id`, and
`chat.enable_thinking` are passed to the exported chat template, so a package
selects the official behavior; any other field under `chat` is rejected.

## Stage execution

- `run_stage()` executes the strategy declared by the manifest: single pass,
  autoregressive generation, or all iterative scheduler steps.
- `begin_stage()` executes the same strategy one step at a time and reports
  each step as a `StageEvent`; `run_stage()` is exactly that run drained to
  completion.
- `step_stage()` executes exactly one pass for callers that manage a loop.
- `overrides` supplies generated tensors or transformed targets explicitly.
- `release_stage()` releases state according to the manifest lifecycle.
- `reset()` clears all session state.

`PipelineSession.run()` executes selected stages in declaration order and
releases each stage after execution. Explicit `run_stage()` calls are clearer
when stages need different prepared inputs.

## Incremental stage execution

`begin_stage()` returns a `StageRun`: a handle over one execution of one stage
that reports what each model or scheduler step produced.

```python
session = pipeline.create_session()

with session.begin_stage("reasoner_decode", options={"max_tokens": 64}) as run:
    for event in run:
        if event.kind == "token":
            print(event.token_ids)          # int64 [batch, 1]
        elif event.finished:
            reasoning = event.outputs       # what run_stage() would return
```

`session.iter_stage(...)` is the same loop without the handle, for callers that
consume every event:

```python
for event in session.iter_stage("world_generation", world_inputs, options=opts):
    progress(event.iteration)
```

Each event carries:

| Field | Meaning |
|---|---|
| `kind` | `"token"` for autoregressive decoding, `"iteration"` for an iterative scheduler step, `"transition"` for the single pass of every other stage kind, `"completed"` for the terminal event. |
| `stage` | The stage this run drives. |
| `iteration` | Zero-based index of this step event within this run; the terminal event reports how many step events preceded it. |
| `token_ids` | The tokens this step generated, on `"token"` events only. |
| `outputs` | The stage's public outputs as of this event, as NumPy arrays. |
| `finished` | True on the terminal `"completed"` event and nowhere else. |

The rules the API guarantees:

- **Parity.** `run.finish()` returns exactly what `run_stage()` returns for the
  same arguments, because `run_stage()` is `begin_stage(...).finish()`. The
  token budget, sampling configuration, seed, end-of-sequence tokens, and
  iterative target are resolved once, when the run begins.
- **Synchronous steps.** One `step()` blocks until one model or scheduler step
  finishes. Events are results, not notifications; there is no background
  thread. A step can be stopped cooperatively — see
  [Cancellation and deadlines](#cancellation-and-deadlines) — but not
  preempted.
- **One terminal event.** A stopping condition — the end-of-sequence token in
  every lane, the token budget, or the last scheduler step — is reported by the
  *next* `step()` as the `"completed"` event, so every run ends the same way.
  Stepping past it raises `WorldModelError`, and `finish()` keeps returning the
  cached result.
- **One run per session.** While a run is unfinished the session raises
  `WorldModelError` from `begin_stage()`, `run_stage()`, `step_stage()`,
  `snapshot()`, `restore()`, `fork()`, `checkpoint()`, `restore_checkpoint()`,
  `drop_checkpoint()`, `reset()`, and `release_stage()`. An in-flight decode
  loop is not part of a snapshot, so the runtime refuses to capture one rather
  than hand back a snapshot that silently drops it. Reading `outputs`,
  `state()`, and `has_checkpoint()` stays legal.
- **No implicit rewind.** `close()`, the context manager, a failed step, and
  dropping the handle all release the session where the run stopped and keep
  everything it already applied. Use `snapshot()` before the run, or
  `checkpoint()`, when a caller wants to rewind.
- **Python materializes.** `token_ids` and `outputs` are independent NumPy
  arrays, exactly as `run_stage()` returns them; in C++ a device-backed output
  stays on its device.

Stopping early releases the session only when the run is closed. A `with`
block or an explicit `close()` does that deterministically; a bare `break` out
of `iter_stage()` waits for the generator to be closed or collected.

The C++ API is the same shape: `PipelineSession::BeginStage` returns a
move-only `StageRun` with `Step()`, `Finish()`, `RequestCancellation()`, and
`Cancel()`, and a `StageEvent` there keeps its device-resident tensors.

## Cancellation and deadlines

Every execution entry point accepts either a `cancellation` token or a
`timeout` in seconds. They are mutually exclusive: a `timeout` builds its own
source, so passing both raises `ValueError`.

```python
import threading

from onnx_world_model import CancellationSource, CancelledError, DeadlineExceededError

# Stop on demand, from anywhere.
source = CancellationSource()
threading.Timer(5.0, source.cancel).start()
try:
    outputs = session.run_stage("world_generation", inputs, cancellation=source.token())
except CancelledError:
    partial = session.outputs        # everything the stage already applied

# Or bound one call by wall-clock time.
try:
    outputs = session.run_stage("world_generation", inputs, timeout=30.0)
except DeadlineExceededError:
    ...
```

`StageRun` adds `request_cancellation()`, the one method that is safe to call
from another thread while `step()` or `finish()` is running, because it takes
no session lock:

```python
run = session.begin_stage("reasoner_decode", prompt)
watchdog = threading.Timer(10.0, run.request_cancellation)
watchdog.start()
try:
    for event in run:
        consume(event)
finally:
    watchdog.cancel()
    run.close()
```

The rules the API guarantees:

- **Two outcomes, one base.** An explicit cancellation raises `CancelledError`
  and an expired deadline raises `DeadlineExceededError`. Both derive from
  `WorldModelError`, so an existing `except WorldModelError` handler still
  catches them, and the C++ `ErrorCode` — `cancelled` or `deadline_exceeded` —
  decides which one, never the message text.
- **First reason wins.** A token is a one-way latch. Once a reason is claimed
  it never changes, and a cancelled token stays cancelled: reusing it fails the
  next call immediately. Build a new `CancellationSource` per request.
- **Never a rollback.** A cancelled call stops at its next boundary, releases
  the session's run slot, and leaves everything it already applied in place —
  exactly like a failed call. The one thing it does undo is internal scratch
  state that was never a result: a guided step that is stopped during its
  unconditional pass restores the conditional conditioning input before it
  unwinds, so the next step is not silently conditioned on the unconditional
  value. Take a `snapshot()` or a `checkpoint()` first if you want to rewind.
- **`request_cancellation()` is not `close()`.** `request_cancellation()`
  signals work that is running and does not take the session lock.
  `close()` takes the lock, so it waits for an in-flight step, and then only
  abandons the handle and releases the slot. A stale handle can do neither to
  a newer run.
- **Boundaries, not preemption.** The token is checked before and after each
  component, between the two classifier-free-guidance passes, around guidance
  combination and the state transforms, around token sampling, and at each
  step. Inside one ONNX Runtime call the runtime terminates the run through a
  fresh per-call `Ort::RunOptions`, which ONNX Runtime honors *between graph
  nodes* — a single long-running kernel still finishes first.
- **Deadlines fire on time, from one shared watchdog.** A deadline still in
  the future is armed on a single lazily started, process-wide watchdog
  thread. There is no thread and no timer per request: it holds every armed
  source weakly, sleeps until the earliest deadline, and claims it there. So a
  deadline stops work that is already blocked inside an ONNX Runtime call
  rather than waiting for the next boundary — bounded only by ORT's
  between-nodes termination check, so a single long kernel can overrun it. A
  deadline that has *already* passed when the source is created is left to the
  next poll instead, which is what lets `source.cancel()` immediately after
  `CancellationSource(timeout=0)` still win the first-reason race.
- **`wait()` blocks without polling.** `CancellationToken.wait()` and
  `CancellationSource.wait()` block until a reason is claimed and return it —
  `"cancelled"` or `"deadline_exceeded"` — rather than raising. They release
  the GIL, so another Python thread can still cancel, and they are what work
  with no boundary of its own uses instead of a polling loop. Waiting on a
  token nothing can cancel raises `WorldModelError` instead of blocking
  forever.
- **One deadline per `run()`.** `PipelineSession.run(..., timeout=...)` builds
  its source once, so a single absolute deadline covers the whole stage
  sequence rather than restarting at each stage.
- **Not yet covered.** The `WorldModel` modality `generate()` methods and the
  fixed `LatentDynamicsModel`, `WorldModel::Step`, and `Rollout` APIs do not
  take a token; drive `PipelineSession` directly when you need one.

## Session snapshots and named checkpoints

A session can capture everything it has accumulated and rewind or branch from
that capture:

```python
session = pipeline.create_session()
session.run_stage("world_generation", inputs, options={"num_inference_steps": 20})

checkpoint = session.snapshot()

# Explore one continuation, then rewind and try another.
first = session.run_stage("world_generation", options={"num_inference_steps": 20})
session.restore(checkpoint)
second = session.run_stage("world_generation", options={"num_inference_steps": 20})

# Or branch, and let both sides advance independently.
branch = session.fork()
```

- `snapshot()` returns an immutable `PipelineSessionSnapshot` holding the
  session's external, endpoint, recurrent-state, and guidance tensors, its
  stage cursors, scheduler histories, position cursors, and random engine.
- `restore()` replaces the session's state with the captured one. It either
  succeeds completely or changes nothing.
- `fork()` returns an independent session on the same pipeline, started from a
  snapshot of this one. Later runs, releases, and resets on either side never
  reach the other.

Snapshots are in-memory values only. They share tensor storage
copy-on-write, so capturing and forking copy no tensor data and never move a
device-resident tensor to the host. They are not written to disk, are not
paged out, and cannot be sent to another process. A snapshot stays bound to
the `Pipeline` it was taken from: restoring it into a session from a
separately loaded `Pipeline` raises `WorldModelError` even when both packages
are byte-for-byte identical. An unfinished `StageRun` also blocks every one of
these calls, because a decode or scheduler loop in flight is not part of the
capture.

### Named checkpoints

When a caller would otherwise track snapshot handles itself, the session can
hold them by name instead:

```python
session.checkpoint("before_branch")

session.run_stage("world_generation", options={"num_inference_steps": 20})
session.restore_checkpoint("before_branch")   # rewind and try again

assert session.has_checkpoint("before_branch")
session.drop_checkpoint("before_branch")
```

- `checkpoint(name)` captures exactly what `snapshot()` captures and stores it
  under `name`, replacing any checkpoint already stored there. The capture and
  the store happen together, so the checkpoint is the state the session held at
  one instant.
- `restore_checkpoint(name)` rewinds the session to that checkpoint with the
  same all-or-nothing behaviour as `restore()`, and leaves the checkpoint
  available for reuse.
- `drop_checkpoint(name)` discards it. Dropping a name the session does not
  hold raises `WorldModelError` rather than silently succeeding.
- `has_checkpoint(name)` reports whether the name is currently held.
- An empty name raises `WorldModelError` from every one of the four methods.

Named checkpoints are control metadata that lives beside the execution state,
not inside it:

- they survive stage execution, `release_stage()`, and `restore()`;
- `restore()` replaces the execution state only, so the session receiving a
  snapshot keeps its own checkpoint names and gains none from the source;
- a snapshot never carries checkpoints, so `fork()` inherits the parent's
  execution state but starts with an empty checkpoint namespace;
- `reset()` clears every named checkpoint along with the session state.

This is in-memory transaction support. It is not paged KV attention, and it is
not disk serialization: no checkpoint is written to disk, paged out, or shared
across processes.

## Conditioned, guided stages

An iterative stage that declares `guidance` and `conditioning` adds two host
responsibilities: run the declared conditioning encoder stage first, and pass
the unconditional value of the guided input alongside the conditional one. The
runtime then runs the stage twice per step — once conditional, once
unconditional — and still applies exactly one scheduler update.

```python
encoded = session.run_stage(
    "encode_video",
    {"video_encoder.sample": conditioning_frames},  # [1, 3, T, H, W] in [-1, 1]
)
session.release_stage("encode_video")

world = session.run_stage(
    "world_generation",
    {
        "text.token_ids": conditional_input_ids,
        "unconditional:generator.input_ids": unconditional_input_ids,
        "diffusion.initial_vision_latent": packed_latents,  # conditioned rows anchored
    },
    overrides={
        # Only the noisy rows carry a timestep, an MSE index, and a scheduler
        # update; the generator still attends to every latent frame.
        "generator.vision_timestep_token_indexes": noisy_token_indexes,
    },
    options={
        "mode": "image_to_video",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)
```

The unconditional value is supplied as `unconditional:<port>`, and the manifest
may name it explicitly through `guidance.unconditional_input`. A guidance scale
of 1 runs one pass; any other scale requires the unconditional value.

`WorldModelPreprocessor.prepare_world(..., image=...)` produces exactly those
tensors: `pipeline_inputs()` and `pipeline_overrides()` carry them, and
`with_conditioning_latent()` folds the encoder's latent into the packed rows.

## Outputs

Outputs remain at their model boundary:

- `generated_token_ids` contains token IDs, not decoded text;
- action state may retain the model's padded width;
- decoded video is a float NCTHW tensor.

The modality APIs on `WorldModel` add preprocessing and structured output
handling.

Model loading and inference release the Python GIL. NumPy inputs are copied
into engine-owned storage, and outputs are returned as independent arrays.

## C++ API

```cpp
#include "onnx_world_model/onnx_world_model.hpp"

using namespace onnx_world_model;

RuntimeOptions runtime;
runtime.ort_library_path = "/path/to/onnxruntime.dll";
runtime.providers = {"cuda", "cpu"};
runtime.provider_options["cuda"] = {
    {"device_id", "0"},
    {"gpu_mem_limit", "25769803776"},
};

Pipeline pipeline = Pipeline::Load(
    "output/cosmos3-edge",
    runtime,
    PipelineSchedulingOptions{
        .max_concurrent_executions = 4,
        .max_concurrent_by_stage_kind = {{"iterative", 2}},
    },
    PipelinePlacementOptions{
        .components =
            {
                {"world_model",
                 ComponentPlacement{
                     .providers = {"cuda", "cpu"},
                     .provider_options = {{"cuda", {{"device_id", "1"}}}},
                 }},
            },
    },
    PipelineTelemetryOptions{
        .enabled = true,
        .trace_directory = "traces",
        .max_trace_records = 256,
    });

for (const PipelineTransfer& transfer : pipeline.transfer_plan().transfers) {
  if (transfer.kind != PipelineTransferKind::direct) {
    // transfer.reason says why in one sentence.
  }
}

for (const ComponentPrecisionReport& entry : pipeline.precision_report()) {
  // Declared by the exporter and never verified against the weights; nullopt
  // when the manifest declared none.
  const std::optional<DataType> declared = entry.declared_parameter_data_type;
  // Read from the loaded session, and deliberately never compared with
  // `declared`: ports carry activations, parameters are weight storage.
  for (const PrecisionPort& port : entry.graph_inputs) {
    (void)port.data_type;
  }
}

PipelineSession session = pipeline.CreateSession();

NamedTensors outputs = session.RunStage(
    "world_generation",
    inputs,
    overrides,
    PipelineRunOptions{
        .strings = {{"mode", "action"}, {"action_domain", "droid_lerobot"}},
        .integers = {{"num_inference_steps", 50}},
    });

PipelineSessionSnapshot checkpoint = session.Snapshot();
PipelineSession branch = session.Fork();
session.Restore(checkpoint);

session.Checkpoint("before_branch");
if (session.HasCheckpoint("before_branch")) {
  session.RestoreCheckpoint("before_branch");
  session.DropCheckpoint("before_branch");
}
```

`PipelineSessionSnapshot` has no public constructor, so it can only come from
`PipelineSession::Snapshot()`. `Restore` and `Fork` take the session lock, and
`Restore` throws `Error` with `ErrorCode::state` when the snapshot came from a
session on a different `PipelinePackage` instance.

`Checkpoint`, `RestoreCheckpoint`, `DropCheckpoint`, and `HasCheckpoint` take
that same lock. `Checkpoint` captures and publishes under one lock hold, so its
linearization point is that single acquisition; `RestoreCheckpoint` linearizes
twice — once where it copies the named checkpoint handle, once where `Restore`
commits — and never holds the lock across the delegation, so it cannot
deadlock. An empty name throws `ErrorCode::invalid_argument`; restoring or
dropping a name the session does not hold throws `ErrorCode::state` and changes
nothing.

Cancellation in C++ is the same contract:

```cpp
CancellationSource source;                      // or WithTimeout(30s)
PipelineRunOptions options;
options.cancellation = source.token();

StageRun run = session.BeginStage("world_generation", inputs, {}, options);
std::jthread watchdog([&run] { run.RequestCancellation(); });
try {
  NamedTensors outputs = run.Finish();
} catch (const Error& error) {
  // ErrorCode::cancelled or ErrorCode::deadline_exceeded.
}
```

`CancellationToken` has no public constructor beyond the default, inert one, so
a cancellable token can only come from a `CancellationSource`. The source is
move-only; the token is copyable and observes the same state. `Cancel()` is
`noexcept` and safe from any thread.

`CancellationToken::WaitForCancellation()` blocks until a reason is claimed and
returns it, so a backend with no boundary of its own can park on the token
instead of polling:

```cpp
NamedTensors MyBackend::Run(
    const NamedTensors& inputs,
    const CancellationToken& cancellation) const {
  StartTheWork();
  (void)cancellation.WaitForCancellation();   // released by cancel or deadline
  cancellation.ThrowIfCancellationRequested();
  ...
}
```

It reports the reason rather than throwing it, and it rejects a token that is
not cancellable with `ErrorCode::invalid_argument` instead of blocking forever.
`CancellationSource::WaitForCancellation()` is the same wait for the owner.

`Model::Run` has a second overload that takes a token, and `ModelBackend::Run`
has a virtual cancellable overload whose default implementation checks the
token before and after the historical one-argument `Run`. An external backend
therefore keeps compiling and still stops at those boundaries; only a backend
that can interrupt work already running needs to override it, as the ONNX
Runtime backend does.

`TensorSpec` gained a final `std::optional<TensorDevice> device` member in
0.6.0. A custom `ModelBackend` written before that keeps compiling and reports
`std::nullopt`, which the transfer plan reads as unknown; a custom backend that
does report placement can populate it and get a real plan. The member is
deliberately not part of the graph signature, so neither `ValidateTensor` nor
the manifest-versus-model check looks at it. A custom backend can also build a
`PipelinePackage` directly and pass `device_outputs_enabled` as the
constructor's final defaulted argument, so an in-memory package still produces
an accurate plan.

## Runtime scope

- Dense FP16, BF16, FP32, integer, and boolean tensors.
- Single-pass, on-demand, state-transition, iterative, and autoregressive
  stages.
- Classifier-free guidance and conditioned diffusion on iterative stages,
  driven by the stage's `guidance` and `conditioning` options rather than by
  model names.
- KV-cache, request, sequence, iteration, and session state lifecycles.
- Per-component load-time placement: execution providers, provider options
  (including `device_id`), graph optimization level, and thread counts, plus a
  conservative, inspection-only `transfer_plan` over the ports ONNX Runtime
  actually assigned. Session warm-up, lazy or deferred component loading,
  offload and eviction, peer-to-peer device-to-device transfers, and executing
  from the plan are **not** included.
- In-memory session snapshot, restore, fork, and named checkpoints. This is
  in-memory transaction support, not paged KV attention: nothing is serialized
  to disk and nothing crosses a process boundary.
- Opt-in runtime telemetry: per-component call, byte, and duration counters,
  per-stage execution, step, and completion counters, per-stage-kind admission
  wait outcomes, and the device-to-host materializations the runtime performed,
  read as an immutable snapshot under a reset-able epoch.
- Opt-in per-run ONNX Runtime node traces on top of that telemetry: one trace
  file per component call under a unique prefix, and one bounded, in-memory
  record per call naming that file, its size, and how the call ended. Trace
  *parsing*, trace file lifetime management, execution-provider peak memory,
  exact host-to-device byte counts, and any aggregation, percentile, or export
  above cumulative counters and raw trace files are **not** included.
- Inspection-only precision reporting: the exporter's declared
  `parameter_dtype` per component, the real port dtypes and registered providers
  of each loaded session, and the state ports each component carries. Detecting
  or verifying quantization is **not** included — initializers and nodes are
  invisible through the ONNX Runtime session API, and the manifest carries no
  quantization provenance — and neither is any precision enforcement policy.
- Explicit cancellation and deadlines on stage execution and generic model
  execution, enforced at execution boundaries, by one shared process-wide
  deadline watchdog, and inside an ONNX Runtime call at graph-node
  granularity — so a single long-running kernel can overrun its deadline.
  Cancellation on the `WorldModel` modality APIs and the fixed
  latent-dynamics API is not included.
- FlowMatch Euler and flow-prediction UniPC order-1/order-2 schedulers.
- Greedy and temperature/top-k/top-p autoregressive sampling.
- Packed layout, attention masks, multimodal positions, scheduler timesteps,
  and action-domain generated inputs.
- Scheduler, cast, reshape, packed video finalization, and audio finalization
  transforms.

Pipeline scheduling and host transforms operate on CPU tensors, but the session
no longer forces every tensor to CPU. Direct connections, recurrent state,
rank adaptation, and `reshape` keep the producer's buffer, so a device-resident
component output can reach the next component and the public outputs untouched.
Each host transform materializes its device operands once at its own boundary.
In C++ this means a stage output may be device-only; call
`Tensor::CopyToCpu()` before reading it. The Python API is unaffected because
it always converts results to independent NumPy arrays.

Scheduler modes are never inferred: `mode` selects a declared `mode_overrides`
entry, and an absent mode uses the stage defaults. Conditioning handoffs
recorded only in manifest metadata are not executed automatically; the
declared `conditioning` stage option is what drives image-to-video, and other
handoffs still require the caller to supply the packed initial latent.
