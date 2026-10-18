# AFD operator test artifacts

## Remaining operators (2026-10-16)

Current suite: **51 CPU tests passed**, **19 Python AST checks passed**.
The 18 new cases cover scheduler state/selection, combineSend route/Completion
contracts and combineRecv weighted arithmetic/protocol models. These are not
C++ compilation or device execution evidence.

`run_pipeline_a5.py` now launches all five actual operators with identity experts;
it does not inject ACK/credit. It is prepared but **not run**. See
`docs/afd/remaining_operators.md` for commands and platform gates.

## Capacity clarification regression (2026-10-16)

`test_capacity_cpu.py` adds six independent specification/oracle tests: two same-DP
owners each C=32768 (DP=65536), each full B=262144 over eight common rounds,
heterogeneous windows/lengths and zero-length exhausted members, mask holes,
TP nonowner B=0, independent Decode slots, hot-F 524288 routes over sixteen
Q=32768 pieces with fixed padding budget, byte expansion retaining full-B source
token indices, and the single-owner workspace regression. Registered slots are
all counted in F memory even if not selected in one DP round. These are CPU
models plus actual-header arithmetic checks, NOT a scheduler implementation or
proof of device round/credit liveness.

```powershell
python -m unittest discover -s tests/afd -p 'test_*cpu.py' -v
```

Result on the development host: **33 CPU tests passed** (27 existing + 6 new).
All 15 `tests/afd/*.py` files parsed with Python AST. The ABI parser now ignores
C++ comments; it still reads and verifies actual field layouts.

Actual target host check, **not executed here**:

```bash
python tests/afd/run_capacity_a5.py --device 0
```

This invokes the compiled `AFDBuffer.create_local_receiver` constructor for
single 32k, same-DP 2x32k and heterogeneous 32k/16k lanes, checks bounded receive
allocation and doubled input reservation, and requires per-lane32769 rejection.
It does not execute a communication kernel. The real HCCL constructor, sender
8-round execution, live ACK/credit closure, scheduler aggregation, graph replay
and A5 kernel output/performance remain separate hardware gates. Do not invoke
multi-round send without a real ACK consumer and mistake a hang for capacity
failure; those downstream operators are not yet implemented.

This is the CPU contract/oracle portion of the approved test plan, not a completed A5 test harness. No torch or pytest dependency is required; NumPy is required. Tests do not import or execute the Ascend kernel.

```powershell
python -B -m unittest discover -s D:/Code/refs/DeepEP-Ascend/tests/afd -p test_cpu.py -v
```

On Linux, `bash tests/afd/run.sh` runs the same CPU suite. `gen_data.py OUTPUT.npz [--fp8] [--batch N]` creates deterministic raw-byte inputs. BF16/FP8 payloads and FP32 weights are represented as bytes/bits so NaN payloads are not canonicalized. `verify_dump.py INPUTS.npz SNAPSHOT.npz` checks a **provided real F snapshot**. It does not collect a snapshot or launch an operator. Its module docstring gives the required keys.

The independent oracle checks distinct target-rank records and every logical expert route. Physical record order may differ. `src_token_idx` is round-local: pass the matching input slice for each communication round. F `prefixes[e]` is the cumulative end of local expert e, not its start. Snapshot payload must be trimmed to `Inbox.num_recv_records * record_stride`; route_refs must be trimmed to the final prefix. Counts/prefix arrays have exactly E/N entries. Metadata padding is zero; unused record-stride holes have no zero requirement.

Three tests inspect real source rather than a substitute model: C++ wire declarations are parsed and compared with fixed ABI sizes/offsets; actual `make_workspace` arithmetic is evaluated against a separately formulated budget; actual TokenLayout expressions are evaluated against the independent byte layout. This restricted evaluator is **not a C++ compiler**: ABI packing/alignment, target compiler semantics and device API overloads remain unverified. Tests fail on grammar they cannot interpret.

Other tests validate the CPU oracle with corruption injections and the abstract protocol specification. They do not claim device branch coverage. Boundary arrays reach B=3*32768+1 without allocating full hidden payloads. Protocol tests cover zero-route F participation, delayed final ACK, 0→1→0, independent Decode slots and delayed Ledger reuse.

## A5 execution gap and required procedure

A two-process, single-round real kernel entry is provided below. There is currently no persistent test receiver kernel / multi-round A5 harness in this directory. Do **not** use a host `tensor.cpu()` poll loop on the same stream as a persistent send kernel: that can deadlock and does not test stream concurrency. There is no fake passing or skipped NPU test hiding this gap.

1. On the actual Ascend950 stack, initialize DeepJIT dependencies, build the DeepEP host extension, and compile the new JIT dispatchSend path. Record compiler, CANN, HCOMM/URMA and driver versions.
2. Add an F test-only device receiver that observes its local InputReady, snapshots Inbox/payload/count/prefix/route before ACK, and writes the matching Consumed to A. Include every F, including zero-route F. This requires the actual owner/control mapping, not a guessed global address.
3. Add an A receiver-side Ledger consumer on a genuinely concurrent stream. It must release each bank only after the test receiver has consumed the matching round. For multiple rounds do not make dispatchSend clear Ledger itself.
4. Use AFDBuffer.dispatchSend with its exact public out arguments. Assert final send_status PUBLISHED can precede final Consumed. Wait for both consumers before owner destruction. Snapshot checker inputs must be sliced by Ledger.token_begin/num_round_tokens.
5. Run plan L0-01..07 first, then L1 and L2. Delay ACK/one jetty/Ledger independently, force SQ wrap, test empty members and cross-transaction persistent state. Use an external all-process watchdog for faults.
6. Only after functional verification measure maximum-rank elapsed time with device events plus cross-rank completion. Report actual transmitted bytes separately from reserved memory and workspace; no performance result is available now.

Not covered by CPU execution: API compilation, host extension validation branches, real SIMT dedup/histogram, source tensor DMA/SF stride, remote offset mapping, queue wrap/CQ completion, URMA→UBMEM visibility, control cache behavior, peer errors/domain exit, simultaneous send/recv residency, memory guards on hardware, graph replay, or performance. None of L0/L1/L2 groups is marked NPU-passed.

## Actual single-round A5 entry (provided, not executed here)

After building the modified DeepEP extension, run from this repository under an external watchdog:

```bash
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --output /tmp/afd-smoke
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --fp8 --transposed-sf
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 0
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --all-masked
```

The script constructs the actual AFDBuffer on A=0/F=1, calls the actual JIT dispatchSend, checks PUBLISHED while final Consumed is still its initial value, then reads the actual F record/route/Inbox/Ready after sender completion and compares against the independent golden. It includes deterministic mixed masks, rank dedup (all 8 experts on F), raw BF16/FP8/SF bits and explicit optional SF transpose. Errors propagate; it does not fall back to CPU or silently skip. `--output` stores a snapshot compatible with verify_dump.py.

This is deliberately a single invocation. No ACK is fabricated, no slot is reused and no Ledger is released. All work is synchronized before explicit resource destruction. It proves neither the live Ready visibility ordering nor ACK, replay, multi-round, multiple members or concurrent progress. It was only AST checked in the current CPU environment; toolchain/API execution remains unverified.

For a future F receiver, the current context peer table lists F destinations only. F does not have an A control destination mapping through that table; extend the agreed receiver resource contract before implementing Consumed writes. This is a known follow-on interface gap, not something this smoke bypasses by guessing a pointer.
# Native-logic restoration regression matrix

`run_reuse_a5.py` invokes the actual local-F `dispatchRecv` kernel. It requires
an A5 SDK/NPU environment and does not implement a scheduler. Run:

```bash
timeout 1800s python tests/afd/run_reuse_a5.py --cores 8 --hidden 7168
```

Default coverage is 16 route counts (1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63,
64, 65, 129, 257, 513) x BF16 / FP8 row SF / FP8 column SF x cores 1 and 8:
96 valid invocations and 96 late-invalid-route invocations. Each case checks
byte-exact hidden/SF/metadata, padding, untouched output tail, untouched input
and control, and no partial outputs on invalid input. Entry-owned Segments are
not sorted by destination; long Segments cross core boundaries, and records
fan out to two experts. H=7168 uses 224 bytes SF per FP8 token. This is a
correctness suite, not a performance benchmark or proof of pipeline overlap.

For actual two-rank `dispatchSend` -> offline fixture -> `dispatchRecv`, exercise
batching across the 32-request drain threshold and four-token metadata tails:

```bash
for batch in 1 3 4 5 7 8 9 31 32 33 63 64 65 255 256 257; do
  timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch "$batch" --piece-size 17 --hidden 7168 --all-active
  timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch "$batch" --piece-size 17 --hidden 7168 --all-active --fp8
  timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch "$batch" --piece-size 17 --hidden 7168 --all-active --fp8 --transposed-sf
done
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 65 --hidden 7168 --fp8 --poison-masked-topk
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 65 --hidden 7168 --fp8 --all-masked --poison-masked-topk
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 257 --hidden 7168 --fp8 --all-active --cores 16
```

The all-active matrix makes token counts equal the per-destination record SQE
counts; default masks otherwise remove every third token. The two masked commands
exercise masked tokens with invalid topk sentinels inside metadata batches and
all-masked no-route publication. This harness synchronizes A
before constructing F Active; it does not validate async visibility, ACK reuse,
online scheduler behavior, or graph replay. The existing single-round sender
entry also uses exactly two ranks (A=0/F=1); multi-F execution still requires a
separate hardware harness. These short single-call cases do not exercise actual
SQ wrap; a long-running/cross-wrap device test remains pending. No C++ compilation or hardware results are
implied by the CPU fixture/oracle tests (`test_reuse_cpu.py`).
The final 16-core run covers more cores than the nine rank/expert prefix columns
in this topology; it requires a device exposing at least 16 vector cores.
