# dispatchSend white-box coverage map

Source: `deep_ep/include/deep_ep/impls/afd/dispatch_send.hpp`, first implementation snapshot 2026-10-16. These are source-derived **expected paths**. Runtime device path coverage is unavailable (no CANN/NPU). CPU models are not counted as device branch hits. Line numbers can move during concurrent implementation; condition text is the stable locator.

| Source branch | Trigger / plan cases | Current evidence |
|---|---|---|
| route_tile: live, token_mask==nullptr, gl<K (46–48) | B=0/1/31/32/33/255/256/257; K=1/8/32; mask absent/mixed/allfalse | oracle mask/boundary tests; device pending |
| value<0 or value>=E, duplicate expert (50,56) | negative/E/2^32+legal ID, duplicate; invalid masked row L2-04 | CPU oracle negative/E/duplicate/masked covered; int64 truncation device pending |
| representative and rank>=0 (62) | same-F multiple experts, all-F, K32 highest representative | expected set oracle; actual SIMT representative pending |
| Emit false/true (64,84) | every nonempty tile; NC>tiles, B not divisible T | actual two-pass device pending |
| wait_control success/timeout (115) | immediate, delayed, never ACK/Ledger | protocol model delays only; hardware timeout pending |
| inflight==32 (127) | 31/32/33 and multiple batches, SQ near 32768 wrap | source reviewed; no queue model counted as transport validation |
| task ABI/lane/src/B/round/layer (167–175) | each invalid field independently L2-05 | device pending |
| collective error exit (182,204,215,278) | one core error, others at local_join | device/watchdog pending |
| previous.has_published (198) | fresh slot vs reused across round/transaction | specification tested; kernel pending |
| round length/tiles loop (205–208) | B=0/C-1/C/C+1/3C+1, common rounds beyond local B | oracle B boundary; device pending |
| core0 prefix columns rank/expert (216–229) | E/N1/8/512, empty experts, tile>1 | golden prefix fixture checked; device pending |
| live=false and S>0 (241,247) | BF16; FP8 with SF token/hidden strides | bitwise SF oracle checked; actual gather pending |
| K==32 mask (262) | K1/8/32 | K32 source path pending |
| routes==0 (290) | no route to selected F; B0/allfalse | expected zero F tested; physical notification pending |
| final code==OK (325) | success vs ABI/route/timeout failure | source only, device pending |

Host constructor and dispatchSend validation branches are in `csrc/buffers/afd.hpp`. No target extension/NPU runtime is available here, so real host executable coverage is zero. Required negative fixtures include H=0/129, E=0/>512, N=0/not dividing E, K=0/33/>E, per-lane C=0/>32768, invalid dtype/SF stride, alias, owner/context identity, odd communication domain, region/uint64 overflow, insufficient UB and transport budget. DP C sum>32768 is a POSITIVE case: `run_capacity_a5.py` exercises local receiver construction for two same-DP 32768 lanes, heterogeneous lanes and per-lane32769 rejection. It remains unexecuted. Do not mark these passed because the independent golden rejects equivalent dimensions.

The CPU workspace test covers C=1/32/256/32768 × T=32/64/128/256 × metadata=128/352 (32 parameter rows). Full-B is deliberately absent from the implementation workspace formula, preserving bounded workspace; this does not measure allocation high water on A5.
