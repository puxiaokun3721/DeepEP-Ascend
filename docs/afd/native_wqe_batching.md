# AFD发送侧WQE批量构建（2026-10-18）

优化前快照：`c44c626`；原生参考：`3b25377`。本轮只改dispatchSend/combineSend的payload SQE构建。dispatchRecv、combineRecv和ffnScheduler未在本轮优化。

## 原生复用与AFD差异

复用 `impls/ep/combine.hpp::load_meta_issue_wqe` 的warp-per-WQE方式：每个lane构建一个uint64 word，经asc_stcg直接写SQ。保留thread_idx、token_idx、lane_idx、task_sqebb_idx、token_sqebb_idx、peer_info_words和wqe_word变量逻辑，以及原生header template、peer words和SGE编码。

最多8个WriteTask描述符保存peer信息地址、目的offset、hidden/metadata源地址及长度。8×40=320B复用原有512B ub_sqe，不增加UB/HBM分配或hidden镜像。描述符准备仍是标量循环，尚未移植原生persistent SIMT worker。

| 路径 | 旧AFD | 当前 |
|---|---|---|
| dispatchSend payload | 逐record构建并DMA一个SQE | 最多8record并行构建，仍每record两SGE、两WQEBB |
| combineSend payload | 逐route标量put | 最多8route并行构建，仍每route一SGE、一WQEBB，保留x-direct |
| 完成机制 | 每WR强序/fence/CQE，最多32WR提交后drain | 不变；没有采用原生末尾NOP合并CQE |
| 控制表 | HcommJetty标量put | 不变 |

AFD record对齐间隙及非连续目标slot不允许直接套用原生六组SGE pair拼接；但不妨碍批量构建独立SQE。本轮恢复后者。

## 同步和容量证明

S填描述符 → S→V → SIMT写SQ并threadfence → V→S → 更新packed_head → 满32或尾批doorbell → drain。

1. V→S完成后才覆盖UB描述符；NIC只引用GM输入/metadata。
2. dispatchSend在tile metadata的MTE3完成后才构建WQE，drain后才进入下一tile；所有核完成后发布Ready。
3. combineSend保留全核预检、credit取得、所有payload完成后Done/Consumed的顺序。
4. 每组最多32个未回收CQE。现有校验要求CQ深度为2幂且大于42，最小64；dispatch每组最多64个SQ块，远小于32768。下一组开始前上组已回收。jetty仍要求kernel间串行使用。
5. 每WR完成计数加一，SQ head按1/2 WQEBB增加，均按uint32回绕。每个WQEBB独立取模，包括两块WQE从最后一个SQ块跨到块0（前序标量控制WR可能留下奇数head）。
6. empty不发VF、不增加完成计数；不足8条的尾批在最终drain前生成并提交。
7. public接口、ABI、maxBS/C_m/Q/Q_alloc及交替位/全F闭合不变。

## 验证与边界

P条payload WR原有P次SQ DMA及逐条S/MTE3往返，改为ceil(P/8)个SIMT任务及S/V交接（dispatch逐tile处理尾批）。payload字节数、WR/CQE数不变，标量目的地址准备仍存在。没有实测时延或带宽结论。

本地53项CPU测试通过；新增2项Python模型检查1/2SGE、SQ owner、奇数head跨环、uint32回绕、空批/尾批与CQ预算。测试没有执行C++，不代表CANN编译、设备精度或性能通过。

A5验收：host/JIT编译两个SIMT实例，确认UB指针描述符和VF参数；BF16/FP8/SF/metadata bitwise；真实五算子闭环；零route/mask/短成员；两种credit策略；SQ环回/CQ owner；与c44c626对比时延和SIMT寄存器spill。

实机压力入口（未运行，host观测不用于计时）：

```bash
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --batch 65536 --window 32768 --q 4096
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --batch 65536 --window 32768 --q 4096 --short-second 0 --precheck
```

脚本单AIV且约80%token有效，可使dispatch与热点F combine的SQ累计越过环尾。不能在测试中手工清零队列来规避环回。
