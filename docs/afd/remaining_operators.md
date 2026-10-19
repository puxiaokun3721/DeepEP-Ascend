# AFD scheduler 与返回算子使用说明

基线 `3b25377d04b24fc6154698ded78a2bcb2c59afff`。本次交付是源码、host/JIT接入与验证入口；尚未通过CANN编译或A5设备验收。原生EP路径未替换。

## 初始化与调用

所有rank先按同样顺序创建AFDBuffer，完成初始化后再启动通信。每个source rank的`num_token_capacity`是完整事务maxBS；每个lane的capacity独立限制通信轮，DP容量没有固定32k上限。图外创建资源，图replay不得重置状态。

F先调用`prepare_recv(q,q_alloc,expert_alignment)`，再调用`prepare_ffn(precheck=False)`。后者申请私有Pending、Completion、两个专家prefix/count张量、返回scratch及BF16 `combine_x[Q_alloc,H]`。生产owner在原payload endpoints上注册这一个实际GMM2输出，无第二份hidden发送镜像。

```python
owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
    owner.active, owner.execution_state, owner.completion,
    owner.psum_num_recv_tokens_per_expert, owner.num_unaligned_recv_tokens_per_expert)
owner.dispatchRecv(owner.payload, owner.context, owner.active, owner.execution_state,
    owner.recv_workspace, owner.recv_x, owner.recv_sf, owner.recv_src_metadata)
# 实际GMM链必须用device Active.valid/kind/layer gate，GMM2写owner.combine_x。
# 此处没有自动集成GMM或权重选层实现；不能用host读Active冒充图模式。
owner.computeReady()  # 同stream，在实际GMM2完成之后；不执行GMM、不验证其精度。
owner.combineSend(owner.combine_x, owner.recv_src_metadata, owner.context,
    owner.active, owner.control_buffer, owner.execution_state,
    owner.combine_send_workspace, owner.completion)
```

`computeReady()`对WORK将INPUT_READY推进到COMPUTE_READY；CONTROL仍为INPUT_READY；NOP/ABORT不访问输出数据。未按状态完成的链进入ABORTED，不发伪成功ACK。上层必须中止失败通信域；本次不新增分布式故障恢复服务。

A先调用`prepare_combine_recv()`，然后在与dispatchSend独立且可共驻的stream提交：

```python
owner.combineRecv(topk_idx, topk_weights, token_mask, task_info, owner.context,
    owner.payload, owner.control_buffer, owner.round_ledger,
    owner.combine_recv_workspace, combined_x, owner.recv_status)
```

`combined_x`为预分配BF16[maxBS,H]；只写实际B行，inactive行清零，容量尾部不写。不要让recv等待多轮send的最终完成事件，否则send可能等ACK而recv尚未启动。host要求2×num_vec_cores不超过可用AIV数，但这只是共驻必要条件；并发kernel、SyncAll资源隔离仍须实机验收。单个owner的scratch要求同类算子调用串行；多D槽通过顺序发布独立任务共享调度，不授权多个dispatchSend并发写同一owner scratch。

## 实现与原生逻辑对照

| 算子 | 保留机制 | AFD必要差异 |
|---|---|---|
| ffnScheduler | 现有route/expert-prefix语义，DMA/cache helper | 原生无独立AF调度器；新Pending、DP到齐、同phase/layer聚合、Q/Q_alloc切片和Completion消费 |
| combineSend | 原生x直发分支、HcommJetty、SQ/CQ、peer缓存、metadata双stage、32请求批提交 | 四列metadata、独立A/F地址、credit、末片全成员Done→Consumed、Completion快照 |
| combineRecv | token分核、GM/UB搬运、双stage输入生命周期 | 原生rank-partial求和改为逐route FP32乘权后升序累加，最终BF16；Ledger逐轮推进及全F闭合 |

scheduler首版单AIV，只扫描metadata，不扫描token或hidden；成员匹配最坏O(L³)，大注册槽数是明确性能风险。默认先算后等；PRECHECK可在选择前绕过无credit候选，不能抢占已计算且占用唯一workspace的任务。

## 新增固定资源与ABI

原Context224B、Active64B及dispatch布局保持不变。Context.reserved_address现在是可选返回目录的**字节offset**：32B目录、32B×N_A source peers和32B×L返回lane。旧fixture没有该扩展时不可调用新增算子。

- F Pending：`32+96L+2*A32(4LR)`；每lane保留Inbox副本、expected bit、grant、首次出现序号，另存count/consumed。
- Completion：`32+64L+A32(4LR)`；先快照，后ACK，scheduler不从已释放Inbox恢复旧任务。
- F prefix/count：`8R`字节；scheduler UB：`256+5*A32(4R)`。
- F返回scratch、A归并scratch各为`256+32NC`，彼此及发送scratch独立。
- GMM2实际输出：`2*Q_alloc*H`字节；只申请/注册一份，属于计算workspace。
- A输出[maxBS,H]由调用者提供，普通HBM单独核算。
- 新增source注册及driver资源实际占用仍需通过transport reserve和实机测量计入，不能以业务scratch替代。

初始Ready/Done/Consumed为1，期待输入bit与首轮return bit为0，F本地credit为0。A每轮读取完毕后写下一credit；F末计算片所有返回写完成后发Done再Consumed。零route成员也参与；短成员参加共同计划的全部零长度尾轮。

## 验证入口与边界

本地：`python -m unittest discover -s tests/afd -p 'test_*cpu.py' -v`。

A5真实五算子入口（尚未执行）：

```bash
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --short-second 0 --precheck
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --all-masked
```

该入口测试两个owner、热点F/零route F、多通信轮、多计算片及真实ACK/credit，用identity expert代替GMM数值，不伪造控制位。启动前应准备JIT cache，避免活跃事务期间冷编译超过设备timeout。它有host观测，仅用于bring-up，不证明图模式或性能。

仍需A5验证：host扩展/JIT编译、裸UB LocalTensor及Cast/Muls/Add重载、短尾DMA、队列注册与CQ完成语义、URMA写到UBMEM控制的可见性、并发核活性、P单槽反复事务、D的M=2/3、混合phase、图replay和性能。禁止将CPU模型测试当作设备精度测试。

当前实现：combineSend payload改用原生warp-per-WQE批量写SQ路径，与dispatchSend共享有界提交实现。每route一SGE、每WR一CQE、credit/Done/Consumed顺序不变；参见[当前实现边界](README.md#当前关键实现边界)。

当前实现：ffnScheduler复用Pending预留字段建立每次调用的成员索引，将成员到齐查找最坏复杂度从O(L³)降至O(L²)，不增加分配。详见[当前实现边界](README.md#当前关键实现边界)。
