# dispatchSend

2026-10-16，developer-doc 初版。按 CANNBot 6.1 算子文档模板编制，描述当前 DeepEP-Ascend 改动；不是平台支持证明。源码基线 `3b25377d04b24fc6154698ded78a2bcb2c59afff`。

## 功能描述

`AFDBuffer.dispatchSend` 将 Attention 的有效 token 按目标 FFN rank 去重发送，生成专家 route reference，写 Inbox 后提交 InputReady。源输入保持完整 B；设备按 lane 的有限通信容量 C 分轮，路由统计分固定 tile 两遍执行。最后一轮发布后返回，不等待最终 DispatchConsumed。

本交付包含发送设备核、JIT 启动、资源 owner、wire ABI 和测试入口；五个算子现均有源码及入口，新增算子见[使用说明](remaining_operators.md)。当前机器未编译扩展或设备核、未执行 A5 测试，不能宣称完整闭环已运行。

## 接口原型

Python 由 pybind 直接导出 `deep_ep.AFDBuffer`，当前构造和方法均按位置传参，没有声明关键字参数或默认值：

```python
owner = deep_ep.AFDBuffer(
    rank_idx, group_name, src_domain_ranks, ffn_domain_ranks,
    lane_specs, layer_ids, hidden, num_sf_packs, num_topk, num_experts,
    num_token_capacity, use_fp8, num_vec_cores,
    num_cpu_timeout_secs, num_gpu_timeout_secs,
    memory_budget_bytes, transport_budget_reserve_bytes,
)
owner.dispatchSend(
    x, sf, topk_idx, topk_weights, token_mask, task_info,
    owner.context, owner.control_buffer, owner.round_ledger,
    owner.workspace, owner.send_status,
)
```

源码入口：[owner](../../csrc/buffers/afd.hpp)、[pybind](../../csrc/python_api.cpp)、[JIT](../../csrc/kernels/afd/dispatch.hpp)。此为 DeepEP JIT/pybind 路线，没有新增 aclnn 或 torch.library 注册框架。

## 参数说明

构造参数均为 Host 输入。同一通信域所有进程提供相同拓扑、lane、family 配置，按相同次序构造 payload/control 两个通信资源；rank_idx 为本进程通信域编号。

| 参数 | 类型 | 含义与当前约束 |
|---|---|---|
| rank_idx | int | 通信域 rank，范围 `[0, A+F)` |
| group_name | str | 已建立 HCCL process group 对应的 communicator 名称 |
| src_domain_ranks / ffn_domain_ranks | list[int] | A/F 域rank，均非空、互不重复；其并集覆盖整个域；当前域大小必须为偶数 |
| lane_specs | list[list[int]] | 每行6个uint32：`src_rank_idx, payload_slot_idx, dp_idx, member_idx, phase, capacity`；src_rank_idx 是 A列表下标，不是通信域rank |
| layer_ids | list[int] | 非空合法 layer ID 白名单；当前 owner 只承载同一 H/dtype/SF family |
| hidden | int | H，正数且为128倍数 |
| num_sf_packs | int | 每token的int16 SF pack数；BF16必须0，FP8必须正数，不从H推导 |
| num_topk | int | K，`0<K<=min(32,E)` |
| num_experts | int | E，`0<E<=512`，能被F数整除 |
| num_token_capacity | int | 单 rank 完整事务 maxBS（B_alloc），输入tensor固定第一维；正数。实际逻辑 token 范围 B由device task.num_tokens决定，可为0；不是有效 mask 计数 |
| use_fp8 | bool | False选择BF16；True选择E4M3FN |
| num_vec_cores | int | 正数，至多64且不超过runtime可用AIV数；并发收发核驻留仍待验证 |
| num_cpu_timeout_secs / num_gpu_timeout_secs | int | 正超时；不等于全域容错能力 |
| memory_budget_bytes | uint64 | owner已知分配与transport reserve之和不得超过该预算；不含调用者输入tensor等完整进程内存 |
| transport_budget_reserve_bytes | uint64 | 必填正估算值。不是驱动实际占用测量，也不证明transport分配必在该值以内 |

lane容量 `0<C_m<=32768` 是当前实现保留的单物理槽单轮限制；不限制同DP成员容量之和。`(src,slot)`唯一；同一逻辑成员的多个microbatch槽仍须满足当前同源、同capacity校验。构造时选择能放入runtime UB的最大 T∈{32,64,128,256}。

容量映射：原生 `num_max_tokens_per_rank` 是一次原生dispatch调用每rank输入上界；AFD在它外层增加完整事务分轮，因此保留该字段名表示 `C_max=max(lane.capacity)`，仅服务单窗口JIT/路由workspace，不能把它当maxBS。完整输入上界单独由 `num_token_capacity` 承载。`task.num_tokens=B<=num_token_capacity`，`lane.capacity=C_m`；不要将同名参数跨层直接传递。

一个DP事务为每成员选择一个物理槽，通信轮数 `W=max(1,max_m ceil(B_m/C_m))`，各成员task携带相同W。第r轮范围为 `begin=min(B_m,r*C_m)`、`length=min(C_m,B_m-begin)`。先耗尽者仍发布零长度Inbox至W轮结束；mask为空不缩短逻辑B。TP非owner使用B=0参与到齐，不重复计数；CP/其他不相交owner按各自B计数。每rank B=256k、C_m=32k的两个owner各8轮，DP每轮可64k；不是分成16个DP轮。

F输入预留对所有已注册物理槽求和（包括当前未选D槽及非owner槽），单F每槽为 `align512(C_m*record_stride)+align32(4*C_m*min(K,R))+2*align32(4*R)`，另算控制/目录；R为本地专家数。A返回窗口仅按自身各槽的C_m计，普通HBM完整输入输出按maxBS单独计。增加一个32k owner确实增加F预留；不改变每片Q/Q_alloc的计算workspace。当前send内部多轮等待真实Consumed，测试不能伪称缺失的combine/scheduler已完成该闭环。

`dispatchSend` 参数全为本 NPU 上的 Tensor（可选项为None），返回 `None`：

| 参数 | 方向 / dtype | shape与要求 |
|---|---|---|
| x | 输入，BF16或float8_e4m3fn | 连续 `[B_alloc,H]`，和owner family一致 |
| sf | 可选输入，int16 | FP8必须 `[B_alloc,num_sf_packs]`；两个stride均正、至少一个为1、都不超过INT_MAX；BF16必须None |
| topk_idx | 输入，int64 | 连续 `[B_alloc,K]`；有效token的专家ID合法且不重复 |
| topk_weights | 输入，float32 | 连续 `[B_alloc,K]`；原始bit搬运，不在F加权 |
| token_mask | 可选输入，bool | 连续 `[B_alloc]`；False的token不校验topk也不发射route |
| task_info | 输入，uint8 | 连续64B、32B对齐；字段见下 |
| context | 只读，uint8 | 必须是owner.context同storage/data_ptr；不得自行构造或改写 |
| control_buffer | 双向，uint8 | 必须owner原tensor；接收Consumed、保存后续控制空间 |
| round_ledger | 双向，uint8 | owner每槽2×64B；send发布，未来A recv释放 |
| workspace | 双向，uint8 | owner固定C预算；含持久send_state，不得每调用清零 |
| send_status | 输出，uint32 | owner `[8]`；异步状态，host查看前须完成相应同步 |

输入不得与owner可变/控制存储非法别名；当前host检查这类alias。context等out实参受owner身份校验。调用者必须保活输入、task、owner直到设备不再引用；同owner工作区不能被多个send stream并发复用。

TaskInfo ABI为16个小端uint32：`abi_version=1, src_rank_idx, payload_slot_idx, microbatch_idx, layer_idx, num_tokens, num_comm_rounds, reserved[9]=0`。`0<=num_tokens<=B_alloc`；共同轮数至少 `max(1,ceil(B/C))`，同DP各A成员必须一致。少token成员后续发布空轮。完整定义见 [context.hpp](../../deep_ep/include/deep_ep/layout/afd/context.hpp)。

## 支持数据类型与布局

纯bit搬运：BF16 hidden或FP8 E4M3FN hidden；SF int16 pack、topk int64、weights FP32。F每token每目标rank一个record，每有效专家route一个uint32 `record_idx*K+topk_slot`。metadata的src_token_idx为**轮内token下标**，用Ledger.token_begin恢复全B下标。F prefix为本rank各专家累计end，专家e区间为`[e==0?0:prefix[e-1],prefix[e])`。

Record/metadata padding见 [TokenLayout](../../deep_ep/include/deep_ep/layout/afd/token.hpp)。metadata对齐padding清零；record_stride中未使用尾部不承诺清零。实际UB/workspace由family/C决定，无完整hidden发送镜像。A端目前仍按每本地lane `C*K*max(15360,align512(2H))`预留未来返回窗口，不能把该预留称作已实现combine功能。

## 约束与生命周期

Ready/Consumed初值1、首次发布bit0，之后交替。send唯一推进输入位。写下一轮前等所有F的上一Consumed（含零route F），并等待目标Ledger bank EMPTY；最后发布返回不等最终ACK。`send_status=PUBLISHED`仅代表发布完毕，不代表槽可复用。业务下一事务须由框架保证前事务完整结束；不得通过清状态绕开ACK。

当前 dispatchRecv 只消费已经选定的 Active，不发送输入 ACK。多轮运行需要真正并发的F消费者和A Ledger消费者，否则等待/超时符合当前接口，不是完整闭环已实现。F的context peer表目前只有F目的地址，缺少F回写A Consumed所需A控制映射；后续receiver资源接口需补齐，禁止猜测指针。

所有rank初始化完成后再启动通信。正常业务销毁前：停止新launch，完成事务含最终ACK，确认所有设备及远端访问结束，再调用 `owner.destroy()`。该方法同步本设备、标记closed并释放owner引用；不是跨rank停止协议。导出的payload/control Tensor通过deleter保留资源owner，外部引用未释放时分配可能继续存活。不要在资源销毁后使用借出的通信tensor，不要重复调用destroy。

隔离单轮smoke不启动消费者或下一事务；全域确认sender已drain且快照完成、无剩余设备工作后销毁。这是测试退出，不能当作正常事务闭合证明。

## 返回值 / 错误码

`send_status[0]`：IDLE=0、RUNNING=1、PUBLISHED=2、FAILED=3；`[1]`为错误码，`[2]`为末轮索引字段（失败时不能视为已完成进度）。错误枚举：OK=0、BAD_ABI=1、BAD_TASK=2、BAD_LAYER=3、BAD_ROUTE=4、CAPACITY=5、STATE_ORDER=6、TIMEOUT=7、TRANSPORT=8；部分为ABI预留，不保证每条错误路径都会提交对应码。

Host不合规参数由现有断言机制拒绝。控制等待可提交TIMEOUT，但底层queue断言/超时仍可能trap，不能保证FAILED总会写回，更不能保证其他rank有界退出；外部全域watchdog和故障退出验证是生产准入阻断项。

## 调用示例与部署

使用本仓现有 [README](../../README.md) 的Linux Ascend依赖和DeepJIT构建方法；初始化子模块，安装匹配torch/torch_npu/CANN/HCOMM栈，再构建修改后的扩展。当前环境只做了CPU检查，没有验证这些步骤能成功。

实际完整调用入口是 [run_a5_single_round.py](../../tests/afd/run_a5_single_round.py)，包括HCCL初始化、17参数构造、Task ABI打包、实际dispatchSend调用、真实F快照oracle与显式destroy：

```bash
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --fp8 --transposed-sf
```

这些命令已提供、尚未实机执行；不是已验证最小示例。更多B=0/allmasked以及CPU命令见 [测试说明](../../tests/afd/README.md)。单轮同步后快照验证不证明实时Ready可见性、ACK复用、多轮、图replay或性能。

## 复用与支持芯片

目标为Ascend950/dav-3510；未声明其他芯片支持，亦未获得本实现A5运行通过证据。

| 原DeepEP机制 | 本次处理 |
|---|---|
| ep/dispatch.hpp SIMT histogram与路由 | 固定tile两遍统计/发射；直接使用common/simt.hpp deduplicate，保留最高lane代表语义 |
| TokenLayout及metadata搬运 | AFD布局增加轮内token/mask；复用原生 preload_metadata/batch_idx/stage_idx 命名及双stage预加载/回填，每stage处理4个token，保留SF连续搬运及NDDMA分支 |
| tile prefix | 各核并行处理独立rank/expert列，汇合后生成有序expert base；不增加scratch |
| HcommJetty/HcommPeer | 每核预加载peer信息；复用原生build_write_sqe、SQ wrap、head及CQ处理；一条record用hidden/metadata两SGE、两WQEBB的SQE，至多32个SQE批量doorbell后drain，尾批显式提交 |
| HCCLContext | 复用初始化/注册/channel；AFD路径允许不等大peer buffer，原EP默认约束保留 |
| EP JIT与pybind | 增加AFD kernel key/owner/public导出，原EP入口保留 |

metadata stage沿用MTE2→S就绪和MTE3→MTE2复用保护，最后一次release也会被消费。当前tile的metadata写完之后才提交引用它的传输；所有在途SQE drain后才覆盖该tile或发布InputReady。恢复的是metadata搬运流水与队列批量提交，没有恢复原生persistent SIMT worker对metadata生产和传输的持续重叠。

原生六组token-pair合并到一个SQE的实现尚未恢复：AFD record_stride带对齐间隙，且SIMT分配的目标slot不等于源token遍历顺序，直接拼接原生SGE会写错目标位置。当前保持一record两SGE；进一步合并需要有界slot排序/填洞方案。每SQE仍请求CQE，route/count/prefix传输及控制发布仍由core0执行；未增加完整hidden镜像、未恢复全rank collective。

本轮联合CPU suite为27项（原sender10+recv15，再加2项复用回归测试）通过，13个Python文件AST检查通过；这些检查不执行C++设备核。新增单卡/双rank矩阵见 [测试说明](../../tests/afd/README.md)，均尚未上卡。A5 API重载、跨URMA/UBMEM可见性、队列完成、并发前进、故障域退出、图replay及性能均未验收。本说明不宣称完整CANNBot流程或生产准入通过。
