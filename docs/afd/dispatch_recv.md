# dispatchRecv

修订：2026-10-16，developer-doc，依据实际 AFD host/device/JIT 和测试入口编制。目标为 A5 / Ascend 950（DAV_3510）；当前仅完成源码、CPU 模型验证和静态审查，host 扩展、Bisheng JIT、NPU 精度、同步、图模式和性能均未验收。

## 功能说明

`AFDBuffer.dispatchRecv` 消费已经选定且保持稳定的 device Active，将 F 本地输入窗口的记录按专家展开到当前计算片输出。它不选择任务、不轮询 DP 到齐、不实现 ffnScheduler、GMM 或 combine，也不发送 DispatchConsumed / 输入 ACK。

对每个 Segment 的第 j 个 route，读取 lane 的整个 route_reference 表中的 `route_begin+j`：`record_idx=reference/K`，`topk_slot=reference%K`。从对应去重 record 逐 bit 拷贝 hidden/SF，输出四列 `int32` metadata：`[source_lane_idx, src_token_idx, topk_slot, entry_idx]`。`src_token_idx` 来自 record，是通信轮内原 token 编号；它不等于去重后的 record_idx。route_begin 是整 lane 表的绝对下标，不能重复加 expert prefix。

每个专家的有效行数 c 来自其所有 Segment 的 num_routes；物理行数为 `align(c, expert_alignment)`。有效行必须紧密覆盖，专家尾部 padding 的 hidden/SF 全 bit 零、metadata 全 -1。`Active.num_padded_tokens` 之后的输出尾部不写。所有有效数据预检成功并完成全部核的输出写入后，才发布 `ExecutionState.INPUT_READY`。

## 接口原型与 owner

```cpp
void AFDBuffer::dispatchRecv(
    const at::Tensor& buffer, const at::Tensor& context,
    const at::Tensor& active, at::Tensor& execution_state,
    at::Tensor& workspace, at::Tensor& recv_x,
    std::optional<at::Tensor>& recv_sf, at::Tensor& recv_src_metadata);
```

| 顺序/参数 | 方向与类型 | 说明 |
|---|---|---|
| 1 buffer | 输入，NPU uint8 连续 | 必须是此 owner 的 `payload`，包含稳定 records/routes/count/prefix |
| 2 context | 输入，NPU uint8 连续 | 必须是此 owner 的 `context`；静态目录，初始化后不可修改 |
| 3 active | 输入，NPU uint8 连续 | 必须是此 owner 的 `active`；固定容量、device 动态内容 |
| 4 execution_state | 输入输出，NPU uint8 连续 | 必须是此 owner 的 `execution_state`，含 header 与 grants |
| 5 workspace | 暂存，NPU uint8 连续 | 必须是 `recv_workspace`，不能传 sender 的 `workspace` |
| 6 recv_x | 输出，连续 `[Q_alloc,H]` | BF16 或 E4M3FN，须与 wire 同类型 |
| 7 recv_sf | 可选输出，int16 `[Q_alloc,S]` | BF16 为 None；FP8 必须存在，保存原始 sf_pack_t bit |
| 8 recv_src_metadata | 输出，连续 int32 `[Q_alloc,4]` | 固定四列，不是原 EP 的 K+2 列 |

Python 的 `active`、`execution_state`、`recv_workspace`、`recv_x`、`recv_sf`、`recv_src_metadata` 均为只读属性 getter，返回可写 device Tensor。不得 resize owner 资源或修改静态 Context。输出可另外预分配，但 shape、dtype、device、stride 和非 alias 条件必须匹配。生产通信 receiver 沿用已有 AFDBuffer 构造函数，再图外调用一次：

```python
owner.prepare_recv(q, q_alloc, expert_alignment, sf_column_major=False)
```

q 是本片有效行上限，q_alloc 是 padding 后容量；要求 `q>0`、`q_alloc>=q`，alignment 是正的 2 次幂。它预分配 Active、state、固定 scratch 和输出，不在每次 dispatchRecv 建链或分配通信资源。内部 launcher 另从 owner 取得 local control pointer 与 F rank；不改变公开八参数接口、不增加可变全局 registry。

单卡测试 factory 的真实参数顺序为：

```python
deep_ep.AFDBuffer.create_local_receiver(
    num_ranks, rank_idx, lane_specs, layer_ids, hidden, num_sf_packs,
    num_topk, num_experts, num_token_capacity, use_fp8,
    num_vec_cores, memory_budget_bytes)
```

`lane_specs` 每项为 `[src_rank_idx,payload_slot_idx,dp_idx,member_idx,phase,capacity]`。local factory 只创建同 ABI 的本地窗口/控制资源，供 fixture 注入；它不创建 HCCL/URMA 通信域，也不是 scheduler 实现。真实通信 owner 的既有构造/销毁约束见 [dispatchSend](dispatch_send.md)。

## 公共 ABI 与 wire

容量口径：`num_token_capacity` 是A侧完整事务maxBS；纯F保留正占位值，dispatchRecv按各lane和Inbox的本轮范围校验源token编号，不以该字段限制DP或来源完整事务。`Context.num_max_tokens_per_rank=max(lane.capacity)` 是兼容原生命名的单次窗口编译上界。各lane仍有 `0<C_m<=32768` 的当前实现限制，但同DP两个32k owner合法，输入预留增加为两槽之和。`prepare_recv(q,q_alloc,expert_alignment,...)` 中q仅限制单片有效专家行，q_alloc限制加padding后的物理行；注册更多来源不能自动扩大二者或输出buffer。scheduler现有独立源码及入口；旧fixture仍只提供Active，不证明设备到齐/分片调度正确。共享通信轮、mask与TP owner语义见[dispatchSend容量说明](dispatch_send.md)。

真实定义在 `deep_ep/include/deep_ep/layout/afd/active.hpp`，全部结构 32B 对齐；所有下列字段除明确标注外均为 uint32。

| 结构 | 大小 | 字段顺序 |
|---|---:|---|
| AFDActive | 64B | valid, kind, phase, layer_idx, num_entries, num_members, num_segments, num_expanded_tokens, num_padded_tokens, reserved, uint64 entries_offset/members_offset/segments_offset |
| AFDActiveEntry | 32B | pending_idx, member_begin, num_members, segment_begin, num_segments, num_rows, is_final, grant_state |
| AFDActiveMember | 32B | source_lane_idx, input_bit, return_bit, delta_begin, num_experts, reserved[3] |
| AFDActiveSegment | 32B | entry_idx, source_lane_idx, local_expert_idx, route_begin, num_routes, dst_row_begin, reserved[2] |
| AFDExecutionState | 32B header | stage, error_code, num_entries, reserved[5]；随后 uint32 grant_state 数组 |

设 L 为 source lane 数、R 为 F 本地专家数，`A32(x)=align32(x)`：entries_offset=64，members_offset=64+32L，segments_offset=64+64L；Active 字节数 `64+64L+32LR`，ExecutionState 字节数 `32+A32(4L)`，workspace 字节数 `32*num_vec_cores+256`。Entry/Member 容量 L，Segment 容量 LR。offset 固定且不能重叠；动态计数不可超过容量。Segment 的 24B 逻辑字段不代表物理 stride，实际 stride 为 32B。

复用 sender 的 Context 224B、Workspace 96B 和 TokenLayout，不改变发送协议。设 X 为 hidden 字节数、B 为 SF 字节数：SF offset=X，topk_idx offset=`X+A32(B)`，topk_weights offset=`X+A32(B)+A32(8K)`，src_token_idx 再加 4K，valid_topk_mask 再加 4。metadata stride M 为 `A32(valid_topk_mask_offset+4-X)`，record stride D 为 `align512(valid_topk_mask_offset+4)`。发送端只传 X+M，recv 也只读 X+M；不能把 D 的未发送尾洞读作有效数据。

K≤32，R≤512，H 为正且 H%128=0；地址/乘法使用 uint64 和减法式边界检查。route_reference 不得为 UINT32_MAX，record_idx 必须落在已发布 record 数内，token 在本轮 token 数内，mask 必须包含 slot，record 的 int64 expert 必须等于 `F_rank*R+local_expert_idx`。Inbox、Ready bit、Member、层、phase、轮、entry/grant 和目录必须一致。padding 不读取无效 route。

## 状态、错误与资源生命周期

| kind | 行为 |
|---|---|
| NOP=0 | valid=0、零计数且 state=EMPTY；保持 state 和输出 |
| WORK=1 | valid=1、有效 route 总数非零；可含零行 Entry；SELECTED→INPUT_READY |
| CONTROL=2 | valid=1、SELECTED、至少一个 Entry；每 Entry.is_final=1，所有 selected Member 的本 F wire route 总数和 Inbox.num_recv_records 均为 0；零 Segment、零展开及 padding；预检后 INPUT_READY |
| ABORT=3 | valid=0；不解引用旧任务表，保持已有 state/error 与输出 |

Execution stage 为 EMPTY0 / SELECTED1 / INPUT_READY2 / COMPUTE_READY3 / COMMITTED4 / ABORTED5。Entry 与 state 的 grant 必须一致，仅 NONE0 或 HELD1 可接收，拒绝 FINISHED2。重复在 INPUT_READY 上调用会因 STATE_ORDER 中止，不能冒充新 SELECTED。

CONTROL 不允许用“非空 wire 已消费完”的空 Active 代替：recv 没有 scheduler 消费游标。其零 route 判据为每 selected Member 的 `prefix[R-1]==0` 且 `Inbox.num_recv_records==0`，违反则 BAD_TASK 且输出不写。`num_round_tokens` 可以大于 0（token 全部路由到其他 F）；Entry.is_final 表示本 F 条目完成，不要求当前通信 round 是事务末轮。

host 的容量、dtype、stride、identity、地址 span 或 alias 错误在 launch 前拒绝。device 预检错误置 ABORTED，保留首个错误且不写输出；错误码沿用 OK0、BAD_ABI1、BAD_TASK2、BAD_LAYER3、BAD_ROUTE4、CAPACITY5、STATE_ORDER6（TIMEOUT7、TRANSPORT8 为公共枚举，不表示 recv 增加了传输轮询）。所有核采用一致 gate/join 路径，空核也参与同步。

输入、Context、Active、state、scratch、control、全部输出以及已存在 sender ledger/workspace/status 的实际 span 不能相互 alias。FP8 SF 接受 pack 单位的两种布局：`stride1=1,stride0>=S` 或 `stride0=1,stride1>=Q_alloc`，正 stride 且不自重叠；实际 span 为 `2*((Q_alloc-1)*stride0+(S-1)*stride1+1)`。SF 不进行格式转换；hidden 的 BF16 NaN/负零及 FP8 特殊 bit 都按字节搬运。

调用异步入当前 stream；外部必须保证整个 kernel 完成之前窗口、Active、owner 和输出仍存活且稳定。取得 INPUT_READY 不代表输入已被 ACK，也不能因此复用输入槽。同一稳定窗口可由测试依次安装不同 Active 消费多片；生产ACK/末片完成由combineSend负责，单独dispatchRecv结束不得据此回收槽位；闭环仍待A5验证。借用 owner Tensor 时必须保持 owner 生存，销毁前同步完成在途访问。device 动态 Active/state 没有改为 host 读取行数选路径，保留了后续图模式所需接口条件；当前不宣称图捕获/replay 支持。

## 调用示例

以下使用测试 fixture 构造真实 ABI 并调用实际 JIT kernel。需在目标仓根目录运行；fixture 模块不是 scheduler。

```python
import sys
import torch
import torch_npu
import deep_ep

sys.path.insert(0, "tests/afd")
from recv_reference import sample_fixture
from recv_npu_support import install_window, install_active

torch.npu.set_device(0)
fixture = sample_fixture(False)
owner = deep_ep.AFDBuffer.create_local_receiver(
    1, 0, [[0, 0, 0, 0, 0, 8], [1, 0, 0, 1, 0, 8]], [0],
    128, 0, 2, 2, 8, False, 8, 64 * 1024**2)
owner.prepare_recv(8, 12, 4)
try:
    install_window(owner, fixture)
    install_active(owner, fixture["segments"], [[0, 1]])
    owner.dispatchRecv(owner.payload, owner.context, owner.active,
        owner.execution_state, owner.recv_workspace,
        owner.recv_x, owner.recv_sf, owner.recv_src_metadata)
    torch.npu.synchronize()
    # Inspect execution_state/error before consuming current-piece outputs.
finally:
    torch.npu.synchronize()
    owner.destroy()
```

## 构建与测试

沿用 DeepEP 原工程，未新增 ACLNN 工程。在匹配的 Linux A5 栈初始化子模块，配置 CANN、torch/torch_npu、HCOMM/HCCL/UBMEM/URMA 后，按仓库 README 构建 host 扩展。首次调用还必须成功编译 DeepJIT 设备核；import 成功或 CMake IDE 配置不代表 kernel 编译完成。

```bash
git submodule update --init --recursive
python -m pip install --no-build-isolation .
python -B -m unittest discover -s tests/afd -p 'test*cpu.py' -v
timeout 300s python tests/afd/run_recv_a5.py --device 0 --cores 8
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 33 --piece-size 7
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 33 --piece-size 7 --fp8
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 33 --piece-size 7 --fp8 --transposed-sf
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 0 --piece-size 1
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 33 --piece-size 7 --all-masked
```

单卡入口不创建 HCCL 通信域；DeepEP 扩展仍需其原构建/链接依赖。入口直接构造 F 输入窗口、routes、Inbox、Active、state，实际执行 dispatchRecv；覆盖 BF16、FP8 两 SF stride、多 lane/专家、非连续 record、多 Segment/片、padding、NOP/CONTROL/ABORT、tail、容量、非法 route/metadata/Active、alias 与重复状态。双 rank 入口真实调用 dispatchSend 并同步，先对原始 A 数据校验发送 wire，再注入 F Active 并实际调用 recv。该同步集成不验证在线调度、异步 Ready 可见性、完整 ACK 或槽复用；当前仅一个 A 和一个 F，多来源 lane 来自独立 fixture。

独立 `recv_reference.py` 无 torch 依赖，执行 byte oracle；`test_recv_cpu.py` 从实际 C++ 声明核对 ABI。本轮联合suite为27个CPU unittest（原sender10+recv15，再加2项复用回归测试）通过，13个Python文件AST检查通过。新增 `run_reuse_a5.py` 覆盖乱序/跨核Segment、同record多专家展开、多stage和尾块、两种SF布局及晚出现非法route时输出不变；执行命令见 [测试说明](../../tests/afd/README.md)。这些CPU检查不能证明 host/JIT 编译或 NPU 搬运正确。设备成功标准为 hidden/SF 逐 byte mismatch=0、四列 metadata 精确一致、padding 零/-1、片外哨兵不变、状态/grant 精确一致，错误调用不改输出。当前所有 NPU case 已运行数量为 0，完整 CP3/CP4/CP5 平台验收未完成。

## DeepEP 复用与性能限制

基线为 DeepEP HEAD `3b25377d04b24fc6154698ded78a2bcb2c59afff`。复用 `impls/ep/dispatch_copy_epilogue.hpp` 的 TokenLayout 思路、AIV/MTE bytecopy、事件和全核完成语义，沿 `csrc/kernels/ep/dispatch.hpp` 的 DeepJIT 编译/launch 方式，接入既有 AFDBuffer owner 与 pybind。AFD 新增公共 Active ABI、本地 route 定位/校验、四列 metadata、专家 padding、状态发布；不改原 EP kernel 或 sender wire。

原 EP epilogue kernel实际只输出row-major SF，host需要列布局时另行转换；不能宣称列布局是原生kernel分支。当前column-major分支先等待本stage hidden输出完成，再复用第一条record的hidden UB区域进行多行SF转列。各列起点32B对齐，每列一次 `2*num_stage_tokens` 字节DMA、每列块统一等待；列多于X/32时分块复用。SF源在各record的+X处，不被hidden区重排覆盖。没有额外GM SF镜像；T=1或尾块仍可能是2B DMA。metadata使用独立对齐stage，尾块可为16B写入；这些长度、bypass读取、地址空间及SyncAll资源仍须真实Bisheng/A5验证。

workspace 仅 `32*num_vec_cores+256`；UB 预算为 `2*A32(T*D)+2*A32(16T)+2*A32(4T)+832`，T 从 1/2/4/8/16 中按真实 UB 和 stage 约束选取。没有全事务展开缓存或额外完整hidden/SF镜像，公开ABI及容量保持不变。

当前先由core0校验公共结构、边界及Segment重叠，再由各核按目的区间交集并行预检selected route/metadata。routeReference按T条批量读取，检查本批所有record地址后才发metadata DMA，每批统一等待；不是将P条间接metadata DMA自动减少到P/T条。各核错误写入既有32B槽，汇总并全核同步之后才允许输出，保持任一预检错误不留下部分输出的语义。

copy采用Segment外层遍历，每核处理与自身目的区间相交的部分，因此支持Entry内Segment目的行乱序及跨核区间，无逐输出行全表扫描。保留原生 `issue_load`、`stage_num_tokens`、`stage_token_base_idx`、`stage_idx` 命名和双stage预加载/回填逻辑；连续record合并搬运，仅区间内部包含record对齐洞，最后一条不读未发送尾洞。MTE2→S保护消费，MTE3→MTE2保护record stage回填，独立MTE3→S保护metadata stage。padding另按专家区间生成，复用原生vector fill逻辑。

剩余限制：core0结构校验仍含Segment两两检查；预检与copy仍有metadata双读；同record跨本地专家Segment仍可能重复读取，没有恢复完整token本地扇出。双stage主要在单Segment内部发挥作用，短Segment及列SF转置前的hidden完成等待会限制重叠；UB转列仍为标量重排。完全恢复token扇出需要有界反向映射/源分桶及目的ownership，不能通过整轮hidden缓存绕过预算。以上仅描述源码机制，不承诺原EP吞吐或任何已实测性能收益。
