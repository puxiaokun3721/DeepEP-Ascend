# DeepEP-AFD 实现入口

本分支基于原生 DeepEP-Ascend `3b25377d04b24fc6154698ded78a2bcb2c59afff` 开发五个 AFD 算子，尽量复用原生阶段、搬运和队列逻辑。原生 EP 与新增 AFD 路径独立。

## 阅读顺序

- [总体设计与算子契约](https://github.com/puxiaokun3721/afd-plugin/blob/afd-five-operator-baseline/docs/design/README.md)：设计位于 AFD Plugin 仓库。
- [dispatchSend](dispatch_send.md)：发送接口、owner、窗口及发布条件。
- [dispatchRecv](dispatch_recv.md)：Active 片内接收、预检及输出布局。
- [ffnScheduler、combineSend、combineRecv](remaining_operators.md)：调度与返回闭环。
- [dispatch 原生结构对照](dispatch_code_map.md)：函数边界、执行顺序和必要差异。
- [测试说明](../../tests/afd/README.md)：CPU 模型与待执行的 A5 测试入口。

## 源码导航

| 内容 | 位置 |
| --- | --- |
| AFD kernel | [impls/afd](../../deep_ep/include/deep_ep/impls/afd/) |
| AFD 布局 | [layout/afd](../../deep_ep/include/deep_ep/layout/afd/) |
| Host/JIT | [kernels/afd](../../csrc/kernels/afd/) |
| Buffer 接口 | [buffers/afd.hpp](../../csrc/buffers/afd.hpp) |
| 原生通信参考 | [impls/ep](../../deep_ep/include/deep_ep/impls/ep/) |

## 当前关键实现边界

发送 payload 复用原生 warp-per-WQE 构建逻辑，最多 8 个 WriteTask 使用原有 512B scratch 中的 320B，不新增 hidden 镜像。dispatch 每 record 两 SGE，combine 每 route 一 SGE。仍逐 WR 请求 CQE，最多 32 个在途 WR 后 drain；CQ 要求 2 的幂且大于 42（最小 64），同一 jetty 必须在 kernel 间串行使用。

同步顺序为：准备描述符 → S/V 交接 → SIMT 写 SQ 并 threadfence → V/S 交接 → 更新 head/doorbell → drain。tile metadata 完成后才能提交引用它的 WQE，drain 后才可复用；payload 全部完成后才提交 Ready 或 Done/Consumed。描述符准备仍有标量循环，未恢复原生 persistent worker 的全部重叠。

ffnScheduler 每次收集 InputReady 快照后重建成员索引，复用 Pending 保留字段；Pending 仍为 96B，Inbox 偏移仍为 32B。成员到齐查找总体最坏 O(L²)，L 为已注册 lane 数。按完整任务 key 检查重复发布，并统计物理成员，不能把同成员多个槽当作多个成员。派生索引每次重建，不跨 replay 信任旧缓存。

## 验证边界

2026-10-21联合回归为70项CPU模型/ABI测试通过；不等于执行了C++算子。尚无CANN/Bisheng/JIT编译、A5精度、图模式或性能通过证据。

优化过程记录保留在本机 `.local-archive/afd-docs-2026-10-19/`，已忽略；已有 Git 历史仍可追溯。

## 2026-10-21 静态返回活性修复

多本地槽与多轮组合不能顺序运行完整事务阻塞recv。默认模式在send发布前及recv入口拒绝该组合；P单槽多轮、D多槽单轮保持兼容。需要该组合时显式启用`prepare_combine_recv(round_step=True, fair_round_robin=True)`，使用同一`combineRecv(..., start=True/False)`及固定公平sweep driver。每槽增加96B持久metadata，单归并workspace和小返回窗口不变，默认先算后等保留。详见[返回推进时序、状态机和图契约](remaining_operators.md#多槽多轮的返回推进2026-10-21静态修复)。

新增8项CPU模型/ABI检查复现旧循环等待，验证公平step推进、最终ACK非阻塞、重复start/TaskInfo变化的持久ERROR及既有场景admission；它们不执行C++ kernel。目标SDK编译、A5多槽多轮、真实图sweep与跨卡可见性仍未验收。

同轮静态修复还包括：scheduler两种策略均过滤缺credit的零route组；dispatchRecv采用两轮有界Segment索引排序（新增最多256KiB metadata）替代区间两两比较，SF列布局每列块改一条多burst DMA。默认非空WORK先算后等、Continuation、Q/Q_alloc及原双stage不变。[接收说明](dispatch_recv.md)明确列出仍未实施的片内record扇出等优化，不能将这些静态事实归为硬件未知。
