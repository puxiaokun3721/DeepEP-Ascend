# 修复、测试和诊断的边界

本分支以 `57b9daf58c473582883ab2c73894d97dcbc92250` 为代码基线。
目标是保存现场通过的源码，不在交付时再次重构通信语义。诊断代码保留并单独标明，
后续删除诊断应另建提交，并重新核对相应入口；不能删除后仍宣称字节等同现场。

## 正式修复

| 位置 | 问题与修改 | 已取得的证据 / 限制 |
|---|---|---|
| DeepJIT `options.hpp` | 显式选择 C++17 / Ascend C 语言；使用生产 `kernel.asc` 路径 | 真实 bisheng/ld.lld 已运行；不是旧离线 cpp probe 的替代框架 |
| AFD layout / 五算子实现 | GM/UB 结构访问改为对应地址空间的 typed 访问和复制；修正 DMA 字节指针及参数接线 | 消除了实际编译失败；CPU 测试不能代替这些目标编译证据 |
| `ffn_scheduler.hpp`、`remaining.hpp` | 按 TU 选择 scheduler 或 compute-ready 单一 kernel 入口 | 不删除 compute-ready，不放宽 DeepJIT 唯一入口检查；分别真实编译、运行 |
| `hccl.hpp` | 明确选择存在的 UBMEM / Jetty layer ID 0，使用 SDK 默认 SQ 配置 | 原 layer/深度配置先后导致 Acquire 超时或参数拒绝；构造和只读元数据验证后才进行真实发送 |
| `combine_recv.hpp` | 权重 DMA 后以 `GetValue` 取标量，再加 S→V 同步后执行 Muls | 修前输出为期望两倍，组合修复后整链通过；没有把两个修改各自认作已独立证实的唯一根因 |
| `hccl.hpp`、context、send / combineSend | AFD world>2 按端点分组资源，统一 grouped 格式；Peer 与 Jetty 都用 peer_base+AIV 对应项 | 四卡有效 IPv6 地址确实不同，不能忽略 padding 来消除原断言；保留每队列 writer、drain 和发布顺序 |
| context / 任务生产者 | ABI v2 布局与任务首字版本一致 | 实际出现过 `[3,1] BAD_ABI`；1→2 修复后的两卡回归通过。这是实现与审查都漏掉的接线错误 |
| 新增资源失败路径 | 记录 endpoint / 已注册 memory 的不同计数；依赖释放失败时停止后续释放，保留 UNKNOWN | 防止释放尚在使用的资源；不把未知状态当正常清理或无条件重试 |

默认 SQ API 深度和设备侧 WQEBB 数量不是同一个单位。现场采样的 SQ/CQ 为
32768×64B；不把设备 `kSQDepth` 随意改成 8192。

## 必要构建与测试支持

- 主机 `setup.py` 扩展与 DeepJIT 设备编译分别构建，不混用 C++20/O3 和 C++17/O2 参数。
- compile-without-load / 逐 world rank 串行预热用于限制 CPU 并发，随后运行原算子入口。
  编译成功本身不算设备功能通过。
- 保留 `tests/afd` 的 CPU 参考模型和设备测试。最终 pipeline 使用真实
  scheduler、dispatchSend/Recv、identity compute-ready、combineSend/Recv 和 ACK/credit。
- 验证任务产生 ABI2，HBM 预算 8GiB/card；不靠手工 Active、假 payload 或假 Completion
  拼出 pipeline PASS。

## 临时诊断：随原样现场版本保留

- 队列元数据、RankGraph 只读入口：为选择通信路径和核对队列参数服务，
  不是新增通用公开 API；普通 pipeline 不需要先跑一遍这些探针。
- DeepJIT `backend.hpp` / `disk.hpp` 日志保留：通过环境变量选择，保留失败日志以定位编译。
  构建修复只需要 `options.hpp`；为复现现场仍交付全部三个补丁。
- 退出阶段 begin/end 标记、FUNCTION_PASS / RUN_PASS 区分：保留现场调用顺序，
  用来确认资源释放、进程组销毁和 main 返回；没有改变通信协议。
- runner 的失败数据抽样仅用于定位；不能用抽样替代完整 golden 检查。

## 为什么一开始没有跑通

问题不全是环境，也不全是算子算法。至少存在四类不同问题：

1. 构建接线与 A5 适配不完整：语言选项、地址空间访问、单 TU 入口、模板参数。
2. 通信实现的假设不成立：旧代码假定跨 peer 共用同一本地端点，四卡现场不满足；
   API SQ 参数和设备队列单位也不能直接等同。
3. 真实功能缺陷：权重标量读取/同步、ABI 任务生产者与消费者不一致。
4. 验证推进方式低效：早期大量时间用于通用执行器、重复冻结和材料交互，
   脚本 PASS 没有尽早转化为真实生产编译。后期改用实际生产入口，才逐层暴露以上问题。

原四卡 630 秒超时发生在功能 PASS 之后。增加退出标记并用 900 秒总观察窗口后，
最终运行约 631 秒、正常清理仅约 139–150ms。这个证据支持旧总预算不足的解释，
但不能倒推出旧运行每个调用的精确耗时或声称唯一根因已被证明。

## 保留的历史与未覆盖项

历史失败、SIGTERM、曾经共驻和被覆盖的旧占用快照记录，仍保存在私有验证仓。
本分支不删除旧记录，不把新 PASS 用来追认旧实验合规。关键证据入口见
[部署首页](README.zh-CN.md)。

当前通过范围是两卡一/三窗口与四卡单窗口 identity 小形状。
真实 GMM、均衡多 F、非空 tail 哨兵、完整多槽复用/ACK/credit、长期稳定性、
graph 和性能仍需分别验证。分支能复现已有测试，不等于这些场景都已完成。

交付期CPU回归：旧ABI1/224B测试模型更新为ABI2/256B，保留已有对齐/offset行为断言；
70项unittest通过。未在整理源码时修改现场运行逻辑或新增设备试验。
