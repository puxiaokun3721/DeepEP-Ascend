# dispatchSend / dispatchRecv 原生结构对照

原生参考为 `3b25377`。本文说明当前函数边界与执行顺序；实现和验证状态见[AFD 入口](README.md)。

## 名称与边界

| 原生 | 当前AFD | 分类/差异 |
|---|---|---|
| dispatch_impl | deep_ep::afd::dispatch_impl | 主函数边界对齐；公开dispatchSend与launch_dispatch_send保持原名 |
| DispatchUBLayout | deep_ep::afd::DispatchUBLayout | 命名对齐，AFD容量/控制字段并非原生布局 |
| simt_persistent_worker | route_tile | 复用warp去重/直方图逻辑；AFD有界tile两遍处理，不冒充persistent worker |
| preload_metadata | preload_metadata（局部lambda） | 复用stage预加载/消费/refill组织；AFD字段及tile作用域 |
| combine::load_meta_issue_wqe | issue_write_wqe | 复用warp-per-WQE编码；不加载原生metadata，因此不套用相同函数名 |
| dispatch_copy_epilogue_impl | deep_ep::afd::dispatch_copy_epilogue_impl | 主函数边界对齐；公开dispatchRecv与launch_dispatch_recv保持原名 |
| epilogue局部UBLayout | DispatchCopyEpilogueUBLayout、局部别名UBLayout | AFD有界refs/control空间；非原生布局原样复制 |
| fill_ubuf | fill_ubuf | 原生vector填充逻辑 |
| PSumIterator | Segment交集遍历 | AFD替代机制，未伪造同名迭代器 |
| issue_load | issue_load（局部lambda） | 保留stage生命周期；AFD间接gather |
| epilogue内联输出循环 | issue_output（局部lambda） | 复用MTE输出机制；AFD四列metadata、列布局SF |

源码使用[REUSE]、[AFD]、[REUSE+AFD]标注。REUSE表示适配了原生逻辑，不能据此宣称逐字复制或完整恢复原生算法。

模板变量对齐为kNumRanks、kNumExperts、kNumTopk、kNumVecCores、kNumHiddenBytes、kNumSFPacks、kNumUbBytes等。AFD独有维度保留明确名称：kNumSourceLanes、kNumLayers、kMaxExpandedTokens(Q)、kMaxPaddedTokens(Q_alloc)。kNumMaxTokensPerRank仍对应单槽通信编译上界，不改为事务maxBS。

## 发送阅读顺序

```text
launch_dispatch_send -> afd::dispatch_impl
  S0 初始化/布局、任务校验、统一退出
  S1 每核jetty/peer准备
  S2 每个通信轮
    S2.1 等上轮Consumed和空Ledger
    S2.2 route_tile<false>：校验、去重、计数
    S2.3 tile/rank/expert prefix
    S2.4 每个tile
      route_tile<true>：产生route/slot
      preload_metadata：预加载 -> 消费/回填 -> 尾部排空
      S2.4b append_write_task -> issue_write_wqe -> flush_writes
    S2.5 全核完成 -> route tables -> Inbox -> Ledger -> Ready -> 翻转位
  S3 发布send_status
```

S2.4中的metadata stage推进仍在主函数原位置，未抽出成串行子kernel。route_tile不是原生persistent worker；本轮没有新承诺metadata与URMA持续重叠。

## 接收阅读顺序

```text
launch_dispatch_recv -> afd::dispatch_copy_epilogue_impl
  R0 初始化/Active门控
  R1 目录及Entry/Member结构预检；Segment索引按输入/输出区间排序校验
  R2 并行route metadata预检、全核错误汇总
  R3 定义issue_output（此处只定义，不执行输出）
  R4 每个相交Segment
    定义issue_load
    预加载两个stage -> 消费当前stage -> issue_output -> refill或排空
  R5 padding
  R6 全核完成 -> INPUT_READY
```

R3是lambda定义位置，实际调用在R4/R5；不能把图中的函数定义误读为提前写输出。AFD仍可能跨Segment重复读record，本轮没有恢复原生record-once/topk扇出。接收算子不做scheduler选取、不轮询远端、不提交ACK/credit。

## 同步与验证边界

- 发送：metadata MTE2→S就绪、MTE3→MTE2复用保护、尾部release消费、WQE发布和drain顺序均不变。
- 接收：refs加载仍位于record refill保护等待之前；MTE2→S、MTE3→MTE2和metadata独立MTE3→S事件及循环位置不变。
- 原生数据搬运大阶段继续内联；2026-10-21增加AFD专用Segment索引排序helper与两份有界GM索引（最多256KiB），不改变数据双stage和原生EP文件。
- JIT生成代码改用限定名称deep_ep::afd::dispatch_impl及deep_ep::afd::dispatch_copy_epilogue_impl；缓存标签、模板参数顺序、launch参数及Python接口不变。
- 2026-10-21联合回归70项CPU模型/ABI测试通过；git diff --check通过。尚未做CANN/JIT编译或A5执行，尤其内部kernel符号实例化及多burst重载须实机确认。没有新增设备精度/性能通过声明。
