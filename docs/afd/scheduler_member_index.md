# ffnScheduler成员索引优化（2026-10-18）

前序代码快照：a61d65e。此优化是AFD必要新增调度器的内部改进，没有原生DeepEP独立AF scheduler可直接复用；没有更改原生EP文件。

## 原实现与变化

L为已注册物理lane数（包含多个microbatch槽），不是成员数。原scheduler_ready对每个候选遍历全部lane，对每个成员再找代表槽及匹配槽，单候选O(L²)，全部候选最坏O(L³)。选择阶段又做同样检查。

现在在本次InputReady快照全部收集完成后，先用两个O(L²)遍历建立：

- member_leader：相同(dp, phase, member_idx)的最小lane下标。
- num_group_members：同(dp, phase)下注册物理成员数，包含当前没有任务的成员。

同时检查相同物理成员是否有两个槽发布相同(microbatch, layer, round, num_comm_rounds)。若有则BAD_TASK，不得用总数相等掩盖重复/缺失。

之后每次ready只需O(L)扫描：排除非最小任务lane，统计与候选任务匹配的有效lane；匹配数量等于num_group_members才到齐。credit预检保持逐匹配成员检查，HELD跳过；先算后等不查询credit。索引构建、全部ready调用及原有oldest扫描合计O(L²)；专家计数与切片的O(LR)逻辑不变，R为本地专家数。

## 存储和生命周期

AFDPendingLane原offset12的reserved改为member_leader，原offset24的8B reserved_address拆为num_group_members及4B reserved。Pending仍96B，Inbox仍offset32，所有workspace/Active/UB/通信容量不变，host分配无变化。这两个字段是私有派生缓存，不是传输协议字段。

每次scheduler调用都重新建立；不持久复用跨replay的映射，不依赖旧缓存值。scalar bypass写后通过DDR barrier再读取。每次ready不再写GM标记，也不增加逐成员同步。只有scheduler的core0访问Pending，选择期间不会重新扫描Inbox或改变valid/task，因此预先完成的唯一性检查一直有效。

Pending对外offset不变，但私有reserved语义已变化；调试程序不能再断言这些字节全零。只有内部代码使用它们，未提升wire ABI版本。

## 正确性推演

以同DP两个member、每member三个D槽为例，注册lane0/1/2属于member0，lane3/4/5属于member1。member_leader分别为0/3，所有lane的num_group_members=2。

- 任务M0发布在lane1和lane5：唯一性通过，匹配数2，到齐；未发布的四个兄弟槽不额外计数。
- 仅lane1发布：匹配数1，不到齐；即使另一个member在后续通信轮为零token，也必须发布该轮Inbox。
- lane0、lane1发布同一M0，member1缺失：预检报重复，不能因匹配数2误判到齐。
- M0/M1/M2使用不同槽同时发布：按完整任务key分别计数，允许多个同DP同层microbatch参与后续聚合。
- 消费完成、槽交替位翻转后：下次调用重新读取Pending并重建索引，没有旧成员计数或mark造成的误判。

对一般情况，所有匹配lane都是注册组的成员；唯一性预检证明它们对应不同物理成员。匹配数等于注册成员数因此等价于集合相等。TP/CP owner与零route约束保持不变。

## 验证与性能边界

本地56项CPU模型测试通过；新增3项覆盖300组随机目录与集合oracle对照（正常/重复各一半）、M=2/3槽、缺失成员、零route、HELD/credit两策略及三次重用。CPU没有执行C++。

静态确认96B/offset32布局断言、无新增分配、无wire控制或Q/Q_alloc变动。未完成CANN编译、A5执行、图replay及性能测量。每次重建有固定开销，小L不保证更快；需实测不同L、DP成员数和D槽数，对比a61d65e的scheduler时延。目录只初始化一次的进一步优化需要额外明确上下文不可变/失效约定，本轮没有引入该依赖。
