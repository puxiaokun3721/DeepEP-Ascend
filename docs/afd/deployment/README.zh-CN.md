> AUDIT025 current delivery (2026-10-27): see
> [current-delivery-manifest.json](current-delivery-manifest.json) and
> [the recommended entry scope](../../../tests/afd/README.md).
> The page below and the existing validated-source/IMAGE019 manifests are
> historical snapshots. Their source hashes and image contents do not certify
> the current runner/stride changes; they remain available for provenance.

# A5 AFD 两卡 / 四卡验证版本部署

本目录面向另一台 Linux ARM64 A5 机器复现。项目的通用 README 介绍上游 EP；
这里记录任务私有 AFD 分支，二者验证范围和依赖版本不能互换。

## 交付内容

- 正式修复：A5 地址空间访问、单 TU 单 kernel、通信层与默认 SQ、
  多端点分组及 peer/AIV 配对、ABI v2、权重标量同步、资源失败释放。
- 必要验证：保留原 CPU 模型测试及设备测试；生产 pipeline 串行预热实际特化，
  校验真实结果，不手工注入 Active 或伪造 Completion。
- 诊断：只读队列/RankGraph 入口、编译日志保留、退出阶段标记。
  保留它们以对应现场通过版本，不等于所有诊断入口都是已验证的公共 API。
- 环境与迁移：[环境说明](environment.zh-CN.md)、[镜像迁移](image-transfer.zh-CN.md)。
- 已打包环境：[IMAGE-019三包清单](image019-artifacts.json)、[加载与启动](image019-runtime.zh-CN.md)。镜像与源码/扩展分别搬迁，宿主驱动不在包内。

## 验收范围

2026-10-26 已核原始日志：两卡 1A1F、BF16/H128/K2/Q7/MB1，batch13 一窗口及
batch33 三窗口；四卡 2A2F/NC1、batch13 一窗口，identity expert 完整链路正常退出。
四卡 F0 热点、F1 零路由；tail 区间为空。
不是实际 GMM、均衡多 F、完整多槽 ACK/credit 矩阵、graph 或性能验收。

最终四卡 run `prod_teardown4_20261026T052009`：05:20:15–05:30:45 +08，
rc0，630.9 秒，6 次 bisheng + 6 次 ld.lld。所有 rank 完成资源释放和退出。
此前两卡回归 `prod_pipe2_b13_20261026T044939` rc0；更早两卡 b33 三窗口通过。
迁移后的机器必须重新验证，不能继承旧机器的 PASS。

历史原文留在用户私有协作仓：
[阶段验收](https://github.com/puxiaokun3721/afd-a5-validation/blob/ddb96369e644e2848e6cbafb6ef7d31126b4632e/reports/PROD-001/two-four-card-scoped-completion.md)。
完整运行证据提交 `e1822eccad0ca80bb8355807f2ddc398ca2e861c`。
不把现场受限日志、地址、EID、句柄或凭据复制到代码仓。

## 使用次序

1. 核对目标机硬件、驱动与容器运行时；不要升级共享宿主环境来强行匹配。
2. 固定本分支的交付提交，初始化固定 DeepJIT，再应用交付补丁。
3. 在任务私有目录构建主机扩展，确认导入来自该目录。
4. 使用前查询全部拟用卡占用；先两卡 b13，再两卡 b33；通过后四卡 b13。
5. 保存每次独立目录的命令、环境白名单、占用快照、日志和真实退出码。
   功能 PASS 不等于完整成功，必须等全部进程正常返回。

构建/运行的完整命令以本目录最终现场清单及环境说明为准。

## 源码一致性与打包方式

现场最终快照为 `21e1b2380c6add42be64f173d064d3fb6d3e3165`。本分支用正常 Git
提交交付，不复制协作仓分片历史。最终增量18文件逐SHA256匹配，早期scheduler、
dispatchRecv修复沿已核源码链复用。DeepJIT全部102文件逐Git blob摘要匹配。

现场将DeepJIT展开在主仓；本分支保留上游固定子模块，并提供3个内容补丁和一个
测试脚本执行位恢复。应用后源码内容一致，提交SHA/打包结构不同。
运行源码和诊断保留原样；本次只额外更新CPU测试的ABI2布局期望、交付文档和校验工具。

```bash
python3 scripts/afd/apply_deepjit_patches.py
python3 scripts/afd/verify_validated_sources.py
python3 -m unittest discover -s tests/afd -p 'test_*cpu.py'
```

[源码清单](validated-source-manifest.json)记录现场快照、证据提交和逐文件摘要。
校验仅证明源码一致，不能代替目标机重建、导入绑定和设备测试。现场so摘要为报告值，
未在Windows重建二进制。历史诊断脚本包含当时的路径/导入绑定，不是可直接套用到新机器
的验收命令；新机器按本文档生产pipeline运行。
