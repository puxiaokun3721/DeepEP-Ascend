# 环境构建与验证

## 版本边界

已经从历史源码或原始日志核到的基线：

| 层 | 本次依据 |
|---|---|
| CPU / 系统 | Linux aarch64，openEuler 24.03 LTS-SP4；Windows x86仅作文件中转 |
| 设备 | A5 / Ascend 950，编译目标 dav-3510 |
| CANN | 9.2.0；固定路径 `/usr/local/Ascend/cann-9.2.0` |
| Python / pip | 3.11.10 / 24.0 |
| PyTorch | 2.10.0 CPU构建，完整wheel后缀未交 |
| torch_npu | 现场原始诊断输出 2.10.0.post7 |
| 主机 C++ | GCC 14.3.1，gcc-toolset-14，需要 C++20/std::format |
| 驱动 / ascendhal / 固件 | 25.6.rc2 / 7.35.23 / 7.0.9.100.203 |
| DeepEP 基线 | `57b9daf58c473582883ab2c73894d97dcbc92250` 加本分支修复 |
| DeepJIT | `65f513952b8771408a19202f65e36398522db6db` 加交付补丁 |
| 官方头文件核验来源 | asc-devkit `89708cdf4b654d2ff5c2b381deee226c44fd0d48` |

完整版本及目录类型见 [环境清单](environment-manifest.json)，来源是执行方016交接报告；
本机Windows未重新安装或运行这套环境。镜像tag为
`vllm-ascend:dev-26.2.0.day20261010-A5-py311-openEuler24.03-lts-aarch64`，
ID为`sha256:e56beaef0c95ec1e5768e6f3effa1af0b456e881a034996ef6c20fecf34d09c7`，
本地导入镜像没有RepoDigest。不要把上游
README 的 Python3.12/torch2.13 验证组合当成本次组合，也不要根据目录名认定
所有 CANN 包和加载库都一致。asc-devkit SHA 是核验来源，不要求用源码替换安装 SDK。

[Ascend 官方兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
要求 torch_npu、PyTorch、CANN、Python 与驱动/固件配套；2026-10-26 查询的主表
未直接列出本次 post7/9.2 组合。该组合以本次现场结果及精确分发包为复现依据，
不能虚构一个公开下载地址或自动改用当前推荐版本。

## 推荐：复用已验证的环境镜像

先按 [镜像迁移说明](image-transfer.zh-CN.md) 区分镜像层和挂载内容。
目标机需同 CPU 架构、兼容 A5 驱动/固件及 UBMEM/UBC_CTP 连接。
驱动归宿主机管理，不能通过搬运容器替代驱动安装。
新建任务容器只暴露选定空闲设备；不要套用默认 privileged/全设备/整个 home 可写挂载。
实际设备节点与驱动目录沿目标机已验证的 Ascend 容器规范配置，不能照抄另一台机编号。

源码、构建、JIT cache、结果使用任务私有目录；驱动只读挂载。
如 SDK 已完整包含在镜像内，不要再挂一个不同版本的宿主 SDK 把它遮住。
安装 wheel 前检查 Python 安装目录和 venv 不属于共享 bind/volume。

## 从头构建

1. 取得与现场一致的 ARM64 基础系统、Python、torch/torch_npu wheel 和完整
   CANN toolkit/runtime/ops/compiler 包。版本和包摘要固定，不做无版本 pip upgrade。
2. 在任务私有容器内安装。CANN 必须同时支持构建与运行；只有 toolkit 能生成
   `.so` 并不保证设备初始化/加载能成功。未知兼容性需先确认，不改共享宿主包。
3. 固定源码及子模块：

```bash
git clone --branch codex/a5-afd-validated-20261026 \
  https://github.com/puxiaokun3721/DeepEP-Ascend.git DeepEP-AFD
cd DeepEP-AFD
# 交付后固定具体 commit，不依赖分支名长期不变。
git submodule update --init third-party/deep_jit
python3 scripts/afd/apply_deepjit_patches.py
python3 scripts/afd/verify_validated_sources.py
```

4. 先设置`TASK_CPUS`为明确分给任务的两个CPU编号；现场为`0,1`，其他机器不可盲抄。激活 CANN 和匹配的 Python 环境。以实际安装的 `set_env.sh` 为准，核验
   ASCEND_HOME_PATH、PATH、LD_LIBRARY_PATH 不混入旧 SDK；只修改当前任务 shell。
   现场 C++ 编译器在 `/opt/openEuler/gcc-toolset-14/root/usr/bin/g++`，
   对应 libstdc++ 也必须可解析。其他发行版不保证相同路径。
5. 主机扩展在独立 clone 内构建，例如：

```bash
set -euo pipefail
export MAX_JOBS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
GX=/opt/openEuler/gcc-toolset-14/root/usr/bin
GL=/opt/openEuler/gcc-toolset-14/root/usr/lib64
export CC="$GX/gcc" CXX="$GX/g++" LDSHARED="$GX/g++ -shared"
export LD_LIBRARY_PATH="$GL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$GL${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LDFLAGS="-Wl,-rpath,$GL"
# 在新clone中构建，避免旧build缓存；CPU编号按目标机任务允许范围设置。
taskset -c "$TASK_CPUS" python3 setup.py build_ext --inplace >host-build.log 2>&1
# 只有上一条返回0才继续；下面检查实际导入，不代表设备验收。
python3 -c 'import deep_ep, deep_ep._C; print(deep_ep.__file__); print(deep_ep._C.__file__)'
sha256sum deep_ep/_C*.so
```

这些命令是移植操作说明，不代表本机 Windows 已编译或目标机已经通过。
主机扩展采用项目 C++20/O3；设备编译是 DeepJIT 的 C++17/Ascend C/O2/dav-3510，
不要混用两组参数。使用真实 `kernel.asc`，不以旧 `.cpp` probe 替代。

## 运行要求

每次启动前保存当刻 `npu-smi info`，所有拟用卡无他人进程；低利用率不代表空闲。
两卡使用任务已允许的空闲组，四卡可任意四张，但核清物理卡→容器→逻辑 rank。
Warning 状态在本任务允许；不能因此忽略运行错误。不 reset、不杀他人进程。

总 CPU2、MAX_JOBS=1、RAM16GiB、每卡HBM8GiB、新增磁盘20GiB，4rank不扩大CPU总额。
四卡初始化在旧现场耗时较长；总预算必须容纳初始化、串行JIT、功能和正常释放。
最后通过运行约631秒，原600秒总额不够。诊断使用900秒+30秒收尾，
不是改全局HCCL超时。超时返回非零，只收尾本次明确归属的进程。

设置新的私有 `EP_JIT_CACHE_DIR`，`EP_JIT_DUMP_ASM=0`，
`EP_JIT_PRINT_COMPILER_COMMAND=1`；需要保留编译日志时设置 `EP_KEEP_TMP=1`。
不要复用别的源码版本缓存来宣称新编译。管道tee需 `set -o pipefail`，保存真实返回码。

基础测试参数：`tests/afd/run_pipeline_a5.py --batch 13 --window 16 --q 7
--microbatches 1 --timeout-seconds 300`，两卡通过再 `--batch 33`，最后四卡b13。
torchrun 的设备映射和完整参数应以最终现场启动清单为准；运行入口会逐rank串行预热。
成功需 golden/status、FUNCTION_PASS、所有rank正常清理及外层rc0，不能只grep PASS。

## 完整运行命令示例

下面保留现场成功的测试参数。`TASK_CPUS`、`TASK_DEVICES`与`RUN_DIR`由
目标机实际分配决定；`RUN_DIR`必须是不存在的新路径。设备可见顺序必须与
LOCAL_RANK 0..N-1一致，不把容器逻辑编号直接当作宿主物理编号。

```bash
set -euo pipefail
NPROC=2                    # 两卡两case通过后改4，batch仍13
BATCH=13                   # 第二个两卡case改33
mkdir "$RUN_DIR"           # 不用 -p：拒绝覆盖旧结果目录
npu-smi info >"$RUN_DIR/npu-before.txt"
# 此处人工核全部拟用卡没有他人进程；存在占用就不要运行下行。
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export EP_JIT_CACHE_DIR="$RUN_DIR/jit_cache"
export EP_JIT_PRINT_COMPILER_COMMAND=1 EP_JIT_DUMP_ASM=0 EP_KEEP_TMP=1
export MAX_JOBS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
set +e
timeout --signal=TERM --kill-after=30s 900s taskset -c "$TASK_CPUS" \
  torchrun --standalone --nproc_per_node="$NPROC" tests/afd/run_pipeline_a5.py \
  --batch "$BATCH" --window 16 --q 7 --microbatches 1 --timeout-seconds 300 \
  >"$RUN_DIR/stdout.log" 2>"$RUN_DIR/stderr.log"
rc=$?
printf '%s\n' "$rc" >"$RUN_DIR/exit-code.txt"
set -e
test "$rc" -eq 0
```

这是目标机使用GNU timeout的复现示例，现场外层使用有界run_pkg；
主torchrun参数与环境来源为执行方交接，不宣称该新shell示例已在NPU上复跑。
检查所有rank退出和残留，只处理本次任务进程；不能为清理使用全局pkill或设备reset。

从零安装还缺精确wheel/安装包文件及摘要，不能保证只靠公开pip命令重建本次组合。
因此最快路径是已验证环境镜像配合本分支；离线包、宿主驱动配置另外准备。
库版本相同也不保证二进制摘要相同，重新构建后记录新so及实际导入位置。
