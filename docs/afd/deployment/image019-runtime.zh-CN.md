# IMAGE-019：加载后如何启动

配合[包清单](image019-artifacts.json)和[传输说明](image-transfer.zh-CN.md)。以下是目标机模板，未在另一台机器实跑；现场已报告新镜像无设备导入rc0，不能代替目标机NPU验收。

镜像含Python3.11.10、torch2.10.0+cpu、torch_npu2.10.0.post7、CANN/ATB9.2.0和GCC14.3.1。宿主驱动、固件、设备和网络配置不随镜像迁移，基线见[环境说明](environment.zh-CN.md)。不得自动升级共享环境。

## 加载和摆放文件

两段传输后逐包字节/SHA256一致才load。在目标机用全新任务目录，勿覆盖旧工作：

```bash
set -euo pipefail
IMAGE=afd-a5-validated-20261026:runtime-clean-v1
docker load -i afd-a5-runtime-clean-v1.docker.tar
docker image inspect "$IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
# 与包清单比较；ENV实际43项，旧模板的48是笔误。
TASK_ROOT=/srv/afd-image019-new  # 改成获准且尚不存在的任务私有路径
mkdir "$TASK_ROOT"
mkdir "$TASK_ROOT/src" "$TASK_ROOT/slog" "$TASK_ROOT/results"
# 先检查归档路径；本包报告无软链，无绝对路径或..逃逸。
tar -tzf final_src.tar.gz
tar --no-same-owner -xzf final_src.tar.gz -C "$TASK_ROOT/src"
cp _C.cpython-311-aarch64-linux-gnu.so "$TASK_ROOT/src/deep_ep/"
```

源码包已展开完整DeepJIT，不需要再次初始化子模块或应用补丁。使用Git分支而非该归档时，按README执行子模块补丁与195文件校验。两种交付不要混用来覆盖已有源码。

## 无设备导入检查

只挂源码和目标机用户态驱动目录；不加任何`--device`，不初始化NPU。driver/dcmi路径先确认存在且属于目标机驱动。使用GCC14运行库与构建说明一致：

```bash
docker run --rm --network none --cpus 2 --memory 16g \
  --mount "type=bind,src=$TASK_ROOT/src,dst=/work/src,readonly" \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
  -e PYTHONPATH=/work/src "$IMAGE" bash -c '
set -euo pipefail
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -V
python3 -c "import torch, torch_npu, deep_ep, deep_ep._C as m; print(torch.__version__, torch_npu.__version__, deep_ep.__file__, m.__file__)"
'
```

预期导入`/work/src/deep_ep/`下随行扩展。无设备告警不等设备验证失败；实际异常/非零必须处理。裸ldd中的libtorch_npu未找到不能单独判定，需结合实际导入和加载库。

## 设备验证容器

复用目标机获准的Ascend容器启动规范，并作以下明确替换：

- 本镜像、CPU总额2、RAM16GiB、单job；任务目录到`/work`可写，私有`$TASK_ROOT/slog`到`/usr/slog`。不要可写挂载共享`/var/log/npu`。
- driver/dcmi只读，不挂另一个宿主SDK遮住镜像内CANN。hccn/rootinfo/hixlep仅在目标机需要时只读挂载已有目标配置，不复制来源机值，不盲建空文件。
- 每次`npu-smi info`检查所有拟用卡无人占用，记录物理→容器→rank映射，仅暴露这些卡及目标驱动必需的管理设备；禁止privileged、全卡和整个home挂载。Warning允许，不抢他人卡。
- 在`/work/src`按[环境说明的完整命令](environment.zh-CN.md#完整运行命令示例)先两卡b13、b33，成功再四卡b13；每卡HBM上限8GiB，新缓存/结果目录。导入使用上述GCC14库路径。

完整成功要求所有rank功能检查、正常清理、外层rc0；仅PASS字样不够。本包不包含真实GMM、均衡多F或性能验收承诺。
