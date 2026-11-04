# CAM 期增量（2026-11-04）

本分支 `a5-afd/cam-era-20261104` 接在 `codex/a5-afd-return-stride-20261026` 之后，
只包含把 AFD 四个算子搬进 CAM 过程中、DeepEP-Ascend 这一侧必须配合的改动。
CAM 仓库那边的四个算子不在这里，见另一条线。

## 改了什么

| 文件 | 改动 | 为什么 |
|---|---|---|
| `csrc/buffers/afd.hpp` | 三处 `torch::from_blob` → `at_npu::native::from_blob`；新增 `cam_config`、`jetty_ptrs` | `torch::from_blob` 造的是普通 `c10::StorageImpl`，没有 `NPUStorageDesc`；CAM 侧 `ConvertType` 读 `npu_desc_` 会越界。`at_npu::native::from_blob` 会构造 `torch_npu::NPUStorageImpl` 并 `StorageDescHelper::SetDesc`，天然带 ND descriptor |
| `csrc/python_api.cpp` | 暴露 `camConfig` 方法与 `jetty_ptrs` 属性 | CAM 算子不自己分配/释放控制面，launch 参数和 jetty 表必须由调用方以张量传入 |
| `csrc/comm/handle.hpp` | 新增 `net_layer_from_env()`，层号由 `EP_HCCL_NET_LAYER` 决定（默认 0） | 原来硬编码 `layers[1]`，本机两个 context 下层数不同会直接断言失败 |
| `csrc/kernels/comm/hccl.hpp` | 同上；`kDiagnosticTransportLayer` 常量换成可配层号，并校验该层存在于 rank graph | 同上 |
| `tests/afd/run_reuse_a5.py` | 复用/多 transaction 用例 | 真实复用路径验证 |
| `tests/afd/run_pipeline_cam.py` | CAM 端到端 pipeline 用例（新） | 用 CAM 算子跑 A→F→A 全链 |
| `tests/afd/run_cam_matrix.sh` | CAM 形状矩阵驱动（新） | 批量形状回归 |
| `tests/afd/run_cycles_cam.py` | CAM 多轮循环（新） | 跨 round 状态机 |
| `tests/afd/run_reuse_cam.py` | CAM 复用（新） | 同上 |
| `tests/afd/run_pipeline_a5_shape.py` / `_topk.py` | 形状/topk 扫描（新） | 边界覆盖 |
| `tests/afd/gen_route_matrix.py` | 路由矩阵生成（新） | 不平衡/可辨路由 |
| `tests/afd/run_discrim_a5.py` | 判别性 oracle（新） | 跨 F、专家可辨变换 |
| `tests/afd/test_combine_weights_cpu_add.py` / `test_reuse_bits_cpu.py` | CPU 单测（新） | 无卡可跑 |

## 没有带上来的

- `extra-info/`（346 MB 编译 dump 与 exception info）：属于现场产物，不入库。
- `build/`、`*.so`、`__pycache__`。
- `*.pre_*` 备份文件。
- 几处 **GitHub 分支更新、现场树更旧** 的文件保持分支版本不动：
  `tests/afd/test_cpu.py`、`tests/afd/README.md`、`docs/afd/dispatch_send.md`、
  `tests/afd/run_send_recv_a5.py`、`tests/afd/run_capacity_a5.py`。

## 上下文

这一支是 DeepEP-Ascend 原生的 JIT 形态；CAM 形态是把同样四个算子重写成
`op_api/op_host/op_kernel` 的标准算子，另有仓。两条线的算法一致，没有分叉。
