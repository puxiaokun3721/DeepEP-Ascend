# 镜像离线迁移：建议与边界

对于相同架构和兼容 A5 硬件的多台服务器，优先准备一个固定的环境镜像，
配合固定源码提交和挂载清单，比每台机器重新安装依赖更容易复现。
本次仅整理流程，不自动打包、下载或上传大镜像。

## 先查清哪些内容会随镜像迁移

| 内容 | 迁移方式 |
|---|---|
| 基础镜像已有的 Python/CANN/系统包 | 镜像 save/load |
| 容器可写层中安装的依赖 | 先制作新镜像，再 save；优先 Dockerfile，临时可用 commit |
| bind mount 或 volume 内的 SDK/源码/venv | 不在镜像内，单独归档，或构建新镜像显式 COPY |
| 宿主驱动、固件、内核、设备节点、网络拓扑 | 不随镜像迁移，在目标机核对兼容性 |
| JIT cache / 旧日志 / 凭据 | cache重建，日志另存；凭据不得打包传播 |

不要执行 `docker save` 原始 image 后以为包含容器后来安装的东西。
`docker commit` 也不包含挂载卷；`docker export` 同样不带卷内容，
不是本项目推荐的环境发布格式。

## 推荐交付包

- `afd-a5-env-<固定标签>.tar`：清理后的任务环境镜像。
- 镜像ID、仓库digest（若有）、OS/CPU架构和文件SHA256。
- 版本清单、Dockerfile或准确构建步骤、容器启动与挂载说明。
- 固定 DeepEP Git 提交及 DeepJIT 补丁，必要时提供 Git bundle 供离线恢复。
- 单独的 SDK/wheel归档仅在分发许可允许时提供；不要上传到代码Git仓。

优先基于固定基础镜像和已归档依赖写 Dockerfile，源码仍由Git版本管理。
若急需搬迁现场环境，可在任务容器停止实验后检查 `docker diff` 和挂载白名单，
确认无凭据、个人目录、受限现场数据，再对这个任务容器 commit 到新标签。
不能对其他作业容器操作；不能把宿主共享目录整包带走。

## save / 中转 / load

以下IMAGE必须指已经包含所需容器私有依赖的新镜像，不是盲用原基础镜像：

```bash
# 来源服务器；输出放在容量足够的任务目录。
docker image inspect IMAGE --format '{{.Id}} {{.Os}}/{{.Architecture}}'
docker image save -o afd-a5-env.tar IMAGE
sha256sum afd-a5-env.tar > afd-a5-env.tar.sha256

# 在目标服务器，传输后校验再导入。
sha256sum -c afd-a5-env.tar.sha256
docker image load -i afd-a5-env.tar
```

内网服务器→Windows本地→另一服务器的两段传输可行；Windows仅作为文件中转，
不需要在本地运行ARM64/NPU镜像。中转时可用：

```powershell
Get-FileHash -Algorithm SHA256 .\afd-a5-env.tar
```

确保两段传输后摘要一致。大量服务器更适合有访问控制的私有registry，
避免重复手工传输；不把含私有依赖的镜像默认发布到公共仓。
目标机启动时重新配置设备、只读驱动和任务目录挂载；保存旧容器的启动参数
并不意味着应原样复用另一台机的绝对路径或设备编号。

目标机先核依赖和host扩展导入，再两卡b13/b33，最后四卡b13。
即使镜像完全相同，驱动/拓扑/资源状态不同，也需要这组验证。

## 官方依据

- [Docker save](https://docs.docker.com/reference/cli/docker/image/save/)：保存镜像层和标签，配合load。
- [Docker commit](https://docs.docker.com/reference/cli/docker/container/commit/)：容器变化可成新镜像，但不含挂载卷；默认暂停该容器。
- [Docker export](https://docs.docker.com/reference/cli/docker/container/export/)：导出容器文件系统，不包含卷内容。

这些说明只涉及容器打包机制，不替代 Ascend 驱动、固件和CANN的兼容性验证。
