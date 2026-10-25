# AI Research Project Template

这是一个“一篇论文一仓”的人工智能科研模板。它把数据版本、训练配置、
checkpoint、评测结果、论文图表和最终 release 串成一条可审计的证据链。

## 快速开始

```bash
uv sync
uv run researchctl init --target iclr-2027
```

然后告诉 Codex：

> 按 `paper/target/venue-research.prompt.md` 搜索并核验当前投稿目标，安装官方模板。

最后运行轻量元数据校验：

```bash
uv run researchctl validate
```

仓库固定 Python 3.11；默认安装只包含模板管理控制面，不会下载 PyTorch。
只有实际开始训练、DVC、MLflow、Slurm 或 Hugging Face 发布时，才按需安装对应
可选依赖。

日常实验必须通过 `researchctl` 登记，不要把数据或 checkpoint 直接提交到
Git。完整工作流见 [docs/workflow/README.md](docs/workflow/README.md)。

## 核心原则

- Git 保存代码、配置、注册表、论文快照和 release manifest。
- DVC 保存数据与 checkpoint 的内容寻址版本，MLflow 保存完整运行轨迹。
- `paper/evidence.yaml` 把论文中的图、表和结论映射回实验资产。
- Overleaf 是论文权威源；本地仓只主动推送初始化内容和生成图表。
- 正式发布前必须通过 `researchctl release check --strict`。

## 环境变量

敏感信息只通过环境或系统凭据管理器提供：

```text
MLFLOW_TRACKING_URI
DVC_REMOTE_URL
OVERLEAF_TOKEN
HF_TOKEN
HF_ORG
GITHUB_TOKEN
```

不要把这些值写入仓库。
