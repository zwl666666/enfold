# Vendored RoboTwin

This directory vendors evaluation code from the upstream RoboTwin repository.

- Upstream: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- License: MIT; the upstream license is preserved in `LICENSE`.

Large assets and task configurations are intentionally not included. From the Enfold repository root, a local upstream checkout can be connected with:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin external/RoboTwin
ln -s ../../external/RoboTwin/assets third_party/RoboTwin/assets
ln -s ../../external/RoboTwin/task_config third_party/RoboTwin/task_config
ln -s ../../../experiments/robotwin/enfold_policy \
  third_party/RoboTwin/policy/enfold_policy
```

Enfold-specific policy code lives in `experiments/robotwin/enfold_policy`; it is not copied from the upstream RoboTwin policies.
