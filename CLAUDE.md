# vehicle_reid

Multi-camera vehicle ReID on RoundaboutHD. The pipeline runs
prepare → detect → track → reid → match → od, with a web app at the end.

- Rules, file formats and folder ownership: `docs/pipeline_contract.md`. Read it first.
- Target device: Raspberry Pi 5 + Hailo-8L (int8, HEF via the Hailo Dataflow Compiler).
  Constraints are in `docs/deployment_target.md`.
- Settings and model choice: `params.yaml` (`class: package.module:ClassName`).
- One agent per stage in `.claude/agents/`:
  `data-eval`, `detection`, `tracking`, `reid`, `matching`, `od-matrix`, `webapp`.
  Delegate stage work to the matching agent.
- Python: `.venv\Scripts\python.exe` (3.12). DVC remote `origin` = DagsHub.
  Git remotes: `origin` pushes to DagsHub and GitHub, and `github` pushes to GitHub only.
