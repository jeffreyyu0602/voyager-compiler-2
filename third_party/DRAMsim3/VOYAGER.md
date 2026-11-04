# DRAMSim3, vendored

Upstream: https://github.com/umd-memsys/DRAMsim3 at commit 2981759
(2021-04-05), MIT license (`LICENSE`).

Copied: `src/` without the thermal model, the CPU trace drivers and
`main.cc`; `ext/headers/` (`INIReader.h`, `json.hpp`) and `ext/fmt/`.
Not copied: the thermal model, tests, scripts and stock configs.

Local changes, each marked `VOYAGER` in the source:

- `SimpleStats::Count` / `VecCount`, `Controller::Stats`,
  `BaseDRAMSystem::Controllers`, `MemorySystem::GetConfig` /
  `GetDRAMSystem`: read the counters and configuration without printing
  stats files.
- `Controller::DrainWrites` (and its `BaseDRAMSystem` / `MemorySystem`
  passthroughs): drain the write buffer whenever no read waits. Upstream
  drains only once more than 8 writes wait, so a short store never
  completes.

Built by `src/voyager_compiler/dram/build.sh` together with the shim the
reporting estimator loads under `--dram_sim`.
