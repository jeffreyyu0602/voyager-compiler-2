// C entry points over DRAMSim3's MemorySystem, loaded by dram/__init__.py
// with ctypes. One handle is one simulated memory system; a DMA runs on it
// with vds_run, and its command and state counters are read back with
// vds_counters.

#include <cstdint>
#include <string>

#include "memory_system.h"

namespace {

struct Handle {
    dramsim3::MemorySystem *memory = nullptr;
    uint64_t clk = 0;
    uint64_t reads_done = 0;
};

// Order of vds_counters' output, summed over channels (and ranks).
const char *const kCounters[] = {
    "num_cycles",      "num_act_cmds",       "num_pre_cmds",
    "num_read_cmds",   "num_write_cmds",     "num_read_row_hits",
    "num_write_row_hits", "num_ref_cmds",    "num_refb_cmds",
};
const char *const kRankCounters[] = {
    "rank_active_cycles",
    "all_bank_idle_cycles",
    "sref_cycles",
};

uint64_t WriteCmds(const Handle &h) {
    uint64_t n = 0;
    for (auto *ctrl : h.memory->GetDRAMSystem().Controllers()) {
        n += ctrl->Stats().Count("num_write_cmds");
    }
    return n;
}

void Tick(Handle &h) {
    h.memory->ClockTick();
    h.clk++;
}

}  // namespace

extern "C" {

void *vds_open(const char *config_file, const char *output_dir) {
    auto *h = new Handle();
    h->memory = new dramsim3::MemorySystem(
        config_file, output_dir, [h](uint64_t) { h->reads_done++; },
        [](uint64_t) {});
    return h;
}

void vds_close(void *handle) {
    auto *h = static_cast<Handle *>(handle);
    delete h->memory;
    delete h;
}

// Issues the n request-aligned byte addresses in order, as many per cycle as
// the controllers accept, stopping at the first refused one until the next
// cycle. Ticks until every read has returned its data, or every write has
// been issued to the DRAM and its data burst has finished. Returns the DRAM
// cycles from the first issue to that point, or UINT64_MAX if the DMA has not
// finished after kStallCycles per request (a simulator that stopped making
// progress).
uint64_t vds_run(void *handle, const uint64_t *addrs, uint64_t n,
                 int is_write) {
    const uint64_t kStallCycles = 10000;
    auto *h = static_cast<Handle *>(handle);
    const auto &config = h->memory->GetConfig();
    uint64_t start = h->clk;
    uint64_t deadline = start + kStallCycles * (n + 1);
    uint64_t reads_before = h->reads_done;
    uint64_t writes_before = WriteCmds(*h);
    for (uint64_t i = 0; i < n;) {
        while (i < n && h->memory->WillAcceptTransaction(addrs[i], is_write)) {
            h->memory->AddTransaction(addrs[i], is_write);
            i++;
        }
        Tick(*h);
    }
    if (is_write) {
        h->memory->DrainWrites(true);
        while (WriteCmds(*h) - writes_before < n) {
            if (h->clk > deadline) return UINT64_MAX;
            Tick(*h);
        }
        h->memory->DrainWrites(false);
        return h->clk - start + config.write_delay;
    }
    while (h->reads_done - reads_before < n) {
        if (h->clk > deadline) return UINT64_MAX;
        Tick(*h);
    }
    return h->clk - start;
}

// Advances the clock by idle cycles.
void vds_tick(void *handle, uint64_t cycles) {
    auto *h = static_cast<Handle *>(handle);
    for (uint64_t i = 0; i < cycles; i++) {
        Tick(*h);
    }
}

// Writes the kCounters then the kRankCounters, each summed over channels and
// ranks, into out.
void vds_counters(void *handle, uint64_t *out) {
    auto *h = static_cast<Handle *>(handle);
    const auto &config = h->memory->GetConfig();
    const int n = sizeof(kCounters) / sizeof(kCounters[0]);
    const int m = sizeof(kRankCounters) / sizeof(kRankCounters[0]);
    for (int i = 0; i < n + m; i++) {
        out[i] = 0;
    }
    for (auto *ctrl : h->memory->GetDRAMSystem().Controllers()) {
        const auto &stats = ctrl->Stats();
        for (int i = 0; i < n; i++) {
            out[i] += stats.Count(kCounters[i]);
        }
        for (int i = 0; i < m; i++) {
            for (int r = 0; r < config.ranks; r++) {
                out[n + i] += stats.VecCount(kRankCounters[i], r);
            }
        }
    }
}

// Writes the configuration facts the Python side prices with into out, in
// the order of dram/__init__.py's _CONFIG_FIELDS. period_bytes is the span
// of every address field below the row when the row is the topmost field
// (shifting a DMA by a multiple of it changes only its rows), else 0.
void vds_config(void *handle, double *out) {
    auto *h = static_cast<Handle *>(handle);
    const auto &c = h->memory->GetConfig();
    const bool row_on_top = c.ro_pos >= c.ch_pos && c.ro_pos >= c.ra_pos &&
                            c.ro_pos >= c.bg_pos && c.ro_pos >= c.ba_pos &&
                            c.ro_pos >= c.co_pos;
    const uint64_t period = uint64_t(1) << (c.shift_bits + c.ro_pos);
    const double period_bytes = row_on_top ? static_cast<double>(period) : 0.0;
    const double values[] = {
        period_bytes,
        c.tCK,
        static_cast<double>(c.request_size_bytes),
        static_cast<double>(c.channels),
        static_cast<double>(c.ranks),
        static_cast<double>(c.tREFI),
        static_cast<double>(c.tRFC),
        static_cast<double>(c.burst_cycle),
        c.act_energy_inc,
        c.read_energy_inc,
        c.write_energy_inc,
        c.ref_energy_inc,
        c.refb_energy_inc,
        c.act_stb_energy_inc,
        c.pre_stb_energy_inc,
        c.sref_energy_inc,
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        out[i] = values[i];
    }
}

}  // extern "C"
