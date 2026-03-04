# Tierra WebGPU

A GPU-accelerated WebGPU implementation of Thomas Ray's **Tierra** artificial life simulator. Watch digital organisms evolve in real-time through mutation, competition, and natural selection.

## Features

- **Real-time GPU rendering** — 60,000-byte soup visualized at 60 FPS on the GPU
- **Full instruction set** — 32 opcodes (mov, jump, call, divide, mal, adro, adrb, adrf, etc.)
- **Mutation mechanics** — Cosmic ray bit-flips and copy errors drive evolution
- **Genotype census & genebank** — Track unique genotypes and save successful lineages
- **Runtime controls** — Adjust mutation rates, debris mode, and CPU visualization without recompiling
- **GPU pre-pass** — CPU position markers rendered as downward-pointing triangles

## Live Demo

[Try it in your browser](https://nor-nrs.github.io/tierra-webgpu/) (requires Chrome 113+, Edge 113+, or other WebGPU-supporting browser)

## How It Works

1. **Simulation** (GPU compute shader)
   - 2,048 organisms (CPUs) execute instructions from a 60 KiB soup
   - `divide` instruction clones organisms with mutations
   - `mal` allocates new memory; `adro`, `adrb`, `adrf` search for targets
   - Reaper kills organisms under population/fragmentation pressure

2. **Rendering** (GPU render pipeline)
   - Fragment shader maps soup addresses to pixels (6 pixels per address)
   - Each pixel colored by organism ownership + instruction type
   - Pre-pass compute shader marks each CPU's IP for visualization

3. **Evolution**
   - Copy mutations during `mov_iab` (~1 in 1,000 instructions)
   - Cosmic ray bit-flips in the soup (~1 in 20,000 per tick)
   - Genotypes tracked by hash; genebank saves successful strains

## Build & Run

### Prerequisites
- Node.js 16+ (for build; runtime requires WebGPU browser)
- npm or yarn

### Development
```bash
npm install
npm run dev
```
Opens at `http://localhost:5173/`

### Production Build
```bash
npm run build
npm run preview
```
Output goes to `dist/`

## Controls

**Settings**
- **Organic debris** — Keep dead code in soup (affects cosmic ray targets)
- **Show CPU pointers** — Display upward triangles marking IP addresses

**Mutation Rates** (slider: adjust without recompilation)
- Cosmic bit-flip, cosmic replace, copy bit-flip, copy replace
- Display shows "1 in N" probability

**Monitors**
- **Soup view** (top left) — Real-time organism colors
- **Timeline** (bottom left) — Population & memory usage over time
- **Histogram** (very bottom) — Genome size distribution
- **Census** (right) — Top 12 live genotypes + genebank

## Architecture

```
src/
├── main.ts              # WebGPU setup, pipelines, render loop
├── monitor.ts           # Canvas visualization (timeline, histogram, census)
├── constants.ts         # Configuration (soup size, organism count, opcodes)
├── style.css            # Dashboard styling
└── shaders/
    ├── compute.wgsl     # Main CPU execution (instruction interpreter)
    ├── soup_render.wgsl # Fragment shader (soup + CPU triangle rendering)
    ├── mark_cpus.wgsl   # Pre-pass compute (marks CPU IP addresses)
    └── helpers.wgsl     # Shared helpers (RNG, template matching, mutation)
```

**Key sizes**
- Soup: 60,000 i32 (240 KB)
- Max organisms: 2,048
- Stack per CPU: 10 slots
- Registers per CPU: 4 (AX, BX, CX, DX)

## References

- [Thomas Ray's Tierra](http://www.his.com/~ray/tierra/)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [Vite Documentation](https://vitejs.dev/)

## License

MIT — See LICENSE file

## Development Notes

- **GPU bottleneck:** Unlikely; compute shader is lean. Most time is in render-to-canvas.
- **CPU readback:** Only 2/50 frames (genotype census) to minimize stalls.
- **Determinism:** Each tick's RNG seeded by `(cpu_index * 1000003u + cycle_counter)` for reproducibility.
- **Atomic ops:** `atomicLoad/Store` for stats and cell slot locks; non-atomic soup writes acceptable (rare same-address races are biologically benign).

## Future Enhancements

- [ ] Parameter presets (classic Tierra, high mutation, etc.)
- [ ] Pause/step simulation frame-by-frame
- [ ] Replay from saved genebank entry
- [ ] Multi-threaded CPU interpretation (if compute shader becomes bottleneck)
- [ ] Custom ancestor genotype upload
