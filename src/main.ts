import "./style.css";
import computeWGSL from "./shaders/compute.wgsl";

const SOUP_SIZE = 60000;
const MAX_ORGANISMS = 1024;
const FIELDS_PER_CELL = 9; // 32 bytes
const FIELDS_PER_CPU = 24; // 96 bytes

interface ICell {
  state: number; // 0 = Dead, 1 = Alive
  mem_start: number;
  mem_size: number;
  d_start: number; // Daughter's memory start
  d_size: number; // Daughter's memory size
  mov_daught: number; // Counter for instructions copied to daughter
  active_cpus: number;
  age: number;
  errors: number;
}

interface ICPU {
  registers: number[]; // [AX, BX, CX, DX]
  IP: number;
  SP: number;
  stack: number[]; // 10 slots
  flags: number[]; // 4 slots
  state: number; // 0 = Dead, 1 = Alive
  cell_index: number; // Points to the cell array
}

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div style="display: flex; gap: 20px; font-family: monospace; background: #222; color: #0f0; padding: 20px;">
    <div style="display: flex; flex-direction: column; gap: 10px;">
      <canvas id="monitor" width="600" height="600" style="border: 1px solid #555;"></canvas>
      <canvas id="timeline" width="600" height="150" style="border: 1px solid #555;"></canvas>
      <canvas id="histogram" width="600" height="120" style="border: 1px solid #555;"></canvas>
    </div>
    <div id="dashboard">
      <h2>TIERRA MONITOR</h2>
      <p>Cycle: <span id="stat-cycle">0</span></p>
      <p>Population: <span id="stat-pop">0</span> / 1024</p>
      <p>Memory Usage: <span id="stat-mem">0</span> / 60000</p>
      <p>Fullness: <span id="stat-full">0</span>%</p>
      <hr>
      <h3>Top Organisms</h3>
      <div id="cell-list"></div>
      <hr>
      <h3>Genotype Census</h3>
      <p style="font-size:11px">Live/Total: <span id="stat-genotypes">—</span></p>
      <p style="font-size:11px">Genebank: <span id="stat-genebank">0</span>
        <button id="genebankBtn" style="font-family:monospace;font-size:10px;background:#333;color:#0f0;border:1px solid #555;padding:1px 5px;cursor:pointer">↓ save</button>
      </p>
      <div id="census-list" style="font-size:11px;line-height:1.6;margin-top:4px"></div>
    </div>
  </div>
  <canvas id="canvas"></canvas>
`;

const timelineCanvas = document.querySelector<HTMLCanvasElement>("#timeline")!;
const histogramCanvas =
  document.querySelector<HTMLCanvasElement>("#histogram")!;

const monitorCanvas = document.querySelector<HTMLCanvasElement>("#monitor")!;
const canvas = document.querySelector<HTMLCanvasElement>("#canvas");
if (!navigator.gpu) {
  throw new Error("WebGPU is not supported in this browser.");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No appropriate GPU adapter found.");
}

const device = await adapter.requestDevice();
if (!device) {
  throw new Error("Failed to create WebGPU device.");
}

const context = canvas?.getContext("webgpu");
if (!context) {
  throw new Error("WebGPU context could not be initialized on the canvas.");
}

// =========== SOUP =================
const soup = new Uint32Array(SOUP_SIZE); // In the soup are instructions
const ownership = new Uint32Array(SOUP_SIZE); // In this are ownership marked
ownership.fill(0); // 0 = unowned memory

// ============== CPUS ================
const cpuData = new Int32Array(MAX_ORGANISMS * FIELDS_PER_CPU); // In this are maps of creatureId: properties. Length = creatures amount
cpuData.fill(0); // Everything defaults to 0 (state = 0 = dead)

// =========== Cells ===========
const cellData = new Int32Array(MAX_ORGANISMS * FIELDS_PER_CELL);
cellData.fill(0);

// Set all d_start fields to -1 initially
for (let i = 0; i < MAX_ORGANISMS; i++) {
  cellData[i * FIELDS_PER_CELL + 3] = -1; // d_start is index 3
}

// ==========Init Ancestor(s)=============
// The 80-instruction genome of 0080aaa
const ancestorGenotype = new Uint32Array([
  0x01, 0x01, 0x01, 0x01, 0x04, 0x02, 0x03, 0x03, 0x18, 0x1c, 0x00, 0x00, 0x00,
  0x00, 0x07, 0x19, 0x1d, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x01, 0x01, 0x00,
  0x01, 0x1e, 0x16, 0x00, 0x00, 0x01, 0x01, 0x1f, 0x14, 0x00, 0x00, 0x01, 0x00,
  0x05, 0x01, 0x01, 0x00, 0x00, 0x0c, 0x0d, 0x0e, 0x01, 0x00, 0x01, 0x00, 0x1a,
  0x0a, 0x05, 0x14, 0x00, 0x01, 0x00, 0x00, 0x08, 0x09, 0x14, 0x00, 0x01, 0x00,
  0x01, 0x05, 0x01, 0x00, 0x01, 0x01, 0x12, 0x11, 0x10, 0x17, 0x01, 0x01, 0x01,
  0x00, 0x05,
]);

// --- Initialize Ancestor CELL (Index 0) at 30000 location ---
const ancestAddr = 30000;
const ancestCellId = 0;
// 1. Put the code in the soup
soup.set(ancestorGenotype, ancestAddr);

// 2. Align the ownership layer
// Owner ID is cell_index + 1 (so 0 means unowned, 1 means Cell 0)
for (let i = 0; i < ancestorGenotype.length; i++) {
  ownership[ancestAddr + i] = ancestCellId + 1;
}

// 3. Initialize Cell 0 (The Organism)
const cellOffset = ancestCellId * FIELDS_PER_CELL;
cellData[cellOffset + 0] = 1; // state (alive)
cellData[cellOffset + 1] = ancestAddr; // mem_start
cellData[cellOffset + 2] = ancestorGenotype.length; // mem_size
cellData[cellOffset + 6] = 1; // active_cpus

// 4. Initialize CPU 0 (The Thread)
const cpuOffset = ancestCellId * FIELDS_PER_CPU;
cpuData[cpuOffset + 4] = ancestAddr; // IP
cpuData[cpuOffset + 20] = 1; // state (alive)
cpuData[cpuOffset + 21] = ancestCellId; // cell_index

// Update the initial global stats
const statData = new Uint32Array(2);
statData[0] = 1; // cells_alive
statData[1] = 80; // memory_used

console.log("Initial Setting");
console.log(soup);
console.log(cpuData);

// ============= Buffer ============

// Uniform Buffer
const uniformBufferSize = 16;
const uniformData = new ArrayBuffer(uniformBufferSize);
const view = new DataView(uniformData);
view.setInt32(0, SOUP_SIZE, true); // soup_size at byte 0
const uniformBuffer = device.createBuffer({
  size: uniformBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformData);

// Storage Buffer
// Buffer to represent the soup, ownership, vCPU, cell
const soupBuffer = device.createBuffer({
  size: soup.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

const ownershipBuffer = device.createBuffer({
  size: ownership.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

const cpuBuffer = device.createBuffer({
  size: cpuData.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

const cellBuffer = device.createBuffer({
  size: cellData.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

const statsBuffer = device.createBuffer({
  size: 16, // Space for a few atomics
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(soupBuffer, 0, soup);
device.queue.writeBuffer(ownershipBuffer, 0, ownership);
device.queue.writeBuffer(cpuBuffer, 0, cpuData);
device.queue.writeBuffer(cellBuffer, 0, cellData);
device.queue.writeBuffer(statsBuffer, 0, statData);

// Slot lock buffers: one atomic<i32> per organism slot.
// 0 = free, 1 = occupied. Used by divide() to claim slots without races.
const cellLocksData = new Uint32Array(MAX_ORGANISMS);
cellLocksData[0] = 1; // Cell 0 (ancestor) is already occupied
const cpuLocksData = new Uint32Array(MAX_ORGANISMS);
cpuLocksData[0] = 1; // CPU 0 (ancestor) is already occupied

const cellLocksBuffer = device.createBuffer({
  size: cellLocksData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const cpuLocksBuffer = device.createBuffer({
  size: cpuLocksData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(cellLocksBuffer, 0, cellLocksData);
device.queue.writeBuffer(cpuLocksBuffer, 0, cpuLocksData);

// Staging buffer

const statsStagingBuffer = device.createBuffer({
  label: "STATS_STAGING_BUFFER",
  size: 16,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const ownershipStagingBuffer = device.createBuffer({
  label: "OWNERSHIP_STAGING_BUFFER",
  size: ownership.byteLength, // Space for a few atomics
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const soupStagingBuffer = device.createBuffer({
  label: "SOUP_STAGING_BUFFER",
  size: soupBuffer.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const cellStagingBuffer = device.createBuffer({
  label: "CELL_STAGING_BUFFER",
  size: cellBuffer.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// ========== Layouts ===========
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 5,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 6,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 7,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
  ],
});

// ========== Pipelines ===========
const computeModule = device.createShaderModule({
  code: computeWGSL,
});
const pipelineLayout = device.createPipelineLayout({
  bindGroupLayouts: [bindGroupLayout],
});
const pipeline = device.createComputePipeline({
  layout: pipelineLayout,
  compute: {
    module: computeModule,
  },
});

// =========== Bind Groups ==========
const computeBindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: uniformBuffer },
    { binding: 1, resource: soupBuffer },
    { binding: 2, resource: ownershipBuffer },
    { binding: 3, resource: cpuBuffer },
    { binding: 4, resource: cellBuffer },
    { binding: 5, resource: statsBuffer },
    { binding: 6, resource: { buffer: cellLocksBuffer } },
    { binding: 7, resource: { buffer: cpuLocksBuffer } },
  ],
});

// How many GPU ticks to advance per visual frame. Raise for faster simulation,
// lower if the browser stutters. 50 ≈ 3000 cycles/sec at 60 fps.
const TICKS_PER_FRAME = 20;

// Pure GPU dispatch — no readback, no await. Submits one simulation step.
function tick() {
  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(64);
  computePass.end();
  device.queue.submit([commandEncoder.finish()]);
  cycleCount++;
}

// ================== Run Sims ================
let isRunning = false;
let cycleCount = 0;
const cycles = 50000;

// Called once per display refresh (~60 fps). Runs TICKS_PER_FRAME simulation
// steps, then does a single GPU→CPU readback for only the data the canvas
// and dashboard actually need (ownership + soup + stats).
// cpuBuffer / cellBuffer are left on the GPU — no wasted transfer.
async function renderFrame() {
  if (!isRunning) return;

  for (let i = 0; i < TICKS_PER_FRAME && cycleCount < cycles; i++) {
    tick();
  }
  if (cycleCount >= cycles) isRunning = false;

  // Single readback for visualization — only 3 buffers instead of 5
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(statsBuffer, 0, statsStagingBuffer, 0, 16);
  commandEncoder.copyBufferToBuffer(
    ownershipBuffer,
    0,
    ownershipStagingBuffer,
    0,
    ownershipBuffer.size,
  );
  commandEncoder.copyBufferToBuffer(
    soupBuffer,
    0,
    soupStagingBuffer,
    0,
    soupBuffer.size,
  );
  commandEncoder.copyBufferToBuffer(
    cellBuffer,
    0,
    cellStagingBuffer,
    0,
    cellBuffer.size,
  );
  device.queue.submit([commandEncoder.finish()]);

  await Promise.all([
    statsStagingBuffer.mapAsync(GPUMapMode.READ),
    ownershipStagingBuffer.mapAsync(GPUMapMode.READ),
    soupStagingBuffer.mapAsync(GPUMapMode.READ),
    cellStagingBuffer.mapAsync(GPUMapMode.READ),
  ]);

  const statsRes = new Int32Array(
    statsStagingBuffer.getMappedRange(0, statsBuffer.size).slice(0),
  );
  const ownershipRes = new Int32Array(
    ownershipStagingBuffer.getMappedRange(0, ownershipBuffer.size).slice(0),
  );
  const soupRes = new Int32Array(
    soupStagingBuffer.getMappedRange(0, soupBuffer.size).slice(0),
  );

  document.getElementById("stat-cycle")!.innerText = cycleCount.toString();
  document.getElementById("stat-pop")!.innerText = statsRes[0].toString();
  document.getElementById("stat-mem")!.innerText = statsRes[1].toString();
  document.getElementById("stat-full")!.innerText = (
    (statsRes[1] / SOUP_SIZE) *
    100
  ).toFixed(2);

  const cellRes = new Int32Array(
    cellStagingBuffer.getMappedRange(0, cellBuffer.size).slice(0),
  );

  // Tally live genome sizes for the histogram
  sizeCounts.fill(0);
  for (let i = 0; i < MAX_ORGANISMS; i++) {
    const off = i * FIELDS_PER_CELL;
    if (cellRes[off] === 1) {
      const sz = cellRes[off + 2];
      if (sz > 0 && sz <= HIST_SIZE_MAX) sizeCounts[sz]++;
    }
  }

  drawSoup(ownershipRes, soupRes);
  recordTimelineSample(cycleCount, statsRes[0], statsRes[1]);
  drawTimeline();
  drawHistogram();

  // Run the genotype census every CENSUS_INTERVAL frames (uses data already mapped)
  censusFrameCounter++;
  if (censusFrameCounter % CENSUS_INTERVAL === 0) {
    runCensus(cellRes, soupRes);
  }

  statsStagingBuffer.unmap();
  ownershipStagingBuffer.unmap();
  soupStagingBuffer.unmap();
  cellStagingBuffer.unmap();

  if (isRunning) requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);

// ============== Get Data from GPU ===================
/**
 * Converts the flat Int32Array from the GPU back into an array of CPU objects.
 */
function getCPUsFromResults(results: Int32Array, count: number): ICPU[] {
  const cpus: ICPU[] = [];
  const STRIDE = FIELDS_PER_CPU;

  for (let i = 0; i < count; i++) {
    const offset = i * STRIDE;
    cpus.push({
      registers: Array.from(results.subarray(offset + 0, offset + 4)),
      IP: results[offset + 4],
      SP: results[offset + 5],
      stack: Array.from(results.subarray(offset + 6, offset + 16)),
      flags: Array.from(results.subarray(offset + 16, offset + 20)),
      state: results[offset + 20],
      cell_index: results[offset + 21],
      // Indices 22 and 23 are padding
    });
  }
  return cpus;
}

function getCellsFromResults(results: Int32Array, count: number): ICell[] {
  const cells: ICell[] = [];
  const STRIDE = FIELDS_PER_CELL;

  for (let i = 0; i < count; i++) {
    const offset = i * STRIDE;
    cells.push({
      state: results[offset + 0],
      mem_start: results[offset + 1],
      mem_size: results[offset + 2],
      d_start: results[offset + 3],
      d_size: results[offset + 4],
      mov_daught: results[offset + 5],
      active_cpus: results[offset + 6],
      age: results[offset + 7],
      errors: results[offset + 8],
    });
  }
  return cells;
}

// ============== Render ===================

function drawSoup(ownershipData: Int32Array, soupData: Int32Array) {
  const ctx = monitorCanvas!.getContext("2d");
  if (!ctx) return;

  const width = monitorCanvas!.width;
  const height = monitorCanvas!.height;
  const imageData = ctx.createImageData(width, height);

  const pixelsPerAddr = (width * height) / SOUP_SIZE;

  for (let i = 0; i < SOUP_SIZE; i++) {
    const owner = ownershipData[i];
    const instruction = soupData[i];

    let r = 0,
      g = 0,
      b = 0;

    if (owner > 0) {
      // 1. Generate base color from owner ID
      // We use prime numbers to spread the colors out
      r = (owner * 53) % 206;
      g = (owner * 97) % 116;
      b = (owner * 151) % 256;

      // 2. Logic to differentiate Instruction vs. Empty Ownership
      // If there is an instruction (value > 0), keep it bright.
      // If the space is owned but empty (value == 0), dim it significantly.
      if (instruction === 0) {
        // Dim the color to 20% brightness to show "Allocated but Empty"
        r = Math.floor(r * 0.2);
        g = Math.floor(g * 0.2);
        b = Math.floor(b * 0.2);
      } else {
        // Boost brightness for instructions so they "glow"
        r = Math.min(255, r + 50);
        g = Math.min(255, g + 50);
        b = Math.min(255, b + 50);
      }
    } else if (instruction !== 0) {
      // OPTIONAL: Draw "Garbage" (Instruction with no owner)
      // This can happen if a cell dies but its code remains in the soup.
      // We'll draw this as dark gray.
      r = 40;
      g = 40;
      b = 40;
    }

    // Fill the pixels
    for (let p = 0; p < pixelsPerAddr; p++) {
      const pixelIndex = (i * pixelsPerAddr + p) * 4;
      imageData.data[pixelIndex + 0] = r;
      imageData.data[pixelIndex + 1] = g;
      imageData.data[pixelIndex + 2] = b;
      imageData.data[pixelIndex + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

// ============== Population Timeline ===================

const TIMELINE_MAX_SAMPLES = 600; // one per render frame; scrolls as history fills

interface TimelineSample {
  cycle: number;
  population: number;
  memUsed: number;
}
const timelineHistory: TimelineSample[] = [];

function recordTimelineSample(
  cycle: number,
  population: number,
  memUsed: number,
) {
  timelineHistory.push({ cycle, population, memUsed });
  if (timelineHistory.length > TIMELINE_MAX_SAMPLES) timelineHistory.shift();
}

function drawTimeline() {
  const ctx = timelineCanvas.getContext("2d");
  if (!ctx || timelineHistory.length < 2) return;

  const W = timelineCanvas.width;
  const H = timelineCanvas.height;
  const PAD = { top: 18, bottom: 22, left: 38, right: 10 };
  const cW = W - PAD.left - PAD.right; // chart area width
  const cH = H - PAD.top - PAD.bottom; // chart area height

  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);

  // Horizontal grid lines + Y-axis labels (population)
  ctx.font = "10px monospace";
  for (let t = 0; t <= 4; t++) {
    const y = PAD.top + (t / 4) * cH;
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(PAD.left + cW, y);
    ctx.stroke();

    ctx.fillStyle = "#555";
    ctx.textAlign = "right";
    ctx.fillText(
      String(Math.round(MAX_ORGANISMS * (1 - t / 4))),
      PAD.left - 4,
      y + 3,
    );
  }

  const n = timelineHistory.length;

  // Memory-used line (orange, normalized 0–1 over SOUP_SIZE)
  ctx.strokeStyle = "#c84";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = PAD.left + (i / (TIMELINE_MAX_SAMPLES - 1)) * cW;
    const y = PAD.top + cH - (timelineHistory[i].memUsed / SOUP_SIZE) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Population line (green, normalized 0–1 over MAX_ORGANISMS)
  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = PAD.left + (i / (TIMELINE_MAX_SAMPLES - 1)) * cW;
    const y =
      PAD.top + cH - (timelineHistory[i].population / MAX_ORGANISMS) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Legend
  ctx.textAlign = "left";
  ctx.fillStyle = "#0f0";
  ctx.fillText("— population", PAD.left + 4, PAD.top - 4);
  ctx.fillStyle = "#c84";
  ctx.fillText("— memory", PAD.left + 110, PAD.top - 4);

  // Cycle stamp (right edge, bottom)
  const last = timelineHistory[n - 1];
  ctx.fillStyle = "#444";
  ctx.textAlign = "right";
  ctx.fillText(`cycle ${last.cycle}`, W - PAD.right, H - 5);
}

// ============== Size Distribution Histogram ===================

const HIST_SIZE_MAX = 200; // track genome sizes 1..200

// Counts per genome size — reset and refilled each frame from cellBuffer readback.
const sizeCounts = new Int32Array(HIST_SIZE_MAX + 1);

function drawHistogram() {
  const ctx = histogramCanvas.getContext("2d");
  if (!ctx) return;

  const W = histogramCanvas.width;
  const H = histogramCanvas.height;
  const PAD = { top: 18, bottom: 18, left: 38, right: 10 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);

  // Find the active size range and peak count
  let minSize = HIST_SIZE_MAX,
    maxSize = 0,
    maxCount = 0;
  for (let s = 1; s <= HIST_SIZE_MAX; s++) {
    if (sizeCounts[s] > 0) {
      if (s < minSize) minSize = s;
      if (s > maxSize) maxSize = s;
      if (sizeCounts[s] > maxCount) maxCount = sizeCounts[s];
    }
  }
  if (maxCount === 0) return; // nothing alive yet

  // Widen the visible range by a small margin for readability
  const rangeStart = Math.max(1, minSize - 5);
  const rangeEnd = Math.min(HIST_SIZE_MAX, maxSize + 5);
  const range = rangeEnd - rangeStart + 1;
  const barW = Math.max(1, Math.floor(cW / range));

  // Horizontal grid + Y-axis labels (count)
  ctx.font = "10px monospace";
  for (let t = 0; t <= 4; t++) {
    const y = PAD.top + (t / 4) * cH;
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(PAD.left + cW, y);
    ctx.stroke();
    ctx.fillStyle = "#555";
    ctx.textAlign = "right";
    ctx.fillText(
      String(Math.round(maxCount * (1 - t / 4))),
      PAD.left - 4,
      y + 3,
    );
  }

  // Bars — same prime-hash color scheme as the soup view
  for (let s = rangeStart; s <= rangeEnd; s++) {
    const count = sizeCounts[s];
    if (count === 0) continue;
    const x = PAD.left + (s - rangeStart) * barW;
    const barH = (count / maxCount) * cH;
    const r = Math.min(255, ((s * 53) % 256) + 50);
    const g = Math.min(255, ((s * 97) % 256) + 50);
    const b = Math.min(255, ((s * 151) % 256) + 50);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, PAD.top + cH - barH, Math.max(1, barW - 1), barH);
  }

  // X-axis size labels — show ~8 evenly spaced ticks
  ctx.fillStyle = "#555";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(range / 8));
  for (let s = rangeStart; s <= rangeEnd; s += step) {
    const x = PAD.left + (s - rangeStart) * barW + barW / 2;
    ctx.fillText(String(s), x, H - 4);
  }

  // Chart title
  ctx.fillStyle = "#444";
  ctx.textAlign = "left";
  ctx.font = "10px monospace";
  ctx.fillText("genome size →", PAD.left + 4, PAD.top - 4);
}

// ============== Genotype Census + Genebank ===================

const CENSUS_INTERVAL = 50; // run census every N render frames
const GENEBANK_THRESHOLD = 4; // min peak population to enter the genebank

interface GenotypeRecord {
  hash: number;
  size: number;
  genome: Int32Array; // full instruction sequence
  firstSeen: number; // cycle number when first observed
  peakPop: number; // highest simultaneous population ever seen
  currentPop: number; // population in the most recent census
  inBank: boolean; // promoted to genebank once peakPop >= threshold
}

const genotypeMap = new Map<number, GenotypeRecord>();
const genebank: GenotypeRecord[] = [];
let censusFrameCounter = 0;

function extractGenome(
  soupRes: Int32Array,
  memStart: number,
  memSize: number,
): Int32Array {
  const genome = new Int32Array(memSize);
  for (let i = 0; i < memSize; i++) {
    genome[i] = soupRes[(((memStart + i) % SOUP_SIZE) + SOUP_SIZE) % SOUP_SIZE];
  }
  return genome;
}

function hashGenome(genome: Int32Array): number {
  let h = 2166136261; // FNV-1a offset basis (32-bit)
  for (let i = 0; i < genome.length; i++) {
    h ^= genome[i];
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function runCensus(cellRes: Int32Array, soupRes: Int32Array) {
  // 1. Count each distinct genotype alive right now
  const currentCounts = new Map<number, number>();
  const firstSeen = new Map<number, Int32Array>(); // hash → genome (first occurrence)

  for (let i = 0; i < MAX_ORGANISMS; i++) {
    const off = i * FIELDS_PER_CELL;
    if (cellRes[off] !== 1) continue; // dead slot
    const memStart = cellRes[off + 1];
    const memSize = cellRes[off + 2];
    if (memSize <= 0 || memSize > HIST_SIZE_MAX) continue;

    const genome = extractGenome(soupRes, memStart, memSize);
    const hash = hashGenome(genome);
    currentCounts.set(hash, (currentCounts.get(hash) ?? 0) + 1);
    if (!firstSeen.has(hash)) firstSeen.set(hash, genome);
  }

  // 2. Reset currentPop for all tracked records (marks extinct ones as 0)
  for (const rec of genotypeMap.values()) rec.currentPop = 0;

  // 3. Upsert into genotypeMap, check genebank threshold
  for (const [hash, count] of currentCounts) {
    let rec = genotypeMap.get(hash);
    if (!rec) {
      const genome = firstSeen.get(hash)!;
      rec = {
        hash,
        size: genome.length,
        genome,
        firstSeen: cycleCount,
        peakPop: count,
        currentPop: count,
        inBank: false,
      };
      genotypeMap.set(hash, rec);
    } else {
      rec.currentPop = count;
      rec.peakPop = Math.max(rec.peakPop, count);
    }
    if (!rec.inBank && rec.peakPop >= GENEBANK_THRESHOLD) {
      rec.inBank = true;
      genebank.push(rec);
    }
  }

  updateCensusDisplay();
}

function updateCensusDisplay() {
  const alive = [...genotypeMap.values()]
    .filter((r) => r.currentPop > 0)
    .sort((a, b) => b.currentPop - a.currentPop)
    .slice(0, 12);

  document.getElementById("stat-genotypes")!.innerText =
    `${alive.length} live / ${genotypeMap.size} total`;
  document.getElementById("stat-genebank")!.innerText = String(genebank.length);

  if (alive.length === 0) {
    document.getElementById("census-list")!.innerHTML = "";
    return;
  }

  const maxPop = alive[0].currentPop;
  const rows = alive.map((r) => {
    const barLen = Math.round((r.currentPop / maxPop) * 16);
    const bar = "█".repeat(barLen).padEnd(16, "░");
    const star = r.inBank ? ' <span style="color:#f80">★</span>' : "";
    return (
      `<div>` +
      `<span style="color:#666">${String(r.size).padStart(3)}</span> ` +
      `<span style="color:#0a0">${bar}</span> ` +
      `<span style="color:#aaa">${r.currentPop}</span>` +
      star +
      `</div>`
    );
  });
  document.getElementById("census-list")!.innerHTML = rows.join("");
}

function downloadGenebank() {
  if (genebank.length === 0) return;
  const entries = genebank.map((r) => ({
    hash: r.hash.toString(16).padStart(8, "0"),
    size: r.size,
    firstSeen: r.firstSeen,
    peakPop: r.peakPop,
    opcodes: Array.from(r.genome).map((op) => OP_NAMES[op] ?? `?${op}`),
  }));
  const blob = new Blob([JSON.stringify(entries, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `tierra_genebank_c${cycleCount}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ============ Report =================
const OP_NAMES: Record<number, string> = {
  0x00: "nop0",
  0x01: "nop1",
  0x02: "not0",
  0x03: "shl",
  0x04: "zero",
  0x05: "ifz",
  0x06: "subCAB",
  0x07: "subAAC",
  0x08: "incA",
  0x09: "incB",
  0x0a: "decC",
  0x0b: "incC",
  0x0c: "pushA",
  0x0d: "pushB",
  0x0e: "pushC",
  0x0f: "pushD",
  0x10: "popA",
  0x11: "popB",
  0x12: "popC",
  0x13: "popD",
  0x14: "jmpo",
  0x15: "jmpb",
  0x16: "call",
  0x17: "ret",
  0x18: "movDC",
  0x19: "movBA",
  0x1a: "mov_iab",
  0x1b: "adro",
  0x1c: "adrb",
  0x1d: "adrf",
  0x1e: "mal",
  0x1f: "divide",
};

class SimulationReport {
  private log: string = "Cycle,CPU_Idx,IP,Instruction,AX,BX,CX,DX,ErrorFlag\n";

  appendCycle(
    cycle: number,
    cpuIdx: number,
    cpuData: Int32Array,
    soupData: Int32Array,
  ) {
    const offset = cpuIdx * 24; // FIELDS_PER_CPU
    if (cpuData[offset + 20] === 0) return; // CPU is dead

    const ip = cpuData[offset + 4];
    const regs = cpuData.subarray(offset + 0, offset + 4);
    const errorFlag = cpuData[offset + 16];

    // Get instruction at IP
    const opcode = soupData[((ip % 60000) + 60000) % 60000] % 32;
    const instName = OP_NAMES[opcode] || "???";

    this.log += `${cycle},${cpuIdx},${ip},${instName},${regs[0]},${regs[1]},${regs[2]},${regs[3]},${errorFlag}\n`;
  }

  download() {
    const blob = new Blob([this.log], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tierra_report_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

const reporter = new SimulationReport();

// Add a button to your HTML
document.body.insertAdjacentHTML(
  "beforeend",
  `<button id="downloadBtn">Download CSV Report</button>`,
);
document.getElementById("downloadBtn")!.onclick = () => reporter.download();
document.getElementById("genebankBtn")!.onclick = downloadGenebank;
