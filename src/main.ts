import "./style.css";
import computeWGSL from "./shaders/compute.wgsl";
import {
  SOUP_SIZE,
  MAX_ORGANISMS,
  FIELDS_PER_CELL,
  FIELDS_PER_CPU,
  OP_NAMES,
} from "./constants";
import {
  initMonitor,
  drawSoup,
  recordTimelineSample,
  drawTimeline,
  tallySizes,
  drawHistogram,
  tickCensus,
  downloadGenebank,
} from "./monitor";

// ============== UI ==============

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
      <hr>
      <h3>Settings</h3>
      <label style="font-size:11px;cursor:pointer">
        <input type="checkbox" id="debrisToggle" checked>
        Organic debris (dead code stays in soup)
      </label>
      <hr>
      <h3>Mutation</h3>
      <div style="font-size:11px;line-height:2.2">
        <div><label>Cosmic bit-flip &nbsp;
          <input type="range" id="cosmicBitSlider" min="0" max="0.0005" step="0.000001" value="0.00005" style="width:90px;vertical-align:middle">
          <span id="cosmicBitVal">1 in 20,000</span>
        </label></div>
        <div><label>Cosmic replace &nbsp;
          <input type="range" id="cosmicRepSlider" min="0" max="0.0005" step="0.000001" value="0" style="width:90px;vertical-align:middle">
          <span id="cosmicRepVal">off</span>
        </label></div>
        <div><label>Copy bit-flip &nbsp;&nbsp;&nbsp;
          <input type="range" id="copyBitSlider" min="0" max="0.05" step="0.0001" value="0.001" style="width:90px;vertical-align:middle">
          <span id="copyBitVal">1 in 1,000</span>
        </label></div>
        <div><label>Copy replace &nbsp;&nbsp;&nbsp;
          <input type="range" id="copyRepSlider" min="0" max="0.05" step="0.0001" value="0" style="width:90px;vertical-align:middle">
          <span id="copyRepVal">off</span>
        </label></div>
      </div>
    </div>
  </div>
  <canvas id="canvas"></canvas>
`;

initMonitor(
  document.querySelector<HTMLCanvasElement>("#monitor")!,
  document.querySelector<HTMLCanvasElement>("#timeline")!,
  document.querySelector<HTMLCanvasElement>("#histogram")!,
);

const canvas = document.querySelector<HTMLCanvasElement>("#canvas");

// ============== WebGPU Setup ==============

if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No appropriate GPU adapter found.");

const device = await adapter.requestDevice();
if (!device) throw new Error("Failed to create WebGPU device.");

const context = canvas?.getContext("webgpu");
if (!context) throw new Error("WebGPU context could not be initialized.");

// ============== Soup / CPU / Cell Data ==============

const soup = new Uint32Array(SOUP_SIZE);
const ownership = new Uint32Array(SOUP_SIZE);

const cpuData = new Int32Array(MAX_ORGANISMS * FIELDS_PER_CPU);
const cellData = new Int32Array(MAX_ORGANISMS * FIELDS_PER_CELL);

// d_start defaults to -1 (no daughter allocated)
for (let i = 0; i < MAX_ORGANISMS; i++) {
  cellData[i * FIELDS_PER_CELL + 3] = -1;
}

// ============== Ancestor (0080aaa) ==============

const ancestorGenotype = new Uint32Array([
  0x01, 0x01, 0x01, 0x01, 0x04, 0x02, 0x03, 0x03, 0x18, 0x1c, 0x00, 0x00, 0x00,
  0x00, 0x07, 0x19, 0x1d, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x01, 0x01, 0x00,
  0x01, 0x1e, 0x16, 0x00, 0x00, 0x01, 0x01, 0x1f, 0x14, 0x00, 0x00, 0x01, 0x00,
  0x05, 0x01, 0x01, 0x00, 0x00, 0x0c, 0x0d, 0x0e, 0x01, 0x00, 0x01, 0x00, 0x1a,
  0x0a, 0x05, 0x14, 0x00, 0x01, 0x00, 0x00, 0x08, 0x09, 0x14, 0x00, 0x01, 0x00,
  0x01, 0x05, 0x01, 0x00, 0x01, 0x01, 0x12, 0x11, 0x10, 0x17, 0x01, 0x01, 0x01,
  0x00, 0x05,
]);

const ancestAddr = 30000;
soup.set(ancestorGenotype, ancestAddr);
for (let i = 0; i < ancestorGenotype.length; i++) ownership[ancestAddr + i] = 1; // owner = cell 0 + 1

const cellOffset = 0 * FIELDS_PER_CELL;
cellData[cellOffset + 0] = 1; // state (alive)
cellData[cellOffset + 1] = ancestAddr; // mem_start
cellData[cellOffset + 2] = ancestorGenotype.length; // mem_size
cellData[cellOffset + 6] = 1; // active_cpus

const cpuOffset = 0 * FIELDS_PER_CPU;
cpuData[cpuOffset + 4] = ancestAddr; // IP
cpuData[cpuOffset + 20] = 1; // state (alive)
cpuData[cpuOffset + 21] = 0; // cell_index

const statData = new Uint32Array([1, 80]); // cells_alive, memory_used

// ============== GPU Buffers ==============

const uniformBufferSize = 32;
const uniformData = new ArrayBuffer(uniformBufferSize);
const view = new DataView(uniformData);
view.setInt32(0, SOUP_SIZE, true);
view.setInt32(4, 1, true); // debris_mode = 1 (on) by default
view.setUint32(8, 0, true); // cycle (updated each tick)
view.setUint32(12, 0, true); // _pad
view.setFloat32(16, 0.00005, true); // cosmic_bit_rate  (≈1 in 20,000)
view.setFloat32(20, 0.0, true); // cosmic_rep_rate  (off)
view.setFloat32(24, 0.001, true); // copy_bit_rate    (≈1 in 1,000)
view.setFloat32(28, 0.0, true); // copy_rep_rate    (off)
const uniformBuffer = device.createBuffer({
  size: uniformBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformData);

function setDebrisMode(on: boolean) {
  view.setInt32(4, on ? 1 : 0, true);
  device.queue.writeBuffer(uniformBuffer, 4, uniformData, 4, 4);
}

function setCosmicBitRate(r: number) {
  view.setFloat32(16, r, true);
  device.queue.writeBuffer(uniformBuffer, 16, uniformData, 16, 4);
}
function setCosmicRepRate(r: number) {
  view.setFloat32(20, r, true);
  device.queue.writeBuffer(uniformBuffer, 20, uniformData, 20, 4);
}
function setCopyBitRate(r: number) {
  view.setFloat32(24, r, true);
  device.queue.writeBuffer(uniformBuffer, 24, uniformData, 24, 4);
}
function setCopyRepRate(r: number) {
  view.setFloat32(28, r, true);
  device.queue.writeBuffer(uniformBuffer, 28, uniformData, 28, 4);
}

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
  size: 16,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(soupBuffer, 0, soup);
device.queue.writeBuffer(ownershipBuffer, 0, ownership);
device.queue.writeBuffer(cpuBuffer, 0, cpuData);
device.queue.writeBuffer(cellBuffer, 0, cellData);
device.queue.writeBuffer(statsBuffer, 0, statData);

// Slot lock buffers: 0 = free, 1 = occupied. Used by divide() to claim slots atomically.
const cellLocksData = new Uint32Array(MAX_ORGANISMS);
const cpuLocksData = new Uint32Array(MAX_ORGANISMS);
cellLocksData[0] = 1; // ancestor cell already occupied
cpuLocksData[0] = 1; // ancestor CPU already occupied

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

// Staging buffers (GPU → CPU readback)
const statsStagingBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const ownershipStagingBuffer = device.createBuffer({
  size: ownership.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const soupStagingBuffer = device.createBuffer({
  size: soupBuffer.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const cellStagingBuffer = device.createBuffer({
  size: cellBuffer.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// ============== Pipeline ==============

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

const pipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
  compute: { module: device.createShaderModule({ code: computeWGSL }) },
});

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

// ============== Simulation Loop ==============

const TICKS_PER_FRAME = 30;
let isRunning = false;
let cycleCount = 0;
const cycles = 1000000;

function tick() {
  // Write current cycle to uniform so the shader can use it as an RNG seed component
  view.setUint32(8, cycleCount, true);
  device.queue.writeBuffer(uniformBuffer, 8, uniformData, 8, 4);

  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(64);
  computePass.end();
  device.queue.submit([commandEncoder.finish()]);
  cycleCount++;
}

async function renderFrame() {
  if (!isRunning) return;

  for (let i = 0; i < TICKS_PER_FRAME && cycleCount < cycles; i++) tick();
  if (cycleCount >= cycles) isRunning = false;

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
  const cellRes = new Int32Array(
    cellStagingBuffer.getMappedRange(0, cellBuffer.size).slice(0),
  );

  document.getElementById("stat-cycle")!.innerText = cycleCount.toString();
  document.getElementById("stat-pop")!.innerText = statsRes[0].toString();
  document.getElementById("stat-mem")!.innerText = statsRes[1].toString();
  document.getElementById("stat-full")!.innerText = (
    (statsRes[1] / SOUP_SIZE) *
    100
  ).toFixed(2);

  tallySizes(cellRes);
  drawSoup(ownershipRes, soupRes);
  recordTimelineSample(cycleCount, statsRes[0], statsRes[1]);
  drawTimeline();
  drawHistogram();
  tickCensus(cellRes, soupRes, cycleCount);

  statsStagingBuffer.unmap();
  ownershipStagingBuffer.unmap();
  soupStagingBuffer.unmap();
  cellStagingBuffer.unmap();

  if (isRunning) requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);

// ============== Debug Report ==============

class SimulationReport {
  private log = "Cycle,CPU_Idx,IP,Instruction,AX,BX,CX,DX,ErrorFlag\n";

  appendCycle(
    cycle: number,
    cpuIdx: number,
    cpuData: Int32Array,
    soupData: Int32Array,
  ) {
    const offset = cpuIdx * FIELDS_PER_CPU;
    if (cpuData[offset + 20] === 0) return;

    const ip = cpuData[offset + 4];
    const regs = cpuData.subarray(offset + 0, offset + 4);
    const errorFlag = cpuData[offset + 16];
    const opcode = soupData[((ip % SOUP_SIZE) + SOUP_SIZE) % SOUP_SIZE] % 32;

    this.log += `${cycle},${cpuIdx},${ip},${OP_NAMES[opcode] ?? "???"},${regs[0]},${regs[1]},${regs[2]},${regs[3]},${errorFlag}\n`;
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

// ============== Buttons ==============

document.body.insertAdjacentHTML(
  "beforeend",
  `<button id="downloadBtn">Download CSV Report</button>`,
);
document.getElementById("downloadBtn")!.onclick = () => reporter.download();
document.getElementById("genebankBtn")!.onclick = () =>
  downloadGenebank(cycleCount);
(document.getElementById("debrisToggle") as HTMLInputElement).onchange = (e) =>
  setDebrisMode((e.target as HTMLInputElement).checked);

function fmtRate(r: number): string {
  if (r <= 0) return "off";
  return `1 in ${Math.round(1 / r).toLocaleString()}`;
}

(document.getElementById("cosmicBitSlider") as HTMLInputElement).oninput = (
  e,
) => {
  const r = parseFloat((e.target as HTMLInputElement).value);
  setCosmicBitRate(r);
  document.getElementById("cosmicBitVal")!.innerText = fmtRate(r);
};
(document.getElementById("cosmicRepSlider") as HTMLInputElement).oninput = (
  e,
) => {
  const r = parseFloat((e.target as HTMLInputElement).value);
  setCosmicRepRate(r);
  document.getElementById("cosmicRepVal")!.innerText = fmtRate(r);
};
(document.getElementById("copyBitSlider") as HTMLInputElement).oninput = (
  e,
) => {
  const r = parseFloat((e.target as HTMLInputElement).value);
  setCopyBitRate(r);
  document.getElementById("copyBitVal")!.innerText = fmtRate(r);
};
(document.getElementById("copyRepSlider") as HTMLInputElement).oninput = (
  e,
) => {
  const r = parseFloat((e.target as HTMLInputElement).value);
  setCopyRepRate(r);
  document.getElementById("copyRepVal")!.innerText = fmtRate(r);
};
