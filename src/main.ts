import "./style.css";
import computeWGSL from "./shaders/compute.wgsl";
import soupRenderWGSL from "./shaders/soup_render.wgsl";
import markCpusWGSL from "./shaders/mark_cpus.wgsl";
import {
  SOUP_SIZE,
  MAX_ORGANISMS,
  FIELDS_PER_CELL,
  FIELDS_PER_CPU,
} from "./constants";
import {
  initMonitor,
  recordTimelineSample,
  drawTimeline,
  tallySizes,
  drawHistogram,
  tickCensus,
  willRunCensus,
  downloadGenebank,
} from "./monitor";

// ============== UI ==============

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<div style="font-family: monospace; background: #222; color: #0f0; padding: 0; margin: 0;">
  <h1 style ="padding: 0; margin: 0; text-align: left;">TIERRA - Artificial Life</h1>  
</div>
<div style="display: flex; gap: 20px; font-family: monospace; background: #222; color: #0f0; padding: 20px;">
    <div style="display: flex; flex-direction: column; gap: 10px;">
      
      <canvas id="monitor" width="800" height="600" style="border: 1px solid #555;"></canvas>
      <canvas id="timeline" width="600" height="100" style="border: 1px solid #555;"></canvas>
      <canvas id="histogram" width="600" height="80" style="border: 1px solid #555;"></canvas>
    </div>
    <div id="dashboard">
      <h2>TIERRA MONITOR</h2>
      <button id="startStopBtn" style="font-family:monospace;font-size:13px;background:#333;color:#0f0;border:1px solid #555;padding:4px 12px;cursor:pointer;margin-bottom:8px">▶ Start</button>
      <p>Cycle: <span id="stat-cycle">0</span></p>
      <p>Population: <span id="stat-pop">0</span> / ${MAX_ORGANISMS}</p>
      <p>Memory Usage: <span id="stat-mem">0</span> / ${SOUP_SIZE}</p>
      <p>Fullness: <span id="stat-full">0</span>%</p>
      <h3>Genotypes</h3>
      <p style="font-size:11px">Live/Total: <span id="stat-genotypes">—</span></p>
      <p style="font-size:11px">Genebank: <span id="stat-genebank">0</span>
        <button id="genebankBtn" style="font-family:monospace;font-size:10px;background:#333;color:#0f0;border:1px solid #555;padding:1px 5px;cursor:pointer">↓ save</button>
      </p>
      <p style="font-size:11px">Top 12 Genotypes</p>
      <div id="census-list" style="font-size:11px;line-height:1.6;margin-top:4px"></div>
      <hr>
      <h3>Settings</h3>
      <label style="font-size:11px;cursor:pointer">
        <input type="checkbox" id="debrisToggle" checked>
        Organic debris (dead code stays in soup)
      </label><br>
      <label style="font-size:11px;cursor:pointer">
        <input type="checkbox" id="cpuMarkersToggle">
        Show CPU pointers
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
          <input type="range" id="copyBitSlider" min="0" max="0.005" step="0.0001" value="0.001" style="width:90px;vertical-align:middle">
          <span id="copyBitVal">1 in 1,000</span>
        </label></div>
        <div><label>Copy replace &nbsp;&nbsp;&nbsp;
          <input type="range" id="copyRepSlider" min="0" max="0.005" step="0.0001" value="0" style="width:90px;vertical-align:middle">
          <span id="copyRepVal">off</span>
        </label></div>
      </div>
    </div>
  </div>
`;

initMonitor(
  document.querySelector<HTMLCanvasElement>("#timeline")!,
  document.querySelector<HTMLCanvasElement>("#histogram")!,
);

// ============== WebGPU Setup ==============

if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No appropriate GPU adapter found.");

const device = await adapter.requestDevice();
if (!device) throw new Error("Failed to create WebGPU device.");

// Configure the monitor canvas as the WebGPU render target for soup visualization.
const monitorCanvas = document.querySelector<HTMLCanvasElement>("#monitor")!;
const monitorContext = monitorCanvas.getContext("webgpu")!;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
monitorContext.configure({ device, format: presentationFormat });

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

const uniformBufferSize = 48;
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
view.setUint32(32, monitorCanvas.width, true);
view.setUint32(36, monitorCanvas.height, true);
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
// soupStagingBuffer is only read on census frames (every 50 frames) for genotype census.
// Soup visualization is handled entirely on GPU via the render pipeline.
const soupStagingBuffer = device.createBuffer({
  size: soupBuffer.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const cellStagingBuffer = device.createBuffer({
  size: cellBuffer.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// CPU marker buffer: one i32 per soup address.
// Cleared before each render pass, then filled by the markCpus compute pre-pass.
// Value: 0 = no CPU here, >0 = cell_index+1 of the CPU whose IP is at that address.
const cpuMarkerBuffer = device.createBuffer({
  size: SOUP_SIZE * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

// ============== Soup Render Pipeline ==============

const soupRenderPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: device.createShaderModule({ code: soupRenderWGSL }),
    entryPoint: "vs_main",
  },
  fragment: {
    module: device.createShaderModule({ code: soupRenderWGSL }),
    entryPoint: "fs_main",
    targets: [{ format: presentationFormat }],
  },
  primitive: { topology: "triangle-list" },
});

const soupRenderBindGroup = device.createBindGroup({
  layout: soupRenderPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: soupBuffer } },
    { binding: 2, resource: { buffer: ownershipBuffer } },
    { binding: 3, resource: { buffer: cpuMarkerBuffer } },
  ],
});

// ============== CPU Mark Pre-pass Pipeline ==============
// Tiny compute pass that runs just before rendering: writes each alive CPU's
// cell_index+1 into cpuMarkerBuffer at its IP address for the render shader to read.

const markCpusPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: device.createShaderModule({ code: markCpusWGSL }),
    entryPoint: "mark_cpus",
  },
});

const markCpusBindGroup = device.createBindGroup({
  layout: markCpusPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: cpuBuffer } },
    { binding: 2, resource: { buffer: cpuMarkerBuffer } },
  ],
});

// ============== Simulation Loop ==============

const TICKS_PER_FRAME = 30;
let isRunning = false;
let cycleCount = 0;

const startStopBtn = document.getElementById(
  "startStopBtn",
) as HTMLButtonElement;
startStopBtn.onclick = () => {
  isRunning = !isRunning;
  startStopBtn.textContent = isRunning ? "⏸ Pause" : "▶ Start";
  if (isRunning) requestAnimationFrame(renderFrame);
};
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

  // Only read soup back to CPU on census frames (every 50 frames).
  // All other frames, soup visualization is handled entirely on the GPU.
  const doSoupReadback = willRunCensus();

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(statsBuffer, 0, statsStagingBuffer, 0, 16);
  commandEncoder.copyBufferToBuffer(
    cellBuffer,
    0,
    cellStagingBuffer,
    0,
    cellBuffer.size,
  );
  if (doSoupReadback) {
    commandEncoder.copyBufferToBuffer(
      soupBuffer,
      0,
      soupStagingBuffer,
      0,
      soupBuffer.size,
    );
  }

  // Clear cpu_markers (always, so toggling off immediately removes all triangles).
  // Only dispatch the mark pass when the CPU pointer overlay is enabled.
  commandEncoder.clearBuffer(cpuMarkerBuffer);
  if (showCpuMarkers) {
    const markPass = commandEncoder.beginComputePass();
    markPass.setPipeline(markCpusPipeline);
    markPass.setBindGroup(0, markCpusBindGroup);
    markPass.dispatchWorkgroups(Math.ceil(MAX_ORGANISMS / 64));
    markPass.end();
  }

  // GPU render pass: draws soup directly from GPU buffers — no CPU pixel loop.
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: monitorContext.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  renderPass.setPipeline(soupRenderPipeline);
  renderPass.setBindGroup(0, soupRenderBindGroup);
  renderPass.draw(6);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);

  const toMap: Promise<void>[] = [
    statsStagingBuffer.mapAsync(GPUMapMode.READ),
    cellStagingBuffer.mapAsync(GPUMapMode.READ),
  ];
  if (doSoupReadback) toMap.push(soupStagingBuffer.mapAsync(GPUMapMode.READ));
  await Promise.all(toMap);

  const statsRes = new Int32Array(
    statsStagingBuffer.getMappedRange(0, statsBuffer.size).slice(0),
  );
  const cellRes = new Int32Array(
    cellStagingBuffer.getMappedRange(0, cellBuffer.size).slice(0),
  );
  const soupRes: Int32Array | null = doSoupReadback
    ? new Int32Array(
        soupStagingBuffer.getMappedRange(0, soupBuffer.size).slice(0),
      )
    : null;

  document.getElementById("stat-cycle")!.innerText = cycleCount.toString();
  document.getElementById("stat-pop")!.innerText = statsRes[0].toString();
  document.getElementById("stat-mem")!.innerText = statsRes[1].toString();
  document.getElementById("stat-full")!.innerText = (
    (statsRes[1] / SOUP_SIZE) *
    100
  ).toFixed(2);

  tallySizes(cellRes);
  recordTimelineSample(cycleCount, statsRes[0], statsRes[1]);
  drawTimeline();
  drawHistogram();
  tickCensus(cellRes, soupRes, cycleCount);

  statsStagingBuffer.unmap();
  cellStagingBuffer.unmap();
  if (doSoupReadback) soupStagingBuffer.unmap();

  if (isRunning) requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);

// ============== Buttons ==============

/* document.body.insertAdjacentHTML(
  "beforeend",
  `<button id="downloadBtn">Download CSV Report</button>`,
); */
/* document.getElementById("downloadBtn")!.onclick = () => reporter.download(); */
document.getElementById("genebankBtn")!.onclick = () =>
  downloadGenebank(cycleCount);
(document.getElementById("debrisToggle") as HTMLInputElement).onchange = (e) =>
  setDebrisMode((e.target as HTMLInputElement).checked);

let showCpuMarkers = false;
(document.getElementById("cpuMarkersToggle") as HTMLInputElement).onchange = (
  e,
) => {
  showCpuMarkers = (e.target as HTMLInputElement).checked;
};

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
