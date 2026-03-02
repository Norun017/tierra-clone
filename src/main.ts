import "./style.css";
import computeWGSL from "./shaders/compute.wgsl";

const SOUP_SIZE = 60000;
const MAX_ORGANISMS = 1024;
const FIELDS_PER_CELL = 8; // 32 bytes
const FIELDS_PER_CPU = 24; // 96 bytes

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div>
    <canvas id="canvas"></canvas>
  </div>
`;
const canvas = document.querySelector<HTMLCanvasElement>("#canvas");
canvas!.width = 600;
canvas!.height = 600;
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
interface CPU {
  registers: Array<number>; // AX, BX, CX, DX
  IP: number; // Instruction Pointer
  SP: number; // Stack Pointer
  stack: Array<number>; // temp memory
  flags: Array<number>; // flags holder
}

const cpuData = new Int32Array(MAX_ORGANISMS * FIELDS_PER_CPU); // In this are maps of creatureId: properties. Length = creatures amount
cpuData.fill(0); // Everything defaults to 0 (state = 0 = dead)

// Helper to update a specific "CPU"
function setCPUData(index: number, data: CPU) {
  const offset = index * FIELDS_PER_CPU;
  cpuData.set(data.registers, offset + 0);
  cpuData[offset + 4] = data.IP;
  cpuData[offset + 5] = data.SP;
  cpuData.set(data.stack, offset + 6);
  cpuData.set(data.flags, offset + 16);
}

// =========== Cells ===========
interface Cell {
  mM: Array<number>; // Memory Main
  mD: Array<number>; // Memory Daughter
  cId: number; // vCPU(s) locator ID
  lD: boolean; // Live = 1, Death = 0
}

const cellData = new Int32Array(MAX_ORGANISMS * FIELDS_PER_CELL);
cellData.fill(0);

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
cellData[cellOffset + 5] = 1; // active_cpus

// 4. Initialize CPU 0 (The Thread)
const cpuOffset = ancestCellId * FIELDS_PER_CPU;
cpuData[cpuOffset + 4] = ancestAddr; // IP
cpuData[cpuOffset + 20] = 1; // state (alive)
cpuData[cpuOffset + 21] = ancestCellId; // cell_index

/* const cell: Cell = {
  mM: [ancestAddr, ancestAddr + ancestorGenotype.length],
  mD: [0, 0],
  cId: 0,
  lD: true,
};
cells.push(cell);

ownership.fill(cell.cId, ancestAddr, ancestAddr + ancestorGenotype.length); // fill ownership with cell Id

const cpu: CPU = {
  registers: [0, 0, 0, 0],
  IP: ancestAddr,
  SP: 0,
  stack: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  flags: [0, 0, 0, 0],
};
setCPUData(0, cpu); */

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
// Buffer to represent the soup, ownership, vCPU

const soupBuffer = device.createBuffer({
  size: soup.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

// For tracking which blocks are owned by whom? // read only
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
  size: cpuData.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

device.queue.writeBuffer(soupBuffer, 0, soup);
device.queue.writeBuffer(ownershipBuffer, 0, ownership);
device.queue.writeBuffer(cpuBuffer, 0, cpuData);
device.queue.writeBuffer(cellBuffer, 0, cellData);

// Output and staging buffer

const stagingBuffer = device.createBuffer({
  size: cpuBuffer.size, // Match the size of CPU buffer
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
  ],
});

const computePassDescriptor: GPUComputePassDescriptor = {};

async function cycle() {
  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();

  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, computeBindGroup);

  computePass.dispatchWorkgroups(64);
  computePass.end();

  //Copy data from buffer in GPU so that CPU can read
  commandEncoder.copyBufferToBuffer(
    cpuBuffer,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    cpuBuffer.size,
  );

  device.queue.submit([commandEncoder.finish()]);

  // Read results back to CPU
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    cpuBuffer.size, // Length
  );
  const copyArrayBuffer = stagingBuffer.getMappedRange(0, cpuBuffer.size);
  // To create JavaScript-owned copy
  const results = new Int32Array(copyArrayBuffer.slice(0));
  stagingBuffer.unmap();

  const activeCPUs = getCPUsFromResults(results, 4);
  console.log(activeCPUs);
}

let count = 0;
const intervalId = setInterval(() => {
  cycle();
  count++;

  if (count === 10) {
    clearInterval(intervalId); // Stops the loop
  }
}, 1000); // 1000ms = 1 second

// =================== Utils ===================
/**
 * Converts the flat Int32Array from the GPU back into an array of CPU objects.
 * Uses the STRIDE (20 i32s / 80 bytes) to find each organism's data.
 */
function getCPUsFromResults(results: Int32Array, count: number): CPU[] {
  const cpus: CPU[] = [];
  const STRIDE = 20; // 80 bytes / 4 bytes per i32

  for (let i = 0; i < count; i++) {
    const offset = i * STRIDE;

    cpus.push({
      // 1. Registers: vec4<i32> at offset 0 (Indices 0, 1, 2, 3)
      registers: Array.from(results.subarray(offset + 0, offset + 4)),

      // 2. IP: i32 at offset 4
      IP: results[offset + 4],

      // 3. SP: i32 at offset 5
      SP: results[offset + 5],

      // 4. Stack: array<i32, 10> starting at offset 6 (Indices 6 to 15)
      stack: Array.from(results.subarray(offset + 6, offset + 16)),

      // 5. Flags: i32 at offset 16
      // Note: Indices 17, 18, 19 are the padding bytes added by WGSL
      // to round the 68-byte struct up to 80 bytes (multiple of 16).
      flags: [results[offset + 16]],
    });
  }

  return cpus;
}
