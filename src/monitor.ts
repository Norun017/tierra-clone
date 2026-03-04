import {
  SOUP_SIZE,
  MAX_ORGANISMS,
  FIELDS_PER_CELL,
  HIST_SIZE_MAX,
  OP_NAMES,
} from "./constants";

// Canvas references — set once via initMonitor()
let timelineCanvas: HTMLCanvasElement;
let histogramCanvas: HTMLCanvasElement;

export function initMonitor(tc: HTMLCanvasElement, hc: HTMLCanvasElement) {
  timelineCanvas = tc;
  histogramCanvas = hc;
}

// ============== Population Timeline ===================

const TIMELINE_MAX_SAMPLES = 600;

interface TimelineSample {
  cycle: number;
  population: number;
  memUsed: number;
}

const timelineHistory: TimelineSample[] = [];

export function recordTimelineSample(
  cycle: number,
  population: number,
  memUsed: number,
) {
  timelineHistory.push({ cycle, population, memUsed });
  if (timelineHistory.length > TIMELINE_MAX_SAMPLES) timelineHistory.shift();
}

export function drawTimeline() {
  const ctx = timelineCanvas.getContext("2d");
  if (!ctx || timelineHistory.length < 2) return;

  const W = timelineCanvas.width;
  const H = timelineCanvas.height;
  const PAD = { top: 18, bottom: 22, left: 38, right: 10 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);

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

  // Memory-used line (orange)
  ctx.strokeStyle = "#c84";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = PAD.left + (i / (TIMELINE_MAX_SAMPLES - 1)) * cW;
    const y = PAD.top + cH - (timelineHistory[i].memUsed / SOUP_SIZE) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Population line (green)
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

  ctx.textAlign = "left";
  ctx.fillStyle = "#0f0";
  ctx.fillText("— population", PAD.left + 4, PAD.top - 4);
  ctx.fillStyle = "#c84";
  ctx.fillText("— memory", PAD.left + 110, PAD.top - 4);

  const last = timelineHistory[n - 1];
  ctx.fillStyle = "#444";
  ctx.textAlign = "right";
  ctx.fillText(`cycle ${last.cycle}`, W - PAD.right, H - 5);
}

// ============== Size Distribution Histogram ===================

const sizeCounts = new Int32Array(HIST_SIZE_MAX + 1);

export function tallySizes(cellRes: Int32Array) {
  sizeCounts.fill(0);
  for (let i = 0; i < MAX_ORGANISMS; i++) {
    const off = i * FIELDS_PER_CELL;
    if (cellRes[off] === 1) {
      const sz = cellRes[off + 2];
      if (sz > 0 && sz <= HIST_SIZE_MAX) sizeCounts[sz]++;
    }
  }
}

export function drawHistogram() {
  const ctx = histogramCanvas.getContext("2d");
  if (!ctx) return;

  const W = histogramCanvas.width;
  const H = histogramCanvas.height;
  const PAD = { top: 18, bottom: 18, left: 38, right: 10 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);

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
  if (maxCount === 0) return;

  const rangeStart = Math.max(1, minSize - 5);
  const rangeEnd = Math.min(HIST_SIZE_MAX, maxSize + 5);
  const range = rangeEnd - rangeStart + 1;
  const barW = Math.max(1, Math.floor(cW / range));

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

  ctx.fillStyle = "#555";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(range / 8));
  for (let s = rangeStart; s <= rangeEnd; s += step) {
    const x = PAD.left + (s - rangeStart) * barW + barW / 2;
    ctx.fillText(String(s), x, H - 4);
  }

  ctx.fillStyle = "#444";
  ctx.textAlign = "left";
  ctx.font = "10px monospace";
  ctx.fillText("genome size →", PAD.left + 4, PAD.top - 4);
}

// ============== Genotype Census + Genebank ===================

const CENSUS_INTERVAL = 50;
const GENEBANK_THRESHOLD = 10;

interface GenotypeRecord {
  hash: number;
  size: number;
  genome: Int32Array;
  firstSeen: number;
  peakPop: number;
  currentPop: number;
  inBank: boolean;
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

function runCensus(
  cellRes: Int32Array,
  soupRes: Int32Array,
  cycleCount: number,
) {
  const currentCounts = new Map<number, number>();
  const firstSeen = new Map<number, Int32Array>();

  for (let i = 0; i < MAX_ORGANISMS; i++) {
    const off = i * FIELDS_PER_CELL;
    if (cellRes[off] !== 1) continue;
    const memStart = cellRes[off + 1];
    const memSize = cellRes[off + 2];
    if (memSize <= 0 || memSize > HIST_SIZE_MAX) continue;

    const genome = extractGenome(soupRes, memStart, memSize);
    const hash = hashGenome(genome);
    currentCounts.set(hash, (currentCounts.get(hash) ?? 0) + 1);
    if (!firstSeen.has(hash)) firstSeen.set(hash, genome);
  }

  for (const rec of genotypeMap.values()) rec.currentPop = 0;

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
    .sort((a, b) => b.currentPop - a.currentPop);
  document.getElementById("stat-genotypes")!.innerText =
    `${alive.length} live / ${genotypeMap.size} total`;
  document.getElementById("stat-genebank")!.innerText = String(genebank.length);

  if (alive.length === 0) {
    document.getElementById("census-list")!.innerHTML = "";
    return;
  }

  const maxPop = alive[0].currentPop;
  const rows = alive.slice(0, 12).map((r) => {
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

export function downloadGenebank(cycleCount: number) {
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

// Returns true if the NEXT tickCensus call will run the census.
// Use this to decide whether to do an expensive soup readback before calling tickCensus.
export function willRunCensus(): boolean {
  return (censusFrameCounter + 1) % CENSUS_INTERVAL === 0;
}

// Called every render frame — runs census on its own interval.
// soupRes may be null on non-census frames (pass null to skip census this tick).
export function tickCensus(
  cellRes: Int32Array,
  soupRes: Int32Array | null,
  cycleCount: number,
) {
  censusFrameCounter++;
  if (censusFrameCounter % CENSUS_INTERVAL === 0 && soupRes !== null) {
    runCensus(cellRes, soupRes, cycleCount);
  }
}

// ============== Debug Report ==============

/* class SimulationReport {
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
} */
