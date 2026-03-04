export const SOUP_SIZE = 60000;
export const MAX_ORGANISMS = 2048;
export const FIELDS_PER_CELL = 9; // 32 bytes
export const FIELDS_PER_CPU = 24; // 96 bytes
export const HIST_SIZE_MAX = 200; // track genome sizes 1..200

export interface ICell {
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

export interface ICPU {
  registers: number[]; // [AX, BX, CX, DX]
  IP: number;
  SP: number;
  stack: number[]; // 10 slots
  flags: number[]; // 4 slots
  state: number; // 0 = Dead, 1 = Alive
  cell_index: number; // Points to the cell array
}

export const OP_NAMES: Record<number, string> = {
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
