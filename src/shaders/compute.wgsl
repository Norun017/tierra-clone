// ========= GLOBAL VAR ===========
const NOP0: i32 = 0x00;
const NOP1: i32 = 0x01;
const NOPS_SUM: i32 = 0x01; // Nop0 + Nop1 = 1 (Complementary sum)
const INST_SET_SIZE = 32;

#include ./helpers.wgsl

// ========= STRUCT ===========
struct Uniform {
    SOUP_SIZE: i32,
}

// 8 fields = 32 bytes
struct Cell {
    state: i32,          // 0 = Dead, 1 = Alive
    mem_start: i32,      // Mother's memory start
    mem_size: i32,       // Mother's memory size
    d_start: i32,        // Daughter's memory start (for phase 2)
    d_size: i32,         // Daughter's memory size (for phase 2)
    active_cpus: i32,    // How many CPUs does this cell currently have?
    pad: vec2<i32>,      // Alignment padding
}

// 24 fields = 96 bytes
struct VCPU {
    registers: array<i32, 4>,   // 0-3 ax, bx, cx, dx
    ip: i32,                    // 4 Instruction Pointer
    sp: i32,                    // 5 Stack Pointer
    stack: array<i32, 10>,      // 6-15 10-word circular stack
    flags: array<i32, 4>,       // 16-19 Bit field for E, S, Z flags
    state: i32,                 // 20 0 = Dead, 1 = Alive
    cell_index: i32,            // 21 "Foreign Key" to the Cell struct
    pad: vec2<i32>,             // 22-23 Alignment padding
}

@group(0) @binding(0) var<uniform> uniforms: Uniform;
@group(0) @binding(1) var<storage, read_write> soup: array<i32>; 
@group(0) @binding(2) var<storage, read_write> ownership: array<i32>;
@group(0) @binding(3) var<storage, read_write> vCPUs: array<VCPU>;
@group(0) @binding(4) var<storage, read_write> cells: array<Cell>;


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let SOUP_SIZE = uniforms.SOUP_SIZE;
    let cpu_index = id.x;
    // Bounds check to prevent out-of-bounds if workgroups > creatures
    if (cpu_index >= arrayLength(&vCPUs)) { return; }

    var cpu = vCPUs[cpu_index];

    // 1. Is this CPU alive?
    if (cpu.state == 0) { return; }

    // 2. Fetch the parent Cell
    let ip = cpu.ip; // Instruction Pointer
    var cell = cells[cpu.cell_index];

    // 3. Fetch instructions
    let opcode = soup[mo(ip, SOUP_SIZE)] % i32(INST_SET_SIZE); // Modulo to ensure the opcode is not null
    let t_size = get_template_size(ip, SOUP_SIZE);

    // 4. Decode & Execute (Example: incA)
    switch (opcode) {
        // --- NOPS ---
        case 0x00, 0x01: { /* nop0, nop1: Do nothing */ }
        
        // --- MATH & LOGIC ---
        case 0x02: { cpu.registers[2] ^= 1; } // not0: Flip bit 0 of CX
        case 0x03: { cpu.registers[2] <<= 1; } // shl: Shift CX left
        case 0x04: { cpu.registers[2] = 0; } // zero: CX = 0
        case 0x05: { // ifz: If CX != 0, skip the next instruction (and its template)
            if (cpu.registers[2] != 0) {
                let next_t_size = get_template_size(mo(cpu.ip + 1, SOUP_SIZE), SOUP_SIZE);
                cpu.ip = mo(cpu.ip + 1 + next_t_size, SOUP_SIZE);
            }
        }
        case 0x06: { cpu.registers[2] = cpu.registers[0] - cpu.registers[1]; } // subCAB: CX = AX - BX
        case 0x07: { cpu.registers[0] = cpu.registers[0] - cpu.registers[2]; } // subAAC: AX = AX - CX
        case 0x08: { cpu.registers[0] += 1; } // incA: AX++
        case 0x09: { cpu.registers[1] += 1; } // incB: BX++
        case 0x0a: { cpu.registers[2] -= 1; } // decC: CX--
        case 0x0b: { cpu.registers[2] += 1; } // incC: CX++

        // --- STACK ---
        case 0x0c: { push(&cpu, cpu.registers[0]); } // pushA
        case 0x0d: { push(&cpu, cpu.registers[1]); } // pushB
        case 0x0e: { push(&cpu, cpu.registers[2]); } // pushC
        case 0x0f: { push(&cpu, cpu.registers[3]); } // pushD
        case 0x10: { cpu.registers[0] = pop(&cpu); } // popA
        case 0x11: { cpu.registers[1] = pop(&cpu); } // popB
        case 0x12: { cpu.registers[2] = pop(&cpu); } // popC
        case 0x13: { cpu.registers[3] = pop(&cpu); } // popD

        // --- CONTROL FLOW ---
        case 0x14: { // jmpo: outward (bi-directional)
            let target_addr = match_template(&cpu, 0, t_size, 4000, SOUP_SIZE);
            if (target_addr != -1) { cpu.ip = target_addr; return; }
        }
        case 0x15: { // jmpb: backward
            let target_addr = match_template(&cpu, 2, t_size, 4000, SOUP_SIZE);
            if (target_addr != -1) { cpu.ip = target_addr; return; }
        }
        case 0x16: { // call: push IP, then jmp forward
            push(&cpu, cpu.ip);
            let target_addr = match_template(&cpu, 1, t_size, 4000, SOUP_SIZE);
            if (target_addr != -1) { cpu.ip = target_addr; return; }
        }
        case 0x17: { // ret: pop IP
            cpu.ip = pop(&cpu);
            return; // Don't do normal IP increment
        }

        // --- MEMORY MOVEMENT ---
        case 0x18: { cpu.registers[3] = cpu.registers[2]; } // movDC: DX = CX
        case 0x19: { cpu.registers[1] = cpu.registers[0]; } // movBA: BX = AX
        case 0x1a: { // movII (mov_iab): Copy instruction from soup[AX] to soup[BX]
            // Note: In Phase 2, we must check the 'ownership' buffer before writing!
            let src_inst = soup[mo(cpu.registers[0], SOUP_SIZE)];
            soup[mo(cpu.registers[1], SOUP_SIZE)] = src_inst;
        }

        // --- TEMPLATE ADDRESSING ---
        case 0x1b, 0x1c, 0x1d: { // adro, adrb, adrf: Find template and put address in AX/CX
            let dir = select(select(0, 2, opcode == 0x1c), 1, opcode == 0x1d); // 0=outward, 1=fwd, 2=bwd
            let target_addr = match_template(&cpu, dir, t_size, 4000, SOUP_SIZE);
            if (target_addr != -1) {
                cpu.registers[0] = target_addr; // AX gets target_addr address
                cpu.registers[2] = t_size; // CX gets template size
            }
        }

        // --- BIOLOGICAL / SYSTEM ---
        case 0x1e: { 
            // mal: Allocate memory for daughter cell
            // TODO Phase 2: Claim memory in Ownership buffer
        }
        case 0x1f: { 
            // divide: Create new organism
            // TODO Phase 2: Wake up a new VCPU
        }
        
        default: {}
    }

    // 5. Increment IP
    cpu.ip = mo(cpu.ip + 1, SOUP_SIZE);

    // Save state back to global buffer
    vCPUs[cpu_index] = cpu;
    cells[cpu.cell_index] = cell; // Save cell back in case CPU modified it (like allocating daughter mem)
}