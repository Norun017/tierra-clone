// ========= GLOBAL CONSTANTS ===========
const NOP0: i32 = 0x00;
const NOP1: i32 = 0x01;
const NOPS_SUM: i32 = 0x01; // Nop0 + Nop1 = 1 (Complementary sum)
const INST_SET_SIZE: i32 = 32;
const SEARCH_LIMIT: i32 = 4000;
const MAX_MAL_MULT: i32 = 3 ;
const MIN_CELL_SIZE: i32 = 10;
const MOV_THRESHOLD_RATIO: f32 = 0.8;

#include ./helpers.wgsl

// ========= STRUCT ===========
struct Uniform {
    SOUP_SIZE: i32,
}

struct GlobalStats {
    cells_alive: atomic<i32>,
    memory_used: atomic<i32>, // Total bytes currently 'mal'-ed
}

// 9 fields = 48 bytes
struct Cell {
    state: i32,          // 0 = Dead, 1 = Alive
    mem_start: i32,      // Mother's memory start
    mem_size: i32,       // Mother's memory size
    d_start: i32,        // Daughter's memory start 
    d_size: i32,         // Daughter's memory size
    mov_daught: i32,     // Counter for instructions copied to daughter
    active_cpus: i32,    // How many CPUs does this cell currently have?
    age: i32,      // Increments every cycle
    errors: i32,   // Increments on failed instructions
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
@group(0) @binding(2) var<storage, read_write> ownership: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> vCPUs: array<VCPU>;
@group(0) @binding(4) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(5) var<storage, read_write> stats: GlobalStats;


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let SOUP_SIZE = uniforms.SOUP_SIZE;
    let search_limit = 4000;
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
    var ip_modified = false;

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
            let target_addr = match_template(&cpu, 0, t_size, search_limit, SOUP_SIZE);
            if (target_addr != -1) { cpu.ip = target_addr; ip_modified = true; }
        }
        case 0x15: { // jmpb: backward
            let target_addr = match_template(&cpu, 2, t_size, search_limit, SOUP_SIZE);
            if (target_addr != -1) { cpu.ip = target_addr; ip_modified = true; }
        }
        case 0x16: { // call: push IP, then jmp outward
            // 2. Calculate the return address (IP + 1 for opcode + template size)
            let ret_addr = mo(cpu.ip + t_size + 1, SOUP_SIZE);
            
            // 3. Push the return address onto the stack
            push(&cpu, ret_addr);
            
            // 4. Search for the complementary template (search_type 0 = "outward")
            let target_addr = match_template(&cpu, 0, t_size, SEARCH_LIMIT, SOUP_SIZE);
            
            if (target_addr != -1) {
                // SUCCESS: Jump to the code found
                cpu.ip = target_addr;
                cpu.flags[0] = 0; // Clear error
            } else {
                // FAILURE: No match found. 
                // We still move the IP to the return address to skip the template
                cpu.ip = ret_addr;
                cpu.flags[0] = 1; // Set Error Flag
            }
            
            // We manually moved the IP, so tell main() not to do IP + 1
            ip_modified = true;
        }
        case 0x17: { // ret: pop IP
            cpu.ip = pop(&cpu);
            ip_modified = true; // Don't do normal IP increment
        }

        // --- MEMORY MOVEMENT ---
        case 0x18: { cpu.registers[3] = cpu.registers[2]; } // movDC: DX = CX
        case 0x19: { cpu.registers[1] = cpu.registers[0]; } // movBA: BX = AX
        case 0x1a: { // mov_iab: Copy instruction from [BX] to [AX]
            let src_addr = cpu.registers[1]; // BX
            let dest_addr = cpu.registers[0]; // AX

            // BIOLOGICAL CHECK: Is the destination owned by me OR my daughter?
            let target_owner = atomicLoad(&ownership[mo(dest_addr, uniforms.SOUP_SIZE)]);
            let my_owner_id = cpu.cell_index + 1;

            // Check: Do I own the destination? (Either my own body or my daughter's space)
            if (target_owner == my_owner_id) {
                let instruction = soup[mo(src_addr, uniforms.SOUP_SIZE)];
                soup[mo(dest_addr, uniforms.SOUP_SIZE)] = instruction;
                
                // As per Tierra C code: track how many instructions were moved to daughter
                if (dest_addr >= cell.d_start && dest_addr < (cell.d_start + cell.d_size)) {
                    cell.mov_daught += 1;
                }
            } else {
                cpu.flags[0] = 1; // Write Fault
            }
        }

        // --- TEMPLATE ADDRESSING ---
        case 0x1b, 0x1c, 0x1d: { // adro, adrb, adrf: Find template and put address in AX/CX
            // 2. Determine search direction
            // 0x1b (adro) -> 0 (outward), 0x1c (adrb) -> 2 (backward), 0x1d (adrf) -> 1 (forward)
            let dir = select(select(0, 2, opcode == 0x1c), 1, opcode == 0x1d);
            
            // 3. Search for the complementary template
            let target_addr = match_template(&cpu, dir, t_size, SEARCH_LIMIT, uniforms.SOUP_SIZE);
            
            if (target_addr != -1) {
                cpu.registers[0] = target_addr; // AX = found address
                cpu.registers[2] = t_size;      // CX = template length
                cpu.flags[0] = 0;               // Success
            } else {
                cpu.flags[0] = 1;               // Failure: Template not found
            }

            // 4. CRITICAL: Skip the template following the opcode
            // Otherwise, the CPU will try to execute the NOPs next cycle
            cpu.ip = mo(cpu.ip + t_size + 1, uniforms.SOUP_SIZE);
            ip_modified = true;
        }

        // --- BIOLOGICAL / SYSTEM ---
        case 0x1e: { 
            mal(&cpu, &cell); 
        }
        case 0x1f: { 
            divide(&cpu, &cell);
        }
        
        default: {}
    }

    // 5. Update health and Reaper check
    update_health(&cpu, &cell);
    reap_check(&cpu, &cell);

    // 6. Increment IP
    if (!ip_modified) {
        cpu.ip = mo(cpu.ip + 1, SOUP_SIZE);
    }

    // Save state back to global buffer
    vCPUs[cpu_index] = cpu;
    cells[cpu.cell_index] = cell; // Save cell back in case CPU modified it (like allocating daughter mem)
}

fn mal(cpu: ptr<function, VCPU>, cell: ptr<function, Cell>) {
    let SOUP_SIZE = uniforms.SOUP_SIZE;
    let requested_size = (*cpu).registers[2]; // CX holds the size requested
    let my_owner_id = (*cpu).cell_index + 1;

    // --- 1. SIZE & GESTATION VALIDATION ---
    
    // Rule: Non-Positive Requests
    if (requested_size <= 0) {
        (*cpu).flags[0] = 1; // Failure
        return;
    }

    // Rule: Absolute Upper Limit (SoupSize)
    if (requested_size >= SOUP_SIZE) {
        (*cpu).flags[0] = 1;
        return;
    }

    // Rule: Minimum Cell Size
    if (requested_size < MIN_CELL_SIZE) {
        (*cpu).flags[0] = 1;
        return;
    }

    // Rule: Proportional Upper Limit (MaxMalMult)
    // Mother cannot allocate a daughter > 3x her own size
    if (requested_size > (MAX_MAL_MULT * (*cell).mem_size)) {
        (*cpu).flags[0] = 1;
        return;
    }

    // Rule: Gestation Limit
    // If d_start is not -1, we already have a daughter being built.
    if ((*cell).d_start != -1) {
        (*cpu).flags[0] = 1;
        return;
    }

    // --- 2. SEARCH FOR SPACE ---
    var found_addr: i32 = -1;
    
    // Simple First-Fit Search
    for (var i: i32 = 0; i < SOUP_SIZE; i++) {
        let check_addr = mo((*cell).mem_start + i, SOUP_SIZE);
        
        var is_free = true;
        for (var j: i32 = 0; j < requested_size; j++) {
            if (get_owner(check_addr + j) != 0) {
                is_free = false;
                break;
            }
        }

        if (is_free) {
            // ATOMIC LOCK
            let compare_result = atomicCompareExchangeWeak(
                &ownership[check_addr], 
                0, 
                my_owner_id
            );

            if (compare_result.exchanged) {
                found_addr = check_addr;
                break;
            }
        }
    }

    // --- 3. ALLOCATION ---
    if (found_addr != -1) {
        for (var k: i32 = 1; k < requested_size; k++) {
            atomicStore(&ownership[mo(found_addr + k, SOUP_SIZE)], my_owner_id);
        }
        
        (*cpu).registers[0] = found_addr; // AX = Address of daughter
        (*cell).d_start = found_addr;     // Track daughter in cell
        (*cell).d_size = requested_size;
        (*cpu).flags[0] = 0;              // Success! Clear error flag

        atomicAdd(&stats.memory_used, requested_size); 
    } else {
        // Rule: If request is valid but no space found, 
        // normally Tierra invokes the Reaper. In our GPU setup, 
        // the Reaper is running in parallel and will eventually 
        // clear space. For now, we set the error flag.
        (*cpu).flags[0] = 1; 
    }
}

fn divide(cpu: ptr<function, VCPU>, cell: ptr<function, Cell>) {
    let SOUP_SIZE = uniforms.SOUP_SIZE;
    let MAX_CELLS = arrayLength(&cells);
    
    // --- 1. TIERRA VALIDATION ---
    // Threshold: Mother must have copied enough (e.g., 80% of daughter size)
    // In C: Thresh = (I32s) (ce->md.s * MovPropThrDiv)
    let threshold = i32(f32((*cell).d_size) * MOV_THRESHOLD_RATIO);
    
    if ((*cell).d_size < MIN_CELL_SIZE) { // MinCellSize check
        (*cpu).flags[0] = 1; 
        return; 
    }
    
    if ((*cell).mov_daught < threshold) {
        (*cpu).flags[0] = 1; // Error: Not enough genetic material copied!
        return;
    }

    // --- 2. FIND RESOURCES ---
    var d_idx: i32 = -1;
    for (var i: u32 = 0u; i < MAX_CELLS; i++) {
        if (cells[i].state == 0) { 
            d_idx = i32(i); 
            break; 
        }
    }

    // Find a CPU for the daughter
    var d_cpu_idx: i32 = -1;
    for (var j: u32 = 0u; j < arrayLength(&vCPUs); j++) {
        if (vCPUs[j].state == 0) { 
            d_cpu_idx = i32(j); 
            break; 
        }
    }

    // --- 3. THE SPLIT ---
    if (d_idx != -1 && d_cpu_idx != -1) {
        let new_owner_id = d_idx + 1;
        let d_start = (*cell).d_start;
        let d_size = (*cell).d_size;

        // A. Handover Ownership: Mother loses write-privileges
        // This is the "Eject" from the mother's control
        for (var k: i32 = 0; k < d_size; k++) {
            atomicStore(&ownership[mo(d_start + k, SOUP_SIZE)], new_owner_id);
        }

        // B. Initialize Daughter Cell
        cells[d_idx].state = 1;
        cells[d_idx].mem_start = d_start;
        cells[d_idx].mem_size = d_size;
        cells[d_idx].active_cpus = 1;
        cells[d_idx].d_start = -1; // New cell has no daughter yet
        cells[d_idx].mov_daught = 0;

        // C. Initialize Daughter CPU
        vCPUs[d_cpu_idx].state = 1;
        vCPUs[d_cpu_idx].cell_index = d_idx;
        vCPUs[d_cpu_idx].ip = d_start; // Starts at beginning of copied code
        vCPUs[d_cpu_idx].registers = (*cpu).registers; // Inherit mother's registers
        
        // D. Mother Reset (ce->md.s = ce->md.p = 0)
        (*cell).d_start = -1;
        (*cell).d_size = 0;
        (*cell).mov_daught = 0;
        (*cpu).flags[0] = 0; // Success

        atomicAdd(&stats.cells_alive, 1); // add cells alive to global stats
    } else {
        (*cpu).flags[0] = 1; // Resource exhaustion
    }
}

// ========= Reaper mech ============
fn reap_check(cpu: ptr<function, VCPU>, cell: ptr<function, Cell>) {
    let SOUP_SIZE = uniforms.SOUP_SIZE;
    let mem_used = f32(atomicLoad(&stats.memory_used));
    let fullness = mem_used / f32(SOUP_SIZE);

    // Only activate Reaper if soup is > 80% full
    if (fullness > 0.80) {
        // Calculate death score with a touch of randomness (using IP as a seed)
        let noise = i32(fract(sin(f32((*cpu).ip)) * 43758.5453) * 50.0);
        let death_score = (*cell).age + ((*cell).errors * 10) + noise;

        // The more full the soup, the lower the 'death_threshold' becomes
        // This creates "pressure"
        let death_threshold = 2000 - i32(fullness * 1000.0);

        if (death_score > death_threshold) {
            kill_cell(cpu, cell);
        }
    }
}

fn kill_cell(cpu: ptr<function, VCPU>, cell: ptr<function, Cell>) {
    let SOUP_SIZE = uniforms.SOUP_SIZE;

    // 1. Free mother's memory
    for (var i: i32 = 0; i < (*cell).mem_size; i++) {
        atomicStore(&ownership[mo((*cell).mem_start + i, SOUP_SIZE)], 0);
    }

    // 2. Free daughter's memory if gestation was in progress
    if ((*cell).d_start != -1) {
        for (var j: i32 = 0; j < (*cell).d_size; j++) {
            atomicStore(&ownership[mo((*cell).d_start + j, SOUP_SIZE)], 0);
        }
        atomicAdd(&stats.memory_used, -(*cell).d_size);
        (*cell).d_start = -1;
        (*cell).d_size = 0;
        (*cell).mov_daught = 0;
    }

    // 3. Decrement global stats
    atomicAdd(&stats.cells_alive, -1);
    atomicAdd(&stats.memory_used, -(*cell).mem_size);

    // 4. Set state to Dead
    (*cell).state = 0;
    (*cpu).state = 0;
}