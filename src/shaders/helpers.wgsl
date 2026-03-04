const MAX_TEMP_SIZE = 20;
// ========= Helper Functions ===========
// WGSL Helper for Circular Memory Addressing
fn mo(addr: i32, size: i32) -> i32 {
    // To prevent negative location
    return (addr % size + size) % size;
}

// Stack Operations (Push / Pop)
fn push(cpu: ptr<function, VCPU>, val: i32) {
    let sp = (*cpu).sp;
    (*cpu).stack[sp] = val;
    (*cpu).sp = (sp + 1) % 10; // Circular stack of 10
}

fn pop(cpu: ptr<function, VCPU>) -> i32 {
    let sp = ((*cpu).sp - 1 + 10) % 10; // Go back 1 circularly
    (*cpu).sp = sp;
    return (*cpu).stack[sp];
}

// Helper to check ownership (Read)
fn get_owner(addr: i32) -> i32 {
    return atomicLoad(&ownership[mo(addr, uniforms.SOUP_SIZE)]);
}

// Update S (sign) and Z (zero) flags after a math/logic operation.
// Mirrors DoFlags() in the original instruct.c.
fn do_flags(cpu: ptr<function, VCPU>, result: i32) {
    (*cpu).flags[1] = select(0, 1, result < 0);  // S: sign
    (*cpu).flags[2] = select(0, 1, result == 0); // Z: zero
}

// Apply range constraint for C/D registers: if |val| > limit, return 0.
// Mirrors the DoMods() dran branch in the original instruct.c.
// A/B registers use mo() instead (address modulus).
fn apply_range(val: i32, limit: i32) -> i32 {
    if (val > limit || val < -limit) { return 0; }
    return val;
}

fn update_health(cpu: ptr<function, VCPU>, cell: ptr<function, Cell>) {
    // Increment age
    (*cell).age += 1;

    // Increment errors if the CPU flag is set
    if ((*cpu).flags[0] != 0) {
        (*cell).errors += 1;
    }
}

// Template Operations
// get template size following the start_ip instructions
fn get_template_size(start_ip: i32, soup_size: i32) -> i32 {
    var size = 0;
    for (var i = 1; i < MAX_TEMP_SIZE; i++) { // Max template size to prevent infinite loops
        let inst = soup[mo(start_ip + i, soup_size)];
        if (!is_nop(inst)) { break; }
        size++;
    }
    return size;
}

// Check if an instruction is a NOP
fn is_nop(inst: i32) -> bool {
    return inst == NOP0 || inst == NOP1;
}

// Check if two NOPs are complementary
fn is_complementary(inst1: i32, inst2: i32) -> bool {
    return (inst1+inst2) == NOPS_SUM;
}

// Match Template
fn match_template(
    cpu: ptr<function, VCPU>, 
    dir: i32,        // 0=outward, 1=forward, 2=backward
    template_size: i32, 
    search_limit: i32,
    soup_size: i32
) -> i32 {
    let start_ip = (*cpu).ip;
    let source_template_origin = mo(start_ip + 1, soup_size);
    
    // Limits check
    if (template_size < 1 || template_size > soup_size) {
        return -1;
    }

    // Directions to search
    var search_f = (dir == 0 || dir == 1);
    var search_b = (dir == 0 || dir == 2);

    // Initial search offsets
    var f_ptr = mo(start_ip + template_size + 1, soup_size);
    var b_ptr = mo(start_ip - template_size - 1, soup_size);

    for (var l: i32 = 1; l <= search_limit; l++) {
        
        // 1. Forward Search Match
        if (search_f) {
            var f_match = true;
            for (var i: i32 = 0; i < template_size; i++) {
                let source_inst = soup[mo(source_template_origin + i, soup_size)];
                let target_inst = soup[mo(f_ptr + i, soup_size)];
                
                if (!is_complementary(source_inst, target_inst)) {
                    f_match = false;
                    break;
                }
            }
            if (f_match) {
                return mo(f_ptr + template_size, soup_size); // Success! [9]
            }
        }

        // 2. Backward Search Match
        if (search_b) {
            var b_match = true;
            for (var i: i32 = 0; i < template_size; i++) {
                let source_inst = soup[mo(source_template_origin + i, soup_size)];
                let target_inst = soup[mo(b_ptr + i, soup_size)];
                
                if (!is_complementary(source_inst, target_inst)) {
                    b_match = false;
                    break;
                }
            }
            if (b_match) {
                return mo(b_ptr + template_size, soup_size); // Success! [9]
            }
        }

        // 3. Increment pointers for next step
        if (search_f) { f_ptr = mo(f_ptr + 1, soup_size); }
        if (search_b) { b_ptr = mo(b_ptr - 1, soup_size); }
    }

    return -1; // Template not found
}

// ========= RNG / Mutation Helpers ===========

// Avalanche hash — fast, good distribution for shader use
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 17u;
    v = v * 0xbf324c81u;
    v ^= v >> 11u;
    v = v * 0x68b665e5u;
    v ^= v >> 16u;
    return v;
}

// Maps a hash uniformly to [0.0, 1.0)
fn rand_f32(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967296.0;
}

// Applies bit-flip and/or replacement mutation to one instruction (0–31).
// base_seed + 0: bit-flip gate, +1: which bit, +2: replace gate, +3: new opcode
fn mutate_instruction(inst: i32, base_seed: u32, bit_rate: f32, rep_rate: f32) -> i32 {
    var result = inst;
    if (rand_f32(base_seed) < bit_rate) {
        let bit = hash_u32(base_seed + 1u) % 5u;
        result = (result ^ i32(1u << bit)) & 31;
    }
    if (rand_f32(base_seed + 2u) < rep_rate) {
        result = i32(hash_u32(base_seed + 3u) % 32u);
    }
    return result;
}