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