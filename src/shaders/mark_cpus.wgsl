// Pre-pass compute shader: runs just before the render pass each frame.
// Each alive CPU thread writes its cell_index+1 into cpu_markers at its IP address.
// The render shader reads cpu_markers in O(1) per pixel instead of iterating all CPUs.
// cpuMarkerBuffer is cleared via commandEncoder.clearBuffer() before this dispatch.

struct Uniform {
    SOUP_SIZE:       i32,
    debris_mode:     i32,
    cycle:           u32,
    _pad:            u32,
    cosmic_bit_rate: f32,
    cosmic_rep_rate: f32,
    copy_bit_rate:   f32,
    copy_rep_rate:   f32,
    canvas_width:    u32,
    canvas_height:   u32,
};

@group(0) @binding(0) var<uniform>            uniforms    : Uniform;
@group(0) @binding(1) var<storage, read>      cpus        : array<i32>;
@group(0) @binding(2) var<storage, read_write> cpu_markers: array<i32>;

const FIELDS_PER_CPU: u32 = 24u;
const MAX_ORGANISMS:  u32 = 2048u;

@compute @workgroup_size(64)
fn mark_cpus(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    if (ci >= MAX_ORGANISMS) { return; }

    let base = ci * FIELDS_PER_CPU;
    if (cpus[base + 20u] == 0) { return; }    // dead CPU

    let ip = u32(cpus[base + 4u]);
    if (ip >= u32(uniforms.SOUP_SIZE)) { return; }

    // Store cell_index+1 so 0 can mean "no CPU here".
    // Last write wins on rare collisions — acceptable for visualization.
    cpu_markers[ip] = cpus[base + 21u] + 1;
}
