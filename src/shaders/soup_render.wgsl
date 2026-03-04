// Render pipeline: maps soup addresses to colored pixels on the monitor canvas.
// Canvas is 600×600 = 360,000 pixels; SOUP_SIZE = 60,000 → 6 pixels per address.
// Reads soupBuffer and ownershipBuffer directly — no CPU readback needed for display.

struct Uniform {
    SOUP_SIZE:       i32,
    debris_mode:     i32,
    cycle:           u32,
    _pad:            u32,
    cosmic_bit_rate: f32,
    cosmic_rep_rate: f32,
    copy_bit_rate:   f32,
    copy_rep_rate:   f32,
    canvas_width:    u32,  // byte 32
    canvas_height:   u32,  // byte 36
};

@group(0) @binding(0) var<uniform>       uniforms     : Uniform;
@group(0) @binding(1) var<storage, read> soup         : array<i32>;
@group(0) @binding(2) var<storage, read> ownership    : array<i32>;
@group(0) @binding(3) var<storage, read> cpu_markers  : array<i32>;

// Full-screen quad: two triangles covering NDC [-1,1]×[-1,1].
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
        vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
    );
    return vec4f(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    // builtin(position) gives pixel-center coords: (0.5,0.5) = top-left pixel.
    // Canvas is 800×600; SOUP_SIZE = 60000; pixels_per_addr = 6.
    let pixels_per_addr = (uniforms.canvas_width * uniforms.canvas_height) / u32(uniforms.SOUP_SIZE);
    let pixel_idx = u32(pos.y) * uniforms.canvas_width + u32(pos.x);
    let addr = i32(pixel_idx / pixels_per_addr);

    if (addr >= uniforms.SOUP_SIZE) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let owner       = ownership[addr];
    let instruction = soup[addr];

    var r = 0.0;
    var g = 0.0;
    var b = 0.0;

    if (owner > 0) {
        r = f32((owner * 53) % 86)  / 255.0;
        g = f32((owner * 97) % 126) / 255.0;
        b = f32((owner * 151) % 256)/ 255.0;

        if (instruction == 0) {
            r *= 0.2;  g *= 0.2;  b *= 0.2;
        } else {
            r = min(1.0, r + 50.0 / 255.0);
            g = min(1.0, g + 50.0 / 255.0);
            b = min(1.0, b + 50.0 / 255.0);
        }
    } else if (instruction != 0) {
        r = 40.0 / 255.0;
        g = 40.0 / 255.0;
        b = 40.0 / 255.0;
    }

    // ── CPU triangle indicator ──────────────────────────────────────────────
    // Triangle shape (points DOWN toward the CPU's instruction pointer):
    //   █████  ← base, 5px wide  (2 display rows above the CPU's address)
    //    ███   ← mid,  3px wide  (1 display row above)
    //     █    ← tip,  1px wide  (at the exact CPU address)
    //
    // addrs_per_row = SOUP_SIZE / canvas_height = 60000 / 600 = 100.
    // subpx: signed sub-pixel offset from the center of the current address strip.
    let addrs_per_row = uniforms.SOUP_SIZE / i32(uniforms.canvas_height);
    let subpx = i32(pixel_idx % pixels_per_addr) - i32(pixels_per_addr / 2u);

    // For each triangle level, we look up whether any CPU sits in the address
    // that is 0, 1, or 2 display rows below the current pixel's row.
    var tri_cell = 0;

    // Base (current pixel is 2 rows above some CPU)
    let base_addr = addr + 2 * addrs_per_row;
    if (abs(subpx) <= 2 && base_addr < uniforms.SOUP_SIZE) {
        let m = cpu_markers[base_addr];
        if (m > 0) { tri_cell = m; }
    }
    // Middle (current pixel is 1 row above some CPU) — overrides base
    let mid_addr = addr + addrs_per_row;
    if (abs(subpx) <= 1 && mid_addr < uniforms.SOUP_SIZE) {
        let m = cpu_markers[mid_addr];
        if (m > 0) { tri_cell = m; }
    }
    // Tip (current pixel IS the CPU's address) — overrides middle
    if (subpx == 0) {
        let m = cpu_markers[addr];
        if (m > 0) { tri_cell = m; }
    }

    if (tri_cell > 0) {
        // Match the organism's color scheme but push it to near-white brightness.
        r = min(1.0, f32((tri_cell * 53) % 86)  / 255.0 + 0.7);
        g = min(1.0, f32((tri_cell * 97) % 126) / 255.0 + 0.7);
        b = min(1.0, f32((tri_cell * 151) % 256)/ 255.0 + 0.7);
    }

    return vec4f(r, g, b, 1.0);
}
