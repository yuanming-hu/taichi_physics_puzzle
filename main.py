import numpy as np
import random
import taichi.math as tm

import taichi as ti

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
max_num_particles = 40000
n_grid = 128 * quality

dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 0.7e-4 / quality
substeps = int(2e-3 // dt)
frame_dt = dt * substeps
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
collider_radius = 0.1

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=float, shape=max_num_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=max_num_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float,
                    shape=max_num_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float,
                    shape=max_num_particles)  # deformation gradient

material = ti.field(dtype=int, shape=max_num_particles)  # material id
Jp = ti.field(dtype=float, shape=max_num_particles)  # plastic deformation
color = ti.field(dtype=ti.u32, shape=max_num_particles)  # color
particle_tag = ti.field(
    dtype=ti.u32, shape=max_num_particles)  # tag (and potentially other bits)
life = ti.field(dtype=float, shape=max_num_particles)  # particle life time

grid_v = ti.Vector.field(2, dtype=float,
                         shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())

max_num_segments = 32
num_segments = ti.field(dtype=ti.i32, shape=())
segments = ti.Vector.field(2, dtype=float, shape=max_num_segments * 2)

max_num_collectors = 32
num_collectors = ti.field(dtype=ti.i32, shape=())
collector_position = ti.Vector.field(2, dtype=float, shape=max_num_collectors)
collector_radius = ti.field(dtype=ti.f32, shape=max_num_collectors)
collector_tag = ti.field(dtype=ti.u32, shape=max_num_collectors)
collector_counter = ti.field(dtype=ti.f32, shape=max_num_collectors)

# Background collider
background_super_sampling = 6
window_res = background_super_sampling * n_grid
background_image = ti.Vector.field(3,
                                   dtype=ti.u8,
                                   shape=(window_res, window_res))
background_image_paint = ti.field(dtype=ti.u8, shape=(window_res, window_res))
background_intensity = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
background_normal = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))

background_collision = True

wrap_y = False


@ti.func
def wrap(I):
    if ti.static(wrap_y):
        j = I[1]
        if j < 0:
            j += n_grid
        if j >= n_grid:
            j -= n_grid
        return ti.Vector([I[0], j])
    else:
        return I


@ti.func
def distance_and_normal_to_line_segment(p, q, x):
    # distance of x to line segment (p, q)
    d = (q - p).normalized()
    l = (q - p).norm()
    f = (x - p).dot(d) / l
    f = max(min(f, 1), 0)
    closest = q * f + p * (1 - f)
    dist = x - closest
    normal = dist.normalized()
    return dist.norm(), normal


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in range(num_particles[None]
                   ):  # Particle state update and scatter to grid (P2G)
        base = ti.floor(x[p] * inv_dx - 0.5, int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) +
                dt * C[p]) @ F[p]  # deformation gradient update
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p])))
                )  # Hardening coefficient: snow gets harder when compressed
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[
                p] == 0:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[wrap(base +
                        offset)] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[wrap(base + offset)] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if ti.static(not wrap_y):
                if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
                if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

            grid_pos = ti.Vector([i, j]) * dx

            if ti.static(background_collision):
                normal = background_normal[i, j]
                if normal[0] != 0 or normal[1] != 0:
                    grid_v[i, j] = grid_v[i, j] - normal * max(
                        normal.dot(grid_v[i, j]), 0)

            for k in range(num_segments[None]):
                p, q = segments[k * 2], segments[k * 2 + 1]
                d, normal = distance_and_normal_to_line_segment(p, q, grid_pos)
                if d < 3 * dx:
                    grid_v[i, j] -= normal * min(normal.dot(grid_v[i, j]), 0)

    for p in range(num_particles[None]):  # grid to particle (G2P)
        base = ti.floor(x[p] * inv_dx - 0.5, int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[wrap(base + ti.Vector([i, j]))]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection

        x[p].y -= ti.floor(x[p].y)
        # v[p].y *= 0.9999


vec2 = ti.types.vector(2, ti.f32)


@ti.kernel
def seed_particles(pos_x: ti.f32, pos_y: ti.f32, radius: ti.f32, shape: ti.i32,
                   material_: ti.i32, count: ti.i32, init_v: vec2,
                   init_color: ti.u32, tag: ti.u32):
    actual_count = count
    if num_particles[None] + count > max_num_particles:
        actual_count = max_num_particles - num_particles[None]
        print("Particle number overflows. Seeding {} particles only.".format(
            actual_count))
    for c in range(actual_count):
        i = c + num_particles[None]
        x[i] = [(ti.random() * 2 - 1) * radius + pos_x,
                (ti.random() * 2 - 1) * radius + pos_y]
        material[i] = material_  # 0: fluid 1: jelly 2: snow
        v[i] = init_v  #[0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)
        life[i] = 0.35
        color[i] = init_color
        particle_tag[i] = tag

    num_particles[None] += actual_count


@ti.kernel
def garbage_collect(dt: ti.f32):
    remaining_particles = 0
    # TODO: this could be slow. Accelerate.
    ti.loop_config(serialize=True)
    for i in range(num_particles[None]):
        life[i] = life[i] - dt

        survive = life[i] > 0

        for k in range(num_collectors[None]):
            p = collector_position[k]
            r = collector_radius[k]
            if p[0] - r < x[i][0] < p[0] + r and p[1] - r < x[i][1] < p[1] + r:
                survive = False
                if (particle_tag[i] & collector_tag[k]) != 0:
                    collector_counter[k] += 1

        if survive:
            x[remaining_particles] = x[i]
            v[remaining_particles] = v[i]
            F[remaining_particles] = F[i]
            Jp[remaining_particles] = Jp[i]
            material[remaining_particles] = material[i]
            C[remaining_particles] = C[i]
            color[remaining_particles] = color[i]
            life[remaining_particles] = life[i]
            particle_tag[remaining_particles] = particle_tag[i]
            remaining_particles += 1
    num_particles[None] = remaining_particles

    for k in range(num_collectors[None]):
        collector_counter[k] *= ti.exp(-dt * 2)


@ti.kernel
def add_segments(x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32):
    i = num_segments[None]

    segments[i * 2] = [x0, y0]
    segments[i * 2 + 1] = [x1, y1]

    num_segments[None] += 1


def make_rgb_int(r, g, b):

    def convert(x):
        return max(min(int(x), 255), 0)

    return convert(r) * 65536 + convert(g) * 256 + convert(b)


brush_radius = 12

background_color = [235, 252, 227]
background_image.fill(background_color)


@ti.kernel
def draw_on_background(x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32,
                       draw: ti.i32):
    x_begin = min(x0, x1)
    x_end = max(x0, x1)
    y_begin = min(y0, y1)
    y_end = max(y0, y1)
    ix_begin = int(x_begin * window_res + 0.5)
    ix_end = int(x_end * window_res + 0.5)
    iy_begin = int(y_begin * window_res + 0.5)
    iy_end = int(y_end * window_res + 0.5)

    for i in range(max(0, ix_begin - brush_radius),
                   min(window_res, ix_end + brush_radius)):
        for j in range(max(0, iy_begin - brush_radius),
                       min(window_res, iy_end + brush_radius)):
            p = tm.vec2((i + 0.5) / window_res, (j + 0.5) / window_res)
            dist, _ = distance_and_normal_to_line_segment(
                tm.vec2(x0, y0), tm.vec2(x1, y1), p)
            if dist < brush_radius / window_res:
                if draw:
                    if dist < brush_radius / window_res * 0.8:
                        background_image_paint[i, j] = ti.u8(255)
                    background_image[i,
                                     j] = [ti.u8(124),
                                           ti.u8(134),
                                           ti.u8(255)]
                else:
                    background_image_paint[i, j] = ti.u8(0)
                    background_image[i, j] = ti.Vector(background_color).cast(
                        ti.u8)


@ti.kernel
def recompute_background_normals():
    for i, j in ti.ndrange(n_grid, n_grid):
        sum = 0
        for p, q in ti.ndrange(background_super_sampling,
                               background_super_sampling):
            sum += background_image_paint[i * background_super_sampling + p,
                                          j * background_super_sampling + q]
        background_intensity[i, j] = sum / (background_super_sampling *
                                            background_super_sampling / 255)

    for i, j in ti.ndrange((1, n_grid - 1), (1, n_grid - 1)):
        # Central difference to compute gradients
        f = ti.static(background_intensity)
        grad_x = (f[i + 1, j + 1] + 2 * f[i + 1, j] + f[i + 1, j - 1]) - (
            f[i - 1, j + 1] + 2 * f[i - 1, j] + f[i - 1, j - 1])
        grad_y = (f[i + 1, j + 1] + 2 * f[i, j + 1] + f[i - 1, j + 1]) - (
            f[i + 1, j - 1] + 2 * f[i, j - 1] + f[i - 1, j - 1])
        normal = tm.vec2(grad_x, grad_y)
        if normal.norm() > 0.1:
            normal = normal.normalized()
        else:
            normal = tm.vec2([0, 0])
        background_normal[i, j] = normal


print(
    "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse bottons to attract/repel. Press R to reset."
)
gui = ti.GUI("Taichi Physics Puzzle Game",
             res=window_res,
             background_color=0x112F41)
gravity[None] = [0, -1]
collider_center = np.array([0.5, 0.5])

mouse = gui.get_cursor_pos()

# gui2 = ti.GUI("debug", res=n_grid)


def add_collector(pos, r, tag):
    i = num_collectors[None]
    collector_position[i] = pos
    collector_radius[i] = r
    collector_tag[i] = tag
    num_collectors[None] += 1


colors = [
    make_rgb_int(92, 193, 255),
    make_rgb_int(252, 41, 255),
    make_rgb_int(115, 200, 46)
]
collector_types = [1, 1, 0, 0, 2, 2]
collector_color = []
for i in range(len(collector_types)):
    collector_color.append(colors[collector_types[i]])
    r = 0.5 / len(collector_types)
    add_collector([(2 * i + 1) * r, r], r * 0.9, 1 << collector_types[i])

for f in range(20000):
    # gui2.set_image(background_normal)
    # gui2.show()
    if f % 3 == 0:
        for i in range(3):
            seed_particles(0.2 + i * 0.3,
                           0.95,
                           0.01,
                           0,
                           material_=i,
                           count=200,
                           init_v=vec2([0, -10]),
                           init_color=colors[i],
                           tag=2**i)

    gui.set_image(background_image)

    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r': reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
    # if gui.event is not None: gravity[None] = [0, 0]  # if had any event
    if gui.is_pressed(ti.GUI.LEFT, 'a'): gravity[None][0] = -1
    if gui.is_pressed(ti.GUI.RIGHT, 'd'): gravity[None][0] = 1
    if gui.is_pressed(ti.GUI.UP, 'w'): gravity[None][1] = 1
    if gui.is_pressed(ti.GUI.DOWN, 's'): gravity[None][1] = -1
    last_mouse = mouse
    mouse = gui.get_cursor_pos()

    for i in range(num_segments[None]):
        gui.line(segments[i * 2], segments[i * 2 + 1], radius=8)

    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=brush_radius)

    if not gui.is_pressed(ti.GUI.LMB) and not gui.is_pressed(ti.GUI.RMB):
        last_mouse = mouse

    if gui.is_pressed(ti.GUI.LMB):
        draw_on_background(last_mouse[0], last_mouse[1], mouse[0], mouse[1], 1)
        background_intensity.fill(0)
        recompute_background_normals()

    if gui.is_pressed(ti.GUI.RMB):
        draw_on_background(last_mouse[0], last_mouse[1], mouse[0], mouse[1], 0)
        background_intensity.fill(0)
        recompute_background_normals()

    mx, my = collider_center
    for s in range(substeps):
        substep()
    garbage_collect(substeps * dt)

    for i in range(num_collectors[None]):
        pos = collector_position[i]
        r = collector_radius[i]
        counter = collector_counter[i]
        ratio = max(0.1, min(counter / 2000, 1))

        color1, color2 = 0xdddddd, collector_color[i]

        y0 = pos[1] - r
        y2 = pos[1] + r
        y1 = ratio * y2 + (1 - ratio) * y0

        def draw_rect(y_begin, y_end, color):
            a = (pos[0] - r, y_begin)
            b = (pos[0] + r, y_begin)
            c = (pos[0] - r, y_end)
            d = (pos[0] + r, y_end)

            gui.triangle(a, b, c, color)
            gui.triangle(c, b, d, color)

        draw_rect(y0, y1, color2)
        draw_rect(y1, y2, color1)

        if ratio >= 0.95:
            gui.text(pos=(pos[0] - 0.04, pos[1] + 0.01),
                     content="Filled!",
                     color=0xFFFFFF,
                     font_size=25)

    particles_to_draw = num_particles[None]
    gui.circles(x.to_numpy()[:particles_to_draw],
                radius=2,
                color=color.to_numpy()[:particles_to_draw])

    # gui.show(f'output/{f:06d}.png')
    gui.show()
