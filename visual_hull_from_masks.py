# visual_hull_from_masks.py — строит плотную 3D-модель синей коробки по маскам (visual hull)
import os, glob
import numpy as np
import cv2
import open3d as o3d

try:
    from skimage.measure import marching_cubes
except Exception:
    from skimage.measure import marching_cubes_lewiner as marching_cubes  # для старых версий skimage

# =======================
# ПАРАМЕТРЫ
# =======================
FRAMES_DIR   = "colmap_images/images"     # кадры (undistorted) из COLMAP
MASKS_DIR    = "masks"             # бинарные маски коробки
MODEL_DIR    = "colmap_text"       # cameras.txt, images.txt, points3D.txt
OUTPUT_DIR   = "output"            # куда сохранить mesh

VOX_RES          = 640             # разрешение воксельной сетки (256–512)
PAD_SCALE        = 0.15            # запас границы вокруг AABB
USE_EVERY_K      = 1               # использовать каждую камеру
MIN_VIEWS        = 3               # минимальное число камер для проверки вокселя
MAX_COLOR_VIEWS  = 5               # сколько кадров брать для усреднения цвета
SMOOTH_ITERS     = 5               # сглаживание
SHOW_BACK_FACE   = True
BACKGROUND_RGB   = (0, 0, 0)       # чёрный фон

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================
def qvec2rotmat(qvec):
    q0, q1, q2, q3 = map(float, qvec)
    return np.array([
        [1 - 2*q2*q2 - 2*q3*q3,   2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,       1 - 2*q1*q1 - 2*q3*q3, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,       2*q2*q3 + 2*q0*q1,     1 - 2*q1*q1 - 2*q2*q2]
    ], dtype=np.float64)

def read_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for ln in f:
            if ln.startswith('#') or not ln.strip():
                continue
            toks = ln.split()
            cam_id = int(toks[0])
            model = toks[1]
            w, h = int(toks[2]), int(toks[3])
            p = list(map(float, toks[4:]))
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = p[0]; cx, cy = p[1], p[2]
            elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE_FULL"):
                fx, fy, cx, cy = p[:4]
            else:
                fx = fy = p[0]; cx, cy = p[1], p[2]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
            cams[cam_id] = {"K": K, "w": w, "h": h}
    return cams

def read_images_txt(path):
    imgs = []
    with open(path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        toks = lines[i].split()
        if toks[0].lstrip('-').isdigit():
            img_id = int(toks[0])
            qvec = np.array(list(map(float, toks[1:5])), dtype=np.float64)
            tvec = np.array(list(map(float, toks[5:8])), dtype=np.float64)
            cam_id = int(toks[8])
            name = toks[9]
            imgs.append({"id": img_id, "q": qvec, "t": tvec, "cam": cam_id, "name": name})
        i += 1
    return imgs

def read_points3d_txt(path):
    pts = []
    with open(path) as f:
        for ln in f:
            if ln.startswith('#') or not ln.strip():
                continue
            toks = ln.split()
            if len(toks) >= 4:
                pts.append(list(map(float, toks[1:4])))
    return np.array(pts, dtype=np.float64)

def project(K, R, t_col, X_world):
    Xc = (R @ X_world.T + t_col).T
    z = Xc[:, 2]
    uv = (K @ (Xc / z[:, None]).T).T
    return uv[:, :2], z

# =======================
# ЗАГРУЗКА МОДЕЛИ COLMAP
# =======================
print("📂 Загрузка модели COLMAP...")
cams  = read_cameras_txt(os.path.join(MODEL_DIR, "cameras.txt"))
imgs  = read_images_txt(os.path.join(MODEL_DIR, "images.txt"))
pts3d = read_points3d_txt(os.path.join(MODEL_DIR, "points3D.txt"))

if len(pts3d) == 0:
    raise RuntimeError("Файл points3D.txt пуст. Проверь COLMAP реконструкцию.")

mins = pts3d.min(0)
maxs = pts3d.max(0)
center = (mins + maxs) / 2
extent = (maxs - mins) * (1.0 + PAD_SCALE)
mins = center - extent / 2
maxs = center + extent / 2

# =======================
# ПОДГОТОВКА КАДРОВ И МАСОК
# =======================
mask_files = {os.path.basename(p): p for p in glob.glob(os.path.join(MASKS_DIR, "*.png")) + glob.glob(os.path.join(MASKS_DIR, "*.jpg"))}
img_files  = {os.path.basename(p): p for p in glob.glob(os.path.join(FRAMES_DIR, "*.png")) + glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))}

pairs = []
for i, info in enumerate(imgs):
    if i % USE_EVERY_K != 0:
        continue
    name = info["name"]
    if name not in mask_files or name not in img_files:
        continue
    cam_id = info["cam"]
    if cam_id not in cams:
        continue
    K = cams[cam_id]["K"]; w = cams[cam_id]["w"]; h = cams[cam_id]["h"]
    R = qvec2rotmat(info["q"]); t = info["t"].reshape(3, 1)
    pairs.append({
        "name": name, "K": K, "w": w, "h": h,
        "R": R, "t": t,
        "mask_path": mask_files[name],
        "img_path": img_files[name]
    })

for p in pairs:
    p["mask"] = cv2.imread(p["mask_path"], cv2.IMREAD_GRAYSCALE)
    p["img"] = cv2.cvtColor(cv2.imread(p["img_path"]), cv2.COLOR_BGR2RGB)

print(f"📸 Загружено кадров: {len(pairs)}")

# =======================
# ВОКСЕЛЬНАЯ РЕШЁТКА
# =======================
nx = ny = nz = VOX_RES
xs = np.linspace(mins[0], maxs[0], nx)
ys = np.linspace(mins[1], maxs[1], ny)
zs = np.linspace(mins[2], maxs[2], nz)
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]; dz = zs[1] - zs[0]
XX, YY = np.meshgrid(xs, ys, indexing='ij')
occupancy = np.ones((nx, ny, nz), dtype=bool)

subset = pairs if len(pairs) <= 16 else pairs[::max(1, len(pairs)//16)]
print("🧊 Space carving (строим визуальный корпус)...")

for iz, z in enumerate(zs):
    slice_pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)], axis=1)
    keep = np.ones(slice_pts.shape[0], dtype=bool)
    for p in subset:
        uv, depth = project(p["K"], p["R"], p["t"], slice_pts)
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        valid = (depth > 0) & (u >= 0) & (u < p["w"]) & (v >= 0) & (v < p["h"])
        inside = np.zeros_like(valid)
        inside[valid] = p["mask"][v[valid], u[valid]] > 127
        keep &= inside
        if not np.any(keep):
            break
    occupancy[:, :, iz] = keep.reshape(nx, ny)

# =======================
# МАРЧИНГ КЬЮБС (MESH)
# =======================
print("🧩 Marching Cubes...")
verts, faces, normals, _ = marching_cubes(occupancy.astype(np.uint8), level=0.5, spacing=(dx, dy, dz))
verts += mins

# === Постобработка меша: сглаживание, очистка, стабилизация ===
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

# Сгладить (метод Таубина даёт мягкие грани)
mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
mesh.compute_vertex_normals()

# Удалить шум и неправильные вершины
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()
mesh.compute_vertex_normals()

# === Опционально: выровнять форму до идеального прямоугольника ===
if True:  # можно поставить False, если хочешь оставить оригинал
    bbox = mesh.get_axis_aligned_bounding_box()
    ideal_box = o3d.geometry.TriangleMesh.create_box(
        width=bbox.get_extent()[0],
        height=bbox.get_extent()[1],
        depth=bbox.get_extent()[2]
    )
    ideal_box.translate(bbox.get_center() - ideal_box.get_center())
    mesh = ideal_box


mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

# Сглаживание
mesh = mesh.filter_smooth_simple(number_of_iterations=SMOOTH_ITERS)
mesh.compute_vertex_normals()

# =======================
# РАСКРАСКА
# =======================
print("🎨 Раскраска вершин...")
verts_np = np.asarray(mesh.vertices)
cols = np.zeros((verts_np.shape[0], 3), dtype=np.float32)

for i, X in enumerate(verts_np):
    colors = []
    for p in pairs[:MAX_COLOR_VIEWS]:
        Xc = p["R"] @ X + p["t"].squeeze()
        if Xc[2] <= 0:
            continue
        uv = p["K"] @ (Xc / Xc[2])
        u, v = int(round(uv[0])), int(round(uv[1]))
        if 0 <= u < p["w"] and 0 <= v < p["h"] and p["mask"][v, u] > 127:
            colors.append(p["img"][v, u] / 255.0)
    if colors:
        cols[i] = np.mean(colors, axis=0)
    else:
        cols[i] = [0.7, 0.7, 0.7]

mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
save_path = os.path.join(OUTPUT_DIR, "box_visual_hull.ply")
o3d.io.write_triangle_mesh(save_path, mesh)
print(f"✅ Модель сохранена: {save_path}")

# =======================
# ПРОСМОТР
# =======================
print("👀 Просмотр модели...")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Visual Hull — Blue Box")
vis.add_geometry(mesh)
opt = vis.get_render_option()
opt.mesh_show_back_face = SHOW_BACK_FACE
opt.background_color = np.array(BACKGROUND_RGB, dtype=float)/255.0
opt.light_on = True
opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
vis.run()
vis.destroy_window()
