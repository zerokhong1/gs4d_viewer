# Phương pháp được hỗ trợ và kết quả kiểm tra

Viewer chỉ đọc một định dạng `.npz` chung ([FORMAT.md](FORMAT.md)); mỗi phương pháp 4DGS có một converter ghi định dạng đó.
Bảng dưới là tóm tắt; phần "Chi tiết" ghi lệnh, số đo và ảnh của từng phương pháp.

- PSNR đo bằng renderer gsplat của viewer trên ảnh test / ảnh giữ lại, ở độ phân giải lúc train (đo ở độ phân giải khác thì
  PSNR tụt vài dB), bằng `tests/check_real.py`, trừ khi ghi khác.
- "Output thật": model đã train (tác giả công bố hoặc tự train) chạy qua converter. "Tổng hợp": chỉ có test trên dữ liệu giả
  đúng layout file thật (`tests/test_<tên>.py`), chưa có model thật.
- Dạng output: **A** Gaussian theo từng frame · **B** Gaussian 4D, cắt theo thời gian · **C** hàm chuyển động tham số ·
  **D** mạng biến dạng, chạy lại trên CPU.

| Dạng | Phương pháp (converter) | Output thật đã kiểm | PSNR | Giới hạn |
|---|---|---|---|---|
| A | Thư mục ply theo frame, kể cả `gaussian_pertimestamp/` của 4DGaussians (`convert_per_frame.py`) | ply xuất từ 4DGaussians `bouncingballs`, `slice-banana` | khớp converter 4DGaussians | số Gaussian đổi theo frame thì không nội suy (`interpolate=False`) |
| A | Dynamic 3D Gaussians `params.npz` (`convert_per_frame.py`) | `basketball` (375.870 Gaussian × 150 frame) | chưa có ảnh gốc | trục up là `-y`; ~14 % màu gốc ngoài [0,1] bị cắt; renderer client chưa đối chiếu với `visualize.py` |
| B | Native 4DGS, Yang et al. (`convert_native4d.py`, `--sh`) | tự train `bouncingballs` (400 px); HyperNeRF `slice-banana` | `bouncingballs` 32,9 dB với `--sh` (repo 33,4); `slice-banana` 26,31 dB (repo 26,30) | không `--sh` thì chỉ SH bậc 0, màu loang (29,0 dB); `--sh` làm file lớn (`--compact` nén còn khoảng 1/3) |
| B | 4D-Rotor GS, SIGGRAPH 2024 (`convert_4drotor.py`) | chưa có | — | chỉ test tổng hợp; màu là SH bậc 3 không đổi theo thời gian (ghi vào `sh`); khoảng thời gian mặc định [0, 1] như D-NeRF, chưa kiểm với camera và dữ liệu thật của Nerfstudio |
| C | Spacetime Gaussians, bản lite (`convert_spacetime.py`) | `n3d_sear_steak` (tác giả công bố); Technicolor `Birthday` (không có ảnh gốc) | 34,37 dB; renderer của repo 34,48 | bản đầy đủ có `rgbdecoder`: converter dừng, `--base-color-only` chuyển bằng màu cơ sở; chưa thử model đầy đủ |
| C | Shape of Motion (`convert_shape_of_motion.py`) | `paper-windmill` (tác giả công bố) | 25,86 dB tại keyframe; `--stride 1` khớp renderer của repo tới 0,000 dB | `--stride` thưa thì giữa hai keyframe thấp hơn (24,67 dB ở stride 8) |
| C/D | MoSca, CVPR 2025 (`convert_mosca.py`) | chưa có (tiền xử lý quá nặng) | — | chỉ test tổng hợp; chỉ Gaussian ĐỘNG (`photometric_d_model_*.pth`), chưa đọc nền tĩnh (`photometric_s_model_*.pth`); `max_sph_order > 0` dừng; cách dùng scale của renderer bên thứ ba chưa đọc |
| D | E-D3DGS, ECCV 2024 (`convert_ed3dgs.py`) | chưa có (môi trường py3.7/torch 1.13 chưa dựng) | — | chỉ test tổng hợp; cần `cfg_args` (hoặc `--set`); thời gian N3V = frame/300, Technicolor = frame/50, HyperNeRF chưa đọc `hyper_loader.py`; giấy phép chỉ cho nghiên cứu |
| D | 4DGaussians, hustvl (`convert_4dgaussians.py`) | tự train `bouncingballs`; HyperNeRF `slice-banana` | 40,8 dB (repo 39,7); `slice-banana` 27,13 dB, bằng repo | dừng với `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False` |
| D | Deformable-3DGS (`convert_deformable3dgs.py`) | tự train: biến thể D-NeRF, mặc định, `--is_6dof`; HyperNeRF `slice-banana` | 40,3 / 39,62 / 40,34 dB; `slice-banana` 24,72 dB | bỏ qua méo ống kính như repo |
| D | SC-GS (`convert_scgs.py`) | tự train `bouncingballs` (thường, có motion mask, có đầu màu); `slice-banana` | 40,3 dB; mask 40,14 dB; đầu màu 37,83 dB; `slice-banana` 26,69 dB | `--use_hash` và `--skinning` chỉ test tổng hợp (repo tự train không hội tụ / sập); đầu màu hash hoặc theo hướng nhìn thì dừng |

Mọi converter cần torch trừ `convert_per_frame.py` và `convert_spacetime.py` (bản CPU là đủ: torch chỉ để mở checkpoint hoặc
chạy lại một MLP nhỏ). `python -m gs4d.open <output>` nhận ra tất cả các loại trên (xem [README](../README.md)).

## Chi tiết từng phương pháp

### A — Thư mục ply theo frame (`convert_per_frame.py`)

```bash
python convert_per_frame.py <thư_mục_ply> -o scene.npz --fps 30 [--stride 2] [--sh] [--compact]
```

- Đầu vào: thư mục `.ply` chuẩn 3DGS, mỗi frame một file, xếp theo thứ tự tự nhiên (`time_2` trước `time_10`): output từng frame của
  model feed-forward, hoặc `gaussian_pertimestamp/time_00000.ply ...` do `export_perframe_3DGS.py` của 4DGaussians xuất ra.
- `--sh` giữ `f_rest_*` (màu theo hướng nhìn, chỉ renderer gsplat dùng); tốn thêm N × K × 12 × (bậc+1)² byte, tức 192 byte mỗi Gaussian
  mỗi keyframe với SH bậc 3.
- Số Gaussian đổi giữa các frame: đệm Gaussian ẩn, viewer nhảy theo keyframe gần nhất, không nội suy.
- Kiểm: ply xuất từ 4DGaussians `bouncingballs` và `slice-banana` cho đúng file của `convert_4dgaussians.py` (xem mục 4DGaussians).

### A — Dynamic 3D Gaussians (`convert_per_frame.py`, `params.npz`)

```bash
python convert_per_frame.py params.npz -o scene.npz --stride 5 --fps 30 --fg-only
python viewer.py scene.npz --up -y
```

- Output thật `basketball` (375.870 Gaussian × 150 frame gốc): `--fg-only` cho 112.592 Gaussian × 30 keyframe, 163 MB; bỏ `--fg-only`:
  375.870 Gaussian, 543 MB, nhưng cảnh bị nền và floater che. Trục up là `-y`; ảnh chụp `shots/wp2/`.
- ~14 % màu gốc nằm ngoài [0,1] và bị cắt; renderer client chưa được đối chiếu với `visualize.py` gốc.

### B — Native 4DGS, Yang et al. (`convert_native4d.py`)

```bash
python convert_native4d.py chkpnt_best.pth -o scene.npz --frames 60 --t0 0 --t1 1 [--sh] [--compact]
```

- `--t0/--t1` là `time_duration` trong config lúc train (cảnh D-NeRF: [0, 1]); không truyền thì script đoán từ tâm thời gian của các
  Gaussian và in ra khoảng đã đoán.
- Không `--sh`: màu chỉ là SH bậc 0. Với `--sh` phần màu theo hướng nhìn và thời gian (`eval_shfs_4d`) được tính sẵn vào hệ số SH bậc 3
  của từng keyframe.
- `bouncingballs` của D-NeRF, tự train 20.000 vòng (38 phút trên RTX 3060, 400 px): 134.260 Gaussian trong checkpoint, giữ 116.193 ×
  60 keyframe, 116 MB, chuyển mất 2 giây. 20 ảnh test ở 400 px: 26,0–31,9 dB, trung bình 29,0 dB (màu loang nhẹ vì bỏ SH bậc cao); với
  `--sh` 27,3–39,6 dB, trung bình 32,9 dB, file 1,4 GB (453 MB với `--compact`), 8 giây. Repo tự đo 33,4 dB; phần chênh 0,5 dB chưa tách
  được nguyên nhân. Ảnh: `shots/real/native4d_bouncingballs.png`.
- HyperNeRF `slice-banana` (297 ảnh train, 33 ảnh giữ lại, ảnh 4x = 268×480; repo không có loader Nerfies nên dữ liệu được ghi theo layout
  D-NeRF bằng `tests/make_hypernerf_blender.py`; config `resolution: 1`, `batch_size: 1`, `densify_until_num_points: 250_000`, 20.000 vòng,
  20 phút): `--frames 110 --sh --compact` cho 21,41–29,17 dB, **trung bình 26,31 dB**; đánh giá của chính repo 26,30 dB.
- Bài học từ lần train đầu (WP19c, WP20): layout cũ ghi `fl_x/fl_y/cx/cy` cho từng frame, khi đó repo để `FovX = FovY = -1`, rasteriser CUDA
  dùng tiêu cự sai và model học bù. Model đó cho 24,23 dB qua viewer (repo 26,14 dB); dựng lại đúng cách vẽ sai đó bằng gsplat
  (`tests/diag_native4d_banana.py`) cho 26,13 dB, nên converter không sai. Layout mới ghi `camera_angle_x`; số 26,31 dB ở trên là của
  model train lại.

### B — 4D-Rotor Gaussian Splatting (`convert_4drotor.py`)

```bash
python convert_4drotor.py nerfstudio_models/step-000019999.ckpt -o scene.npz --frames 60 --t0 0 --t1 1
```

- Đầu vào: checkpoint Nerfstudio `step-*.ckpt` (`pipeline` state dict, khoá `gauss_params.*`); cần torch.
- Vị trí và opacity đổi theo thời gian; covariance và SH bậc 3 ghi một lần. Thời gian là `time` thô của file transforms (D-NeRF: [0, 1]).
- Chưa có output thật: chỉ kiểm bằng checkpoint tổng hợp (`tests/test_4drotor.py`, so với phép lấy điều kiện của Gaussian 4D).

### C — Spacetime Gaussians, bản lite (`convert_spacetime.py`)

```bash
python convert_spacetime.py point_cloud/iteration_25000/point_cloud.ply -o scene.npz --frames 50 --t0 0 --t1 1
```

- Chỉ màu cơ sở `f_dc_0..2`; MLP `rgbdecoder` của bản đầy đủ không được tái hiện. Gặp bản đầy đủ (ply có `f_dc_3..` hoặc file
  `rgbdecoder` cạnh ply) converter dừng; `--base-color-only` để chuyển bằng màu cơ sở.
- Màu được giữ thô, không cắt về [0,1]: model lưu màu ngoài khoảng đó ở 30 % Gaussian (−5 đến 6,9), repo trộn màu thô rồi mới cắt ở
  pixel. Trước 16/11/2026 converter cắt màu và chỉ đạt 31,43 dB; file chuyển trước ngày đó cần chuyển lại (`gs4d.open` tự chuyển lại khi
  converter mới hơn file cache). Renderer client vẫn cắt màu.
- `n3d_sear_steak_lite_allcam` (tác giả công bố): 108.317 Gaussian trong ply, giữ 98.993 × 50 keyframe, 199 MB, 3 giây. So với video gốc
  `sear_steak` của Neural 3D Video (21 camera × frame 0, 12, 25, 37, 49 = 105 ảnh, 1352×1014, thời gian = frame/50): 27,76–37,81 dB,
  **trung bình 34,37 dB**. Renderer của chính repo (`test_ours_lite` + rasteriser `forward_lite`, `tests/repo_psnr.py spacetime`) trên cùng
  105 ảnh: 27,79–38,20 dB, trung bình 34,48 dB; 0,11 dB chênh là do `--min-peak-opacity 0.02` (để 0 thì khớp tới 0,01 dB).
- Technicolor `Birthday` (`ours_lite`, 658.999 Gaussian): `--frames 50 --compact` giữ 596.601 Gaussian, 583 MB, 20 giây; render bằng camera 0
  của `cameras.json` ở t = 0; 0,5; 1 cho cảnh đúng, người chuyển động; không có ảnh gốc nên không có PSNR.

### C — Shape of Motion (`convert_shape_of_motion.py`)

```bash
python convert_shape_of_motion.py <work_dir>/checkpoints/last.ckpt -o scene.npz --fps 30 [--stride 8] [--fg-only]
```

- Motion bases SE(3) + hệ số theo Gaussian, nền tĩnh (nền chỉ lưu một lần trong file); cần torch để đọc checkpoint. Thời gian = frame / `--fps`.
- `paper-windmill` (tác giả công bố), `--stride 8`: 71.867 Gaussian tiền cảnh + 924.075 nền × 35 keyframe (277 frame gốc), 147 MB (1,27 GB
  trước khi có nhóm tĩnh), 10 giây. Render từ camera huấn luyện (`Ks`, `w2cs` trong checkpoint) khớp cảnh; ảnh `shots/real/`.
- So với ảnh 1x (720×960) của bộ iPhone DyCheck tại camera huấn luyện, mỗi 4 frame (70 ảnh): tại keyframe **25,86 dB** (min 24,28); ở frame nằm
  giữa hai keyframe của `--stride 8` 24,67 dB (min 21,61). Với `--stride 1` (mọi frame là keyframe, 416 MB khi `--compact`) 25,85 dB ở mọi nơi:
  phần giảm 1,2 dB là do nội suy thẳng giữa các keyframe thưa của cối xay đang quay.
- Renderer của chính repo (`SceneModel.render`, `tests/repo_psnr.py som`) trên cùng 70 ảnh: 24,28–27,66 dB, trung bình 25,85 dB; file
  `--stride 1` lệch 0,000 dB ở từng ảnh.

### C/D — MoSca (`convert_mosca.py`)

```bash
python convert_mosca.py photometric_d_model_native_add3.pth -o scene.npz --fps 30 --stride 2
```

- Scaffold node + skinning theo K node gần nhất, trộn dual quaternion (hoặc LBS); cần torch. Thời gian = frame / `--fps`.
- Chỉ Gaussian động; nền tĩnh (`photometric_s_model_*.pth`) chưa đọc. `max_sph_order > 0` thì dừng.
- Chưa có output thật: chỉ kiểm bằng state dict tổng hợp (`tests/test_mosca.py`, so với phép trộn dual quaternion chuẩn hoá theo sách).

### D — E-D3DGS (`convert_ed3dgs.py`)

```bash
python convert_ed3dgs.py <thư_mục_model> -o scene.npz --frames 60 --t0 0 --t1 0.9967 --sh
```

- Thư mục model có `point_cloud/iteration_*/{point_cloud.ply,deformation.pth}` và `cfg_args`. Embedding theo Gaussian + bảng thời gian,
  chạy lại MLP thô và tinh trên CPU; cần torch.
- Thời gian: N3V = frame/300 (`--t1 0.9967` cho 300 frame), Technicolor = frame/50. `cfg_args` quyết định đầu nào bật (`no_do`, `no_dc`,
  `use_anneal`, ...); `--set KEY=VALUE` thay một giá trị. Màu và opacity theo keyframe khi `no_dc` / `no_do` là False.
- Chưa có output thật: chỉ kiểm bằng model tổng hợp so với một module torch viết theo repo (`tests/test_ed3dgs.py`).

### D — 4DGaussians, hustvl (`convert_4dgaussians.py`)

```bash
python convert_4dgaussians.py <thư_mục_model> -o scene.npz --frames 60 [--sh]
```

- Thư mục model có `point_cloud/iteration_*/{point_cloud.ply,deformation.pth}` và `cfg_args`. HexPlane + MLP biến dạng chạy lại trên CPU
  (cần torch), không cần môi trường của repo gốc. Cách cũ vẫn dùng được: chạy `export_perframe_3DGS.py` trong repo rồi dùng dòng A.
- Dừng (nêu tên biến thể) với `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False` (SH đổi theo thời gian).
- `bouncingballs` của D-NeRF, tự train (20.000 vòng, 692 s trên RTX 3060, `arguments/dnerf/bouncingballs.py`, nền trắng): 27.043 Gaussian,
  `--frames 60 --sh` cho file 64 MB, 4 giây. 20 ảnh test ở 800 px (ghép nền trắng): 36,7–45,9 dB, trung bình 40,8 dB (repo tự đo 39,7 dB ở
  vòng 14.000). So với `export_perframe_3DGS.py` trên cùng model, tại 20 thời điểm test: vị trí lệch tối đa 1,2e-6, covariance lệch tương đối
  7e-6, opacity giống hệt.
- HyperNeRF `slice-banana` (reader HyperNeRF của repo, `arguments/hypernerf/banana.py`, 14.000 vòng, 18 phút, ảnh 4x 268×480, 297 ảnh train /
  33 ảnh giữ lại): 52.741 Gaussian, `--frames 330 --sh --compact` cho file 325 MB, 35 giây, RAM đỉnh 1,5 GB. 33 ảnh giữ lại ở 268 px:
  22,05–30,39 dB, trung bình 27,13 dB; repo tự render cùng 33 ảnh cho đúng số đó (từng ảnh lệch tối đa 0,003 dB). Repo này giữ camera và
  điểm HyperNeRF ở toạ độ gốc, thời gian = warp_id / max.

### D — Deformable-3DGS (`convert_deformable3dgs.py`)

```bash
python convert_deformable3dgs.py <thư_mục_model> -o scene.npz --frames 60 [--sh]
```

- Thư mục model có `point_cloud/` và `deform/`; MLP biến dạng chạy lại trên CPU (cần torch). Biến thể D-NeRF (timenet), mặc định và 6-DoF
  được đọc từ state dict.
- `bouncingballs` (biến thể D-NeRF, tự train 20.000 vòng, 34 phút, 800 px): `--sh` cho 64.571 Gaussian × 60 keyframe, SH bậc 3, 153 MB
  (84 MB với `--compact`), 49 giây. 20 ảnh test ở 800 px: 37,2–43,6 dB, trung bình 40,3 dB (ở 400 px chỉ 29,3–33,2 dB, vì model train ở
  800 px). Ảnh (400 px): `shots/real/deformable3dgs_bouncingballs.png`.
- Biến thể mặc định (không `--is_blender`): 35,35–43,14 dB, trung bình 39,62 dB. Biến thể 6-DoF (`--is_6dof`, 27 phút): 36,99–43,75 dB,
  trung bình 40,34 dB. Converter không phải sửa cho cả hai.
- HyperNeRF `slice-banana` (video một camera, 297 ảnh train / 33 giữ lại, ảnh 4x 268×480, 20.000 vòng, 73 phút, 392.000 Gaussian; dữ liệu
  dựng bằng `tests/make_hypernerf_split.py`): `--frames 330 --sh --compact` (đúng 330 thời điểm của video), 33 ảnh giữ lại ở 268 px:
  19,28–29,78 dB, trung bình 24,72 dB (bỏ qua méo ống kính như loader của repo).

### D — SC-GS (`convert_scgs.py`)

```bash
python convert_scgs.py <thư_mục_model>_node -o scene.npz --frames 60 [--knn 3] [--sh]
```

- Thư mục model có `point_cloud/` và `deform/`; control node + MLP + nội suy theo K node gần nhất, chạy trên CPU (cần torch).
  `--knn` = `--K` lúc train nếu khác 3; `--max-d-scale` = `--max_d_scale` lúc train nếu có.
- `bouncingballs` (512 node, `--local_frame`, 20.000 vòng sau 10.000 bước khởi tạo node, 19 phút, 400 px): 165.900 Gaussian × 60 keyframe,
  10 giây, 361 MB (393 MB với `--sh`). 20 ảnh test ở 400 px: 35,4–43,0 dB, trung bình 40,3 dB với `--sh`; 27,8–33,5 dB (5 ảnh) khi chỉ dùng
  màu cơ sở. Ảnh: `shots/real/scgs_bouncingballs.png`.
- Có motion mask (`--gs_with_motion_mask --gt_alpha_mask_as_dynamic_mask`, ply có `fea_0..8`, 20.000 vòng, 20 phút, 170.091 Gaussian):
  35,48–43,14 dB, trung bình 40,14 dB; converter tự nhận ra mask.
- Đầu màu `--pred_color`: d_color theo node được chạy lại và cộng vào hệ số SH DC, nên màu (và `sh` khi có `--sh`) ghi theo từng keyframe;
  `--ignore-pred-color` giữ màu cố định (file nhỏ hơn, màu sai). Kiểm trên `bouncingballs` nhuộm màu theo thời gian, tự train: 37,83 dB có
  đầu màu, 24,48 dB khi bỏ (mục `scgs_predcolor`). Đầu màu hash hoặc theo hướng nhìn: converter dừng.
- HyperNeRF `slice-banana` (512 node, hyper_dim 8, 20.000 vòng, 52 phút, 603.000 Gaussian; bản clone SC-GS cần sửa `np.float` → `np.float64`):
  `--frames 110 --sh --compact` 20,21–30,22 dB, trung bình 26,69 dB; `--frames 330` (mỗi ảnh giữ lại trùng một keyframe; 3,5 phút, file 3,7 GB)
  21,27–30,05 dB, trung bình 26,87 dB.
- `--use_hash` (hash grid viết lại bằng numpy, khớp tinycudann tới 2e-4) và `--skinning`: chỉ test tổng hợp; bản tự train của repo không
  hội tụ (hash, 6,8 dB) hoặc sập sau densification (skinning).

## Dừng có thông báo (converter không đoán)

- SC-GS: `--deform_type mlp` (đó là Deformable-3DGS), hash với `tcnn_mlp`, đầu màu hash / theo hướng nhìn, ply và `deform.pth` lệch số cột `fea_*`.
- Spacetime: model đầy đủ có `rgbdecoder` (dùng `--base-color-only` để chuyển mà bỏ nó).
- 4DGaussians: `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False`.
- E-D3DGS: ply không có SH bậc 3 hoặc không có cột `embedding_*`; `--set` với khoá không có trong ModelHiddenParams.
- MoSca: `max_sph_order > 0`; file nền tĩnh `photometric_s_model_*.pth` (`gs4d.open` nói cần file động).
- Native 4DGS: checkpoint không có đúng 19 trường `capture()` (không phải model 4D), hoặc ít kênh SH hơn yêu cầu.
- Mọi converter: thiếu `torch` khi cần (báo cách cài), file torch không đọc được, thư mục model không có `iteration_*` (hoặc không có
  `--iteration` đã chọn), file không đúng layout. Thông báo là một dòng, không phải traceback.

`python -m gs4d.open <output>` nhận ra các loại trên và chuyển; biến thể không hỗ trợ dừng và nêu cờ để chuyển tiếp.
Để thêm phương pháp mới: [ADD_A_CONVERTER.md](ADD_A_CONVERTER.md).

## Ghi chú tra repo trước khi viết ba converter mới (17/11/2026)

Giữ lại để tra cứu: đây là ghi chú đọc repo trước khi viết `convert_4drotor.py`, `convert_ed3dgs.py`, `convert_mosca.py`. Chỗ nào ghi "chưa xác minh" đã được đọc nốt khi viết converter (xem docstring của từng converter).

Tra từ repo chính thức bằng cách đọc README và mã nguồn (không chạy). Mục nào ghi "chưa xác minh" là chỗ phần đọc được không bao gồm. Ước lượng công viết converter là ước lượng của người viết, dựa trên converter tương tự đã có; repo không công bố số GPU, RAM hay thời gian train nào.

### 4D-Rotor Gaussian Splatting (4DRotorGS, SIGGRAPH 2024) — đã có `convert_4drotor.py` (17/11/2026)
Công thức vị trí `slice_4d_to_3d` đã đọc: `mean(t) = xyz + speed·(t − t_g)`, `speed = Σ[:3,3]/Σ[3,3]`; không chuẩn hoá rotor khi render. Checkpoint là `pipeline` state dict với khoá kết thúc bằng `gauss_params.<tên>`; tiền tố `_model.` chưa xác minh trên file thật nên converter tìm theo đuôi khoá.
- Repo: https://github.com/weify627/4D-Rotor-Gaussians, giấy phép Apache-2.0, xây trên Nerfstudio. Paper arXiv 2402.03307.
- Output: `step-<N>.ckpt` (`torch.save` với `step, pipeline, optimizers, ...`); tham số Gaussian nằm trong `gauss_params`: `xyz, t, scales_xyz, scales_t, quats1, quats2, features_dc, features_rest, opacities`. Không thấy xuất ply theo Gaussian.
- Công thức (`splatfacto.py`, `general_utils.py::slice_4d`): rotor 8 số → ma trận quay 4D, `scale = exp(cat(scales_xyz, scales_t))`, Σ = L·Lᵀ (4×4). `w = 1/Σ[3,3]`, covariance 3D điều kiện `cov3 = Σ[:3,:3] − Σ[:3,3]·Σ[3,:3]·w` (không phụ thuộc t); opacity `sigmoid(o)·exp(−0,5·w·(t_g − t)²)`, bỏ Gaussian khi mũ ≥ 16; màu SH bậc 3 không đổi theo thời gian. Công thức vị trí `slice_4d_to_3d` và khoảng thời gian: chưa xác minh.
- Dạng: B (cùng họ native 4DGS; `convert_native4d.py` là khuôn mẫu, gồm cách bỏ Gaussian và bake opacity theo keyframe).
- Dữ liệu: D-NeRF (`splatfacto`, 20.000 vòng) và N3V (`splatfacto-big`, cần `scripts/n3v2blender.py`).
- Cài đặt: Python 3.8, torch 2.1.2+cu118, tiny-cuda-nn, PyTorch3D, ba submodule CUDA riêng (`diff-gaussian-rasterization-confidence`, `knn`, `knn_ops_3_fwd_bwd_mask`) và `simple_knn`. Rasteriser tuỳ biến, README nói bản FPS nhanh không công bố.
- Công viết converter: nhỏ–vừa (khoảng 1 ngày): chỉ numpy + torch để mở ckpt, công thức đã rõ trừ phép cắt vị trí cần đọc nốt `slice_4d_to_3d`. Dùng `--sh` kiểu native 4DGS vì màu là SH.
- Train trên máy 12 GB GPU / 15 GB RAM: D-NeRF có vẻ khả thi (cảnh nhỏ, như native 4DGS đã train), nhưng phải dựng môi trường py3.8/cu118 và build ba submodule; N3V chưa biết, không có số liệu bộ nhớ.

### E-D3DGS (ECCV 2024, Per-Gaussian Embedding-Based Deformation) — đã có `convert_ed3dgs.py` (17/11/2026)
Đã đọc đủ `scene/deformation.py`, renderer và `render.py`: đầu vào MLP chỉ là [embedding thời gian, embedding Gaussian], hai nhánh thô/tinh chỉ cộng dồn (không nối đầu ra nhánh thô vào nhánh tinh như ghi chú ban đầu); delta cộng TRƯỚC kích hoạt (log scale, logit opacity, quaternion rồi normalize).
- Repo: https://github.com/JeongminB/E-D3DGS (trang dự án https://jeongminb.github.io/e-d3dgs/). Giấy phép là bản Inria/MPII của Gaussian Splatting: chỉ nghiên cứu, không thương mại.
- Output: `point_cloud/iteration_N/point_cloud.ply` (x, y, z, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`, thêm `embedding_0..E-1`; giá trị trước kích hoạt) và `deformation.pth` (state_dict của `deform_network`), cùng `cfg_args`.
- Biến dạng (`scene/deformation.py`): mỗi Gaussian có embedding riêng; thời gian là bảng latent học được (`weight`, nội suy song tuyến tại t); hai nhánh MLP thô và tinh, nhánh tinh nhận đầu ra của nhánh thô. Đầu ra cộng vào: vị trí, log-scale, quaternion, logit opacity, và hệ số SH (cả màu theo hướng nhìn). Thời gian `t = frame / duration` (dynerf 300, Technicolor 50). Số Gaussian cố định theo thời gian.
- Dạng: D (giống Deformable-3DGS: chạy lại MLP trên CPU), nhưng thêm delta opacity và SH, nên cần `sh` theo keyframe như `convert_scgs.py --pred_color`. Hai thiết lập chưa xác minh: kích thước embedding (file `arguments/`) và iteration dùng khi render.
- Dữ liệu: N3V (nhiều camera), Technicolor, HyperNeRF (một camera); cần COLMAP.
- Cài đặt cũ: Python 3.7, torch 1.13.1+cu116, mmcv 1.6.0, open3d, submodule `diff-gaussian-rasterization` và `simple-knn`.
- Công viết converter: vừa (1–2 ngày): MLP thô + tinh và lấy mẫu bảng thời gian đều ở numpy, kiểm bằng so với `deformation.pth` chạy bằng torch như `test_deformable3dgs.py`.
- Train trên máy này: không chắc. N3V là nhiều camera × 300 frame (hàng chục GB, như Spacetime full model đã không làm được); HyperNeRF `slice-banana` (đã có sẵn trong `../data_real`) thì có khả năng, nhưng môi trường py3.7/torch 1.13 phải dựng riêng.

### MoSca (CVPR 2025, Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds) — đã có `convert_mosca.py` (17/11/2026)
Đã đọc `mosca.py`, `dynamic_gs.py`, `dualquat_helper.py`, `photo_recon.py`, `mosca_evaluate.py`. File cần: `photometric_d_model_<backend>.pth` (state_dict của `DynSCFGaussian`, scaffold ở `scf.*`). Sigma node = `max_sigma·sigmoid(logit)`; DQB trộn cả với đơn vị bằng `dyn_o` (mặc định `1 − 1e-3`), nên chuyển động không rigid tuyệt đối. Nền tĩnh nằm ở file riêng, converter chưa đọc.
- Repo: https://github.com/JiahuiLei/MoSca, MIT cho `lib_moca`, `lib_mosca`; phần rasteriser và model bên thứ ba có giấy phép riêng.
- Output: `mosca/mosca.pth` (scaffold: `_node_xyz [T,M,3]`, `_node_rotation` theo frame, `_node_sigma_logit`, `topo_knn_ind`, ...); model Gaussian động `DynSCFGaussian` lưu `_xyz` (toạ độ cục bộ theo node gắn), `_rotation`, `_scaling`, `_opacity`, `_features_*`, `_skinning_weight`, `attach_ind`, `ref_time`. File lưu model Gaussian (`photo_recon.py`) chưa đọc nên tên file chưa xác minh.
- Chuyển động: vị trí = quay/tịnh tiến của node gắn tại `ref_time` rồi `warp` sang frame t bằng skinning K node gần nhất (K = 16 khi reconstruct) trọng số `exp(−|x−node|²/(2σ²))`; pha trộn dual quaternion (`dqb`, mặc định) hoặc `lbs`. Scale, opacity và màu không đổi theo thời gian (trừ `nn_fusion` làm opacity về 0 ngoài các frame hỗ trợ). Thời gian là chỉ số frame. Phần DQB (`scaffold_utils/dualquat_helper.py`) chưa đọc.
- Dạng: D hoặc C lai (gần SC-GS: control node + K node gần nhất, thêm DQB); `convert_scgs.py` và `convert_shape_of_motion.py` là khuôn mẫu.
- Dữ liệu: video thường (iPhone/DyCheck, NVIDIA, TUM, Sintel, wild); cần tiền xử lý nặng: tracker (BootsTAPIR, CoTracker, SpaTracker), độ sâu (DepthCrafter, Metric3D-v2, UniDepth), RAFT, rồi bundle adjustment camera; trọng số tải tay.
- Cài đặt: Python 3.10, torch 2.1.0+cu118, pytorch3d, torch_geometric; ba rasteriser tuỳ biến (TODO của repo: chuyển sang gsplat). `cuda:0` cố định trong mã.
- Công viết converter: lớn hơn hai cái trên (vài ngày): phải chép đúng DQB và `get_skinning_weights`, `resample_time` từ keyframe thưa (mỗi 4 frame) ra mọi frame, và Gaussian có `ref_time` nên tập Gaussian sống thay đổi theo frame.
- Train trên máy này: khả năng thấp. Cần chạy nhiều model tiền xử lý nặng; repo không công bố yêu cầu bộ nhớ. Cảnh `paper-windmill` là DyCheck iPhone mà MoSca hỗ trợ nên có thể so sánh với Shape of Motion nếu tự train được.

### Gợi ý thứ tự (theo công / lợi ích, chưa phải quyết định)
4D-Rotor GS (công thức rõ nhất, cùng họ native 4DGS) → E-D3DGS (cần dựng môi trường cũ) → MoSca (tiền xử lý và converter đều nặng).
