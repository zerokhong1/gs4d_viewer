# Phương pháp và biến thể được hỗ trợ

Số lấy từ README và mục Status của `CLAUDE.md` (không đo lại). PSNR đo bằng renderer gsplat của viewer trên ảnh giữ lại / ảnh test, ở độ phân giải lúc train.
"Output thật" là model đã chạy qua converter; "tổng hợp" là chỉ có test trên dữ liệu giả đúng layout file thật (`tests/test_<tên>.py`).

| Dạng | Phương pháp (converter) | Output thật đã kiểm | PSNR | Giới hạn |
|---|---|---|---|---|
| A | Thư mục ply theo frame, kể cả `gaussian_pertimestamp/` của 4DGaussians (`convert_per_frame.py`) | ply xuất từ 4DGaussians `bouncingballs`, `slice-banana` | khớp converter 4DGaussians | số Gaussian đổi theo frame thì không nội suy (`interpolate=False`) |
| A | Dynamic 3D Gaussians `params.npz` (`convert_per_frame.py`) | `basketball` (375.870 Gaussian × 150 frame) | chưa có ảnh gốc | trục up là `-y`; ~14 % màu gốc ngoài [0,1] bị cắt; renderer client chưa đối chiếu với `visualize.py` |
| B | Native 4DGS, Yang et al. (`convert_native4d.py`, `--sh`) | tự train `bouncingballs` (400 px); HyperNeRF `slice-banana` | `bouncingballs` 32,9 dB với `--sh` (repo 33,4); `slice-banana` 26,31 dB (repo 26,30) | không `--sh` thì chỉ SH bậc 0, màu loang (29,0 dB); `--sh` làm file lớn (`--compact` nén được từ WP21) |
| B | 4D-Rotor GS, SIGGRAPH 2024 (`convert_4drotor.py`) | chưa có (không train được ở đây) | — | chỉ test tổng hợp (`tests/test_4drotor.py`); màu là SH bậc 3 không đổi theo thời gian (ghi vào `sh`); khoảng thời gian mặc định [0, 1] như D-NeRF, chưa kiểm với camera và dữ liệu thật của Nerfstudio |
| C | Spacetime Gaussians, bản lite (`convert_spacetime.py`) | `n3d_sear_steak` (tác giả công bố); Technicolor `Birthday` (không có ảnh gốc) | 34,37 dB; renderer của repo 34,48 | bản đầy đủ có `rgbdecoder`: converter dừng, `--base-color-only` chuyển bằng màu cơ sở; chưa thử model đầy đủ |
| C | Shape of Motion (`convert_shape_of_motion.py`) | `paper-windmill` (tác giả công bố) | 25,86 dB tại keyframe; `--stride 1` khớp repo 0,000 dB | cần torch; `--stride` thưa thì giữa keyframe thấp hơn (24,67 dB ở stride 8) |
| D | E-D3DGS, ECCV 2024 (`convert_ed3dgs.py`) | chưa có (môi trường py3.7/torch 1.13 chưa dựng) | — | chỉ test tổng hợp so với module torch viết theo repo (`tests/test_ed3dgs.py`); cần `cfg_args` (cờ `no_do`, `no_dc`, `use_anneal`...), hoặc `--set`; thời gian N3V = frame/300, Technicolor = frame/50, HyperNeRF chưa đọc `hyper_loader.py`; giấy phép chỉ nghiên cứu |
| D | 4DGaussians, hustvl (`convert_4dgaussians.py`) | tự train `bouncingballs`; HyperNeRF `slice-banana` | 40,8 dB (repo 39,7); `slice-banana` 27,13 dB, bằng repo | dừng với `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False`; cần torch |
| D | Deformable-3DGS (`convert_deformable3dgs.py`) | tự train: D-NeRF, mặc định, `--is_6dof`; HyperNeRF `slice-banana` | 40,3 / 39,62 / 40,34 dB; `slice-banana` 24,72 dB | cần torch; bỏ qua méo ống kính như repo |
| D | SC-GS (`convert_scgs.py`) | tự train `bouncingballs` (kể cả có motion mask); `slice-banana` | 40,3 dB; mask 40,14 dB; `slice-banana` 26,69 dB | `--pred_color` (mặc định hoặc một lớp linear) có công thức đúng nhưng model thật chưa phân biệt được (`--ignore-pred-color` giữ màu cố định); `--use_hash` và `--skinning` chỉ có test tổng hợp (repo tự train không hội tụ / sập); đầu màu hash hoặc theo hướng nhìn thì dừng |

## Dừng có thông báo (converter không đoán)

- SC-GS: `--deform_type mlp` (đó là Deformable-3DGS), hash với `tcnn_mlp`, đầu màu hash / theo hướng nhìn, ply và `deform.pth` lệch số cột `fea_*`.
- Spacetime: model đầy đủ có `rgbdecoder` (dùng `--base-color-only` để chuyển mà bỏ nó).
- 4DGaussians: các biến thể kể trên.
- Native 4DGS: checkpoint không có đúng 19 trường `capture()` (không phải model 4D), hoặc ít kênh SH hơn yêu cầu.
- Mọi converter: thiếu `torch` khi cần (báo cách cài), thư mục model không có `iteration_*`, file không đúng layout.

`python -m gs4d.open <output>` nhận ra các loại trên và chuyển; biến thể không hỗ trợ dừng và nêu cờ để chuyển tiếp.
Để thêm phương pháp mới: [ADD_A_CONVERTER.md](ADD_A_CONVERTER.md).

## Phương pháp có thể thêm (tra repo 17/11/2026, chưa có converter)

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

### MoSca (CVPR 2025, Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds)
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
