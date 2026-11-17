# Phương pháp và biến thể được hỗ trợ

Số lấy từ README và mục Status của `CLAUDE.md` (không đo lại). PSNR đo bằng renderer gsplat của viewer trên ảnh giữ lại / ảnh test, ở độ phân giải lúc train.
"Output thật" là model đã chạy qua converter; "tổng hợp" là chỉ có test trên dữ liệu giả đúng layout file thật (`tests/test_<tên>.py`).

| Dạng | Phương pháp (converter) | Output thật đã kiểm | PSNR | Giới hạn |
|---|---|---|---|---|
| A | Thư mục ply theo frame, kể cả `gaussian_pertimestamp/` của 4DGaussians (`convert_per_frame.py`) | ply xuất từ 4DGaussians `bouncingballs`, `slice-banana` | khớp converter 4DGaussians | số Gaussian đổi theo frame thì không nội suy (`interpolate=False`) |
| A | Dynamic 3D Gaussians `params.npz` (`convert_per_frame.py`) | `basketball` (375.870 Gaussian × 150 frame) | chưa có ảnh gốc | trục up là `-y`; ~14 % màu gốc ngoài [0,1] bị cắt; renderer client chưa đối chiếu với `visualize.py` |
| B | Native 4DGS, Yang et al. (`convert_native4d.py`, `--sh`) | tự train `bouncingballs` (400 px); HyperNeRF `slice-banana` | `bouncingballs` 32,9 dB với `--sh` (repo 33,4); `slice-banana` 26,31 dB (repo 26,30) | không `--sh` thì chỉ SH bậc 0, màu loang (29,0 dB); `--sh` làm file lớn (`--compact` nén được từ WP21) |
| C | Spacetime Gaussians, bản lite (`convert_spacetime.py`) | `n3d_sear_steak` (tác giả công bố); Technicolor `Birthday` (không có ảnh gốc) | 34,37 dB; renderer của repo 34,48 | bản đầy đủ có `rgbdecoder`: converter dừng, `--base-color-only` chuyển bằng màu cơ sở; chưa thử model đầy đủ |
| C | Shape of Motion (`convert_shape_of_motion.py`) | `paper-windmill` (tác giả công bố) | 25,86 dB tại keyframe; `--stride 1` khớp repo 0,000 dB | cần torch; `--stride` thưa thì giữa keyframe thấp hơn (24,67 dB ở stride 8) |
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
