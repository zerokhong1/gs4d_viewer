# gs4d_viewer — xem 4D Gaussian Splatting theo thời gian

Viewer chạy trên trình duyệt (viser), có thanh trượt thời gian, play/pause, bước từng keyframe.
Viewer chỉ đọc **một file `.npz` theo định dạng chung** ("Gaussian theo từng keyframe"); mỗi loại
output 4DGS có một script chuyển về định dạng đó.

## Chạy thử trong 1 phút

```bash
pip install -r requirements.txt
python make_toy.py -o toy.npz        # scene động giả lập
python viewer.py toy.npz             # mở http://localhost:8080
```

## Dùng với output thật

| Dạng output | Phương pháp | Lệnh |
|---|---|---|
| A. Gaussian theo từng frame | thư mục `.ply` mỗi frame (feed-forward, hoặc `gaussian_pertimestamp/` do `export_perframe_3DGS.py` của 4DGaussians xuất ra) | `python convert_per_frame.py <thư_mục> -o scene.npz --fps 30` |
| A. | Dynamic 3D Gaussians (`params.npz`) | `python convert_per_frame.py params.npz -o scene.npz` |
| B. Gaussian 4D | Native 4DGS của Yang et al. (`chkpnt*.pth`) | `python convert_native4d.py chkpnt_best.pth -o scene.npz --frames 60 --t0 0 --t1 10` |
| C. Hàm chuyển động tham số | Spacetime Gaussians, Shape of Motion | chưa có |
| D. Mạng biến dạng | 4DGaussians: đi qua `export_perframe_3DGS.py` rồi dùng dòng A. Deformable-3DGS, SC-GS: chưa có | |

Sau đó: `python viewer.py scene.npz`.

Màu phụ thuộc hướng nhìn: thêm `--sh` vào `convert_per_frame.py <thư_mục>` để giữ `f_rest_*` của ply
(tốn thêm N × K × 12 × (bậc+1)² byte, tức 192 byte mỗi Gaussian mỗi keyframe với SH bậc 3). Thử nhanh:
`python make_toy.py --sh -o toy_sh.npz && python viewer.py toy_sh.npz --renderer gsplat` rồi xoay camera.

`--t0/--t1` của native 4DGS là `time_duration` trong config lúc train. Không truyền thì script đoán
từ tâm thời gian của các Gaussian.

## Renderer

- `--renderer gsplat`: rasterize bằng gsplat trên GPU NVIDIA của máy chạy viewer rồi gửi ảnh về trình
  duyệt. Dùng cho scene thật (hàng trăm nghìn đến hàng triệu Gaussian). Cần `pip install gsplat` và
  torch có CUDA.
- `--renderer client`: gửi Gaussian sang trình duyệt, vẽ bằng WebGL. Không cần CUDA, hợp với laptop
  và scene nhỏ; mặc định giới hạn 300.000 Gaussian (`--max-splats`).
- `--renderer auto` (mặc định): chọn gsplat nếu có CUDA, không thì client.

Cài cho renderer gsplat (không cần CUDA toolkit; wheel dựng sẵn của gsplat chỉ có cho Python 3.10):

```bash
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python "numpy<2" "viser>=1.0" playwright pillow
uv pip install --python .venv/bin/python "gsplat==1.5.3+pt24cu124" --find-links https://docs.gsplat.studio/whl/gsplat/ --index-strategy unsafe-best-match
```

Tốc độ đo được của renderer gsplat với `toy.npz` (15.000 Gaussian × 60 keyframe) trên RTX 3060 12 GB,
torch 2.4.1+cu124, gsplat 1.5.3, đang play, trình duyệt cùng máy:

| Độ phân giải render | Thời gian một frame (rasterize + JPEG + gửi) | Frame/giây gửi đi khi "Update rate" = 60 Hz |
|---|---|---|
| 1280×720 | 14 ms | 57 |
| 1920×1080 | 23 ms | 42 |

"Update rate" mặc định là 20 Hz nên mặc định viewer gửi 20 frame/giây ở cả hai độ phân giải. Con số trên
đo ở phía server; tốc độ hiển thị trong trình duyệt thật chưa đo.

Xem từ máy khác qua SSH: `ssh -L 8080:localhost:8080 <server>` rồi mở `http://localhost:8080`.
Scene từ COLMAP thường cần `--up -y`.

## Định dạng chung (`gs4d/format.py`)

| Khóa | Kích thước | Ý nghĩa |
|---|---|---|
| `times` | `[K]` | mốc thời gian của các keyframe |
| `means` | `[K,N,3]` hoặc `[N,3]` | tâm |
| `covs` | `[K,N,6]` hoặc `[N,6]` | nửa trên của covariance 3×3: xx xy xz yy yz zz |
| `opacities` | `[K,N]` hoặc `[N]` | trong [0,1] |
| `colors` | `[K,N,3]` hoặc `[N,3]` | RGB trong [0,1] |
| `interpolate` | bool | `False` nếu các keyframe không chung danh tính Gaussian |
| `sh` (tùy chọn) | `[K,N,S,3]` hoặc `[N,S,3]`, S = (bậc+1)² | hệ số SH theo quy ước 3DGS (màu = SH(hướng nhìn) + 0,5; `sh[...,0,:]` là `f_dc`). Chỉ renderer gsplat dùng |

Trường không có trục K là hằng theo thời gian. Giữa hai keyframe viewer nội suy tuyến tính.
Thêm phương pháp mới: viết một script tạo `Sequence4D(...)` rồi `.save()`.

## Giới hạn hiện tại

- Màu phụ thuộc hướng nhìn chỉ có khi file có trường `sh` và dùng renderer gsplat; renderer client luôn
  vẽ bằng `colors` (SH bậc 0). Hiện chỉ `convert_per_frame.py --sh` (thư mục ply) ghi trường này.
- Dung lượng ≈ N × K × 40 byte cho phần thay đổi theo thời gian; giảm `--frames` hoặc `--stride` nếu file lớn.
- Số Gaussian thay đổi giữa các frame (feed-forward): viewer nhảy theo keyframe gần nhất, không nội suy.

## Kiểm tra

```bash
python tests/test_all.py                                  # 4 phép kiểm, không cần GPU
python tests/test_all.py /path/to/4d-gaussian-splatting   # thêm so khớp với hàm gốc của repo
python tests/test_sh.py                                   # trường sh; phần render tự bỏ qua nếu không có CUDA
python tests/check_gsplat.py                              # cần CUDA + Chrome: so ảnh gsplat với client ở 3 thời điểm
```
