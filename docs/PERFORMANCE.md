# Tốc độ và bộ nhớ

Mọi số đo trên máy phát triển: RTX 3060 12 GB, 15 GB RAM, SSD NVMe, torch 2.4.1+cu124, gsplat 1.5.3, trình duyệt cùng máy (trừ khi ghi khác).
Máy dùng chung với phiên khác nên số phía CPU dao động khoảng ±20 %. Lệnh đo ghi cạnh từng bảng.

## Renderer gsplat: một frame

`toy.npz` (15.000 Gaussian × 60 keyframe), đang play:

| Độ phân giải render | Thời gian một frame (rasterize + JPEG + gửi) | Frame/giây gửi đi khi "Update rate" = 60 Hz |
|---|---|---|
| 1280×720 | 14 ms | 57 |
| 1920×1080 | 23 ms | 42 |

"Update rate" mặc định là 20 Hz nên mặc định viewer gửi 20 frame/giây ở cả hai độ phân giải.

## Giải mã và nội suy trên GPU khi play

Trước đây mỗi frame khi play: `at(t)` giải mã và nội suy bằng numpy, đổi covariance sang 3×3, lọc opacity, rồi chép mọi thứ lên GPU. Đo bước 0
(`python tests/bench_large.py`, 500.000 Gaussian × 100 keyframe, dạng nén): chép khối lượng đĩa 3 ms, giải mã 41, nội suy 10, chép lên GPU 4;
cả frame 100 ms. Thời gian nằm ở giải mã trên CPU và các bước numpy sau `at(t)`, không ở đọc đĩa.

Bây giờ `gs4d.GpuFrames` giữ trên GPU các keyframe quanh thời điểm hiện tại (bộ đệm 3 keyframe mỗi trường, float32): mảng thô
(float16 / int16 / uint8) được chép lên một keyframe một lần khi thanh thời gian sang khoảng mới, rồi giải mã và nội suy bằng torch. Trường
không đổi và nhóm tĩnh chép lên GPU một lần lúc khởi tạo. `viewer.py --cpu-decode` dùng lại đường CPU để so sánh.

Thời gian lấy đầu vào gsplat cho một frame khi play (`viewer.frame_cpu` / `viewer.frame_gpu`, trung bình 31 frame qua 2 keyframe):

| | CPU trước | GPU sau |
|---|---|---|
| 500.000 × 100, float32 | 63 ms | 2,3 ms |
| 500.000 × 100, nén | 100 ms (`at(t)` 48 ms) | 2,2 ms |
| `paper-windmill` (995.942 × 35), nén + tách tĩnh | 214 ms (`at(t)` 11 ms) | 3,1 ms |
| `paper-windmill`, float32 + tách tĩnh | 204 ms | 3,2 ms |
| `paper-windmill`, file gốc 1271 MB | 225 ms | 4,6 ms |

Ảnh render khác đường CPU tối đa 1/255 mỗi kênh (0/255 ở `paper-windmill` nén; 1/255 ở scene ngẫu nhiên do làm tròn float32).
`tests/check_real.py` chạy lại, và chạy thêm với `Sequence4D.at` thay bằng `GpuFrames`: cả hai cho PSNR y hệt. Từ 23/11/2026
`tests/test_gpu_frames.py` so `GpuFrames` với `Sequence4D.at` trên CPU cho mọi dạng file (chênh tương đối lớn nhất 1,1·10⁻⁷).

`sh` theo keyframe (native 4DGS `--sh --compact`), lấy frame khi play (`python tests/bench_large.py --file`):

| | `slice-banana` 330 keyframe, `sh` float32 | `sh` uint8 | `bouncingballs` 60 keyframe, `sh` float32 | `sh` uint8 |
|---|---|---|---|---|
| Đường GPU (`GpuFrames`) | 2,2 ms | 1,4 ms | 1,1 ms | 0,8 ms |
| Đường CPU (`--cpu-decode`) | 60 ms | 128 ms | 29 ms | 49 ms |
| Thời gian chuyển / RAM đỉnh | 4 phút 58 giây / 1,3 GB | 4 phút 04 giây / 1,5 GB | 10 giây / 0,8 GB | 14 giây / 0,8 GB |

Đường CPU chậm hơn vì phải giải mã `sh` bằng numpy mỗi frame; renderer gsplat mặc định dùng đường GPU, renderer client không dùng `sh`.

## Hiển thị trong Chrome có GPU

`python tests/bench_display.py --rates 20 30 40 50 60` (cần màn hình không khoá và GPU không train): `paper-windmill` nén + tách tĩnh
(995.942 Gaussian × 35 keyframe), đang play, Chrome 157 trên cùng máy (WebGL qua ANGLE/OpenGL), màn hình 120 Hz, mỗi dòng đo 10 giây.

Chế độ `splats`, 1280×720 (15/11/2026):

| "Update rate" | Server gửi (frame/giây) | Trình duyệt hiện frame mới (frame/giây) | Trình duyệt vẽ lại (frame/giây) |
|---|---|---|---|
| 20 Hz (mặc định) | 20,0 | 20,0 | 120 |
| 30 Hz | 29,9 | 30,0 | 120 |
| 40 Hz | 39,9 | 39,9 | 119 |
| 50 Hz | 49,5 | 49,3 | 119 |
| 60 Hz | 54,3 | 52,9 | 117 |

Chế độ `points` (17/11/2026, thêm `--draw-as points`, và `--size 1920 1080`):

| | 20 Hz | 30 Hz | 40 Hz | 50 Hz | 60 Hz |
|---|---|---|---|---|---|
| 1280×720: server gửi / trình duyệt hiện | 20,0 / 20,0 | 29,9 / 30,0 | 39,9 / 39,8 | 48,7 / 48,2 | 52,3 / 49,5 |
| 1920×1080: server gửi / trình duyệt hiện | 20,1 / 20,0 | | | | 38,8 / 34,2 |

Trình duyệt hiện gần như mọi frame server gửi; giới hạn là phía server. "Frame mới" đếm số lần trang nạp một ảnh đủ kích thước vào texture
WebGL; cột cuối là tốc độ `requestAnimationFrame`. Màn hình khoá hoặc một job train trên GPU làm phép đo sai (rAF tụt còn 1/giây). Chép
canvas ra để so pixel mỗi frame làm tốc độ trình duyệt giảm một nửa: đừng đo theo cách đó. Chưa đo: trình duyệt trên máy khác qua SSH, và
khi bật vệt chuyển động.

## Chế độ điểm (Draw as: points)

Thời gian một frame ở phía server khi play, renderer gsplat, 1280×720 (giải mã + rasterize + chép ảnh về CPU; `tests/check_points.py`
in ra; 2–3 lần chạy, 16/11/2026):

| File | Gaussian | `splats` | `points` (3 px) |
|---|---|---|---|
| SC-GS `bouncingballs_sh.npz` | 165.900 × 60 keyframe | 12–15 ms | 10–11 ms (khoảng 90 frame/giây) |
| Shape of Motion `paper-windmill.npz` | 995.942 × 35 keyframe | 27–40 ms | 24–27 ms (khoảng 40 frame/giây) |

Chế độ điểm không chậm hơn splats. Renderer client chưa được đo trên GPU thật (test chạy WebGL phần mềm).

## Vệt chuyển động

500 vệt, renderer gsplat: cập nhật vệt 0,5 ms mỗi frame (không tính gửi qua mạng); giải mã + render 1280×720 + vệt 17 ms (SC-GS
`bouncingballs_sh`), 7 ms (4DGaussians `slice-banana`, 52.741 Gaussian). Renderer client trên toy: 0,5 ms cho vệt. Lần bật đầu đọc mọi
keyframe hai lượt (`paper-windmill` 277 keyframe: vài giây).

## Xuất video

- Camera cố định, renderer gsplat: 60 frame 1280×720 của `toy.npz` khoảng 1 giây.
- Camera cố định, renderer client: mỗi frame chờ trình duyệt vẽ rồi gửi ảnh về, 30 frame mất 24 giây với Chrome headless vẽ bằng phần mềm.
- Theo quỹ đạo camera, gsplat, 1280×720, 300 frame: SC-GS `bouncingballs_sh` 5,2 s, `paper-windmill` 10,0 s; renderer client trên toy
  480×300, 20 frame: 8 s.

## Dạng nén và nạp lười

`python tests/bench_large.py`: 500.000 Gaussian × 100 keyframe sinh ngẫu nhiên (means, covs, opacities đổi theo thời gian), file vừa ghi
nên còn trong cache của hệ điều hành:

| | Float32, nạp hết | Float32, nạp lười | Nén, nạp hết | Nén, nạp lười |
|---|---|---|---|---|
| Dung lượng file | 2006 MB | 2006 MB | 964 MB | 964 MB |
| Thời gian nạp | 1,04 s | dưới 0,01 s | 0,50 s | dưới 0,01 s |
| RAM sau khi nạp và lấy 1 frame | 1963 MB | 89 MB | 980 MB | 94 MB |
| Lấy một frame nội suy (`at(t)`, CPU) | 14 ms | 17 ms | 77 ms | 77 ms |
| Thời gian lưu | 2,2 s | 2,2 s | 13,5 s | 13,5 s |
| Sai số vị trí lớn nhất | 0 | 0 | 0,002 | 0,002 |

Nhóm Gaussian tĩnh trên file thật (dung lượng, RAM, `at(t)`): [FORMAT.md](FORMAT.md#gaussian-tĩnh-chỉ-lưu-một-lần).

## Converter: RAM đỉnh và thời gian

Converter ghi từng keyframe ra đĩa qua `Sequence4DWriter`, nên RAM đỉnh không tăng theo số keyframe. Đo bằng `/usr/bin/time -v`; file ra
giống code cũ (dựng cả chuỗi trong RAM) từng bit ở mọi mảng.

| Input thật | Lệnh | RAM đỉnh trước → sau |
|---|---|---|
| SC-GS `slice-banana` | `--frames 110 --sh --compact` | 4,5 → 1,4 GB |
| Native 4DGS `slice-banana` | `--frames 110 --sh --compact` | 8,3 → 1,3 GB |
| Deformable-3DGS `slice-banana` | `--frames 330 --sh --compact` | 7,2 → 1,0 GB |
| 4DGaussians `bouncingballs` | `--frames 60 --sh --compact` | 538 → 473 MB |
| 4DGaussians `slice-banana` | `--frames 330 --sh --compact` | 1,49 → 0,60 GB |
| Shape of Motion `paper-windmill` | `--stride 8 --compact` (35 keyframe) | 3,2 → 0,88 GB |
| Shape of Motion `paper-windmill` | `--compact`, cả 277 frame | không chạy được (cần > 11 GB, làm sập app) → 0,88 GB, 69 giây, file 416 MB |
| Spacetime `n3d_sear_steak` lite | `--compact` | 590 → 102 MB |
| Spacetime Technicolor `Birthday` lite | `--compact` | 3,28 → 0,44 GB |
| ply theo frame (4DGaussians `gaussian_pertimestamp`) | `--sh --compact` / không cờ | 300 → 97 MB / 97 → 53 MB |

330 keyframe với `slice-banana`: SC-GS 3,5 phút, file 3,7 GB; native 4DGS 5 phút, file 19,9 GB (`sh` float32; 5,8 GB khi nén `sh`);
Deformable-3DGS hơn 10 phút. Spacetime đọc opacity hai lượt (lượt đầu chỉ tính đỉnh opacity cho `--min-peak-opacity`). Thư mục ply có số
Gaussian đổi theo frame vẫn dựng trong RAM (writer cần N cố định); `params.npz` của Dynamic 3D Gaussians nạp cả file nhưng ghi covariance
từng frame.
