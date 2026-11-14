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
| A. | Dynamic 3D Gaussians (`params.npz`) — chạy với output thật `basketball` (375.870 Gaussian x 150 frame gốc) | `python convert_per_frame.py params.npz -o scene.npz --stride 5 --fps 30 --fg-only` rồi `python viewer.py scene.npz --up -y`. Đã đo: `--fg-only` cho 112.592 Gaussian x 30 keyframe, 163 MB (bỏ `--fg-only`: 375.870 Gaussian, 543 MB, nhưng cảnh bị nền và floater che). Trục up là `-y`; ảnh chụp ở `shots/wp2/`. Lưu ý: ~14% màu gốc nằm ngoài [0,1] và bị cắt; renderer client chưa đối chiếu với `visualize.py` gốc. |
| B. Gaussian 4D | Native 4DGS của Yang et al. (`chkpnt*.pth`; chỉ màu SH bậc 0) — chạy với checkpoint thật tự train trên cảnh `bouncingballs` của D-NeRF (20.000 vòng, 38 phút trên RTX 3060) | `python convert_native4d.py chkpnt_best.pth -o scene.npz --frames 60 --t0 0 --t1 10` (cảnh D-NeRF có thời gian trong [0, 1]: dùng `--t1 1`). Đã đo: 134.260 Gaussian trong checkpoint, giữ 116.193 x 60 keyframe, 116 MB, chuyển mất 2 giây. Render ở camera và thời điểm của 5 ảnh test: hình và chuyển động khớp, PSNR 26,0–31,1 dB (repo gốc tự đo 33,4 dB); màu loang nhẹ vì bỏ SH bậc cao theo hướng nhìn và thời gian. Ảnh: `shots/real/native4d_bouncingballs.png` |
| C. Hàm chuyển động tham số | Spacetime Gaussians (`point_cloud.ply`; chỉ màu cơ sở, bỏ MLP `rgbdecoder`) — chạy với model thật `n3d_sear_steak_lite_allcam` do tác giả công bố | `python convert_spacetime.py point_cloud.ply -o scene.npz --frames 50 --t0 0 --t1 1`. Đã đo: 108.317 Gaussian trong ply, giữ 98.993 x 50 keyframe, 199 MB, chuyển mất 3 giây. Render gsplat từ hai camera huấn luyện (`cameras.json`) ra đúng cảnh, người và vật chuyển động theo thời gian. Model lite này chỉ có `f_dc_0..2` và không kèm `rgbdecoder`; model đầy đủ (có `rgbdecoder`) chưa thử. |
| C. | Shape of Motion (`checkpoints/last.ckpt`; motion bases SE(3) + hệ số, nền tĩnh; cần torch để đọc checkpoint) — chạy với checkpoint thật `paper-windmill` do tác giả công bố | `python convert_shape_of_motion.py last.ckpt -o scene.npz --fps 30` (`--fg-only` bỏ nền, `--stride n` lấy thưa frame). Đã đo với `--stride 8`: 71.867 Gaussian tiền cảnh + 924.075 nền x 35 keyframe (277 frame gốc), 1,27 GB, chuyển mất 10 giây, RAM đỉnh 3,2 GB. Render gsplat từ camera huấn luyện (`Ks`, `w2cs` trong checkpoint) khớp cảnh, ảnh ở `shots/real/`. Nền tĩnh vẫn bị lặp lại ở mọi keyframe nên file lớn: dùng `--stride` hoặc `python -m gs4d` để nén. |
| D. Mạng biến dạng | 4DGaussians: đi qua `export_perframe_3DGS.py` rồi dùng dòng A | |
| D. | Deformable-3DGS (thư mục model có `point_cloud/` và `deform/`; chạy lại MLP biến dạng trên CPU, cần torch; nhận cả biến thể D-NeRF và 6-DoF) — chạy với model thật tự train trên cảnh `bouncingballs` của D-NeRF (biến thể D-NeRF, 20.000 vòng, 34 phút trên RTX 3060); biến thể mặc định và 6-DoF chưa thử với output thật | `python convert_deformable3dgs.py <thư_mục_model> -o scene.npz --frames 60` (thêm `--sh` để giữ màu theo hướng nhìn). Đã đo với `--sh`: 64.571 Gaussian x 60 keyframe, SH bậc 3, 153 MB, chuyển mất 49 giây. Render ở camera và thời điểm của 5 ảnh test: PSNR 29,3–33,2 dB. Ảnh: `shots/real/deformable3dgs_bouncingballs.png` |
| D. | SC-GS (thư mục model `<tên>_node` có `point_cloud/` và `deform/`; control node + MLP + nội suy theo K node gần nhất, chạy trên CPU, cần torch; không hỗ trợ `--use_hash`, `--skinning`; bỏ qua `--pred_color`) — chạy với model thật tự train trên cảnh `bouncingballs` của D-NeRF (512 node, `--local_frame`, 20.000 vòng sau 10.000 bước khởi tạo node, 19 phút trên RTX 3060) | `python convert_scgs.py <thư_mục_model> -o scene.npz --frames 60` (`--knn` = `--K` lúc train nếu khác 3; thêm `--sh` để giữ màu theo hướng nhìn). Đã đo: 165.900 Gaussian x 60 keyframe, chuyển mất 10 giây; 361 MB, với `--sh` (SH bậc 3) 393 MB. Render ở camera và thời điểm của 5 ảnh test: PSNR 36,2–41,3 dB với `--sh`, 27,8–33,5 dB khi chỉ dùng màu cơ sở. Ảnh: `shots/real/scgs_bouncingballs.png` |

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

## So sánh hai sequence

```bash
python viewer.py ours.npz --compare baseline.npz
```

Hai file dùng chung thanh thời gian, nút play và camera. Ô **Show** trong bảng điều khiển chọn `A` (file thứ
nhất), `B` (file `--compare`) hoặc `A | B` (cạnh nhau, mặc định). Đổi giữa `A` và `B` ở cùng một góc nhìn để
soi khác biệt tại chỗ.

- Renderer gsplat: `A | B` chia đôi màn hình, mỗi nửa vẽ một sequence từ **cùng một camera**, tên file ghi ở
  cạnh dưới. Bảng điều khiển che góc trên bên phải của nửa B; bấm dấu `−` trên bảng để thu gọn.
- Renderer client: `A | B` đặt B sang bên phải A trong cùng một scene (cách 2,4 lần bán kính scene), có nhãn
  tên file phía trên. Giới hạn `--max-splats` áp dụng riêng cho từng file.

Thanh thời gian phủ từ mốc sớm nhất đến mốc muộn nhất của cả hai file; file nào hết keyframe thì đứng ở
keyframe cuối của nó. Ô "Keyframe" và hai nút bước keyframe đi theo các mốc của file A. Hai file phải cùng hệ
tọa độ và cùng trục up. Xuất video hoạt động ở cả ba chế độ. Ảnh chụp: `shots/compare/`.

## Xuất video

Trong bảng điều khiển mở mục **Export video**, đặt camera ở góc muốn quay, chọn "Frames per second" và
"Video length (s)", rồi bấm **Export mp4 from this camera**. Viewer render toàn bộ khoảng thời gian của
sequence từ camera hiện tại (camera đứng yên) thành file `<tên_file>_<ngày_giờ>.mp4` trong thư mục
`--video-dir` (mặc định là thư mục đang chạy) trên máy chạy viewer, đồng thời hiện link tải file về trong
trình duyệt. Cần `pip install imageio imageio-ffmpeg` (đã kèm ffmpeg).

- Renderer gsplat: render thẳng trên GPU ở độ phân giải đang hiển thị (giới hạn bởi "Max render height");
  60 frame 1280×720 của `toy.npz` mất khoảng 1 giây.
- Renderer client: mỗi frame phải chờ trình duyệt vẽ rồi gửi ảnh về nên chậm hơn nhiều (30 frame mất 24 giây
  với Chrome headless vẽ bằng phần mềm).

Trong lúc xuất, viewer tạm dừng cập nhật cho mọi trình duyệt đang kết nối.

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

## File lớn: lưu nén và nạp lười

```bash
python -m gs4d scene.npz scene_small.npz             # lưu lại ở dạng nén (có mất mát nhỏ)
python -m gs4d scene_small.npz scene.npz --float32   # trả về float32 thường
```

Trong code: `seq.save(path, compact=True)`; mặc định `compact=False` (float32, không mất mát). `Sequence4D.load`
đọc được cả hai dạng và mọi file cũ. Dạng nén chỉ đổi các trường thay đổi theo thời gian: `means` thành
float16 theo offset so với keyframe đầu, `covs` thành thừa số Cholesky lượng tử hóa int16 (giải mã luôn ra ma
trận nửa xác định dương), `opacities` và `colors` thành uint8. Viewer nạp lười (`load(path, lazy=True)`, ánh xạ
file vào bộ nhớ): chỉ keyframe đang xem mới được đọc từ đĩa, nên không được ghi đè file khi viewer đang mở nó.

Đo bằng `python tests/bench_large.py` với 500.000 Gaussian × 100 keyframe sinh ngẫu nhiên (means, covs,
opacities thay đổi theo thời gian), SSD NVMe, file vừa ghi nên còn trong cache của hệ điều hành:

| | Trước (float32, nạp hết) | Sau: float32, nạp lười | Sau: nén, nạp hết | Sau: nén, nạp lười |
|---|---|---|---|---|
| Dung lượng file | 2006 MB | 2006 MB | 964 MB | 964 MB |
| Thời gian nạp | 1,04 s | dưới 0,01 s | 0,50 s | dưới 0,01 s |
| RAM sau khi nạp và lấy 1 frame | 1963 MB | 89 MB | 980 MB | 94 MB |
| Lấy một frame nội suy (`at(t)`) | 14 ms | 17 ms | 77 ms | 77 ms |
| Thời gian lưu | 2,2 s | 2,2 s | 13,5 s | 13,5 s |
| Sai số vị trí lớn nhất | 0 | 0 | 0,002 | 0,002 |

Sai số vị trí 0,002 là với Gaussian dịch tới 5,35 đơn vị so với keyframe đầu; sai số tỉ lệ với độ dịch
(float16 có 11 bit có nghĩa). Sai số opacity lớn nhất 0,002; sai số covariance lớn nhất 7·10⁻⁵ lần phương sai
lớn nhất của Gaussian đó. Đổi lại, dạng nén giải mã mỗi frame chậm hơn (77 ms so với 14 ms ở 500.000
Gaussian), nên khi play scene lớn tốc độ cập nhật thấp hơn dạng float32.

## Giới hạn hiện tại

- Màu phụ thuộc hướng nhìn chỉ có khi file có trường `sh` và dùng renderer gsplat; renderer client luôn
  vẽ bằng `colors` (SH bậc 0). `convert_per_frame.py --sh` (thư mục ply), `convert_deformable3dgs.py --sh` và `convert_scgs.py --sh` ghi trường này.
- Dung lượng ≈ N × K × 40 byte cho phần thay đổi theo thời gian (≈ 19 byte ở dạng nén); giảm `--frames` hoặc
  `--stride`, hoặc lưu nén như mục trên nếu file lớn. Các converter chưa có cờ nén: chạy `python -m gs4d` sau khi chuyển.
- Số Gaussian thay đổi giữa các frame (feed-forward): viewer nhảy theo keyframe gần nhất, không nội suy.

## Kiểm tra

```bash
python tests/test_all.py                                  # 4 phép kiểm, không cần GPU
python tests/test_all.py /path/to/4d-gaussian-splatting   # thêm so khớp với hàm gốc của repo
python tests/test_sh.py                                   # trường sh; phần render tự bỏ qua nếu không có CUDA
python tests/test_spacetime.py                            # converter Spacetime Gaussians
python tests/test_deformable3dgs.py                       # converter Deformable-3DGS (cần torch, không thì tự bỏ qua)
python tests/test_shape_of_motion.py                      # converter Shape of Motion (cần torch, không thì tự bỏ qua)
python tests/test_scgs.py                                 # converter SC-GS (cần torch, không thì tự bỏ qua)
python tests/test_compact.py                              # lưu nén, nạp lười
python tests/check_real.py                              # cần CUDA + dữ liệu thật (data_real, gs4d_train): PSNR trên toàn bộ ảnh test của 3 method; Spacetime/Shape of Motion chỉ kiểm số Gaussian và dung lượng
python tests/check_gsplat.py                              # cần CUDA + Chrome: so ảnh gsplat với client ở 3 thời điểm
python tests/check_compare.py                             # cần Chrome: --compare ở cả hai renderer (gsplat tự bỏ qua nếu không có CUDA)
```

## Giấy phép và nguồn dữ liệu

Code trong repo này theo giấy phép MIT (xem `LICENSE`).

Ba ảnh `shots/wp2/basketball_fg_up-y_t*.png` **không** thuộc giấy phép MIT. Chúng được render từ model đã train
`basketball` của Dynamic 3D Gaussians, vốn dựng từ bản ghi của CMU Panoptic Studio. Bộ dữ liệu CMU Panoptic
Studio chỉ được dùng cho mục đích nghiên cứu, phi thương mại; ba ảnh này cũng chỉ được dùng theo điều kiện đó.

- J. Luiten, G. Kopanas, B. Leibe, D. Ramanan. *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View
  Synthesis.* 3DV 2024. <https://github.com/JonathonLuiten/Dynamic3DGaussians>
- H. Joo, H. Liu, L. Tan, L. Gui, B. Nabbe, I. Matthews, T. Kanade, S. Nobuhara, Y. Sheikh. *Panoptic Studio:
  A Massively Multiview System for Social Motion Capture.* ICCV 2015.
  <https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox>

Bốn ảnh trong `shots/real/` có hàng trên là render của viewer, hàng dưới (hoặc ảnh đối chiếu) là ảnh của bộ dữ liệu
gốc, nên cũng **không** thuộc giấy phép MIT của repo này:

- `shape_of_motion_paper-windmill.png`: cảnh `paper-windmill` của bộ iPhone trong DyCheck. Repo DyCheck ghi cả code
  và dataset phát hành theo Apache 2.0. Checkpoint do tác giả Shape of Motion công bố (repo MIT).
- `deformable3dgs_bouncingballs.png`, `scgs_bouncingballs.png`, `native4d_bouncingballs.png`: cảnh tổng hợp
  `bouncingballs` của D-NeRF. Repo D-NeRF theo giấy phép MIT và không ghi điều khoản riêng cho dataset. Model do
  chúng tôi tự train bằng code của Deformable-3DGS, SC-GS và native 4DGS (cả ba repo đều MIT).

- H. Gao, R. Li, S. Tulsiani, B. Russell, A. Kanazawa. *Monocular Dynamic View Synthesis: A Reality Check.*
  NeurIPS 2022. <https://github.com/KAIR-BAIR/dycheck>
- A. Pumarola, E. Corona, G. Pons-Moll, F. Moreno-Noguer. *D-NeRF: Neural Radiance Fields for Dynamic Scenes.*
  CVPR 2021. <https://github.com/albertpumarola/D-NeRF>
- Q. Wang, V. Ye, H. Gao, W. Zeng, J. Austin, Z. Li, A. Kanazawa. *Shape of Motion: 4D Reconstruction from a Single
  Video.* ICCV 2025. <https://github.com/vye16/shape-of-motion>
- Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, X. Jin. *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic
  Scene Reconstruction.* CVPR 2024. <https://github.com/ingra14m/Deformable-3D-Gaussians>
- Y. Huang, Y. Sun, Z. Yang, X. Lyu, Y. Cao, X. Qi. *SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic
  Scenes.* CVPR 2024. <https://github.com/yihua7/SC-GS>
- Z. Yang, H. Yang, Z. Pan, L. Zhang. *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D
  Gaussian Splatting.* ICLR 2024. <https://github.com/fudan-zvg/4d-gaussian-splatting>
