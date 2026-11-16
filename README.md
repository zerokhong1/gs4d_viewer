# gs4d_viewer — xem 4D Gaussian Splatting theo thời gian

Viewer chạy trên trình duyệt (viser), có thanh trượt thời gian, play/pause, bước từng keyframe.
Viewer chỉ đọc **một file `.npz` theo định dạng chung** ("Gaussian theo từng keyframe"); mỗi loại
output 4DGS có một script chuyển về định dạng đó.

## Mở thẳng một output

```bash
python -m gs4d.open <đường_dẫn_output> [--port 8080 --renderer gsplat ...]
```

Tự nhận ra loại output, gọi đúng `convert_*.py`, rồi mở viewer. Các đối số khác được chuyển cho `viewer.py`.
Nhận ra: `.npz` đã ở định dạng chung (mở luôn), `params.npz` hoặc thư mục ply từng frame, `chkpnt*.pth`,
`last.ckpt` hoặc thư mục có `checkpoints/last.ckpt`, thư mục có `point_cloud/` + `deform/` (có cột `fea_*` thì SC-GS,
không thì Deformable-3DGS), `point_cloud.ply` của Spacetime Gaussians. File chuyển đổi là `<tên>.gs4d.npz` cạnh output,
dùng lại nếu output chưa đổi (`--reconvert` để ép chuyển lại, `--dry-run` để chỉ xem lệnh). Scene D-NeRF
(`cfg_args` ghi dnerf/is_blender) tự đặt thời gian [0, 1]. Không nhận ra thì in các dấu hiệu đã kiểm tra và thoát mã 2.

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
| B. Gaussian 4D | Native 4DGS của Yang et al. (`chkpnt*.pth`; mặc định chỉ màu SH bậc 0, `--sh` giữ màu theo hướng nhìn và thời gian) — chạy với checkpoint thật tự train trên cảnh `bouncingballs` của D-NeRF (20.000 vòng, 38 phút trên RTX 3060) Cảnh thật không phải D-NeRF: HyperNeRF `slice-banana` (297 ảnh train, 33 ảnh giữ lại, ảnh 4x = 268x480; repo không có loader Nerfies nên dùng `tests/make_hypernerf_blender.py` để ghi layout kiểu D-NeRF; config tự viết `resolution: 1`, `batch_size: 1` và `densify_until_num_points: 250_000` thay cho 2 và -1 vì GPU 12 GB còn chừng 8 GB trống, 20.000 vòng, 26 phút): PSNR 21,14–26,19 dB, trung bình 24,23 dB ở 268 px (`--frames 110 --sh --compact`); với `--frames 330` (mỗi ảnh giữ lại nằm đúng một keyframe; converter ghi từng keyframe ra đĩa từ WP19d, RAM đỉnh 1,3 GB thay cho 8,3 GB ở 110 keyframe trước đó, 5 phút, file 19,9 GB vì `sh` theo keyframe là float32): 21,22–26,12 dB, trung bình 24,27 dB (nội suy giữa keyframe không giải thích được gap: chỉ hơn 110 keyframe 0,04 dB); đánh giá của chính repo trên cùng ảnh: 26,14 dB. Kém repo 1,9 dB, nhưng đó là lỗi của repo chứ không phải của converter (WP19c, `tests/diag_native4d_banana.py`): layout Blender của chúng tôi cho từng frame `fl_x/fl_y/cx/cy`, khi đó `readCamerasFromTransforms` để `FovX = FovY = -1`, nên rasterizer CUDA tính Jacobian covariance 2D với tiêu cự giả `W/(2 tan 0,5)` (245 px thay vì 429 px theo x) và ghim `x/z` ở -0,71; model đã học bù cho lỗi đó. Dựng lại đúng cách vẽ này bằng gsplat cho trung bình 26,13 dB (min 21,58, max 29,33), khớp 26,14 của repo. Các bước khác không đổi gì: keyframe đặt đúng 33 thời điểm test 24,27; không nén compact 24,27; cắt `marginal_t` ≤ 0,05 24,29; giữ cả Gaussian opacity đỉnh ≤ 0,02 24,27; near plane 0,2 24,27; hướng SH từ vị trí chưa dịch 24,27. Khi render bằng viewer (tiêu cự đúng) model đó luôn kém 1,9 dB. WP20 (15/11/2026): `tests/make_hypernerf_blender.py` nay ghi `camera_angle_x` ở đầu file và không ghi `fl_x/fl_y/cx/cy` từng frame; train lại cùng config (20.000 vòng, 20 phút, `banana_fov.yaml`): đánh giá của repo 26,30 dB; converter `--frames 110 --sh --compact` + gsplat ở 268 px, 33 ảnh giữ lại: PSNR 21,41–29,17 dB, **trung bình 26,31 dB**, khớp repo trong 0,01 dB, gap 1,9 dB đã hết. Số 24,2 ở trên là của model cũ, train với camera sai. | `python convert_native4d.py chkpnt_best.pth -o scene.npz --frames 60 --t0 0 --t1 10` (cảnh D-NeRF có thời gian trong [0, 1]: dùng `--t1 1`). Đã đo: 134.260 Gaussian trong checkpoint, giữ 116.193 x 60 keyframe, 116 MB, chuyển mất 2 giây. Render ở camera và thời điểm của cả 20 ảnh test, ở 400 px (độ phân giải lúc train): hình và chuyển động khớp, PSNR 26,0–31,9 dB, trung bình 29,0 dB (repo gốc tự đo 33,4 dB); màu loang nhẹ vì bỏ SH bậc cao theo hướng nhìn và thời gian. Với `--sh` (tính sẵn phần cos theo thời gian vào hệ số SH bậc 3 của từng keyframe, WP13): PSNR 27,3–39,6 dB, trung bình 32,9 dB; file 1,4 GB cho 116.193 x 60 keyframe (trường `sh` luôn float32, `--compact` không nén được phần này; dùng `--frames` nhỏ hơn hoặc `--max-gaussians` để giảm), chuyển mất 8 giây; vẫn thấp hơn 33,4 dB của repo gốc, nguyên nhân chưa tách (nội suy giữa keyframe, lọc `marginal_t` > 0,05 của repo gốc mà converter không làm). Ảnh: `shots/real/native4d_bouncingballs.png` |
| C. Hàm chuyển động tham số | Spacetime Gaussians (`point_cloud.ply`; chỉ màu cơ sở, bỏ MLP `rgbdecoder`; bản đầy đủ có `rgbdecoder` thì converter dừng, thêm `--base-color-only` để chuyển bằng màu cơ sở) — chạy với model thật `n3d_sear_steak_lite_allcam` do tác giả công bố | `python convert_spacetime.py point_cloud.ply -o scene.npz --frames 50 --t0 0 --t1 1`. Đã đo: 108.317 Gaussian trong ply, giữ 98.993 x 50 keyframe, 199 MB, chuyển mất 3 giây. Render gsplat từ hai camera huấn luyện (`cameras.json`) ra đúng cảnh, người và vật chuyển động theo thời gian. Model lite này chỉ có `f_dc_0..2` và không kèm `rgbdecoder`; model đầy đủ (có `rgbdecoder`) chưa thử.; model lite thứ hai (Technicolor `Birthday`, `ours_lite`, 658.999 Gaussian, không có `rgbdecoder`): `--frames 50 --compact` giữ 596.601 Gaussian, 583 MB, 20 giây; render gsplat bằng camera 0 của `cameras.json` ở t = 0, 0,5, 1 cho cảnh đúng và người chuyển động; KHÔNG có ảnh gốc nên không có PSNR. **PSNR (WP19c phần 2, 16/11/2026):** so với video gốc `sear_steak` của Neural 3D Video (21 camera × frame 0, 12, 25, 37, 49 = 105 ảnh, 1352×1014 = độ phân giải lúc train, thời gian = frame/50): 27,76–37,81 dB, **trung bình 34,37 dB**. **Đối chiếu với renderer của chính repo (WP31, 16/11/2026):** `test_ours_lite` + rasteriser CUDA `forward_lite` của repo trên đúng 105 ảnh đó cho 27,79–38,20 dB, trung bình 34,48 dB (`tests/repo_psnr.py spacetime`); file của ta thấp hơn 0,11 dB vì `--min-peak-opacity 0.02` bỏ Gaussian gần như trong suốt (để 0 thì khớp tới 0,01 dB). Trước 16/11/2026 converter cắt màu từng Gaussian về [0, 1] và chỉ đạt 31,43 dB: model lưu màu ngoài khoảng đó ở 30 % Gaussian (−5 đến 6,9), repo trộn màu thô rồi mới cắt ở pixel; nay converter giữ màu thô (renderer client vẫn cắt). File `.npz` Spacetime chuyển trước ngày này cần chuyển lại (`gs4d.open` tự chuyển lại khi converter mới hơn cache). Mục `spacetime_psnr` trong `tests/check_real.py` (ngưỡng min 27,26) |
| C. | Shape of Motion (`checkpoints/last.ckpt`; motion bases SE(3) + hệ số, nền tĩnh; cần torch để đọc checkpoint) — chạy với checkpoint thật `paper-windmill` do tác giả công bố | `python convert_shape_of_motion.py last.ckpt -o scene.npz --fps 30` (`--fg-only` bỏ nền, `--stride n` lấy thưa frame). Đã đo với `--stride 8`: 71.867 Gaussian tiền cảnh + 924.075 nền x 35 keyframe (277 frame gốc), 147 MB (nền tĩnh chỉ lưu một lần; trước khi có nhóm tĩnh là 1,27 GB), chuyển mất 10 giây, RAM đỉnh 3,2 GB. Render gsplat từ camera huấn luyện (`Ks`, `w2cs` trong checkpoint) khớp cảnh, ảnh ở `shots/real/`. **PSNR (WP19c phần 1, 16/11/2026):** so với ảnh gốc 1x (720×960) của bộ iPhone DyCheck tại camera huấn luyện, mỗi 4 frame (70 ảnh): tại keyframe trung bình **25,86 dB** (min 24,28); ở frame nằm giữa hai keyframe của `--stride 8` trung bình 24,67 dB (min 21,61). Với `--stride 1` (mọi frame là keyframe, 416 MB khi `--compact`) cùng 70 ảnh đó cho 25,85 dB ở mọi nơi: mức giảm 1,2 dB là do nội suy thẳng giữa các keyframe thưa của cối xay đang quay. **Đối chiếu với renderer của chính repo (WP31, 16/11/2026):** `flow3d.scene_model.SceneModel.render` của repo trên cùng 70 ảnh cho 24,28–27,66 dB, trung bình 25,85 dB; file `--stride 1` của ta lệch 0,000 dB ở từng ảnh, file `--stride 8` lệch 0,000 dB tại keyframe (`tests/repo_psnr.py som`; repo chạy với gsplat 1.5.3 của `.venv`, không phải bản repo ghim). Mục `som_psnr` trong `tests/check_real.py` (ngưỡng min 21,11) |
| D. Mạng biến dạng | 4DGaussians (hustvl): thư mục model có `point_cloud/iteration_*/{point_cloud.ply,deformation.pth}` và `cfg_args`; chạy lại HexPlane + MLP biến dạng trên CPU (cần torch), không cần môi trường của repo gốc. Chạy với output thật tự train: `bouncingballs` của D-NeRF (20.000 vòng, 692 s trên RTX 3060, config `arguments/dnerf/bouncingballs.py`, nền trắng): 27.043 Gaussian, `--frames 60 --sh` cho file 64 MB, chuyển mất 4 giây. So với ảnh test (gsplat, 800 px, 20 ảnh, ghép nền trắng): PSNR 36,7–45,9 dB, trung bình 40,8 dB (repo gốc tự đo 39,7 dB ở vòng 14.000). So với `export_perframe_3DGS.py` chạy trên cùng model: vị trí lệch tối đa 1,2e-6, covariance lệch tương đối 7e-6, opacity giống hệt, tại 20 thời điểm test. Không hỗ trợ `no_grid`, `empty_voxel`, `static_mlp`, `no_dshs=False` (dừng và nêu tên). Cảnh thật HyperNeRF `slice-banana` (tự train bằng reader HyperNeRF của repo với `arguments/hypernerf/banana.py`, 14.000 vòng, 18 phút, ảnh 4x 268×480, 297 ảnh train / 33 ảnh giữ lại): 52.741 Gaussian, `--frames 330 --sh --compact` cho file 325 MB, chuyển mất 35 giây, RAM đỉnh 1,5 GB. PSNR trên 33 ảnh giữ lại (gsplat, 268 px): 22,05–30,39 dB, trung bình 27,13 dB; repo tự render cùng 33 ảnh: 22,05–30,39 dB, trung bình 27,13 dB (từng ảnh lệch tối đa 0,003 dB). Không phải sửa converter. Cách cũ vẫn dùng được: `export_perframe_3DGS.py` rồi dòng A | `python convert_4dgaussians.py <model> -o scene.npz --frames 60` (thêm `--sh` để giữ SH) |
| D. | Deformable-3DGS (thư mục model có `point_cloud/` và `deform/`; chạy lại MLP biến dạng trên CPU, cần torch; nhận cả biến thể D-NeRF và 6-DoF) — chạy với model thật tự train trên cảnh `bouncingballs` của D-NeRF (biến thể D-NeRF, 20.000 vòng, 34 phút trên RTX 3060); biến thể mặc định (không `--is_blender`, không timenet) cũng đã chạy với model tự train: PSNR 35,35–43,14 dB, trung bình 39,62 dB trên 20 ảnh test ở 800 px, converter không phải sửa; Cảnh thật không phải D-NeRF: HyperNeRF `slice-banana` (video một camera, 297 ảnh train, 33 ảnh giữ lại, ảnh 4x = 268x480, 20.000 vòng, 73 phút, 392.000 Gaussian): PSNR 19,28–29,78 dB, trung bình 24,72 dB trên 33 ảnh giữ lại ở 268 px (bỏ qua méo ống kính như loader của repo; ảnh giữ lại nằm giữa các ảnh train theo thời gian); thời gian của loader đã nằm trong [0, 1], converter không phải sửa. Dữ liệu dựng bằng `tests/make_hypernerf_split.py`; `--frames 330 --sh --compact` cho đúng 330 thời điểm của video; biến thể 6-DoF (`--is_6dof`, cũng tự train, 20.000 vòng, 27 phút) 36,99–43,75 dB, trung bình 40,34 dB, converter không phải sửa | `python convert_deformable3dgs.py <thư_mục_model> -o scene.npz --frames 60` (thêm `--sh` để giữ màu theo hướng nhìn). Đã đo với `--sh`: 64.571 Gaussian x 60 keyframe, SH bậc 3, 153 MB, chuyển mất 49 giây. Render ở camera và thời điểm của cả 20 ảnh test, ở 800 px (độ phân giải lúc train): PSNR 37,2–43,6 dB, trung bình 40,3 dB (ở 400 px chỉ còn 29,3–33,2 dB trên 5 ảnh, vì model train ở 800 px). Ảnh (400 px): `shots/real/deformable3dgs_bouncingballs.png` |
| D. | SC-GS (thư mục model `<tên>_node` có `point_cloud/` và `deform/`; control node + MLP + nội suy theo K node gần nhất, chạy trên CPU, cần torch; không hỗ trợ `--use_hash`, `--skinning`; model train với `--pred_color` (WP30): đầu màu mặc định của repo được chạy lại, màu và hệ số SH DC ghi theo từng keyframe; `--ignore-pred-color` giữ màu cố định cho file nhỏ hơn; đầu màu kiểu hash hoặc theo hướng nhìn thì converter dừng. Model thật `bouncingballs` train với `--pred_color` (20k vòng, 400 px): 35,62–43,75 dB, trung bình 40,70 trên 20 ảnh test; đầu màu của model này gần như còn ở giá trị khởi tạo (|d_color| ≤ 2,3e-4) nên bỏ đầu màu cũng cho đúng số đó: công thức màu mới chỉ được kiểm bằng test tổng hợp) — chạy với model thật tự train trên cảnh `bouncingballs` của D-NeRF (512 node, `--local_frame`, 20.000 vòng sau 10.000 bước khởi tạo node, 19 phút trên RTX 3060) Cảnh thật không phải D-NeRF: HyperNeRF `slice-banana` (loader Nerfies của repo, ảnh 4x = 268x480, 297 ảnh train, 33 ảnh giữ lại, 512 node, hyper_dim 8, 20.000 vòng, 52 phút, 603.000 Gaussian; không `--is_blender`): PSNR 20,21–30,22 dB, trung bình 26,69 dB ở 268 px (`--frames 110 --sh --compact`, thời điểm ảnh giữ lại nằm giữa các keyframe; bỏ qua méo ống kính); với `--frames 330` (mỗi ảnh giữ lại nằm đúng một keyframe; converter ghi từng keyframe ra đĩa từ WP19d, RAM đỉnh 1,4 GB thay cho 4,5 GB ở 110 keyframe trước đó, 3,5 phút, file 3,7 GB): 21,27–30,05 dB, trung bình 26,87 dB; thời gian của loader đã nằm trong [0, 1], converter không phải sửa. Cần sửa `np.float` -> `np.float64` trong bản clone SC-GS để train. | `python convert_scgs.py <thư_mục_model> -o scene.npz --frames 60` (`--knn` = `--K` lúc train nếu khác 3; thêm `--sh` để giữ màu theo hướng nhìn). Đã đo: 165.900 Gaussian x 60 keyframe, chuyển mất 10 giây; 361 MB, với `--sh` (SH bậc 3) 393 MB. Render ở camera và thời điểm của cả 20 ảnh test, ở 400 px (độ phân giải lúc train): PSNR 35,4–43,0 dB, trung bình 40,3 dB với `--sh`; 27,8–33,5 dB trên 5 ảnh khi chỉ dùng màu cơ sở. Ảnh: `shots/real/scgs_bouncingballs.png`; model train thêm `--gs_with_motion_mask --gt_alpha_mask_as_dynamic_mask` (ply có `fea_0..8`, converter nhận ra mask): `bouncingballs` tự train 20.000 vòng, 20 phút, 170.091 Gaussian, PSNR 35,48–43,14 dB, trung bình 40,14 dB ở 400 px, converter không phải sửa |

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
đo ở phía server; tốc độ hiển thị trong Chrome có GPU xem bảng ở mục "Giải mã và nội suy trên GPU khi play" bên dưới (WP19b).

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

### Camera huấn luyện và ảnh gốc (`--cameras`)

```bash
python viewer.py bouncingballs.npz --cameras ../data_real/dnerf/data/bouncingballs/transforms_test.json
```

`--cameras` nhận `transforms_*.json` (NeRF / D-NeRF / instant-ngp, camera OpenGL, tự lật cột 1 và 2 của c2w) hoặc
một `cameras.json` (`{"convention": "opencv"|"opengl", "frames": [{"c2w": [[4x4]], "fx" hoặc "fov_x"/"fov_y", "w", "h",
"time", "image", "name"}]}`). Bảng điều khiển có mục **Cameras**: chọn camera trong danh sách (hoặc bấm
`◀ previous camera` / `next camera ▶`) thì góc nhìn nhảy tới camera đó và thanh thời gian nhảy tới thời điểm của ảnh nếu
file có `time`. Với renderer gsplat, ô **Ground truth beside render** vẽ ảnh render (trái) và ảnh gốc (phải) của camera
đang chọn, kèm **PSNR** của frame đang xem. Các điểm cần biết:

- PSNR tính ở kích thước ảnh gốc, bị chặn bởi **Max render height** (đặt 400 cho model huấn luyện ở 400 px, ví dụ SC-GS
  và 4DGS gốc; kích thước khác kích thước huấn luyện làm PSNR tụt vài dB). Công thức giống `tests/check_real.py`:
  ảnh RGBA ghép lên nền đen (ô **Image background** đổi sang trắng cho model huấn luyện nền trắng như 4DGaussians),
  resize Lanczos. Hai thanh trượt Display (ngưỡng opacity, kích thước Gaussian) không áp dụng cho phép so này, nên
  số hiển thị là của chính model. `tests/check_cameras.py`: lệch so với `check_real.py` tối đa 0,0002 dB trên 20 ảnh test
  của SC-GS (400 px) và Deformable-3DGS (800 px).
- Nếu thanh thời gian rời thời điểm của ảnh thì bảng ghi rõ hai bên lệch thời gian (PSNR vẫn tính, nhưng không còn là
  so cùng một thời điểm).
- Cạnh nhau chỉ có ở renderer gsplat; renderer client vẫn có danh sách camera và nhảy camera/thời điểm.
- Cùng `--compare` (WP29): chế độ `A | B` hiện ba ô **render A | ảnh gốc | render B**, bảng ghi PSNR của cả hai (`A: 38.32 dB · B: 32.39 dB`);
  chế độ `A` hoặc `B` hiện hai ô như khi chỉ có một file. Camera lấy từ `--cameras`, nên hai file phải cùng hệ toạ độ. `tests/check_cameras.py`:
  PSNR của A và B bằng số của viewer một file (lệch ≤ 1e-3 dB trên 20 ảnh, SC-GS và Deformable-3DGS `bouncingballs`). Ảnh chụp `shots/cameras/ground_truth_compare.png`.
- Dataset kiểu Nerfies / HyperNeRF (`camera/<id>.json`, `metadata.json`, `rgb/<N>x/`; WP27): `python -m gs4d.open <thư mục model>` của 4DGaussians,
  Deformable-3DGS hoặc SC-GS đọc `source_path` trong `cfg_args`, ghi các camera giữ lại (`val_ids`; không có thì mọi camera) ra
  `<tên>.gs4d.cameras.json` cạnh output và truyền cho viewer. 4DGaussians giữ toạ độ gốc và thời gian = `warp_id`/max; hai phương pháp kia
  dùng `(position - center) * scale` của `scene.json` và `time_id`/max; ảnh lấy ở thư mục `rgb/<N>x` nhỏ nhất, bỏ qua méo ống kính.
  Đã kiểm trên `slice-banana`: PSNR qua danh sách camera này của 4DGaussians là 22,05–30,39 dB, trung bình 27,13 dB (bằng `check_real.py`);
  camera của ba model khớp loader của `check_real.py` tới 3e-7.
- `python -m gs4d.open` tự truyền `--cameras` khi tìm thấy `cameras.json` đọc được (có `c2w`) hoặc `transforms_test.json` của dataset trong `cfg_args`; tự gõ `--cameras` thì giữ của bạn; có `--compare` vẫn truyền (WP29). Ảnh chụp: `shots/cameras/`.

## Xem dạng điểm và di chuyển camera

Trong bảng điều khiển, mục **Display** có ô **Draw as**: `splats` (vẽ Gaussian như trước) hoặc `points` (mỗi Gaussian là
một điểm tại tâm của nó, mang màu của Gaussian) và thanh **Point size**. Mở thẳng ở chế độ điểm: `python viewer.py scene.npz --draw-as points`
(`python -m gs4d.open <output> --draw-as points` cũng được). Chế độ điểm dùng được khi đang play, với `--compare`, khi xuất mp4, ở cả hai renderer.
Thanh **Hide opacity below** vẫn áp dụng: Gaussian dưới ngưỡng không có điểm. Thanh "Gaussian size (x)" không ảnh hưởng tới điểm.

- Renderer gsplat: điểm được rasterize bằng chính gsplat, mỗi tâm là một Gaussian tròn, đục, có đường kính **Point size (px)** pixel
  trên màn hình ở mọi độ sâu (sigma = cỡ điểm / 2,355 × độ sâu / tiêu cự). Chọn cách này thay cho gửi point cloud sang trình duyệt vì
  không phải truyền một triệu điểm mỗi frame, điểm gần che điểm xa đúng thứ tự, màu theo hướng nhìn (`sh`) vẫn đúng, và chia đôi
  màn hình của `--compare` cùng xuất mp4 chạy không cần code riêng.
- Renderer client: point cloud của viser (`add_point_cloud`). viser tính cỡ điểm theo đơn vị của cảnh nên thanh là
  **Point size (radius / 1000)** (phần nghìn bán kính cảnh), điểm ở xa nhỏ hơn; màu là màu cơ sở `colors`; vẫn giới hạn `--max-splats` điểm.
- Ô "Ground truth beside render" (`--cameras`) luôn vẽ splats, vì PSNR là của model.

Tốc độ khi play ở chế độ điểm, renderer gsplat, 1280×720, RTX 3060 (thời gian một frame ở phía server: giải mã + rasterize + chép ảnh về CPU;
`tests/check_points.py` in ra; hai tới ba lần chạy, 16/11/2026):

| File | Gaussian | `splats` | `points` (3 px) |
|---|---|---|---|
| `../data_real/scgs/bouncingballs_sh.npz` | 165.900 × 60 keyframe | 12–15 ms | 10–11 ms (khoảng 90 frame/giây) |
| `../data_real/som/paper-windmill.npz` | 995.942 × 35 keyframe | 27–40 ms | 24–27 ms (khoảng 40 frame/giây) |

Chế độ điểm không chậm hơn splats. Chưa đo số frame trình duyệt thực sự hiện ở chế độ điểm (WP19b đo cho splats), và chưa đo tốc độ
của renderer client trên GPU thật (test chạy WebGL phần mềm). Lệnh đo đã sẵn (WP28, cần màn hình không khoá):
`python tests/bench_display.py --draw-as points --rates 20 30 40 50 60` và thêm `--size 1920 1080`. Lần thử 16/11/2026 màn hình đang khoá
nên phía trình duyệt không hợp lệ; riêng phía server gửi 19,9 / 58,8 frame/giây ở 20 / 60 Hz (`paper-windmill`, 1280×720, chế độ điểm). Ảnh chụp: `shots/points/` (3 thời điểm × 2 renderer cho hai file trên và cảnh toy).

Di chuyển camera (viser 1.1.1, đã thử từng thao tác trong Chrome và đọc lại vị trí camera):

| Thao tác | Kết quả |
|---|---|
| Kéo chuột trái | xoay quanh điểm đang nhìn |
| Kéo chuột phải | trượt camera ngang/dọc (điểm đang nhìn đi theo) |
| Lăn chuột, hoặc kéo chuột giữa | lại gần / ra xa điểm đang nhìn |
| `W` / `S` | tiến / lùi theo phương ngang (không đổi độ cao) |
| `A` / `D` | sang trái / phải |
| `Q` / `E` | xuống / lên theo trục up của cảnh |
| Phím mũi tên | xoay quanh điểm đang nhìn (trái/phải, lên/xuống) |

Phím chỉ có tác dụng khi con trỏ nằm trên vùng cảnh (không phải trên bảng điều khiển). Trục up đặt bằng `--up`.

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
Thêm phương pháp mới: viết một script tạo `Sequence4D(...)` rồi `.save()`. Cảnh dài không vừa RAM: dùng `Sequence4DWriter`
(`gs4d/format.py`): trường hằng theo thời gian đưa vào lúc tạo, các trường còn lại đưa vào `add_keyframe(t, ...)` từng keyframe;
RAM đỉnh không phụ thuộc K, file ra giống hệt `save` từng bit (cả dạng nén và nhóm tĩnh). Lúc ghi cần thêm chỗ đĩa tạm
≈ N × K × 40 byte cạnh file ra (thư mục `.gs4d_writer_*`, tự xoá; còn sót lại nếu tiến trình bị kill). `convert_scgs.py`,
`convert_native4d.py`, `convert_deformable3dgs.py` và (WP22) `convert_4dgaussians.py`, `convert_shape_of_motion.py`,
`convert_spacetime.py`, `convert_per_frame.py` dùng cách này; Deformable-3DGS `slice-banana` 330 keyframe: RAM đỉnh 7,2 → 1,0 GB.

RAM đỉnh của bốn converter WP22 (`/usr/bin/time -v`, cùng input, file ra giống code cũ từng bit ở mọi mảng):

| Input thật | Lệnh | RAM đỉnh trước → sau |
|---|---|---|
| 4DGaussians `bouncingballs` | `--frames 60 --sh --compact` | 538 → 473 MB |
| 4DGaussians `slice-banana` | `--frames 330 --sh --compact` | 1,49 → 0,60 GB |
| Shape of Motion `paper-windmill` | `--stride 8 --compact` (35 keyframe) | 3,2 → 0,88 GB |
| Shape of Motion `paper-windmill` | `--compact`, cả 277 frame | không chạy được (cần > 11 GB, làm sập app) → 0,88 GB, 69 giây, file 416 MB |
| Spacetime `n3d_sear_steak` lite | `--compact` | 590 → 102 MB |
| Spacetime Technicolor `Birthday` lite | `--compact` | 3,28 → 0,44 GB |
| ply theo frame (4DGaussians `gaussian_pertimestamp`) | `--sh --compact` / không cờ | 300 → 97 MB / 97 → 53 MB |

Spacetime đọc opacity hai lượt (lượt đầu chỉ tính đỉnh opacity để lọc `--min-peak-opacity`, lượt hai ghi). `convert_per_frame.py`:
folder ply có số Gaussian đổi theo frame (`interpolate=False`, đệm Gaussian ẩn) giữ đường cũ trong RAM, vì Writer cần N cố định và
biết trước; params.npz của Dynamic 3D Gaussians vẫn nạp cả file nhưng ghi covariance từng frame (không có output thật để đo, kiểm bằng
dữ liệu tổng hợp: 8 tổ hợp giống code cũ từng bit).

## File lớn: lưu nén và nạp lười

```bash
python -m gs4d scene.npz scene_small.npz             # lưu lại ở dạng nén (có mất mát nhỏ)
python -m gs4d scene_small.npz scene.npz --float32   # trả về float32 thường
```

Trong code: `seq.save(path, compact=True)`; mặc định `compact=False` (float32, không mất mát). `Sequence4D.load`
đọc được cả hai dạng và mọi file cũ. Dạng nén chỉ đổi các trường thay đổi theo thời gian: `means` thành
float16 theo offset so với keyframe đầu, `covs` thành thừa số Cholesky lượng tử hóa int16 (giải mã luôn ra ma
trận nửa xác định dương), `opacities` và `colors` thành uint8, `sh` theo keyframe thành uint8 (mục dưới). Viewer nạp lười (`load(path, lazy=True)`, ánh xạ
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

### `sh` theo keyframe trong dạng nén

Trước đây dạng nén giữ `sh` ở float32. Native 4DGS với `--sh` ghi `sh` theo từng keyframe (`[K,N,16,3]`), nên gần như
toàn bộ file là `sh`. Bây giờ mỗi hệ số của mỗi Gaussian được lượng tử hóa thành uint8 trên khoảng giá trị của chính nó
theo thời gian: file có `sh_lo [N,S,3]` (giá trị nhỏ nhất qua các keyframe), `sh_step [N,S,3]` (= (lớn nhất − nhỏ nhất) / 255)
và `sh_u8 [K,N,S,3]`; `sh = sh_lo + sh_u8 · sh_step`. `sh` hằng theo thời gian vẫn là float32. `Sequence4DWriter`, nạp lười
và `GpuFrames` đều đọc/ghi dạng này; file cũ (nén, `sh` float32) vẫn đọc được, `python -m gs4d cũ.npz mới.npz` chuyển sang
dạng mới. File có `sh_u8` không đọc được bằng bản viewer cũ. `sh` có giá trị không hữu hạn thì giữ float32.

Bước 0, phân bố `sh` trên file thật (12 keyframe rải đều; "khoảng theo thời gian" là lớn nhất − nhỏ nhất của một hệ số
của một Gaussian):

| | Bậc 0: khoảng giá trị | \|x\| ở phân vị 99 | khoảng theo thời gian, trung vị / lớn nhất | Bậc 1–3: khoảng giá trị | \|x\| ở phân vị 99 | khoảng theo thời gian, trung vị / lớn nhất |
|---|---|---|---|---|---|---|
| `bouncingballs` | −2,10 … 4,40 | 1,88 | 0,026 / 0,36 | −0,72 … 0,64 | 0,14–0,21 | 0,035–0,040 / 0,62 |
| `slice-banana` | −2,09 … 6,79 | 2,76 | 0,028 / 0,47 | −0,94 … 0,94 | 0,16–0,18 | 0,032–0,038 / 0,95 |

Một hệ số đổi theo thời gian ít hơn nhiều so với độ lớn của nó, nên lượng tử hóa trên khoảng riêng chính xác hơn hẳn
thang chung. Sai số RMS của hệ số bậc 0 trên `slice-banana`: int8 một thang cho mỗi bậc 1,6·10⁻²; int8 một thang cho mỗi
hệ số của mỗi Gaussian 2,3·10⁻³; float16 2,3·10⁻⁴ (file chỉ nhỏ đi một nửa); uint8 trên khoảng riêng theo thời gian
5,3·10⁻⁵ (bậc 1–3: 6–8·10⁻⁵, lớn nhất 1,8·10⁻³).

Đo với output thật của native 4DGS (15/11/2026, `convert_native4d.py --sh --compact`, trước = code cũ, sau = code mới,
cùng checkpoint; PSNR bằng code của `tests/check_real.py`, nạp lười; lấy frame bằng `python tests/bench_large.py --file`):

| | `slice-banana` 330 keyframe, trước | sau | `bouncingballs` 60 keyframe, trước | sau |
|---|---|---|---|---|
| Gaussian | 297.745 | 297.745 | 116.193 | 116.193 |
| Dung lượng file | 19.858 MB | 5.824 MB (0,29 lần) | 1.412 MB | 453 MB (0,32 lần) |
| PSNR trung bình (nhỏ nhất – lớn nhất) | 24,2687 dB (21,220–26,119), 33 ảnh | 24,2687 dB (21,220–26,119) | 32,9117 dB (27,281–39,594), 20 ảnh | 32,9117 dB (27,281–39,593) |
| PSNR giảm nhiều nhất ở một ảnh | — | 0,0002 dB | — | 0,0004 dB |
| Lấy frame khi play, đường GPU (`GpuFrames`) | 2,2 ms | 1,4 ms | 1,1 ms | 0,8 ms |
| Lấy frame khi play, đường CPU (`--cpu-decode`) | 60 ms | 128 ms | 29 ms | 49 ms |
| Thời gian chuyển / RAM đỉnh | 4 phút 58 giây / 1,3 GB | 4 phút 04 giây / 1,5 GB | 10 giây / 0,8 GB | 14 giây / 0,8 GB |

Mục tiêu (dung lượng không quá 1/3, PSNR giảm không quá 0,1 dB) đạt ở cả hai. Đường CPU chậm hơn vì phải giải mã `sh`
bằng numpy ở mỗi frame (trước đây chỉ đọc float32 từ đĩa); renderer gsplat mặc định dùng đường GPU, renderer client không
dùng `sh`. Lúc ghi, `Sequence4DWriter` vẫn cần đĩa tạm cho `sh` float32 (25 GB với `slice-banana` 330 keyframe).

### Giải mã và nội suy trên GPU khi play (renderer gsplat)

Trước đây mỗi frame khi play: `at(t)` giải mã và nội suy bằng numpy, đổi covariance sang 3×3, lọc opacity, rồi chép
mọi thứ lên GPU. Bước 0 (`python tests/bench_large.py`, ms cho một frame ở 500.000 Gaussian × 100 keyframe, nén):
chép khối lượng đĩa 3, giải mã 41, nội suy 10, chép lên GPU 4; cộng phần còn lại của đường CPU (covariance 3×3, mặt nạ
opacity) cả frame là 100 ms. Thời gian nằm ở giải mã trên CPU và ở các bước numpy sau `at(t)`, không ở đọc đĩa.

Bây giờ `gs4d.GpuFrames` (trong `gs4d/format.py`) giữ trên GPU các keyframe quanh thời điểm hiện tại (bộ nhớ đệm 3 keyframe
mỗi trường, dạng float32): mảng thô (float16/int16/uint8) được chép lên một keyframe một lần khi thanh thời gian
sang khoảng mới, giải mã và nội suy bằng torch. Trường không đổi và nhóm tĩnh (WP11) chép lên GPU một lần lúc khởi tạo.
Renderer client và đường CPU không đổi; `viewer.py --cpu-decode` buộc dùng đường CPU để so sánh.

Thời gian lấy đầu vào gsplat cho một frame khi play (`viewer.frame_cpu` / `viewer.frame_gpu`, trung bình 31 frame qua
2 keyframe, RTX 3060; máy dùng chung với phiên khác nên số CPU dao động ±20%):

| | CPU trước | GPU sau |
|---|---|---|
| 500.000 × 100, float32 | 63 ms | 2,3 ms |
| 500.000 × 100, nén | 100 ms (`at(t)` 48 ms) | 2,2 ms |
| `paper-windmill` (995.942 × 35), nén + tách tĩnh | 214 ms (`at(t)` 11 ms) | 3,1 ms |
| `paper-windmill`, float32 + tách tĩnh | 204 ms | 3,2 ms |
| `paper-windmill`, file gốc 1271 MB | 225 ms | 4,6 ms |

Mục tiêu "lấy frame ≤ 20 ms ở dạng nén" đạt (2–3 ms). Ảnh render khác đường CPU tối đa 1/255 mỗi kênh (0/255 ở
`paper-windmill` nén; 1/255 ở scene ngẫu nhiên do làm tròn float32). `tests/check_real.py` chạy lại và chạy thêm với
`Sequence4D.at` thay bằng `GpuFrames`: cả hai cho PSNR y hệt (Deformable-3DGS 37,20–43,56 dB, SC-GS 35,35–42,99, native 4DGS 26,04–31,94).

Số trên là phía server (một frame lấy dữ liệu, chưa kể rasterize và mã hóa JPEG).

Tốc độ hiển thị trong Chrome có GPU (WP19b, 15/11/2026): `paper-windmill` nén + tách tĩnh (995.942 Gaussian × 35 keyframe),
render 1280×720, đang play, Chrome 157.0.8103.50 trên cùng máy (RTX 3060, WebGL qua ANGLE/OpenGL), màn hình 120 Hz, đo 10 giây
mỗi dòng bằng `python tests/bench_display.py --rates 20 30 40 50 60`:

| "Update rate" | Server gửi (frame/giây) | Trình duyệt hiện frame mới (frame/giây) | Trình duyệt vẽ lại (frame/giây) |
|---|---|---|---|
| 20 Hz (mặc định) | 20,0 | 20,0 | 120 |
| 30 Hz | 29,9 | 30,0 | 120 |
| 40 Hz | 39,9 | 39,9 | 119 |
| 50 Hz | 49,5 | 49,3 | 119 |
| 60 Hz | 54,3 | 52,9 | 117 |

Trình duyệt hiện mọi frame server gửi; giới hạn là phía server (khoảng 54 frame/giây ở 1280×720 với scene này). "Frame mới" đếm số
lần trang nạp một ảnh 1280×720 vào texture WebGL; cột cuối là tốc độ `requestAnimationFrame`. Chưa đo ở 1920×1080 và trên máy khác qua SSH.

### Gaussian tĩnh chỉ lưu một lần

Khi lưu, `Sequence4D.save` tự tìm các Gaussian không đổi qua mọi keyframe ở mọi trường (sai khác so với keyframe
đầu không quá 10⁻⁶ lần độ lớn của giá trị) và lưu chúng một lần thay vì K lần; `save(path, split_static=False)`
tắt việc này. Người gọi không phải biết: `at(t)` vẫn trả về đủ N Gaussian theo đúng thứ tự cũ, `viewer.py` không
sửa dòng nào. File cũ vẫn đọc được; `python -m gs4d cũ.npz mới.npz --float32` chuyển file cũ sang dạng mới không
mất mát (bỏ `--float32` để nén thêm, thêm `--no-split` để ghi theo kiểu cũ). File dạng mới không đọc được bằng
bản viewer cũ.

Đo với output thật (14/11/2026, nạp lười, `at(t)` là trung bình 39 thời điểm nằm giữa các keyframe):

| | Shape of Motion `paper-windmill`, trước | sau | sau + nén | Deformable-3DGS `bouncingballs`, trước | sau |
|---|---|---|---|---|---|
| Gaussian tĩnh / tổng | — | 924.075 / 995.942 | 924.075 / 995.942 | — | 0 / 64.571 |
| Dung lượng file | 1271 MB | 147 MB | 103 MB | 153 MB | 153 MB |
| RAM sau khi nạp hết (không lười) | 1248 MB | 177 MB | 135 MB | 181 MB | 181 MB |
| RAM sau khi play hết dải thời gian (nạp lười) | 1248 MB | 205 MB | 161 MB | 166 MB | 167 MB |
| Lấy một frame nội suy (`at(t)`) | 34 ms | 8,7 ms | 10,9 ms | 1,7 ms | 1,6 ms |

`bouncingballs` không có Gaussian tĩnh nào vì MLP biến dạng của Deformable-3DGS dịch mọi Gaussian một chút, nên
file giữ nguyên. Ảnh gsplat render từ file trước và sau: `bouncingballs` giống hệt ở cả 3 thời điểm; `paper-windmill`
(715 × 983, camera huấn luyện) giống hệt tại keyframe, còn giữa hai keyframe có 0–7 trên 702.845 pixel lệch quá
1/255 (lớn nhất 8,6/255). Nguyên nhân đã kiểm: bản cũ nội suy cả Gaussian tĩnh với chính nó nên giá trị bị làm
tròn lệch tới 5·10⁻⁷; thay các giá trị đó bằng giá trị gốc thì ảnh cũ trùng ảnh mới từng pixel.

## Giới hạn hiện tại

- Màu phụ thuộc hướng nhìn chỉ có khi file có trường `sh` và dùng renderer gsplat; renderer client luôn
  vẽ bằng `colors` (SH bậc 0). `convert_per_frame.py --sh` (thư mục ply), `convert_deformable3dgs.py --sh` và `convert_scgs.py --sh` ghi trường này.
- Dung lượng ≈ N × K × 40 byte cho phần thay đổi theo thời gian (≈ 19 byte ở dạng nén); giảm `--frames` hoặc
  `--stride`, hoặc lưu nén như mục trên nếu file lớn. Mọi `convert_*.py` có cờ `--compact` (lưu nén ngay khi chuyển); `python -m gs4d.open` tự thêm cờ này khi file ước tính lớn hơn 1 GB. Với Deformable-3DGS `bouncingballs` (có `--sh`): 153 → 84 MB.
- Số Gaussian thay đổi giữa các frame (feed-forward): viewer nhảy theo keyframe gần nhất, không nội suy.

## Kiểm tra

Một lệnh chạy mọi `tests/test_*.py` và `tests/check_*.py` (lần lượt từng file), in bảng qua / bỏ qua (kèm lý do: không CUDA, không Chrome, không torch, không dữ liệu) / hỏng; mã thoát 1 nếu có file hỏng:

```bash
python tests/run_all.py            # thêm vài từ để chỉ chạy file có tên chứa chúng: run_all.py sh static
```

Từng lệnh riêng:

```bash
python tests/test_all.py                                  # 4 phép kiểm, không cần GPU
python tests/test_all.py /path/to/4d-gaussian-splatting   # thêm so khớp với hàm gốc của repo
python tests/test_sh.py                                   # trường sh; phần render tự bỏ qua nếu không có CUDA
python tests/test_spacetime.py                            # converter Spacetime Gaussians
python tests/test_deformable3dgs.py                       # converter Deformable-3DGS (cần torch, không thì tự bỏ qua)
python tests/test_shape_of_motion.py                      # converter Shape of Motion (cần torch, không thì tự bỏ qua)
python tests/test_scgs.py                                 # converter SC-GS (cần torch, không thì tự bỏ qua)
python tests/test_compact.py                              # lưu nén, nạp lười
python tests/test_static.py                               # nhóm Gaussian tĩnh lưu một lần
python tests/test_writer.py                               # ghi từng keyframe (Sequence4DWriter) == ghi một lần, từng bit
python tests/check_real.py                              # cần CUDA + dữ liệu thật (data_real, gs4d_train): PSNR có ngưỡng trên ảnh gốc của mọi method, kể cả Spacetime lite (`spacetime_psnr`, cần video `sear_steak`) và Shape of Motion (`som_psnr`, cần ảnh iPhone DyCheck)
python tests/repo_psnr.py som                           # WP31, không nằm trong run_all: PSNR từ renderer của chính repo Shape of Motion / Spacetime (cần clone repo, xem đầu file)
python tests/check_gsplat.py                              # cần CUDA + Chrome: so ảnh gsplat với client ở 3 thời điểm
python tests/check_cameras.py                            # cần CUDA, ../data_real, Chrome: --cameras và PSNR hiển thị so với check_real
python tests/check_points.py                              # cần Chrome: "Draw as: points" ở cả hai renderer (gsplat tự bỏ qua nếu không có CUDA); `--file` để chạy trên file thật
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
