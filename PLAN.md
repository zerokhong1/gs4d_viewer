# Kế hoạch và phân việc — gs4d_viewer

Cập nhật 14/11/2026. Việc chia thành 8 gói (WP) và giao cho 3 phiên Claude Code, mở trong thư mục `gs4d_viewer`.
Phiên nào cũng tự đọc `CLAUDE.md` nên không phải giải thích lại bối cảnh.

## Trạng thái (14/11/2026)

Cả 8 gói đã đạt điều kiện xong và đã merge vào `main`. Phần còn lại của file là kế hoạch gốc, giữ để tra cứu.

| WP | Trạng thái | Nhánh | Ghi chú |
|---|---|---|---|
| 1 | Xong | `cc-a/wp1-gsplat` | Bảng FPS trong README (RTX 3060) |
| 2 | Xong | `cc-b/wp2-real-data` | Output thật `basketball` của Dynamic 3D Gaussians; ảnh ở `shots/wp2/` |
| 3 | Xong | `cc-c/wp3-spacetime` | Đã chạy với model thật `n3d_sear_steak_lite_allcam` (bản lite, không có `rgbdecoder`) |
| 4 | Xong | `cc-c/wp4-deformable3dgs` | 3 biến thể (mặc định, D-NeRF, 6-DoF); đã chạy biến thể D-NeRF với model thật tự train (`bouncingballs`) |
| 5 | Xong | `cc-c/wp5-shape-of-motion`, `cc-c/wp5-scgs` | Làm cả hai phương pháp; đã chạy với checkpoint thật `paper-windmill` (Shape of Motion) và model tự train `bouncingballs` (SC-GS) |
| 6 | Xong | `cc-a/wp6-sh` | |
| 7 | Xong | `cc-a/wp7-video`, `cc-a/wp7-compare` | Làm cả (a) so sánh hai sequence và (b) xuất mp4 |
| 8 | Xong | `cc-a/wp8-large` | Bảng trước/sau trong README |

Còn mở, nằm ngoài điều kiện xong của các gói:

- Đã chạy với output thật (14/11/2026): cả bốn converter của WP3–5 và native 4DGS; số đo ở bảng "Dùng với output
  thật" trong README, ảnh so với ảnh gốc ở `shots/real/`. Chưa thử: model Spacetime Gaussians bản đầy đủ (có
  `rgbdecoder`), Deformable-3DGS biến thể mặc định và 6-DoF, SC-GS với motion mask, cảnh thật (không phải D-NeRF)
  cho Deformable-3DGS, SC-GS và native 4DGS.
- Câu hỏi cho giáo sư bên dưới chưa có câu trả lời được ghi lại; WP3–5 đã được làm mà không chờ.
- Tốc độ hiển thị trong trình duyệt thật chưa đo (số trong README là phía server).
- Danh sách đầy đủ những gì đã và chưa kiểm nằm ở mục Status của `CLAUDE.md`.

## Phân công: 3 phiên Claude Code

| Phiên | Vai trò | Máy | Model | Việc, theo đúng thứ tự | File được sửa (chỉ phiên này) |
|---|---|---|---|---|---|
| **CC-A** | Render và giao diện | máy có GPU NVIDIA | mạnh (Opus/Fable) | WP1 → WP6 → WP7 → WP8 | `viewer.py`, `gs4d/format.py`, `gs4d/ply.py`, `make_toy.py`, phần thân README |
| **CC-B** | Dữ liệu thật | máy có file output | Sonnet | WP2 | `convert_per_frame.py`, `convert_native4d.py`, `tests/test_all.py` |
| **CC-C** | Converter mới | bất kỳ, CPU là đủ | Sonnet | WP3 → WP4 → WP5, chỉ những gói lab cần | chỉ file mới: `convert_<tên>.py`, `tests/test_<tên>.py` |

Quy ước chung:

- Phiên nào chỉ sửa file trong cột của mình. Cần thay đổi ở file của phiên khác thì ghi vào báo cáo, không tự sửa.
- CC-B và CC-C chỉ được thêm đúng một dòng vào bảng "Dùng với output thật" trong README cho mỗi converter.
- CC-A và CC-B bắt đầu cùng lúc. CC-C chỉ mở sau khi giáo sư trả lời câu hỏi bên dưới.
- CC-A giữ cùng một phiên cho cả 4 gói để không phải nạp lại bối cảnh về `viewer.py`; giữa hai gói thì merge nhánh rồi mới làm tiếp.
- Nếu CC-B phát hiện output thật cần đổi định dạng chung (ví dụ thiếu trường), CC-B dừng và báo; thay đổi đó thuộc CC-A.
- Nhánh git: `cc-a/wp1-gsplat`, `cc-b/wp2-real-data`, `cc-c/wp3-spacetime`, ... CC-B và CC-C làm trong worktree riêng.

## Trước khi mở CC-C: một câu hỏi cho giáo sư

**Lab dùng phương pháp 4DGS nào, và file output nằm ở đâu?**

Đây là chỗ tiết kiệm lớn nhất. WP3–WP5 mỗi gói phục vụ một phương pháp; viết converter cho phương pháp
lab không dùng là lãng phí hoàn toàn. Chỉ WP1 và WP2 là chắc chắn cần.

## Bảng gói việc

| WP | Việc | File được sửa | Cần | Phụ thuộc | Khi nào làm |
|---|---|---|---|---|---|
| 1 | Chạy và sửa renderer gsplat | `viewer.py` (chỉ phần gsplat) | GPU NVIDIA + CUDA | không | Ngay |
| 2 | Chạy trọn vẹn với một output thật | converter tương ứng | file output thật | câu trả lời của giáo sư | Ngay khi có file |
| 3 | Converter Spacetime Gaussians (dạng C) | `convert_spacetime.py`, test mới | CPU | không | Chỉ khi lab dùng |
| 4 | Converter Deformable-3DGS (dạng D) | `convert_deformable3dgs.py`, test mới | CPU + torch | không | Chỉ khi lab dùng |
| 5 | Converter Shape of Motion / SC-GS | file converter mới, test mới | CPU + torch | không | Chỉ khi lab dùng |
| 6 | Màu SH phụ thuộc hướng nhìn | `gs4d/format.py`, `viewer.py` | GPU | WP1 xong | Sau demo đầu |
| 7 | Tính năng demo: so sánh 2 sequence cạnh nhau, xuất video | `viewer.py` | GPU nếu dùng gsplat | WP1 xong | Khi giáo sư yêu cầu |
| 8 | File lớn: nén float16/lượng tử hóa, đọc lười | `gs4d/format.py` | CPU | WP6 xong hoặc bỏ WP6 | Khi file > vài GB |

## Thứ tự chạy

- **Đợt 1 (song song, tối đa 3 phiên):** WP1 + WP2 + nhiều nhất một trong WP3/4/5.
  Ba gói này không sửa chung file nào.
- **Đợt 2 (tuần tự, từng phiên một):** WP6 → WP7 → WP8. Cả ba cùng sửa `viewer.py` hoặc `format.py`,
  chạy song song sẽ xung đột và phải làm lại.

## Luật chống lãng phí

1. **Một file, một chủ.** `viewer.py` và `gs4d/format.py` mỗi lúc chỉ một phiên được sửa. Converter chỉ
   thêm file mới.
2. **Git trước.** Chạy `git init && git add -A && git commit -m "baseline"` một lần. Mỗi WP một nhánh
   `wp<N>-<tên>`, xong thì merge. Hai phiên không dùng chung working tree: dùng `git worktree add ../gs4d-wp3 -b wp3-spacetime`.
3. **Không nghiên cứu lại.** API của viser, gsplat và cấu trúc checkpoint đã ghi trong `CLAUDE.md`.
4. **Chọn model theo việc.** Converter (WP3–5) là việc port công thức có test rõ ràng: dùng model rẻ hơn
   (Sonnet). WP1, WP6, WP7 đụng tới render và giao diện: dùng model mạnh hơn.
5. **Dừng đúng lúc.** Mỗi prompt bên dưới có điều kiện xong. Phiên đạt điều kiện thì báo cáo 5 dòng và dừng,
   không tự mở rộng phạm vi.
6. **Thiếu dữ liệu thì dừng, không đoán.** Nếu không có GPU hoặc không có file output thật, phiên phải báo
   ngay ở lượt đầu thay vì viết thêm code không kiểm được.

## Prompt cho từng phiên

Dán nguyên khối vào một phiên Claude Code mới mở trong `gs4d_viewer`.

### WP1 — renderer gsplat
```
Đọc CLAUDE.md. Việc: làm cho `python viewer.py toy.npz --renderer gsplat` chạy đúng trên máy này.
Bước 0: kiểm tra `nvidia-smi` và `python -c "import torch, gsplat; print(torch.cuda.is_available())"`.
Nếu không có CUDA thì báo và dừng.
Chỉ sửa phần gsplat trong viewer.py (_draw_gsplat và chỗ khởi tạo liên quan). Không sửa GUI, format, converter.
Xong khi: (1) toy.npz render đúng, so bằng mắt với renderer client qua ảnh chụp headless ở cùng thời điểm t;
(2) kéo thanh thời gian và xoay camera đều cập nhật ảnh; (3) ghi vào README số FPS đo được với toy.npz
và độ phân giải đã đo. Báo cáo 5 dòng rồi dừng.
```

### WP2 — chạy với output thật
```
Đọc CLAUDE.md. Output thật nằm ở: <ĐƯỜNG DẪN>. Phương pháp: <TÊN>.
Việc: chuyển output này sang định dạng chung bằng converter có sẵn và mở được trong viewer.
Nếu converter báo lỗi vì định dạng thật khác với giả định trong code, sửa converter đó (và test của nó),
không sửa viewer.py hay gs4d/format.py. Nếu phương pháp chưa có converter thì báo và dừng (đó là WP3–5).
Xong khi: có file .npz, viewer mở được, có ảnh chụp ở 3 thời điểm, và README ghi lệnh đã dùng cùng
dung lượng file, số Gaussian. Ghi rõ điều gì trông sai (màu, hướng trục up, tỉ lệ). Báo cáo 5 dòng rồi dừng.
```

### WP3 — Spacetime Gaussians
```
Đọc CLAUDE.md. Việc: viết convert_spacetime.py đọc point_cloud.ply của Spacetime Gaussians
(oppo-us-research/SpacetimeGaussians) và xuất Sequence4D.
Clone repo để ĐỌC công thức (chuyển động đa thức theo thời gian, xoay theo thời gian, opacity theo RBF thời gian)
và tên thuộc tính trong ply; không chạy code của repo. Chỉ thêm: convert_spacetime.py, tests/test_spacetime.py,
một dòng trong bảng README. Màu: dùng thành phần màu cơ sở, bỏ phần phụ thuộc hướng nhìn và ghi rõ giới hạn này.
Xong khi: test tạo ply giả đúng layout, chạy converter, và so vị trí/opacity tại 3 thời điểm với công thức
viết độc lập trong test; `python tests/test_all.py` vẫn qua. Báo cáo 5 dòng rồi dừng.
```

### WP4 — Deformable-3DGS
```
Đọc CLAUDE.md. Việc: viết convert_deformable3dgs.py đọc point_cloud.ply + deform.pth của Deformable-3DGS
(ingra14m/Deformable-3D-Gaussians) và xuất Sequence4D ở K mốc thời gian.
Clone repo để ĐỌC kiến trúc MLP biến dạng (positional encoding, số lớp, skip, đầu ra d_xyz/d_rotation/d_scaling)
rồi viết lại forward pass độc lập bằng torch trên CPU, nạp trọng số từ state_dict; không import code của repo.
Chạy theo lô để không hết RAM. Chỉ thêm: convert_deformable3dgs.py, tests/test_deformable3dgs.py, một dòng README.
Xong khi: test dựng MLP ngẫu nhiên cùng kiến trúc, lưu state_dict + ply giả, chạy converter, và so với
forward pass tham chiếu viết trong test; `python tests/test_all.py` vẫn qua. Báo cáo 5 dòng rồi dừng.
```

### WP5 — Shape of Motion hoặc SC-GS
```
Đọc CLAUDE.md. Phương pháp: <Shape of Motion | SC-GS>. Việc và ràng buộc giống WP4: đọc repo để port công thức
(motion bases SE(3) + hệ số, hoặc control node + LBS), viết convert_<tên>.py + tests/test_<tên>.py + một dòng README,
không import code của repo, không sửa file khác. Xong khi test so khớp với công thức viết độc lập và
`python tests/test_all.py` vẫn qua. Báo cáo 5 dòng rồi dừng.
```

### WP6 — màu SH
```
Đọc CLAUDE.md. Điều kiện: WP1 đã merge. Việc: thêm trường tùy chọn `sh` [N,(d+1)^2,3] vào Sequence4D
(file cũ không có trường này vẫn đọc được), cho renderer gsplat dùng sh_degree khi có; renderer client vẫn dùng `colors`.
Cập nhật gs4d/ply.py và convert_per_frame.py để giữ f_rest_* khi có cờ --sh. Thêm test round-trip.
Xong khi: toy có SH giả đổi màu theo hướng nhìn trong renderer gsplat, và mọi test cũ vẫn qua. Báo cáo 5 dòng rồi dừng.
```

### WP7 — tính năng demo
```
Đọc CLAUDE.md. Điều kiện: WP1 đã merge. Việc: <CHỌN MỘT: (a) mở 2 file .npz và bật/tắt hoặc đặt cạnh nhau,
chung thanh thời gian; (b) nút xuất video mp4 theo quỹ đạo camera hiện tại qua toàn bộ thời gian>.
Chỉ sửa viewer.py. Xong khi tính năng chạy với toy.npz, có ảnh chụp hoặc file video làm bằng chứng,
và README có một đoạn hướng dẫn. Báo cáo 5 dòng rồi dừng.
```

### WP8 — file lớn
```
Đọc CLAUDE.md. Việc: giảm dung lượng và RAM cho sequence lớn. Đo trước: dung lượng và thời gian nạp của một
sequence 500k Gaussian x 100 keyframe sinh ngẫu nhiên. Sau đó thêm lưu means dạng float16 theo offset so với
keyframe đầu và covs/opacities lượng tử hóa, có cờ tắt; file cũ vẫn đọc được. Chỉ sửa gs4d/format.py và thêm test.
Xong khi README có bảng trước/sau (dung lượng, thời gian nạp, sai số vị trí lớn nhất) và mọi test qua.
Báo cáo 5 dòng rồi dừng.
```
