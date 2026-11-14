# Kế hoạch giai đoạn 2 — gs4d_viewer

Cập nhật 14/11/2026. Giai đoạn 1 (WP1–WP8) đã xong và đã merge vào `main`; kế hoạch gốc nằm ở phụ lục cuối file.
Phiên Claude Code nào cũng tự đọc `CLAUDE.md` nên không phải giải thích lại bối cảnh.

## Làm ngay

1. **Dọn worktree cũ** (người làm, 2 phút). `gs4d-wp2` còn giữ `PLAN.md` và `README.md` bản cũ, dễ mở nhầm.
   ```bash
   git -C ../gs4d-wp2 status          # phải sạch; nếu còn thay đổi thì commit hoặc bỏ trước
   git worktree remove ../gs4d-wp2
   git branch --merged main           # xoá các nhánh cc-*/wp* đã merge
   ```
2. **Demo cho giáo sư và chốt câu hỏi bên dưới** (WP9, người làm). Mọi gói ở đợt 2 chờ câu trả lời này.
3. ~~Mở đợt 1~~: WP10, WP11, WP12 đã xong và đã merge vào `main` (14/11/2026).

## Câu hỏi cho giáo sư (WP9)

Ghi câu trả lời vào đây, kèm ngày. Giai đoạn 1 đã làm WP3–5 mà không chờ; giai đoạn 2 không lặp lại việc đó.

| Câu hỏi | Trả lời | Ngày |
|---|---|---|
| Lab dùng phương pháp 4DGS nào? (đặc biệt: có dùng 4DGaussians của hustvl không) | | |
| File output nằm ở đâu, cảnh nào, bao nhiêu Gaussian? | | |
| Xem trên máy nào: máy có GPU NVIDIA hay laptop qua SSH? | | |
| Cần gì thêm khi xem: so với ảnh gốc, camera huấn luyện, xuất video theo quỹ đạo? | | |

Mang theo khi demo: `python viewer.py ../data_real/scgs/bouncingballs_sh.npz --renderer gsplat`, chế độ `--compare`,
và bảng "Dùng với output thật" trong README.

## Trạng thái giai đoạn 2

| WP | Việc | Trạng thái | Nhánh | Ghi chú |
|---|---|---|---|---|
| 9 | Demo + chốt câu hỏi với giáo sư | Chưa | — | Người làm |
| 10 | Lệnh mở thẳng một output | Xong | `claude/practical-wiles-00a752` | `python -m gs4d.open <output>`; phiên WP10 báo mở được cả 6 output thật |
| 11 | Tách Gaussian tĩnh và động trong định dạng | Xong | `cc-a/wp11-static` | `paper-windmill` 1271 → 147 MB; `viewer.py` không sửa. Ảnh trước/sau lệch quá 1/255 ở tối đa 7 pixel giữa hai keyframe (do làm tròn của bản cũ) |
| 12 | Kiểm tự động với output thật (PSNR có ngưỡng) | Xong | `claude/friendly-feynman-9aa271` | Mỗi model đo ở độ phân giải train; 20 ảnh test; Spacetime và Shape of Motion chỉ kiểm số Gaussian và dung lượng |
| 13 | Native 4DGS: màu SH theo keyframe | Chờ WP9 | `cc-c/wp13-native4d-sh` | Chỉ khi lab dùng native 4DGS |
| 14 | Giải mã và nội suy trên GPU khi play | Chưa | `cc-a/wp14-gpu-decode` | Sau WP11 |
| 15 | Converter trực tiếp cho 4DGaussians (hustvl) | Chờ WP9 | `cc-c/wp15-4dgaussians` | Chỉ khi lab dùng |
| 16 | Biến thể chưa thử của phương pháp lab dùng | Chờ WP9 | `cc-c/wp16-<tên>` | Chỉ biến thể lab dùng |
| 17 | Cờ `--compact` cho converter, một lệnh chạy mọi test | Xong | `claude/wp17-housekeeping-56ac95` | `python tests/run_all.py`: 13 file qua (check_real chạy riêng sau khi sửa dò dữ liệu trong worktree); `--compact` ở cả 6 converter, `gs4d.open` tự dùng khi ước tính > 1 GB |
| 18 | Camera huấn luyện và ảnh gốc trong viewer | Chờ WP9 | `cc-a/wp18-cameras` | Chỉ khi giáo sư yêu cầu; sau WP14 |

## Phân công: 3 phiên Claude Code

| Phiên | Vai trò | Model | Việc, theo đúng thứ tự | File được sửa (chỉ phiên này) |
|---|---|---|---|---|
| **CC-A** | Định dạng và render | mạnh (Opus/Fable) | WP11 → WP14 → (WP18) | `viewer.py`, `gs4d/format.py`, `gs4d/__main__.py`, `tests/test_compact.py`, `tests/bench_large.py`, phần thân README |
| **CC-B** | Công cụ dùng hằng ngày | Sonnet | WP10 → WP17 | file mới `gs4d/open.py`, `tests/test_open.py`, `tests/run_all.py`; ở WP17: phần đối số dòng lệnh của các `convert_*.py` |
| **CC-C** | Kiểm chứng và converter | Sonnet | WP12 → (WP13, WP15, WP16 theo câu trả lời) | file mới `tests/check_real.py`; `convert_native4d.py` (WP13); file mới `convert_4dgaussians.py`, `tests/test_4dgaussians.py` (WP15) |

Quy ước chung (giữ từ giai đoạn 1, sửa chỗ đã lệch):

- Phiên nào chỉ sửa file trong cột của mình. Cần thay đổi ở file của phiên khác thì ghi vào báo cáo, không tự sửa.
- Nhánh đặt theo `cc-<phiên>/wp<N>-<tên>`. CC-B và CC-C làm trong worktree riêng (`git worktree add ../gs4d-wp<N> -b <nhánh>`),
  merge xong thì xoá worktree ngay.
- Cả ba phiên dùng chung một GPU (RTX 3060) và 15 GB RAM: không train; lệnh render hoặc đo PSNR chạy từng lệnh một.
- WP17 sửa các `convert_*.py`, nên chỉ chạy khi CC-C không có gói converter nào đang mở.
- Gói ghi "Chờ WP9" không được mở trước khi bảng câu hỏi ở trên có câu trả lời.

## Thứ tự chạy

- **Đợt 1 (song song, không chờ giáo sư):** WP10 + WP11 + WP12. Ba gói không sửa chung file nào và có ích cho mọi phương pháp.
- **Đợt 2 (sau WP9):** CC-A làm WP14; CC-B làm WP17; CC-C làm những gói trong WP13/15/16 mà câu trả lời yêu cầu, từng gói một.
- **Đợt 3 (khi được yêu cầu):** WP18.

## Điều kiện xong, áp dụng cho mọi gói

Giai đoạn 1 chỉ yêu cầu test tổng hợp, nên việc chạy với output thật phải làm bù sau. Từ giờ:

1. Gói đụng tới converter hoặc định dạng chỉ xong khi đã chạy với **ít nhất một output thật** trong `../data_real`
   và ghi số đo (PSNR so với ảnh gốc, dung lượng, thời gian) vào README.
2. Gói đổi định dạng hoặc renderer phải chạy lại `tests/check_real.py` (sau khi WP12 có) và không làm PSNR giảm quá 0,2 dB.
3. Mọi test cũ vẫn qua; file `.npz` cũ vẫn đọc được.
4. Cập nhật dòng của gói trong bảng trạng thái ở trên và mục Status của `CLAUDE.md`.
5. Báo cáo 5 dòng rồi dừng, không tự mở rộng phạm vi. Thiếu dữ liệu hoặc GPU thì báo ngay ở lượt đầu, không đoán.

## Prompt cho từng phiên

Dán nguyên khối vào một phiên Claude Code mới mở trong `gs4d_viewer`.

### WP10 — lệnh mở thẳng một output
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Việc: viết gs4d/open.py để `python -m gs4d.open <đường_dẫn_output> [đối số của viewer]` tự nhận dạng loại output,
gọi đúng converter, rồi mở viewer. Không sửa converter hay viewer.py: gọi chúng qua subprocess.
Nhận dạng: `params.npz` hoặc thư mục ply theo frame -> convert_per_frame; `chkpnt*.pth` -> convert_native4d;
`last.ckpt` hoặc thư mục có `checkpoints/last.ckpt` -> convert_shape_of_motion; thư mục có `point_cloud/` và `deform/`
-> convert_scgs nếu ply có cột `fea_*`, ngược lại convert_deformable3dgs; `point_cloud.ply` có thuộc tính chuyển động
của Spacetime Gaussians -> convert_spacetime; file `.npz` đã ở định dạng chung -> mở thẳng.
File chuyển xong lưu cạnh output (`<tên>.gs4d.npz`) và dùng lại nếu output không đổi (so mtime). Không nhận dạng được
thì in các dấu hiệu đã kiểm rồi thoát với mã lỗi, không đoán. Cảnh D-NeRF: tự đặt khoảng thời gian [0, 1].
Chỉ thêm: gs4d/open.py, tests/test_open.py, một đoạn "Mở thẳng một output" ở đầu README.
Xong khi: test dựng layout giả của từng loại và kiểm nhận dạng đúng; và cả 6 output thật trong ../data_real
mở được bằng đúng một lệnh (liệt kê lệnh và kết quả trong báo cáo). Báo cáo 5 dòng rồi dừng.
```

### WP11 — tách Gaussian tĩnh và động
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Vấn đề: trường nào có trục K thì mọi Gaussian đều bị lưu ở mọi keyframe. Shape of Motion `paper-windmill` có
71.867 Gaussian tiền cảnh động và 924.075 Gaussian nền tĩnh, file 1,27 GB.
Việc: cho Sequence4D lưu một nhóm Gaussian tĩnh (không có trục K) bên cạnh nhóm động, trong cùng một file;
`at(t)` trả về cả hai nhóm ghép lại nên viewer.py không phải biết. Khi lưu, tự phát hiện Gaussian không đổi qua mọi
keyframe (mọi trường, sai khác trong ngưỡng float32) và đưa vào nhóm tĩnh; có cờ tắt. Dạng nén và nạp lười vẫn chạy.
File cũ vẫn đọc được; `python -m gs4d in.npz out.npz` chuyển file cũ sang dạng mới.
Chỉ sửa gs4d/format.py, gs4d/__main__.py, tests/test_compact.py (hoặc thêm tests/test_static.py); viewer.py chỉ sửa
nếu không thể tránh, và ghi rõ lý do.
Xong khi: test round-trip với scene giả có cả hai nhóm; README có bảng trước/sau cho ../data_real/som/paper-windmill.npz
và ../data_real/d3dgs/bouncingballs.npz (dung lượng, RAM sau khi nạp, thời gian `at(t)`); ảnh gsplat render từ file
trước và sau tại 3 thời điểm khác nhau không quá 1/255 mỗi kênh. Báo cáo 5 dòng rồi dừng.
```

### WP12 — kiểm tự động với output thật
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Việc: viết tests/check_real.py gom các phép so đã làm tay ngày 14/11/2026 thành một lệnh. Với mỗi output thật
có ảnh gốc trong ../data_real (Deformable-3DGS, SC-GS, native 4DGS trên `bouncingballs`): chạy converter, render bằng
gsplat ở camera và thời điểm của TOÀN BỘ ảnh test (không chỉ 5 ảnh), tính PSNR từng ảnh, in bảng (nhỏ nhất, trung bình,
lớn nhất), và trả mã lỗi nếu PSNR nhỏ nhất thấp hơn ngưỡng. Ngưỡng ban đầu = số nhỏ nhất đo được lần này trừ 0,5 dB,
ghi thành hằng số ở đầu file kèm ngày đo. Với Spacetime Gaussians và Shape of Motion (chưa có so PSNR): dùng camera
huấn luyện trong `cameras.json` / checkpoint và ảnh gốc nếu có trong ../data_real; nếu không có ảnh gốc thì chỉ kiểm
converter chạy được, số Gaussian và dung lượng khớp README, và ghi rõ là chưa so ảnh.
Không có CUDA hoặc thiếu thư mục dữ liệu thì bỏ qua từng mục và in lý do, không báo qua.
Chỉ thêm: tests/check_real.py, một dòng trong mục "Kiểm tra" của README. Không sửa converter; converter sai thì ghi
vào báo cáo.
Xong khi: lệnh chạy hết trên máy này, bảng PSNR dán vào báo cáo, và số trong README khớp với bảng (nếu lệch thì ghi
số nào lệch, không tự sửa README ngoài dòng của mình). Báo cáo 5 dòng rồi dừng.
```

### WP13 — native 4DGS: màu SH theo keyframe (chỉ khi lab dùng)
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md. Điều kiện: WP12 đã merge.
Vấn đề: convert_native4d.py chỉ ghi màu SH bậc 0; PSNR trên `bouncingballs` là 26,0–31,1 dB, repo gốc tự đo 33,4 dB.
Việc: thêm cờ --sh. Đọc repo fudan-zvg/4d-gaussian-splatting để port cách màu phụ thuộc hướng nhìn và thời gian;
tại mỗi keyframe, tính sẵn phần phụ thuộc thời gian và ghi hệ số SH 3D còn lại vào trường `sh` [K,N,S,3].
Không import code của repo. Chỉ sửa convert_native4d.py, thêm tests/test_native4d_sh.py, sửa dòng native 4DGS
trong README và ngưỡng tương ứng trong tests/check_real.py.
Xong khi: test so màu tại 3 thời điểm và 3 hướng nhìn với công thức viết độc lập; tests/check_real.py cho PSNR nhỏ nhất
trên `bouncingballs` tăng so với trước. Mục tiêu trung bình >= 32 dB; không đạt thì báo nguyên nhân còn lại
(ví dụ lọc theo marginal_t, nội suy giữa keyframe) thay vì chỉnh cho khớp. Báo cáo 5 dòng rồi dừng.
```

### WP14 — giải mã và nội suy trên GPU khi play
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md. Điều kiện: WP11 đã merge.
Vấn đề: `at(t)` ở dạng nén mất 77 ms với 500.000 Gaussian (float32: 14 ms), nên play scene lớn chỉ được khoảng
13 frame/giây trước cả khi rasterize.
Bước 0: đo để biết thời gian nằm ở đâu (đọc đĩa, giải mã, nội suy, chép lên GPU) với tests/bench_large.py; ghi số.
Việc: với renderer gsplat, giữ hai keyframe kề nhau ở dạng tensor trên GPU, giải mã và nội suy bằng torch, chỉ nạp
keyframe mới khi thanh thời gian qua mốc; nhóm tĩnh (WP11) chép lên GPU một lần. Renderer client và đường CPU giữ nguyên.
Chỉ sửa viewer.py, gs4d/format.py, tests/bench_large.py.
Đo thêm tốc độ hiển thị trong trình duyệt thật (Chrome có GPU, không phải headless swiftshader) ở 1280x720 và ghi
cạnh số phía server trong README.
Xong khi: README có bảng trước/sau thời gian một frame khi play (500.000 Gaussian x 100 keyframe, float32 và nén)
và với ../data_real/som/paper-windmill.npz; mục tiêu phần lấy frame <= 20 ms ở dạng nén; ảnh render không đổi quá
1/255 mỗi kênh so với đường CPU. Báo cáo 5 dòng rồi dừng.
```

### WP15 — converter trực tiếp cho 4DGaussians (chỉ khi lab dùng)
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Output thật nằm ở: <ĐƯỜNG DẪN>. Việc: viết convert_4dgaussians.py đọc thư mục model của 4DGaussians (hustvl/4DGaussians:
`point_cloud/iteration_*/point_cloud.ply` + file trọng số mạng biến dạng) và xuất Sequence4D ở K mốc thời gian, để không
phải chạy `export_perframe_3DGS.py` trong môi trường của repo gốc. Ràng buộc giống converter Deformable-3DGS: đọc repo
để port công thức (HexPlane đa độ phân giải + MLP biến dạng vị trí/xoay/scale), viết lại forward pass bằng torch trên CPU,
chạy theo lô, không import code của repo. Chỉ thêm: convert_4dgaussians.py, tests/test_4dgaussians.py, một dòng README,
một mục trong tests/check_real.py.
Xong khi: test tổng hợp so với công thức viết độc lập; và với output thật ở trên, kết quả khớp với ply từng frame do
`export_perframe_3DGS.py` xuất (nếu có sẵn) hoặc PSNR so với ảnh test được ghi vào README. Báo cáo 5 dòng rồi dừng.
```

### WP16 — biến thể chưa thử của phương pháp lab dùng
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Phương pháp và biến thể: <MỘT trong: Spacetime Gaussians bản đầy đủ có rgbdecoder | Deformable-3DGS mặc định | Deformable-3DGS
6-DoF | SC-GS có motion mask | cảnh thật (không phải D-NeRF) cho <tên>>. Output thật nằm ở: <ĐƯỜNG DẪN>.
Việc: chạy converter tương ứng với output này, so với ảnh gốc bằng tests/check_real.py (thêm một mục), sửa converter và
test của nó nếu định dạng thật khác giả định. Không sửa viewer.py hay gs4d/format.py.
Xong khi: README và mục Status của CLAUDE.md ghi PSNR đo được và chuyển biến thể này từ "NOT tested" sang "Tested".
Báo cáo 5 dòng rồi dừng.
```

### WP17 — cờ `--compact` và một lệnh chạy mọi test
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md. Điều kiện: WP10 đã merge; không có gói
converter nào của CC-C đang mở.
Việc: (1) thêm cờ --compact vào mọi convert_*.py, chỉ truyền `compact=True` cho `Sequence4D.save`; không đổi gì khác
trong converter. gs4d/open.py dùng --compact khi file ước tính lớn hơn 1 GB. (2) viết tests/run_all.py chạy mọi
test_*.py và check_*.py, in một bảng qua / bỏ qua (kèm lý do: không CUDA, không Chrome, không dữ liệu) / hỏng,
trả mã lỗi nếu có mục hỏng. Thay danh sách lệnh trong mục "Kiểm tra" của README bằng lệnh này, giữ danh sách cũ bên dưới.
(3) Đối chiếu requirements.txt với các gói README bảo cài (viser, numpy<2, imageio, imageio-ffmpeg, pillow, playwright)
và sửa cho khớp.
Xong khi: `python tests/run_all.py` qua trên máy này, và một converter chạy với --compact cho file đọc được.
Báo cáo 5 dòng rồi dừng.
```

### WP18 — camera huấn luyện và ảnh gốc trong viewer (chỉ khi được yêu cầu)
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md. Điều kiện: WP14 đã merge.
Việc: thêm đối số `--cameras <cameras.json | transforms_*.json>` cho viewer.py: danh sách camera trong bảng điều khiển,
bấm để nhảy tới camera đó (và tới thời điểm của ảnh nếu file có), và với renderer gsplat có ô bật ảnh gốc cạnh ảnh render
kèm PSNR của frame đang xem. Quy ước camera đã ghi trong CLAUDE.md (D-NeRF là OpenGL: lật cột 1 và 2 của c2w).
Chỉ sửa viewer.py; gs4d/open.py tự truyền --cameras khi tìm thấy file (ghi vào báo cáo cho CC-B nếu cần sửa).
Xong khi: chạy với ../data_real/dnerf/data/bouncingballs và một model đã chuyển, có ảnh chụp trong shots/, PSNR hiển thị
khớp tests/check_real.py trong 0,1 dB, README có một đoạn hướng dẫn. Báo cáo 5 dòng rồi dừng.
```

---

# Phụ lục: kế hoạch giai đoạn 1 (đã xong, giữ để tra cứu)

Chỗ đã lệch so với thực tế: luật 2 ghi nhánh `wp<N>-<tên>` nhưng nhánh thật là `cc-<phiên>/wp<N>-<tên>`; quy ước
"CC-C chỉ mở sau khi giáo sư trả lời" đã không được giữ; WP7 ghi "chọn một" nhưng đã làm cả hai.

### Trạng thái (14/11/2026)

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

### Phân công: 3 phiên Claude Code

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

### Trước khi mở CC-C: một câu hỏi cho giáo sư

**Lab dùng phương pháp 4DGS nào, và file output nằm ở đâu?**

Đây là chỗ tiết kiệm lớn nhất. WP3–WP5 mỗi gói phục vụ một phương pháp; viết converter cho phương pháp
lab không dùng là lãng phí hoàn toàn. Chỉ WP1 và WP2 là chắc chắn cần.

### Bảng gói việc

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

### Thứ tự chạy

- **Đợt 1 (song song, tối đa 3 phiên):** WP1 + WP2 + nhiều nhất một trong WP3/4/5.
  Ba gói này không sửa chung file nào.
- **Đợt 2 (tuần tự, từng phiên một):** WP6 → WP7 → WP8. Cả ba cùng sửa `viewer.py` hoặc `format.py`,
  chạy song song sẽ xung đột và phải làm lại.

### Luật chống lãng phí

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

### Prompt cho từng phiên

Dán nguyên khối vào một phiên Claude Code mới mở trong `gs4d_viewer`.

#### WP1 — renderer gsplat
```
Đọc CLAUDE.md. Việc: làm cho `python viewer.py toy.npz --renderer gsplat` chạy đúng trên máy này.
Bước 0: kiểm tra `nvidia-smi` và `python -c "import torch, gsplat; print(torch.cuda.is_available())"`.
Nếu không có CUDA thì báo và dừng.
Chỉ sửa phần gsplat trong viewer.py (_draw_gsplat và chỗ khởi tạo liên quan). Không sửa GUI, format, converter.
Xong khi: (1) toy.npz render đúng, so bằng mắt với renderer client qua ảnh chụp headless ở cùng thời điểm t;
(2) kéo thanh thời gian và xoay camera đều cập nhật ảnh; (3) ghi vào README số FPS đo được với toy.npz
và độ phân giải đã đo. Báo cáo 5 dòng rồi dừng.
```

#### WP2 — chạy với output thật
```
Đọc CLAUDE.md. Output thật nằm ở: <ĐƯỜNG DẪN>. Phương pháp: <TÊN>.
Việc: chuyển output này sang định dạng chung bằng converter có sẵn và mở được trong viewer.
Nếu converter báo lỗi vì định dạng thật khác với giả định trong code, sửa converter đó (và test của nó),
không sửa viewer.py hay gs4d/format.py. Nếu phương pháp chưa có converter thì báo và dừng (đó là WP3–5).
Xong khi: có file .npz, viewer mở được, có ảnh chụp ở 3 thời điểm, và README ghi lệnh đã dùng cùng
dung lượng file, số Gaussian. Ghi rõ điều gì trông sai (màu, hướng trục up, tỉ lệ). Báo cáo 5 dòng rồi dừng.
```

#### WP3 — Spacetime Gaussians
```
Đọc CLAUDE.md. Việc: viết convert_spacetime.py đọc point_cloud.ply của Spacetime Gaussians
(oppo-us-research/SpacetimeGaussians) và xuất Sequence4D.
Clone repo để ĐỌC công thức (chuyển động đa thức theo thời gian, xoay theo thời gian, opacity theo RBF thời gian)
và tên thuộc tính trong ply; không chạy code của repo. Chỉ thêm: convert_spacetime.py, tests/test_spacetime.py,
một dòng trong bảng README. Màu: dùng thành phần màu cơ sở, bỏ phần phụ thuộc hướng nhìn và ghi rõ giới hạn này.
Xong khi: test tạo ply giả đúng layout, chạy converter, và so vị trí/opacity tại 3 thời điểm với công thức
viết độc lập trong test; `python tests/test_all.py` vẫn qua. Báo cáo 5 dòng rồi dừng.
```

#### WP4 — Deformable-3DGS
```
Đọc CLAUDE.md. Việc: viết convert_deformable3dgs.py đọc point_cloud.ply + deform.pth của Deformable-3DGS
(ingra14m/Deformable-3D-Gaussians) và xuất Sequence4D ở K mốc thời gian.
Clone repo để ĐỌC kiến trúc MLP biến dạng (positional encoding, số lớp, skip, đầu ra d_xyz/d_rotation/d_scaling)
rồi viết lại forward pass độc lập bằng torch trên CPU, nạp trọng số từ state_dict; không import code của repo.
Chạy theo lô để không hết RAM. Chỉ thêm: convert_deformable3dgs.py, tests/test_deformable3dgs.py, một dòng README.
Xong khi: test dựng MLP ngẫu nhiên cùng kiến trúc, lưu state_dict + ply giả, chạy converter, và so với
forward pass tham chiếu viết trong test; `python tests/test_all.py` vẫn qua. Báo cáo 5 dòng rồi dừng.
```

#### WP5 — Shape of Motion hoặc SC-GS
```
Đọc CLAUDE.md. Phương pháp: <Shape of Motion | SC-GS>. Việc và ràng buộc giống WP4: đọc repo để port công thức
(motion bases SE(3) + hệ số, hoặc control node + LBS), viết convert_<tên>.py + tests/test_<tên>.py + một dòng README,
không import code của repo, không sửa file khác. Xong khi test so khớp với công thức viết độc lập và
`python tests/test_all.py` vẫn qua. Báo cáo 5 dòng rồi dừng.
```

#### WP6 — màu SH
```
Đọc CLAUDE.md. Điều kiện: WP1 đã merge. Việc: thêm trường tùy chọn `sh` [N,(d+1)^2,3] vào Sequence4D
(file cũ không có trường này vẫn đọc được), cho renderer gsplat dùng sh_degree khi có; renderer client vẫn dùng `colors`.
Cập nhật gs4d/ply.py và convert_per_frame.py để giữ f_rest_* khi có cờ --sh. Thêm test round-trip.
Xong khi: toy có SH giả đổi màu theo hướng nhìn trong renderer gsplat, và mọi test cũ vẫn qua. Báo cáo 5 dòng rồi dừng.
```

#### WP7 — tính năng demo
```
Đọc CLAUDE.md. Điều kiện: WP1 đã merge. Việc: <CHỌN MỘT: (a) mở 2 file .npz và bật/tắt hoặc đặt cạnh nhau,
chung thanh thời gian; (b) nút xuất video mp4 theo quỹ đạo camera hiện tại qua toàn bộ thời gian>.
Chỉ sửa viewer.py. Xong khi tính năng chạy với toy.npz, có ảnh chụp hoặc file video làm bằng chứng,
và README có một đoạn hướng dẫn. Báo cáo 5 dòng rồi dừng.
```

#### WP8 — file lớn
```
Đọc CLAUDE.md. Việc: giảm dung lượng và RAM cho sequence lớn. Đo trước: dung lượng và thời gian nạp của một
sequence 500k Gaussian x 100 keyframe sinh ngẫu nhiên. Sau đó thêm lưu means dạng float16 theo offset so với
keyframe đầu và covs/opacities lượng tử hóa, có cờ tắt; file cũ vẫn đọc được. Chỉ sửa gs4d/format.py và thêm test.
Xong khi README có bảng trước/sau (dung lượng, thời gian nạp, sai số vị trí lớn nhất) và mọi test qua.
Báo cáo 5 dòng rồi dừng.
```
