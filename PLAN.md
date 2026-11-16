# Kế hoạch giai đoạn 2 — gs4d_viewer

Cập nhật 15/11/2026 (15:15). Giai đoạn 1 (WP1–WP8) đã xong; kế hoạch gốc nằm ở phụ lục cuối file. Giai đoạn 2: WP10–WP18, WP19a,
WP19d và WP19c phần 3 đã làm, đã commit và merge vào `main` (13:45). Giáo sư đã trả lời (bảng bên dưới): không còn gói nào phải chờ.
Phiên Claude Code nào cũng tự đọc `CLAUDE.md` nên không phải giải thích lại bối cảnh.

## Làm ngay

1. ~~**WP20**~~ (xong 15/11/2026) (một phiên, cần GPU trống khoảng 30 phút): train lại native 4DGS `slice-banana` với camera đúng. Con số 24,23 dB
   hiện tại đo trên một model train với camera sai. Đây là gói duy nhất còn chặn dòng native 4DGS của WP9.
2. **WP19b: xong 15/11/2026.** `python tests/bench_display.py --rates 20 30 40 50 60` (cửa sổ Chrome có GPU; cần màn hình không khoá và GPU không train).
   `paper-windmill` 1280×720: trình duyệt hiện đủ mọi frame server gửi, 20,0 ở 20 Hz đến 52,9 frame/giây ở 60 Hz (server gửi 54,3). Bảng trong README.
3. **Mở song song với WP20** (không dùng GPU lúc đang train): ~~WP22~~ (CC-B, bốn converter, xong 15/11/2026). WP21 (CC-A) đã xong 15/11/2026, chờ merge nhánh `cc-a/wp21-compact-sh`.
4. **Sau WP20, từng gói một vì chung GPU:** ~~WP23~~ (xong 15/11/2026) → WP19c phần 1 → WP19c phần 2.
5. **Dọn worktree** (người làm). Git còn đăng ký 10 worktree, kể cả `gs4d-wp19c` (đã merge) và `gs4d-wp2` từ giai đoạn 1.
   ```bash
   git worktree list
   git branch --no-merged main        # phải trống trước khi xoá
   git worktree remove ../gs4d-wp19c && git worktree remove ../gs4d-wp2
   git worktree remove <đường dẫn từng worktree claude/* đã merge> && git worktree prune
   ```
6. **Xác nhận lại một ý của giáo sư** trước khi mở WP24: "quan sát các điểm 4DGS" có nghĩa là cần chế độ vẽ tâm Gaussian
   thành điểm hay không (xem ghi chú dưới bảng câu trả lời).

## WP9 — sản phẩm: đọc và xem được mọi file 4DGS

Đổi ngày 14/11/2026: WP9 trước đây là "demo + chốt câu hỏi với giáo sư"; nay là mục tiêu của sản phẩm.

**Mục tiêu.** Đưa một output 4DGS bất kỳ vào `python -m gs4d.open <output>` là xem được với thanh thời gian, không cần
cài môi trường hay chạy code của repo gốc. WP9 không có nhánh riêng: nó xong khi mọi dòng trong bảng dưới ở trạng thái "Xong".

"Mọi file" là các loại trong bảng này. Gặp phương pháp mới thì thêm một dòng, một `convert_<method>.py` và một nhánh
nhận dạng trong `gs4d/open.py`.

| Loại file | Converter | Hiện tại | Còn thiếu | Gói |
|---|---|---|---|---|
| Thư mục ply 3DGS từng frame (mọi phương pháp xuất được) | `convert_per_frame.py` | Xong: `gaussian_pertimestamp/` của 4DGaussians `bouncingballs`, mục `4dgaussians_export` | — | — |
| Dynamic 3D Gaussians `params.npz` | `convert_per_frame.py` | Xong (`basketball`) | — | — |
| Shape of Motion `last.ckpt` | `convert_shape_of_motion.py` | Xong (`paper-windmill`, kiểm bằng mắt; `check_real.py` chỉ kiểm số Gaussian và dung lượng) | Số PSNR so với ảnh gốc (không chặn) | WP19c phần 1 |
| Native 4DGS `chkpnt*.pth` | `convert_native4d.py` | `--sh`: `bouncingballs` 27,3–39,6 dB, trung bình 32,9 (repo gốc 33,4). Cảnh thật `slice-banana` (train lại với camera đúng, WP20): 26,31 dB (110 keyframe), repo gốc tự đo 26,30 | Xong. Gap 1,9 dB cũ là do layout cũ của `make_hypernerf_blender.py` (`FovX = FovY = -1`), đã sửa và train lại. File 330 keyframe: 19,9 → 5,8 GB từ WP21 (`sh` nén uint8) | WP20 |
| 4DGaussians (hustvl), thư mục model | `convert_4dgaussians.py` | Converter xong: `bouncingballs` 36,7–45,9 dB, trung bình 40,8; khớp `export_perframe_3DGS.py` đến 1e-6 | Xong (WP19a): `gs4d.open` nhận thư mục này, mở có danh sách camera. Cảnh thật `slice-banana` (WP23): 22,05–30,39 dB, trung bình 27,13, khớp repo; `gs4d.open` mở được nhưng không có danh sách camera cho layout HyperNeRF | — |
| Spacetime Gaussians `point_cloud.ply` | `convert_spacetime.py` | Hai model lite (`n3d_sear_steak`, Technicolor `Birthday`), kiểm bằng mắt, chưa có PSNR. Bản đầy đủ (`rgbdecoder`): converter dừng và nêu tên, `--base-color-only` để chuyển tiếp (WP19a) | PSNR cho bản lite. Bản đầy đủ là giới hạn đã ghi nhận: không có model công khai, train vượt 15 GB RAM | WP19c phần 2 |
| Deformable-3DGS, thư mục model | `convert_deformable3dgs.py` | Xong: D-NeRF 37,2–43,6 dB; mặc định 35,35–43,14; 6-DoF 36,99–43,75; cảnh thật `slice-banana` 19,28–29,78 (trung bình 24,72) | — | — |
| SC-GS, thư mục model | `convert_scgs.py` | Xong: D-NeRF 35,4–43,0 dB; motion mask 35,48–43,14; cảnh thật `slice-banana` trung bình 26,69 dB (110 keyframe), 26,87 dB (330 keyframe, RAM đỉnh 1,4 GB từ WP19d). `--use_hash`, `--skinning`, tắt `d_rot_as_res`, `--pred_color` (`--ignore-pred-color` để chuyển tiếp): dừng có thông báo | — | — |
| File không thuộc loại nào ở trên | `gs4d/open.py` | Báo đã thử nhận dạng những gì | — | — |

**Điều kiện xong của WP9** (thêm vào "Điều kiện xong, áp dụng cho mọi gói"):

1. Mỗi dòng mở được bằng một lệnh `python -m gs4d.open <output>` trên một output thật, và có một mục trong `tests/check_real.py`.
2. Biến thể mà converter chưa hỗ trợ phải dừng với thông báo nêu tên biến thể, không được cho ra ảnh sai mà không báo.
3. Bảng "Dùng với output thật" trong README liệt kê đủ các dòng trên, kèm số đo.

Không có output thật của một dòng thì tự train trên `bouncingballs` trong `/home/cont/gs4d_train` (từng job một, 15 GB RAM),
như đã làm với Deformable-3DGS, SC-GS và native 4DGS.

### Câu trả lời của giáo sư (15/11/2026)

| Câu hỏi | Trả lời | Ngày | Hệ quả cho kế hoạch |
|---|---|---|---|
| Lab dùng phương pháp 4DGS nào? | Tất cả | 15/11/2026 | Không bỏ phương pháp nào trong bảng WP9. Các gói trước đây ghi "khi lab dùng" đều mở: WP19c phần 1–2, WP21, WP22 |
| File output nằm ở đâu, cảnh nào, bao nhiêu Gaussian? | Chưa có output của lab: tự chạy các phương pháp và dùng output đó | 15/11/2026 | Tiếp tục tự train trong `/home/cont/gs4d_train` và dùng checkpoint tác giả công bố. Thêm WP23 để 4DGaussians cũng có một cảnh thật như ba phương pháp kia. Spacetime bản đầy đủ vẫn là giới hạn (train vượt 15 GB RAM) |
| Xem trên máy nào? | Tạm thời máy này (RTX 3060 12 GB, 15 GB RAM) | 15/11/2026 | Renderer `gsplat` là đường chính; WP19b đo trên Chrome của máy này. Xem qua SSH hoặc trên laptop chưa cần kiểm |
| Cần gì thêm khi xem? | Di chuyển được camera để quan sát các điểm 4DGS | 15/11/2026 | Xem ghi chú dưới |

Ghi chú cho câu cuối. Viewer đã cho di chuyển camera tự do bằng chuột (xoay, kéo, zoom của viser) ngay cả khi đang play, ở cả
hai renderer; nếu chỉ cần vậy thì không phải làm thêm. Viewer chưa có chế độ vẽ tâm Gaussian thành điểm: nếu "các điểm 4DGS"
nghĩa là muốn nhìn thấy từng Gaussian như một đám mây điểm thì đó là WP24 (chưa mở, chờ xác nhận).

Dùng khi demo: `python -m gs4d.open <thư mục output>` (tự chọn renderer `gsplat` trên máy này), chế độ `--compare`,
và bảng "Dùng với output thật" trong README.

## Trạng thái giai đoạn 2

| WP | Việc | Trạng thái | Nhánh | Ghi chú |
|---|---|---|---|---|
| 9 | Sản phẩm: đọc và xem được mọi file 4DGS | Đang làm | — | 7/9 dòng trong bảng WP9 đã xong; còn native 4DGS (WP20) và Spacetime (PSNR bản lite, WP19c phần 2). Giáo sư: lab dùng tất cả các phương pháp |
| 10 | Lệnh mở thẳng một output | Xong | `claude/practical-wiles-00a752` | `python -m gs4d.open <output>`; phiên WP10 báo mở được cả 6 output thật |
| 11 | Tách Gaussian tĩnh và động trong định dạng | Xong | `cc-a/wp11-static` | `paper-windmill` 1271 → 147 MB; `viewer.py` không sửa. Ảnh trước/sau lệch quá 1/255 ở tối đa 7 pixel giữa hai keyframe (do làm tròn của bản cũ) |
| 12 | Kiểm tự động với output thật (PSNR có ngưỡng) | Xong | `claude/friendly-feynman-9aa271` | Mỗi model đo ở độ phân giải train; 20 ảnh test; Spacetime và Shape of Motion chỉ kiểm số Gaussian và dung lượng |
| 13 | Native 4DGS: màu SH theo keyframe | Xong (đã merge) | `cc-c/wp13-native4d-sh` | `--sh`: PSNR `bouncingballs` min 26,04 → 27,28, mean 29,0 → 32,9 dB; file 1,4 GB (116k x 60 keyframe) |
| 14 | Giải mã và nội suy trên GPU khi play | Xong | `claude/wp14-gpu-decode-559de8` (đã merge) | Lấy frame khi play 100 → 2,2 ms (500.000 × 100, nén), 214 → 3,1 ms (`paper-windmill`); ảnh lệch tối đa 1/255; PSNR check_real không đổi. FPS trong Chrome có GPU: đo ở WP19b |
| 15 | Converter trực tiếp cho 4DGaussians (hustvl) | Xong (đã merge) | `cc-c/wp15-4dgaussians` | `bouncingballs` tự train: 36,7–45,9 dB (trung bình 40,8), khớp `export_perframe_3DGS.py` đến 1e-6. `gs4d/open.py` nhận dạng thư mục 4DGaussians từ WP19a |
| 16 | Các biến thể chưa thử (Spacetime đầy đủ, Deformable-3DGS, SC-GS) | Xong, trừ Spacetime đầy đủ (đã merge) | `claude/wp16-variants-8fe2f3`, `claude/wp16-real-scene-5cd4d8` | `bouncingballs`: Deformable-3DGS mặc định 35,35–43,14 dB, 6-DoF 36,99–43,75, SC-GS có motion mask 35,48–43,14. Cảnh thật HyperNeRF `slice-banana` (15/11/2026, 33 ảnh giữ lại, 268 px): Deformable-3DGS trung bình 24,72 dB, SC-GS 26,69, native 4DGS 24,23 (model train với camera sai, xem WP20). Spacetime: thêm model lite `Birthday`, chưa có ảnh gốc; bản đầy đủ (`rgbdecoder`) không làm được trên máy này |
| 17 | Cờ `--compact` cho converter, một lệnh chạy mọi test | Xong | `claude/wp17-housekeeping-56ac95` | `python tests/run_all.py`: 13 file qua (check_real chạy riêng sau khi sửa dò dữ liệu trong worktree); `--compact` ở cả 6 converter, `gs4d.open` tự dùng khi ước tính > 1 GB |
| 18 | Camera huấn luyện và ảnh gốc trong viewer | Xong 14/11/2026 (đã merge) | `claude/wp18-cameras-398403` | `--cameras`, danh sách camera, ảnh gốc + PSNR (gsplat); PSNR khớp check_real 0,0002 dB; không dùng được cùng `--compare`; `gs4d/open.py` tự truyền `--cameras` từ WP19a |
| 19a | `open.py` nhận 4DGaussians và `--cameras`; converter dừng ở biến thể chưa hỗ trợ | Xong 15/11/2026 (đã commit) | làm thẳng trên `main` | `gs4d.open` nhận 4DGaussians và truyền `--cameras` (bỏ qua `cameras.json` kiểu 3DGS vì `load_cameras` không đọc được); SC-GS `--pred_color` và Spacetime đầy đủ dừng, có cờ chuyển tiếp |
| 19b | Đo FPS hiển thị trong Chrome có GPU | Xong 15/11/2026 (chưa commit) | làm thẳng trên `main` | `tests/bench_display.py`; `paper-windmill` 1280×720, Chrome 157, RTX 3060: hiện 20,0 / 30,0 / 39,9 / 49,3 / 52,9 frame/giây ở 20 / 30 / 40 / 50 / 60 Hz, bằng số server gửi; chưa đo 1920×1080 |
| 19c | PSNR cho Spacetime và Shape of Motion; nguyên nhân lệch 1,9 dB của native 4DGS trên cảnh thật | Phần 3 xong 15/11/2026 (đã merge); phần 1, 2 chưa, đã mở | `cc-c/wp19c-psnr` (`../gs4d-wp19c`) | Phần 3: lệch do model train với `FovX = FovY = -1` (layout của `make_hypernerf_blender.py`), không phải converter; giả lập trong gsplat cho 26,13 dB (repo 26,14); `tests/diag_native4d_banana.py`. Phần 1, 2: làm sau WP23 vì chung GPU |
| 19d | Converter ghi từng keyframe ra đĩa để không giữ cả sequence trong RAM | Xong 15/11/2026 (đã commit) | làm thẳng trên `main` | `gs4d.Sequence4DWriter`, `tests/test_writer.py` (bằng `save` từng bit); SC-GS, native 4DGS, Deformable-3DGS dùng nó. RAM đỉnh: SC-GS 4,5 → 1,4 GB, native 4DGS 8,3 → 1,3 GB, Deformable-3DGS 7,2 → 1,0 GB; 330 keyframe của `slice-banana` chuyển được. `check_real.py` vẫn đo ở 110 keyframe |
| 20 | Train lại native 4DGS `slice-banana` với camera đúng, đo lại | Xong 15/11/2026 (chưa merge) | `cc-c/wp20-native4d-banana` | Train lại với `camera_angle_x` (không `fl_x` từng frame), 20.000 vòng, 20 phút: repo tự đo 26,30 dB; converter + gsplat 21,41–29,17, trung bình **26,31 dB** (trước 24,23): gap 1,9 dB hết. `make_hypernerf_blender.py` đã sửa; `check_real.py` trỏ `output/banana_fov`, ngưỡng min 20,91 |
| 21 | Nén trường `sh` theo keyframe | Xong 15/11/2026 (chưa merge) | `cc-a/wp21-compact-sh` | `sh` theo keyframe thành uint8 trên khoảng riêng theo thời gian của từng hệ số (`sh_lo`, `sh_step`, `sh_u8`). Native 4DGS `slice-banana` 330 keyframe: 19,86 → 5,82 GB (0,29 lần), PSNR 24,2687 dB không đổi (ảnh xấu nhất −0,0002 dB); `bouncingballs`: 1412 → 453 MB, 32,9117 dB không đổi. Lấy frame đường GPU 2,2 → 1,4 ms; đường CPU 60 → 128 ms. Còn lại cho CC-B: `estimate_bytes` trong `gs4d/open.py` chưa tính `sh` theo keyframe |
| 22 | `Sequence4DWriter` cho bốn converter còn lại | Chưa | `cc-b/wp22-writer-rest` | 4DGaussians, Shape of Motion, Spacetime, per-frame vẫn dựng cả sequence trong RAM. Đã mở (giáo sư: tất cả phương pháp) |
| 23 | 4DGaussians trên cảnh thật `slice-banana` | Xong 15/11/2026 (chưa commit) | làm thẳng trên `main` | Train 14.000 vòng (config hypernerf của repo), 18 phút, 52.741 Gaussian. PSNR 33 ảnh giữ lại: của ta 22,05–30,39, trung bình 27,13; repo tự render 27,13 (từng ảnh lệch ≤ 0,003 dB). Không sửa converter; mục `4dgaussians_banana` trong `check_real.py`. Camera của repo đã kiểm: FovX/FovY tính từ focal, đúng |
| 24 | Chế độ vẽ tâm Gaussian thành điểm | Chờ xác nhận | `cc-a/wp24-points` | Chỉ làm nếu "quan sát các điểm 4DGS" nghĩa là xem dạng đám mây điểm; di chuyển camera tự do thì đã có |

## Phân công: 3 phiên Claude Code

| Phiên | Vai trò | Model | Việc, theo đúng thứ tự | File được sửa (chỉ phiên này) |
|---|---|---|---|---|
| **CC-A** | Định dạng và render | mạnh (Opus/Fable) | WP11 → WP14 → (WP18) | `viewer.py`, `gs4d/format.py`, `gs4d/__main__.py`, `tests/test_compact.py`, `tests/bench_large.py`, phần thân README |
| **CC-B** | Công cụ dùng hằng ngày | Sonnet | WP10 → WP17 | file mới `gs4d/open.py`, `tests/test_open.py`, `tests/run_all.py`; ở WP17: phần đối số dòng lệnh của các `convert_*.py` |
| **CC-C** | Kiểm chứng và converter | Sonnet | WP12 → WP15 → WP13 → WP16 (từng biến thể) | file mới `tests/check_real.py`; `convert_native4d.py` (WP13); file mới `convert_4dgaussians.py`, `tests/test_4dgaussians.py` (WP15) |

Quy ước chung (giữ từ giai đoạn 1, sửa chỗ đã lệch):

- Phiên nào chỉ sửa file trong cột của mình. Cần thay đổi ở file của phiên khác thì ghi vào báo cáo, không tự sửa.
- Nhánh đặt theo `cc-<phiên>/wp<N>-<tên>`. CC-B và CC-C làm trong worktree riêng (`git worktree add ../gs4d-wp<N> -b <nhánh>`),
  merge xong thì xoá worktree ngay.
- Cả ba phiên dùng chung một GPU (RTX 3060) và 15 GB RAM: không train; lệnh render hoặc đo PSNR chạy từng lệnh một.
- WP17 sửa các `convert_*.py`, nên chỉ chạy khi CC-C không có gói converter nào đang mở.
- Các gói của WP9 (WP13, WP15, WP16) mở được ngay, từng gói một vì chung GPU. Ngoại lệ so với dòng "không train" ở trên:
  gói thiếu output thật được train một model `bouncingballs` trong `/home/cont/gs4d_train`, khi không phiên nào khác dùng GPU.

## Thứ tự chạy

- **Đợt 1 (song song, không chờ giáo sư):** WP10 + WP11 + WP12. Ba gói không sửa chung file nào và có ích cho mọi phương pháp.
- **Đợt 2 (WP9, không chờ giáo sư):** CC-A làm WP14; CC-B đã xong WP17; CC-C làm WP15 → WP13 → WP16, từng gói một.
  WP15 đi trước vì 4DGaussians là phương pháp duy nhất chưa đọc trực tiếp được; câu trả lời của giáo sư có thể đổi thứ tự này.
- **Đợt 3 (khi được yêu cầu):** WP18 (đã làm 14/11/2026).
- **Đợt 4 (15/11/2026):** WP19a, WP19d và WP19c phần 3 đã làm. WP19b xong 15/11/2026 (`tests/bench_display.py`).
- **Đợt 5 (15/11/2026, sau câu trả lời của giáo sư):** WP20 trước. Song song với nó, không dùng GPU khi đang train: WP21 (`gs4d/format.py`)
  và WP22 (bốn converter), hai gói không sửa chung file. Sau WP20, lần lượt vì chung GPU: WP23 → WP19c phần 1 → WP19c phần 2.
  WP21 và WP22 chạy `check_real.py` ở cuối, khi không gói nào đang train. WP24 chờ xác nhận.

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

### WP13 — native 4DGS: màu SH theo keyframe (thuộc WP9)
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

### WP15 — converter trực tiếp cho 4DGaussians (thuộc WP9)
```
Đọc CLAUDE.md và các mục "WP9" và "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Output thật nằm ở: <ĐƯỜNG DẪN>. Nếu chưa có: train hustvl/4DGaussians trên D-NeRF `bouncingballs` trong /home/cont/gs4d_train
(một job, không chạy gì khác trên GPU), rồi chạy `export_perframe_3DGS.py` một lần để có ply từng frame làm đối chứng. Việc: viết convert_4dgaussians.py đọc thư mục model của 4DGaussians (hustvl/4DGaussians:
`point_cloud/iteration_*/point_cloud.ply` + file trọng số mạng biến dạng) và xuất Sequence4D ở K mốc thời gian, để không
phải chạy `export_perframe_3DGS.py` trong môi trường của repo gốc. Ràng buộc giống converter Deformable-3DGS: đọc repo
để port công thức (HexPlane đa độ phân giải + MLP biến dạng vị trí/xoay/scale), viết lại forward pass bằng torch trên CPU,
chạy theo lô, không import code của repo. Chỉ thêm: convert_4dgaussians.py, tests/test_4dgaussians.py, một dòng README,
hai mục trong tests/check_real.py (converter mới, và thư mục ply từng frame qua convert_per_frame.py). Nhánh nhận dạng trong
gs4d/open.py là file của CC-B: ghi vào báo cáo.
Xong khi: test tổng hợp so với công thức viết độc lập; và với output thật ở trên, kết quả khớp với ply từng frame do
`export_perframe_3DGS.py` xuất (nếu có sẵn) hoặc PSNR so với ảnh test được ghi vào README. Báo cáo 5 dòng rồi dừng.
```

### WP16 — các biến thể chưa thử (thuộc WP9; mỗi biến thể một phiên)
```
Đọc CLAUDE.md và các mục "WP9" và "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Phương pháp và biến thể: <MỘT trong: Spacetime Gaussians bản đầy đủ có rgbdecoder | Deformable-3DGS mặc định | Deformable-3DGS
6-DoF | SC-GS có motion mask | SC-GS --use_hash | SC-GS --skinning | SC-GS --pred_color | SC-GS tắt d_rot_as_res |
cảnh thật (không phải D-NeRF) cho <tên>>. Output thật nằm ở: <ĐƯỜNG DẪN>; nếu chưa có thì train biến thể đó trong
/home/cont/gs4d_train (một job).
Việc: chạy converter tương ứng với output này, so với ảnh gốc bằng tests/check_real.py (thêm một mục), sửa converter và
test của nó nếu định dạng thật khác giả định. Biến thể không hỗ trợ được thì converter phải dừng với thông báo nêu tên
biến thể. Không sửa viewer.py hay gs4d/format.py.
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

### WP19a — `open.py` nhận 4DGaussians và `--cameras`; dừng ở biến thể chưa hỗ trợ
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Việc 1: trong gs4d/open.py, thêm nhánh nhận dạng thư mục model của 4DGaussians (hustvl) và gọi convert_4dgaussians.py.
Đọc phần đầu convert_4dgaussians.py để lấy đúng các file nó cần; phân biệt với Deformable-3DGS / SC-GS (có `deform/`)
và Spacetime Gaussians bằng dấu hiệu trong file, không bằng tên thư mục. Thêm trường hợp này vào tests/test_open.py.
Việc 2: gs4d/open.py tự truyền `--cameras` cho viewer khi tìm thấy `cameras.json` trong thư mục model, hoặc
`transforms_test.json` của dataset (đọc đường dẫn dataset từ `cfg_args` nếu có); người dùng tự truyền thì giữ của người dùng.
Không tìm thấy thì không truyền và không báo lỗi.
Việc 3 (điều kiện 2 của WP9): biến thể chưa hỗ trợ phải dừng và nêu tên biến thể.
 - convert_scgs.py: model có đầu màu (`--pred_color`) hiện chỉ in cảnh báo. Đổi thành dừng, trừ khi có cờ `--ignore-pred-color`.
 - convert_spacetime.py: ply của bản đầy đủ (có `f_dc_3..`, hoặc có file `rgbdecoder` cạnh ply) hiện được chuyển bằng màu cơ sở
   mà không báo. Đổi thành dừng và nêu "Spacetime Gaussians full model (rgbdecoder)", trừ khi có cờ `--base-color-only`.
   Hai model lite đang có trong ../data_real phải vẫn chạy như cũ, không cần cờ.
 Thông báo dừng phải ghi luôn cờ để chạy tiếp. Cập nhật test của hai converter cho hai trường hợp này.
Việc 4: sửa câu đã cũ trong mục Status của CLAUDE.md ("SH degree 0 only, so colours are mottled" sau số đo của `--sh`).
Chỉ sửa: gs4d/open.py, tests/test_open.py, convert_scgs.py, convert_spacetime.py, tests/test_scgs.py, tests/test_spacetime.py,
README (dòng tương ứng), CLAUDE.md, và bảng trạng thái trong PLAN.md.
Xong khi: `python -m gs4d.open <thư mục 4DGaussians bouncingballs trong /home/cont/gs4d_train>` mở được viewer có danh sách camera;
`python tests/run_all.py` và `python tests/check_real.py` qua, PSNR không đổi; bảng WP9 trong PLAN.md cập nhật.
Báo cáo 5 dòng rồi dừng.
```

### WP19b — đo FPS hiển thị trong Chrome có GPU (xong 15/11/2026)

Các bước ở mục "Làm ngay", điểm 2. Xong khi README có số FPS trong trình duyệt ở 1280×720 cạnh số phía server, kèm tên GPU
và phiên bản Chrome; xoá dòng "display FPS in a real browser" khỏi phần NOT tested của `CLAUDE.md`.

### WP19c — PSNR cho Spacetime và Shape of Motion; lệch 1,9 dB của native 4DGS (phần 3 đã xong; phần 1, 2 đã mở)
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md. Mỗi phần một phiên; nêu phần nào khi dán.
Phần 1 — Shape of Motion `paper-windmill`: lấy ảnh gốc của cảnh từ bộ iPhone của DyCheck (chỉ cảnh này), render ở camera
huấn luyện (`Ks`, `w2cs` trong checkpoint) và thêm PSNR có ngưỡng vào mục tương ứng của tests/check_real.py.
Báo dung lượng cần tải ở lượt đầu và dừng nếu vượt 5 GB.
Phần 2 — Spacetime Gaussians `n3d_sear_steak` (lite): tương tự với video `sear_steak` của Neural 3D Video và `cameras.json`.
Báo dung lượng cần tải ở lượt đầu và dừng nếu vượt 5 GB. Ảnh có người: không commit ảnh vào repo, chỉ ghi số.
Phần 3 — native 4DGS trên `slice-banana`: của ta 24,23 dB, repo gốc tự đo 26,14 dB. Chỉ chẩn đoán, không chỉnh cho khớp.
Thử lần lượt và ghi PSNR sau mỗi bước: (a) keyframe đặt đúng tại 33 thời điểm test thay vì nội suy; (b) ngưỡng lọc
marginal_t và cách nhân opacity theo thời gian so với renderer gốc; (c) nền và độ phân giải dùng khi đo so với eval của repo;
(d) SH: bậc và phần phụ thuộc thời gian có được tính đủ không. Dừng ở bước giải thích được phần lớn khoảng lệch.
Chỉ sửa: tests/check_real.py; convert_native4d.py và tests/test_native4d_sh.py nếu phần 3 tìm ra lỗi trong converter;
dòng tương ứng trong README, CLAUDE.md và PLAN.md.
Xong khi: README có PSNR (phần 1, 2) hoặc bảng PSNR theo từng bước và kết luận (phần 3). Báo cáo 5 dòng rồi dừng.
```

### WP19d — converter ghi từng keyframe ra đĩa (khi cần cảnh dài) — XONG 15/11/2026
Kết quả: `Sequence4DWriter` trong `gs4d/format.py` (giữ cả nhóm tĩnh và dạng nén, file bằng `save` từng bit), `tests/test_writer.py`;
`slice-banana` 330 keyframe: SC-GS RAM đỉnh 1,4 GB, 26,87 dB; native 4DGS 1,3 GB, 24,27 dB; Deformable-3DGS 7,2 → 1,0 GB.
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Vấn đề: converter dựng cả sequence [K,N,...] trong RAM rồi mới lưu. Trên `slice-banana` (392k–603k Gaussian), SC-GS và
native 4DGS chỉ chuyển được 110 keyframe; 330 keyframe thì hết 15 GB RAM và tiến trình bị kill.
Bước 0: đo RAM đỉnh của `convert_scgs.py --frames 110 --sh --compact` trên `slice-banana` và xác định mảng nào chiếm chỗ
(thường là `sh` [K,N,16,3]).
Việc: thêm vào gs4d/format.py một cách ghi tăng dần (ví dụ `Sequence4DWriter`: mở file, `add_keyframe(t, means, covs, ...)`,
`close()`), ghi được cả dạng thường và dạng nén, RAM đỉnh không phụ thuộc K; file ra đọc được bằng `Sequence4D.load` hiện tại.
Phát hiện nhóm tĩnh (WP11) có thể bỏ qua ở đường ghi này nếu cần hai lượt; ghi rõ nếu bỏ. Sau đó chuyển convert_scgs.py,
convert_native4d.py và convert_deformable3dgs.py sang dùng nó; các converter khác giữ nguyên.
Chỉ sửa: gs4d/format.py, ba converter trên, thêm tests/test_writer.py, dòng tương ứng trong README.
Xong khi: test so file ghi tăng dần với file ghi một lần (bằng nhau từng bit ở dạng thường); SC-GS và native 4DGS chuyển được
330 keyframe của `slice-banana` với RAM đỉnh dưới 8 GB; tests/check_real.py qua, và ghi PSNR `scgs_banana`, `native4d_banana`
ở 330 keyframe cạnh số 110 keyframe. Báo cáo 5 dòng rồi dừng.
```

### WP20 — train lại native 4DGS `slice-banana` với camera đúng
```
Đọc CLAUDE.md (mục WP19c phần 3) và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md. Điều kiện: nhánh cc-c/wp19c-psnr đã merge.
Vấn đề: tests/make_hypernerf_blender.py ghi `fl_x/fl_y/cx/cy` theo từng frame; reader Blender của fudan-zvg/4d-gaussian-splatting
khi đó để `FovX = FovY = -1`, nên model `slice-banana` được train với camera sai và 24,23 dB của ta (repo tự đo 26,14) không
phản ánh chất lượng converter.
Bước 0: `nvidia-smi`; nếu GPU còn dưới 10 GB trống thì báo và dừng (lần trước một game giữ ~3 GB và ba lần train bị CUDA OOM).
Việc: đọc reader của repo để biết nó lấy FoV từ trường nào, sửa tests/make_hypernerf_blender.py để ghi đúng trường đó
(không vá repo gốc), tạo lại layout, train lại trong /home/cont/gs4d_train với cùng config (20k vòng, batch_size 1,
densify_until_num_points 250_000), rồi chạy lại eval của repo và `convert_native4d.py --frames 110 --sh --compact`.
Chỉ sửa: tests/make_hypernerf_blender.py, mục `native4d_banana` và ngưỡng của nó trong tests/check_real.py,
tests/diag_native4d_banana.py (bỏ bước giả lập nếu không còn cần), dòng tương ứng trong README, CLAUDE.md, PLAN.md.
Không sửa convert_native4d.py; nếu sau khi train lại vẫn lệch quá 0,5 dB so với eval của repo thì báo số theo từng bước
của diag script và dừng, không chỉnh cho khớp.
Xong khi: README ghi PSNR mới của repo và của ta trên 33 ảnh giữ lại; hai số lệch không quá 0,5 dB; check_real.py qua.
Báo cáo 5 dòng rồi dừng.
```

### WP21 — nén trường `sh` theo keyframe
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Vấn đề: dạng nén (WP8) không đụng tới `sh`. Native 4DGS ghi `sh` theo từng keyframe [K,N,S,3] ở float32:
`slice-banana` 330 keyframe ra 19,9 GB, và check_real.py không nạp nổi file đó nếu không `load(lazy=True)`.
Bước 0: đo phân bố giá trị của `sh` trên file thật (bậc 0 so với bậc cao) để chọn cách lượng tử hóa; ghi số.
Việc: trong dạng nén, lưu `sh` thay đổi theo thời gian ở dạng lượng tử hóa (ví dụ float16, hoặc int8 với thang riêng cho từng bậc),
`sh` hằng theo thời gian giữ nguyên; `Sequence4DWriter`, nạp lười và `GpuFrames` đều đọc được; file cũ vẫn đọc được.
Chỉ sửa: gs4d/format.py, tests/test_compact.py, tests/test_writer.py, tests/bench_large.py, đoạn "File lớn" trong README.
Xong khi: README có bảng trước/sau cho native 4DGS `slice-banana` 330 keyframe và `bouncingballs --sh` (dung lượng, thời gian
lấy frame khi play, PSNR); mục tiêu dung lượng không quá 1/3 bản float32 và PSNR giảm không quá 0,1 dB; check_real.py qua.
Báo cáo 5 dòng rồi dừng.
```

### WP22 — `Sequence4DWriter` cho bốn converter còn lại — XONG 15/11/2026

Kết quả: bốn converter ghi bằng Writer, file ra giống code cũ từng bit ở mọi mảng trên output thật (số RAM trong README). `convert_per_frame.py` ở chế độ số Gaussian đổi theo frame giữ đường cũ (Writer cần N cố định). Shape of Motion cả 277 frame nay chuyển được (RAM đỉnh 0,88 GB).

Prompt gốc:
```
Đọc CLAUDE.md (mục WP19d) và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Việc: chuyển convert_4dgaussians.py, convert_shape_of_motion.py, convert_spacetime.py và convert_per_frame.py sang ghi bằng
`gs4d.Sequence4DWriter`, theo đúng cách convert_scgs.py đã làm; không đổi công thức nào. Với convert_per_frame.py ở chế độ
số Gaussian thay đổi theo frame (`interpolate=False`): nếu Writer không hỗ trợ thì giữ đường cũ cho chế độ đó và ghi rõ.
Chỉ sửa bốn converter trên và dòng tương ứng trong README, CLAUDE.md, PLAN.md.
Xong khi: với mỗi converter, file ra trên output thật trong ../data_real giống file của code cũ ở mọi mảng (so bằng script, dán
kết quả vào báo cáo); README có RAM đỉnh trước/sau; `python tests/run_all.py` và check_real.py qua. Báo cáo 5 dòng rồi dừng.
```

### WP23 — 4DGaussians trên cảnh thật `slice-banana`
```
Đọc CLAUDE.md (các mục WP15 và WP16 ngày 15/11/2026) và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Điều kiện: WP20 đã xong (GPU trống). Bước 0: `nvidia-smi`; GPU còn dưới 10 GB trống thì báo và dừng.
Việc: train hustvl 4DGaussians trên HyperNeRF `slice-banana` trong /home/cont/gs4d_train/4dgaussians (clone đã vá sẵn), dùng
reader HyperNeRF của chính repo và cùng cách chia 297 ảnh train / 33 ảnh giữ lại như ba phương pháp kia (xem
tests/make_hypernerf_split.py), ảnh 268x480, 20k vòng hoặc config hypernerf mặc định của repo nếu vừa 12 GB GPU và 15 GB RAM.
Sau đó `convert_4dgaussians.py --sh --compact`, mở bằng `python -m gs4d.open`, và đo PSNR trên 33 ảnh giữ lại cạnh số eval của repo.
Kiểm trước khi tin số của repo: camera mà repo dùng khi train có đúng không (bài học WP19c phần 3: FovX/FovY).
Chỉ sửa: convert_4dgaussians.py và tests/test_4dgaussians.py nếu định dạng thật khác giả định; thêm mục `4dgaussians_banana`
vào tests/check_real.py; dòng tương ứng trong README, CLAUDE.md, PLAN.md.
Xong khi: README ghi PSNR của ta và của repo trên 33 ảnh, số Gaussian, dung lượng, thời gian chuyển; lệch quá 0,5 dB thì báo
nguyên nhân đã thử, không chỉnh cho khớp. Báo cáo 5 dòng rồi dừng.
```

### WP24 — chế độ vẽ tâm Gaussian thành điểm (chờ xác nhận)
```
Đọc CLAUDE.md và mục "Điều kiện xong, áp dụng cho mọi gói" trong PLAN.md.
Việc: thêm ô chọn "Display" trong bảng điều khiển của viewer.py: `splats` (như hiện tại) hoặc `points` (mỗi Gaussian là một điểm
tại tâm, màu của Gaussian, có thanh "Point size"), dùng được khi đang play, với `--compare`, và ở cả hai renderer
(client: point cloud của viser; gsplat: rasterize với scale nhỏ cố định theo pixel, hoặc gửi point cloud sang client nếu đơn giản hơn;
chọn một và ghi lý do). Thanh "Hide opacity below" vẫn áp dụng. Ghi vào README cách di chuyển camera (chuột và phím) đúng theo
phiên bản viser đang cài, sau khi thử thật.
Chỉ sửa viewer.py, README, CLAUDE.md, PLAN.md; thêm tests/check_points.py (ảnh chụp headless ở 3 thời điểm).
Xong khi: chạy với `../data_real/scgs/bouncingballs_sh.npz` và `../data_real/som/paper-windmill.npz`, có ảnh chụp trong shots/points/,
tốc độ khi play ở chế độ điểm được ghi trong README. Báo cáo 5 dòng rồi dừng.
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
  `rgbdecoder`), Deformable-3DGS biến thể mặc định và 6-DoF, cảnh thật (không phải D-NeRF)
  cho Deformable-3DGS, SC-GS và native 4DGS.
- Câu hỏi cho giáo sư bên dưới chưa có câu trả lời được ghi lại; WP3–5 đã được làm mà không chờ.
- Tốc độ hiển thị trong trình duyệt thật: đã đo ở WP19b (15/11/2026), bảng trong README.
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
