# Định dạng `.npz` chung

Viewer chỉ đọc một loại file: "Gaussian theo từng keyframe", class `Sequence4D` trong `gs4d/format.py` (docstring đầu file là bản
tham chiếu chính xác nhất). Mỗi converter ghi file này; viewer không import code của phương pháp nào.

## Các trường

| Khoá | Kích thước | Ý nghĩa |
|---|---|---|
| `times` | `[K]` | mốc thời gian của các keyframe, tăng dần (đơn vị tuỳ phương pháp) |
| `means` | `[K,N,3]` hoặc `[N,3]` | tâm |
| `covs` | `[K,N,6]` hoặc `[N,6]` | nửa trên của covariance 3×3: xx xy xz yy yz zz |
| `opacities` | `[K,N]` hoặc `[N]` | trong [0,1] |
| `colors` | `[K,N,3]` hoặc `[N,3]` | RGB không phụ thuộc hướng nhìn, thường trong [0,1]. Giá trị ngoài khoảng được giữ và đến renderer gsplat nguyên vẹn (Spacetime Gaussians trộn màu thô và chỉ cắt ở pixel); renderer client cắt về [0,1] |
| `sh` (tuỳ chọn) | `[K,N,S,3]` hoặc `[N,S,3]`, S = (bậc+1)² | hệ số SH theo quy ước 3DGS: màu = max(SH(hướng nhìn) + 0,5; 0), `sh[...,0,:]` là `f_dc`. Chỉ renderer gsplat dùng; `colors` vẫn phải có |
| `interpolate` | bool | `False` khi các keyframe không chung danh tính Gaussian (viewer nhảy theo keyframe gần nhất) |
| `source` | str | chú thích tự do, hiện trong viewer |

Trường không có trục K là hằng theo thời gian. Giữa hai keyframe viewer nội suy tuyến tính (`Sequence4D.at(t)`).
Dung lượng phần thay đổi theo thời gian ≈ N × K × 40 byte ở dạng float32 (≈ 19 byte ở dạng nén).

## Ghi file

- Cả chuỗi vừa RAM: tạo `Sequence4D(times, means, covs, opacities, colors, sh=...)` rồi `.save(path)`.
- Cảnh dài: `Sequence4DWriter` ghi từng keyframe, RAM đỉnh không phụ thuộc K. Trường hằng đưa vào lúc tạo, các trường còn lại đưa vào
  `add_keyframe(t, ...)`; file ra giống hệt `save` từng bit, cả dạng nén và nhóm tĩnh (`tests/test_writer.py`). Lúc ghi cần thêm chỗ đĩa
  tạm ≈ N × K × 40 byte cạnh file ra (thư mục `.gs4d_writer_*`, tự xoá; còn sót nếu tiến trình bị kill), nhiều hơn khi có `sh` theo keyframe.

```python
from gs4d import Sequence4DWriter

with Sequence4DWriter("scene.npz", compact=True, colors=colors, source="my method") as w:   # trường hằng
    for t in times:
        w.add_keyframe(t, means=means_at(t), covs=covs_at(t), opacities=opacities_at(t))  # trường đổi theo t
```

Mọi converter đều ghi qua `Sequence4DWriter` (trừ thư mục ply có số Gaussian đổi theo frame, vì writer cần N cố định). RAM đỉnh trước
và sau khi chuyển sang writer: [PERFORMANCE.md](PERFORMANCE.md#converter-ram-đỉnh-và-thời-gian).

## Dạng nén và nạp lười

```bash
python -m gs4d scene.npz scene_small.npz             # lưu lại ở dạng nén (có mất mát nhỏ)
python -m gs4d scene_small.npz scene.npz --float32   # trả về float32 thường
```

Trong code: `seq.save(path, compact=True)`; mặc định `compact=False` (float32, không mất mát). Mọi `convert_*.py` có cờ `--compact`;
`python -m gs4d.open` tự thêm cờ này khi file ước tính lớn hơn 1 GB. `Sequence4D.load` đọc được cả hai dạng và mọi file cũ.

Dạng nén chỉ đổi các trường thay đổi theo thời gian:

| Trường | Lưu thành |
|---|---|
| `means` | `means_ref [N,3]` float32 (keyframe đầu) + `means_d16 [K,N,3]` float16 (độ lệch so với nó) |
| `covs` | `covs_scale [N]` float32 + `covs_q [K,N,6]` int16: thừa số Cholesky L (cov = L Lᵀ) chia cho `covs_scale`, nhân 32767; giải mã luôn ra ma trận nửa xác định dương |
| `opacities` | `opacities_u8 [K,N]` uint8 |
| `colors` | `colors_u8 [K,N,3]` uint8 (giữ float32 khi có giá trị ngoài [0,1]) |
| `sh` | `sh_lo [N,S,3]` + `sh_step [N,S,3]` float32 + `sh_u8 [K,N,S,3]` uint8: mỗi hệ số của mỗi Gaussian lượng tử hoá trên khoảng giá trị của chính nó theo thời gian, `sh = sh_lo + sh_u8 · sh_step` |

Trường không có trục K vẫn là float32. Viewer nạp lười (`load(path, lazy=True)`, ánh xạ file vào bộ nhớ): chỉ keyframe đang xem mới được
đọc từ đĩa, nên **không được ghi đè file khi viewer đang mở nó**.

Sai số, đo bằng `python tests/bench_large.py` với 500.000 Gaussian × 100 keyframe sinh ngẫu nhiên: vị trí lớn nhất 0,002 với Gaussian dịch
tới 5,35 đơn vị so với keyframe đầu (sai số tỉ lệ với độ dịch: float16 có 11 bit có nghĩa); opacity 0,002; covariance 7·10⁻⁵ lần phương sai
lớn nhất của Gaussian đó. Dung lượng 2006 → 964 MB; nạp lười dưới 0,01 s và 94 MB RAM thay cho 1,04 s và 1963 MB khi nạp hết. Đổi lại, giải mã
mỗi frame trên CPU chậm hơn (77 ms so với 14 ms); renderer gsplat giải mã trên GPU nên không bị ảnh hưởng. Bảng đầy đủ:
[PERFORMANCE.md](PERFORMANCE.md#dạng-nén-và-nạp-lười).

### `sh` theo keyframe trong dạng nén

Native 4DGS với `--sh` ghi `sh` theo từng keyframe (`[K,N,16,3]`), nên gần như toàn bộ file là `sh`. Một hệ số đổi theo thời gian ít hơn
nhiều so với độ lớn của nó, nên lượng tử hoá trên khoảng riêng theo thời gian chính xác hơn hẳn một thang chung. Phân bố đo trên file thật
(12 keyframe rải đều; "khoảng theo thời gian" là lớn nhất − nhỏ nhất của một hệ số của một Gaussian):

| | Bậc 0: khoảng giá trị | \|x\| ở phân vị 99 | khoảng theo thời gian, trung vị / lớn nhất | Bậc 1–3: khoảng giá trị | \|x\| ở phân vị 99 | khoảng theo thời gian, trung vị / lớn nhất |
|---|---|---|---|---|---|---|
| `bouncingballs` | −2,10 … 4,40 | 1,88 | 0,026 / 0,36 | −0,72 … 0,64 | 0,14–0,21 | 0,035–0,040 / 0,62 |
| `slice-banana` | −2,09 … 6,79 | 2,76 | 0,028 / 0,47 | −0,94 … 0,94 | 0,16–0,18 | 0,032–0,038 / 0,95 |

Sai số RMS của hệ số bậc 0 trên `slice-banana`: int8 một thang cho mỗi bậc 1,6·10⁻²; int8 một thang cho mỗi hệ số của mỗi Gaussian
2,3·10⁻³; float16 2,3·10⁻⁴ (file chỉ nhỏ đi một nửa); uint8 trên khoảng riêng theo thời gian 5,3·10⁻⁵ (bậc 1–3: 6–8·10⁻⁵, lớn nhất 1,8·10⁻³).
`sh` có giá trị không hữu hạn thì giữ float32.

Với output thật của native 4DGS (`convert_native4d.py --sh --compact`, cùng checkpoint, trước = `sh` float32, sau = uint8; PSNR bằng code của
`tests/check_real.py`, nạp lười):

| | `slice-banana` 330 keyframe, trước | sau | `bouncingballs` 60 keyframe, trước | sau |
|---|---|---|---|---|
| Gaussian | 297.745 | 297.745 | 116.193 | 116.193 |
| Dung lượng file | 19.858 MB | 5.824 MB (0,29 lần) | 1.412 MB | 453 MB (0,32 lần) |
| PSNR trung bình (nhỏ nhất – lớn nhất) | 24,2687 dB (21,220–26,119), 33 ảnh | 24,2687 dB (21,220–26,119) | 32,9117 dB (27,281–39,594), 20 ảnh | 32,9117 dB (27,281–39,593) |
| PSNR giảm nhiều nhất ở một ảnh | — | 0,0002 dB | — | 0,0004 dB |

(Số `slice-banana` ở đây là của model native 4DGS train lần đầu, trước khi sửa camera; xem [SUPPORTED.md](SUPPORTED.md).) Tốc độ lấy frame
của hai dạng: [PERFORMANCE.md](PERFORMANCE.md#giải-mã-và-nội-suy-trên-gpu-khi-play). Lúc ghi, `Sequence4DWriter` vẫn cần đĩa tạm cho `sh`
float32 (25 GB với `slice-banana` 330 keyframe).

## Gaussian tĩnh chỉ lưu một lần

Khi lưu, `Sequence4D.save` tự tìm các Gaussian không đổi qua mọi keyframe ở mọi trường (sai khác so với keyframe đầu không quá 10⁻⁶ lần độ lớn
của giá trị) và lưu chúng một lần thay vì K lần; `save(path, split_static=False)` tắt việc này. Người gọi không phải biết: `at(t)` vẫn trả về
đủ N Gaussian theo đúng thứ tự cũ. File có thêm `static_index [Ns]` (vị trí các Gaussian tĩnh) và `<trường>_static [Ns,...]`; các trường theo
keyframe chỉ giữ Nd = N − Ns Gaussian còn lại. `python -m gs4d cũ.npz mới.npz --float32` chuyển file cũ sang dạng mới không mất mát (bỏ
`--float32` để nén thêm, thêm `--no-split` để ghi theo kiểu cũ).

Đo với output thật (nạp lười, `at(t)` là trung bình 39 thời điểm nằm giữa các keyframe):

| | Shape of Motion `paper-windmill`, trước | sau | sau + nén | Deformable-3DGS `bouncingballs`, trước | sau |
|---|---|---|---|---|---|
| Gaussian tĩnh / tổng | — | 924.075 / 995.942 | 924.075 / 995.942 | — | 0 / 64.571 |
| Dung lượng file | 1271 MB | 147 MB | 103 MB | 153 MB | 153 MB |
| RAM sau khi nạp hết (không lười) | 1248 MB | 177 MB | 135 MB | 181 MB | 181 MB |
| RAM sau khi play hết dải thời gian (nạp lười) | 1248 MB | 205 MB | 161 MB | 166 MB | 167 MB |
| Lấy một frame nội suy (`at(t)`) | 34 ms | 8,7 ms | 10,9 ms | 1,7 ms | 1,6 ms |

`bouncingballs` không có Gaussian tĩnh nào vì MLP biến dạng của Deformable-3DGS dịch mọi Gaussian một chút. Ảnh gsplat render từ file trước
và sau: `bouncingballs` giống hệt ở cả 3 thời điểm; `paper-windmill` (715 × 983, camera huấn luyện) giống hệt tại keyframe, còn giữa hai
keyframe có 0–7 trên 702.845 pixel lệch quá 1/255 (lớn nhất 8,6/255): bản cũ nội suy cả Gaussian tĩnh với chính nó nên giá trị bị làm tròn
lệch tới 5·10⁻⁷; thay các giá trị đó bằng giá trị gốc thì ảnh cũ trùng ảnh mới từng pixel.

## Tương thích giữa các phiên bản

`Sequence4D.load` đọc mọi file các bản trước ghi. Ngược lại, bản viewer cũ không đọc được file có nhóm tĩnh (`static_index`, từ 14/11/2026)
và file nén có `sh_u8` (từ 15/11/2026); chuyển bằng `python -m gs4d cũ.npz mới.npz --float32 --no-split` nếu cần mở bằng bản cũ.
