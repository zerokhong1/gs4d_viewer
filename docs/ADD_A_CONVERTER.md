# Thêm một phương pháp 4DGS mới

Viewer chỉ đọc một file `.npz` chung (`gs4d/format.py`, class `Sequence4D`). Thêm phương pháp = viết một converter ghi file đó; không sửa `viewer.py`, `gs4d/format.py` hay converter khác.

## Các bước

1. `cp convert_template.py convert_<tên>.py` và `cp tests/test_template.py tests/test_<tên>.py`. Template là phương pháp giả (vị trí = `xyz + vel * t`) đã chạy được.
2. **Đọc repo gốc, không chạy nó.** Tìm: file output (checkpoint, ply, npz), cách tính vị trí / quaternion / scale / opacity / màu tại thời điểm `t`, các hàm kích hoạt (`exp` cho scale, `sigmoid` cho opacity), thời gian nằm trong khoảng nào. Chép công thức sang numpy. Torch chỉ để mở checkpoint hoặc chạy lại một MLP nhỏ trên CPU. Đừng thêm repo gốc làm dependency.
3. Xếp phương pháp vào một dạng (README, bảng "Dạng output"): A Gaussian theo từng frame; B Gaussian 4D; C hàm chuyển động tham số; D mạng biến dạng chạy lại trên CPU.
4. Thay `load_model` và `keyframe` trong template. Điều format cần: `means [N,3]`, `covs [N,6]` (nửa trên của 3×3; dùng `cov_from_quat_scale(quat_wxyz, scale_tuyến_tính)`), `opacities [N]` trong [0,1], `colors [N,3]` (có thể ngoài [0,1] nếu method lưu màu thô như Spacetime), `sh [N,S,3]` tùy chọn. Trường không đổi theo thời gian đưa vào constructor của `Sequence4DWriter`, trường đổi thì đưa vào từng `add_keyframe(t, ...)`.
5. Ghi bằng `Sequence4DWriter` (RAM đỉnh chỉ vài keyframe, nhóm tĩnh và `--compact` có sẵn), không dựng cả chuỗi trong RAM.
6. Biến thể chưa hỗ trợ thì `raise SystemExit("... nêu tên biến thể ...")`; đừng im lặng chuyển sai.
7. Test: dữ liệu giả đúng layout file thật, so với một cách tính độc lập (như `tests/test_template.py`: ma trận covariance đầy đủ `R S Sᵀ Rᵀ`). `tests/run_all.py` tự nhặt `tests/test_*.py`; test cần torch thì thêm vào `NEEDS` trong `tests/run_all.py`.
8. Thêm một dòng vào bảng trong README và một dòng vào `docs/SUPPORTED.md`. Nếu muốn `gs4d.open` nhận ra output, đó là việc riêng ở `gs4d/open.py`.
9. Kiểm trên output thật (xem dưới) rồi mới ghi PSNR.

## Bài học

- **Kiểm camera của repo gốc trước khi tin số eval của nó.** Native 4DGS trên `slice-banana` kém repo 1,9 dB; nguyên nhân là camera dựng từ layout của chúng tôi để `FovX = FovY = -1`, rasteriser CUDA dùng tiêu cự giả và model học bù. Converter không sai. Sau khi sửa camera, hai số khớp trong 0,01 dB.
- **Đo ở độ phân giải lúc train.** Deformable-3DGS train ở 800 px cho 37,2–43,6 dB; render ở 400 px chỉ 29,3–33,2 dB.
- **Đối chiếu với renderer của chính repo** khi có thể (`tests/repo_psnr.py`): Shape of Motion khớp 0,000 dB; làm vậy mới phát hiện Spacetime lưu màu thô ngoài [0,1] mà converter đã cắt (31,43 → 34,37 dB).
- Cộng đúng thứ tự: Deformable-3DGS và SC-GS cộng `d_scaling` vào `exp(scale)` (sau kích hoạt), không phải vào log-scale. Kiểm từng thứ trong repo, đừng suy ra.
- Thời gian: D-NeRF dùng [0, 1], cảnh video dùng frame hoặc frame/fps. Chọn `--t0 --t1` cho khớp và nói trong `--help`.
- Máy có 15 GB RAM: nếu chuỗi dài, kiểm RAM đỉnh; so sánh file ghi bằng `Sequence4DWriter` với `Sequence4D.save` trên dữ liệu nhỏ.
