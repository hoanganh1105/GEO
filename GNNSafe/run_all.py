import os
import datetime
import re
import csv
import subprocess

# 1. Danh sách mạng (Backbone)
backbones = ['gcn', 'gat'] 

# 2. Cấu hình các Dataset
datasets = [
    ("cora", "structure", -5, -1, 0.01),
    ("cora", "feature", -5, -1, 0.01),
    ("cora", "label", -5, -1, 0.01),
    ("amazon-photo", "structure", -9, -1, 1.0),
    ("amazon-photo", "feature", -9, -1, 1.0),
    ("amazon-photo", "label", -9, -1, 1.0),
    ("coauthor-cs", "structure", -9, -2, 0.01),
    ("coauthor-cs", "feature", -9, -2, 0.01),
    ("coauthor-cs", "label", -9, -2, 0.01),
    ("arxiv", "structure", -9, -2, 0.01),
]

# 3. Tổ hợp cấu hình
experiments = [
    ("1. MSP (No - No - No)", "--method msp"),
    ("2. (No - No - Yes)", "--method gnnsafe --use_bn --use_occ --beta 0.1 --nu 0.01"),
    ("3. (No - Yes - Yes)", "--method gnnsafe --use_bn --use_reg --m_in {m_in} --m_out {m_out} --lamda {lamda} --use_occ --beta 0.1 --nu 0.01"),
    ("4. (Yes - No - Yes)", "--method gnnsafe --use_bn --use_prop --use_occ --beta 0.1 --nu 0.01"),
    ("5. GNNSafe++ (Yes - Yes - No)", "--method gnnsafe --use_bn --use_prop --use_reg --m_in {m_in} --m_out {m_out} --lamda {lamda}"),
    ("6. GEO Full (Yes - Yes - Yes)", "--method gnnsafe --use_bn --use_prop --use_reg --m_in {m_in} --m_out {m_out} --lamda {lamda} --use_occ --beta 0.1 --nu 0.01")
]

log_file = "ket_qua_thi_nghiem.txt"
csv_file = "bang_ket_qua_so_sanh.csv"

# BƯỚC MỚI: Khởi tạo file CSV ngay từ đầu và ghi luôn dòng Tiêu đề
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["Backbone & Model"] + [f"{ds[0]} ({ds[1]})" for ds in datasets]
    writer.writerow(header)

with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n\n{'='*50}\nBẮT ĐẦU CHẠY THÍ NGHIỆM - {datetime.datetime.now()}\n{'='*50}\n")

total_runs = len(backbones) * len(datasets) * len(experiments)
current_run = 0

# 4. Vòng lặp chạy và Ghi Real-time
for backbone in backbones:
    for exp_name, flag_template in experiments:
        
        # Tạo một mảng để hứng dữ liệu cho hàng này
        row_data = [f"{backbone.upper()} - {exp_name}"]
        
        for ds_name, ood_type, m_in, m_out, lamda in datasets:
            current_run += 1
            
            flags = flag_template.format(m_in=m_in, m_out=m_out, lamda=lamda)
            cmd = f"python main.py --backbone {backbone} --dataset {ds_name} --ood_type {ood_type} --mode detect {flags}"
            if ds_name in ["amazon-photo", "coauthor-cs", "arxiv"]:
                cmd += " --cpu"
            
            print(f"[{current_run}/{total_runs}] Đang chạy: {backbone.upper()} | {exp_name} | {ds_name} ({ood_type})")
            
            # Ghi tiêu đề vào log text
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{backbone.upper()} - {exp_name}] - {ds_name} ({ood_type})\n")
            
            # DÙNG SUBPROCESS ĐỂ CHẠY VÀ CHỤP KẾT QUẢ NGAY LẬP TỨC
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = process.stdout + process.stderr
            
            # Ghi phần text chi tiết vào file log để lưu vết
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(output)
            
            # Quét tìm con số AUROC trong kết quả vừa chạy xong
            match = re.search(r"OOD Test 1 Final AUROC:\s+([\d\.]+)", output)
            if match:
                auroc = match.group(1)
                print(f"   => Đã lấy được AUROC: {auroc}%")
            else:
                auroc = "Lỗi"
                print(f"   => Lỗi: Không tìm thấy điểm AUROC!")
                
            # Đưa điểm vừa lấy vào mảng của hàng ngang hiện tại
            row_data.append(auroc)
            
        # QUAN TRỌNG: Sau khi chạy xong 10 dataset của hàng này, Ghi ngay hàng đó vào file CSV!
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        print(f"-----> Đã chốt và lưu kết quả của Hàng '{exp_name}' vào file CSV!\n")

print("\nĐÃ HOÀN THÀNH TOÀN BỘ THÍ NGHIỆM! Dữ liệu đã nằm gọn trong file", csv_file)