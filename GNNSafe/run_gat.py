import os
import datetime
import re
import csv
import subprocess

# 1. Chỉ chạy duy nhất mạng GAT
backbones = ['gat'] 

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

# 3. Tổ hợp cấu hình (Đã loại bỏ --use_bn khỏi Baseline MSP)
experiments = [
    ("1. MSP (No - No - No)", "--method msp"),
    ("2. (No - No - Yes)", "--method gnnsafe --use_bn --use_occ --beta 0.1 --nu 0.01"),
    ("3. (No - Yes - Yes)", "--method gnnsafe --use_bn --use_reg --m_in {m_in} --m_out {m_out} --lamda {lamda} --use_occ --beta 0.1 --nu 0.01"),
    ("4. (Yes - No - Yes)", "--method gnnsafe --use_bn --use_prop --use_occ --beta 0.1 --nu 0.01"),
    ("5. GNNSafe++ (Yes - Yes - No)", "--method gnnsafe --use_bn --use_prop --use_reg --m_in {m_in} --m_out {m_out} --lamda {lamda}"),
    ("6. GEO Full (Yes - Yes - Yes)", "--method gnnsafe --use_bn --use_prop --use_reg --m_in {m_in} --m_out {m_out} --lamda {lamda} --use_occ --beta 0.1 --nu 0.01")
]

log_file = "ket_qua_log_gat.txt"
csv_file = "bang_ket_qua_gat.csv"

# Khởi tạo file CSV với tiêu đề
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["Backbone & Model"] + [f"{ds[0]} ({ds[1]})" for ds in datasets]
    writer.writerow(header)

with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n\n{'='*50}\nBẮT ĐẦU CHẠY THÍ NGHIỆM GAT - {datetime.datetime.now()}\n{'='*50}\n")

total_runs = len(backbones) * len(datasets) * len(experiments)
current_run = 0

# 4. Vòng lặp chạy và Ghi Real-time
for backbone in backbones:
    for exp_name, flag_template in experiments:
        
        row_data = [f"{backbone.upper()} - {exp_name}"]
        
        for ds_name, ood_type, m_in, m_out, lamda in datasets:
            current_run += 1
            
            flags = flag_template.format(m_in=m_in, m_out=m_out, lamda=lamda)
            
            # Đã bỏ --cpu để GPU trên Colab phát huy tác dụng
            cmd = f"python main.py --backbone {backbone} --dataset {ds_name} --ood_type {ood_type} --mode detect {flags}"
            
            print(f"[{current_run}/{total_runs}] Đang chạy: {backbone.upper()} | {exp_name} | {ds_name} ({ood_type})")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{backbone.upper()} - {exp_name}] - {ds_name} ({ood_type})\n")
            
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = process.stdout + process.stderr
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(output)
            
            match = re.search(r"OOD Test 1 Final AUROC:\s+([\d\.]+)", output)
            if match:
                auroc = match.group(1)
                print(f"   => Đã lấy được AUROC: {auroc}%")
            else:
                # Nếu Colab vẫn bị OOM ở Arxiv, nó sẽ ghi "OOM" vào bảng thay vì bị tắt ngang
                auroc = "Lỗi/OOM"
                print(f"   => Lỗi: Không tìm thấy điểm AUROC (Có thể do OOM)!")
                
            row_data.append(auroc)
            
        # Ghi từng hàng ngang vào CSV
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        print(f"-----> Đã chốt và lưu kết quả của Hàng '{exp_name}' vào file CSV!\n")

print("\nĐÃ HOÀN THÀNH TOÀN BỘ THÍ NGHIỆM GAT! Vui lòng tải file", csv_file, "về máy.")