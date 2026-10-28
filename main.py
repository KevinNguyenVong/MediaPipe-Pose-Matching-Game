import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading

# Nạp thư viện âm thanh (đã có sẵn trong venv)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ==========================================
# 1. ÂM THANH PHẢN HỒI KHI HOÀN THÀNH TƯ THẾ
# ==========================================
def play_sound_effect(effect_type="SUCCESS"):
    """Phát âm thanh báo hiệu khi người chơi tạo dáng đúng hoặc chiến thắng."""
    if not AUDIO_AVAILABLE:
        return

    def _worker():
        sr = 44100
        if effect_type == "SUCCESS":
            freqs = [523.25, 659.25, 783.99, 1046.50] # C5 - E5 - G5 - C6
            dur = 0.55
        else:
            freqs = [523.25, 659.25, 783.99, 1046.50, 1318.51]
            dur = 0.9

        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        wave = np.zeros_like(t)
        for idx, f in enumerate(freqs):
            delay = int(idx * 0.08 * sr)
            t_note = np.maximum(0, t - delay / sr)
            env = np.clip(t_note / 0.01, 0, 1) * np.exp(-5.0 * t_note)
            env[:delay] = 0
            wave += np.sin(2 * np.pi * f * t_note) * env

        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = (wave / max_val) * 0.5
        sd.stop()
        sd.play(wave.astype(np.float32), sr)

    threading.Thread(target=_worker, daemon=True).start()


# ==========================================
# 2. KHỞI TẠO MEDIAPIPE POSE
# ==========================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

WIDTH, HEIGHT = 1280, 720

# ==========================================
# 3. ĐỊNH NGHĨA CÁC KHỚP XƯƠNG VÀ TƯ THẾ MẪU
# ==========================================
# Tên các góc khớp và bộ 3 điểm Landmark (A, B, C) tạo nên góc tại đỉnh B:
# 11: Vai trái, 12: Vai phải, 13: Khuỷu trái, 14: Khuỷu phải,
# 15: Cổ tay trái, 16: Cổ tay phải, 23: Hông trái, 24: Hông phải,
# 25: Gối trái, 26: Gối phải, 27: Cổ chân trái, 28: Cổ chân phải.
JOINT_TRIPLETS = {
    "left_elbow":     (11, 13, 15),
    "right_elbow":    (12, 14, 16),
    "left_shoulder":  (23, 11, 13),
    "right_shoulder": (24, 12, 14),
    "left_hip":       (11, 23, 25),
    "right_hip":      (12, 24, 26),
    "left_knee":      (23, 25, 27),
    "right_knee":     (24, 26, 28)
}

# Danh sách 6 Tư thế mẫu (Mỗi tư thế gồm góc mục tiêu theo độ cho các khớp cần chấm)
POSES_DB = [
    {
        "name": "1. T-POSE (Dang ngang 2 tay)",
        "instruction": "Dung thang, dang 2 canh tay ngang vai (180 do)",
        "targets": {
            "left_shoulder": 90, "right_shoulder": 90,
            "left_elbow": 175, "right_elbow": 175,
            "left_knee": 175, "right_knee": 175
        },
        "stickman": {
            "l_arm": (-65, 0, -125, 0), "r_arm": (65, 0, 125, 0),
            "l_leg": (-25, 90, -25, 165), "r_leg": (25, 90, 25, 165)
        }
    },
    {
        "name": "2. VICTORY V-POSE (Gio 2 tay chu V)",
        "instruction": "Gio cao 2 tay thang len troi tao hinh chu V",
        "targets": {
            "left_shoulder": 165, "right_shoulder": 165,
            "left_elbow": 170, "right_elbow": 170
        },
        "stickman": {
            "l_arm": (-45, -55, -75, -110), "r_arm": (45, -55, 75, -110),
            "l_leg": (-25, 90, -25, 165), "r_leg": (25, 90, 25, 165)
        }
    },
    {
        "name": "3. DOUBLE BICEPS (Gong co 2 tay)",
        "instruction": "Nang 2 khuyu tay ngang vai va gap cang tay len 90 do",
        "targets": {
            "left_shoulder": 90, "right_shoulder": 90,
            "left_elbow": 85, "right_elbow": 85
        },
        "stickman": {
            "l_arm": (-65, 0, -65, -60), "r_arm": (65, 0, 65, -60),
            "l_leg": (-25, 90, -25, 165), "r_leg": (25, 90, 25, 165)
        }
    },
    {
        "name": "4. RIGHT HAND WAVE (Gio tay phai, ha tay trai)",
        "instruction": "Tay phai gio cao vuông goc, tay trai duoi doc than nguoi",
        "targets": {
            "left_shoulder": 20, "right_shoulder": 100,
            "left_elbow": 170, "right_elbow": 90
        },
        "stickman": {
            "l_arm": (-35, 45, -40, 95), "r_arm": (65, 0, 65, -60),
            "l_leg": (-25, 90, -25, 165), "r_leg": (25, 90, 25, 165)
        }
    },
    {
        "name": "5. TREE YOGA POSE (Tu the Cai Cay)",
        "instruction": "Dang 2 tay ngang va co 1 dau goi chan len cao",
        "targets": {
            "left_shoulder": 90, "right_shoulder": 90,
            "left_elbow": 170, "right_elbow": 170,
            "right_knee": 95
        },
        "stickman": {
            "l_arm": (-65, 0, -120, 0), "r_arm": (65, 0, 120, 0),
            "l_leg": (-25, 90, -25, 165), "r_leg": (55, 55, 20, 95)
        }
    },
    {
        "name": "6. WIDE SQUAT GUARD (Xuong tan & Thu the)",
        "instruction": "Chung 2 dau goi xuong va gap 2 khuyu tay truoc nguc",
        "targets": {
            "left_elbow": 75, "right_elbow": 75,
            "left_knee": 125, "right_knee": 125
        },
        "stickman": {
            "l_arm": (-45, 30, -20, -10), "r_arm": (45, 30, 20, -10),
            "l_leg": (-55, 70, -55, 150), "r_leg": (55, 70, 55, 150)
        }
    }
]


# ==========================================
# 4. CÁC HÀM THUẬT TOÁN XỬ LÝ TƯ THẾ
# ==========================================

def calculate_angle_3p(a, b, c):
    """
    Thuật toán tính góc (0 - 180 độ) tạo bởi 3 điểm Landmark A - B - C (với B là đỉnh khớp).
    Sử dụng hàm lượng giác atan2 để đảm bảo tính chính xác ở mọi góc phần tư.
    """
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def evaluate_pose_similarity(landmarks, target_dict):
    """
    Thuật toán so sánh tư thế người dùng với tư thế mẫu:
    1. Tính góc hiện tại của từng khớp mục tiêu.
    2. Tính sai số tuyệt đối |Góc_người_dùng - Góc_mẫu|.
    3. Quy đổi sai số thành điểm tương đồng (0% - 100%) cho từng khớp (dung sai 45 độ).
    4. Trả về độ tương đồng trung bình, trạng thái Đúng/Sai của từng khớp và góc đo được.
    """
    joint_scores = []
    joint_status = {}  # Lưu thông tin (tọa độ đỉnh khớp, góc hiện tại, đã khớp hay chưa)

    for joint_name, target_angle in target_dict.items():
        idx_a, idx_b, idx_c = JOINT_TRIPLETS[joint_name]
        lm_a, lm_b, lm_c = landmarks[idx_a], landmarks[idx_b], landmarks[idx_c]

        # Kiểm tra độ tin cậy hiển thị (visibility) của khớp trên camera
        if lm_b.visibility < 0.45 or lm_c.visibility < 0.45:
            joint_scores.append(0.0)
            joint_status[joint_name] = (int(lm_b.x * WIDTH), int(lm_b.y * HEIGHT), 0, False)
            continue

        pt_a = (lm_a.x * WIDTH, lm_a.y * HEIGHT)
        pt_b = (lm_b.x * WIDTH, lm_b.y * HEIGHT)
        pt_c = (lm_c.x * WIDTH, lm_c.y * HEIGHT)

        user_angle = calculate_angle_3p(pt_a, pt_b, pt_c)
        error = abs(user_angle - target_angle)

        # Công thức ánh xạ sai số góc sang điểm tương đồng [0, 100]:
        # Sai số <= 12 độ được tính là 100%, sai số >= 55 độ là 0%
        if error <= 12.0:
            score = 100.0
        else:
            score = max(0.0, 100.0 * (1.0 - (error - 12.0) / 45.0))

        is_joint_matched = (error <= 25.0)
        joint_scores.append(score)
        joint_status[joint_name] = (int(pt_b[0]), int(pt_b[1]), int(user_angle), is_joint_matched)

    overall_similarity = float(np.mean(joint_scores)) if joint_scores else 0.0
    return round(overall_similarity, 1), joint_status


def draw_reference_stickman(frame, stickman_cfg, similarity):
    """Vẽ bảng hình người mẫu (Reference Pose) ở góc phải trên màn hình."""
    box_x1, box_y1, box_x2, box_y2 = WIDTH - 290, 95, WIDTH - 20, 415
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    border_col = (0, 255, 0) if similarity >= 80 else (0, 215, 255)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), border_col, 2)
    cv2.putText(frame, "TARGET POSE", (box_x1 + 65, box_y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Tâm vai của Stickman
    cx, cy = (box_x1 + box_x2) // 2, box_y1 + 115
    hip_x, hip_y = cx, cy + 70

    # Vẽ đầu và thân
    cv2.circle(frame, (cx, cy - 32), 20, (255, 255, 255), 3)
    cv2.line(frame, (cx, cy - 12), (hip_x, hip_y), (255, 255, 255), 4)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 4)
    cv2.line(frame, (hip_x - 20, hip_y), (hip_x + 20, hip_y), (255, 255, 255), 4)

    # Lưu ý: Do ảnh camera đã lật gương (flip), tay trái người chơi nằm bên trái màn hình
    # Vẽ tay trái & tay phải mẫu
    la = stickman_cfg["l_arm"]
    ra = stickman_cfg["r_arm"]
    cv2.line(frame, (cx - 25, cy), (cx + la[0], cy + la[1]), (0, 255, 180), 4)
    cv2.line(frame, (cx + la[0], cy + la[1]), (cx + la[2], cy + la[3]), (0, 255, 180), 4)
    cv2.line(frame, (cx + 25, cy), (cx + ra[0], cy + ra[1]), (0, 255, 180), 4)
    cv2.line(frame, (cx + ra[0], cy + ra[1]), (cx + ra[2], cy + ra[3]), (0, 255, 180), 4)

    # Vẽ chân trái & chân phải mẫu
    ll = stickman_cfg["l_leg"]
    rl = stickman_cfg["r_leg"]
    cv2.line(frame, (hip_x - 20, hip_y), (cx + ll[0], cy + ll[1]), (255, 200, 0), 4)
    cv2.line(frame, (cx + ll[0], cy + ll[1]), (cx + ll[2], cy + ll[3]), (255, 200, 0), 4)
    cv2.line(frame, (hip_x + 20, hip_y), (cx + rl[0], cy + rl[1]), (255, 200, 0), 4)
    cv2.line(frame, (cx + rl[0], cy + rl[1]), (cx + rl[2], cy + rl[3]), (255, 200, 0), 4)


def draw_help_overlay(frame):
    """Bảng hướng dẫn chơi Pose Matching Game."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (160, 120), (WIDTH - 160, HEIGHT - 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (160, 120), (WIDTH - 160, HEIGHT - 120), (0, 255, 255), 3)

    cv2.putText(frame, "HUONG DAN CHOI POSE MATCHING GAME", (340, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    lines = [
        "1. Dung cach camera khoang 1.5m - 2.5m de thay ro co the.",
        "2. Quan sat hinh nguoi mau (TARGET POSE) o goc tren ben phai.",
        "3. Khop xuong hien MAU XANH LA la ban da dat dung goc do.",
        "   Khop xuong hien MAU DO la can dieu chinh lai goc tay/chan do.",
        "4. Khi Do tuong dong (Similarity) >= 80%, giu nguyen tu the trong",
        "   1.5 giay de nap day thanh HOLD PROGRESS va ghi diem!",
        "5. Phim tat: 'N' (Doi tu the tiep theo), 'R' (Choi lai), 'H' (Help), 'Q' (Thoat)."
    ]
    y = 235
    for line in lines:
        cv2.putText(frame, line, (210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
        y += 45


# ==========================================
# 5. VÒNG LẶP CHÍNH CỦA TRÒ CHƠI
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    current_pose_idx = 0
    total_score = 0
    hold_progress = 0.0
    HOLD_REQUIRED_FRAMES = 35  # Giữ đúng tư thế khoảng 1.2 - 1.5 giây để qua màn
    pose_time_limit = 25.0     # 25 giây cho mỗi tư thế
    pose_start_time = time.time()

    show_help = False
    game_completed = False
    celebration_timer = 0

    print("Đang chạy Pose Matching Game... Nhấn 'h' để xem hướng dẫn, 'n' để qua bài, 'q' để thoát.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Lật ảnh gương để người chơi dễ căn chỉnh trái/phải
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose_detector.process(rgb_frame)

        current_pose = POSES_DB[current_pose_idx]
        similarity = 0.0
        joint_status = {}

        # ----------------------------------------------------
        # XỬ LÝ DỮ LIỆU MEDIAPIPE POSE & TÍNH ĐỘ TƯƠNG ĐỒNG
        # ----------------------------------------------------
        if results.pose_landmarks and not game_completed:
            # Vẽ khung xương cơ bản
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(120, 120, 120), thickness=2, circle_radius=2)
            )

            # Lật lại nhãn trái/phải trong ảnh đã flip để khớp với góc nhìn gương của người chơi
            landmarks = results.pose_landmarks.landmark
            similarity, joint_status = evaluate_pose_similarity(landmarks, current_pose["targets"])

            # Vẽ vòng tròn phản hồi Đúng (Xanh) / Sai (Đỏ) kèm số đo góc tại từng khớp
            for j_name, (jx, jy, j_angle, is_ok) in joint_status.items():
                col = (0, 255, 0) if is_ok else (0, 0, 255)
                cv2.circle(frame, (jx, jy), 14, col, -1)
                cv2.circle(frame, (jx, jy), 16, (255, 255, 255), 2)
                cv2.putText(frame, f"{j_angle}d", (jx + 18, jy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            # Kiểm tra điều kiện giữ tư thế chuẩn (>= 80%)
            if not show_help and celebration_timer == 0:
                if similarity >= 80.0:
                    hold_progress += 1.0
                    if hold_progress >= HOLD_REQUIRED_FRAMES:
                        # Người chơi hoàn thành tư thế!
                        time_left = max(0, pose_time_limit - (time.time() - pose_start_time))
                        bonus = int(time_left * 5)
                        earned_points = int(similarity) + bonus
                        total_score += earned_points

                        play_sound_effect("SUCCESS")
                        celebration_timer = 35
                        hold_progress = 0.0
                else:
                    hold_progress = max(0.0, hold_progress - 0.5)

        # Xử lý chuyển sang tư thế tiếp theo sau khi ăn mừng
        if celebration_timer > 0:
            celebration_timer -= 1
            cv2.putText(frame, "PERFECT POSE! +POINTS!", (WIDTH // 2 - 250, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)
            if celebration_timer == 0:
                if current_pose_idx + 1 < len(POSES_DB):
                    current_pose_idx += 1
                    pose_start_time = time.time()
                else:
                    game_completed = True
                    play_sound_effect("WIN_GAME")

        # Kiểm tra hết giờ cho 1 tư thế
        elapsed = time.time() - pose_start_time
        remaining_time = max(0.0, pose_time_limit - elapsed)
        if remaining_time == 0 and not game_completed and celebration_timer == 0 and not show_help:
            if current_pose_idx + 1 < len(POSES_DB):
                current_pose_idx += 1
                pose_start_time = time.time()
                hold_progress = 0.0
            else:
                game_completed = True

        # ----------------------------------------------------
        # VẼ GIAO DIỆN NGƯỜI DÙNG (HUD & STICKMAN)
        # ----------------------------------------------------
        # Thanh Header phía trên
        cv2.rectangle(frame, (0, 0), (WIDTH, 82), (30, 30, 30), -1)
        cv2.putText(frame, f"POSE: {current_pose['name']}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(frame, current_pose["instruction"], (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

        cv2.putText(frame, f"SCORE: {total_score}", (WIDTH - 260, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        cv2.putText(frame, f"TIME: {remaining_time:.1f}s | Stage {current_pose_idx + 1}/{len(POSES_DB)}",
                    (WIDTH - 340, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)

        # Vẽ bảng Stickman mẫu
        if not game_completed:
            draw_reference_stickman(frame, current_pose["stickman"], similarity)

        # Thanh đo Độ tương đồng (Similarity Meter) và Thanh giữ tư thế (Hold Progress)
        cv2.rectangle(frame, (0, HEIGHT - 75), (WIDTH, HEIGHT), (25, 25, 25), -1)

        sim_color = (0, 255, 0) if similarity >= 80 else ((0, 200, 255) if similarity >= 50 else (0, 80, 255))
        cv2.putText(frame, f"SIMILARITY: {similarity}%", (25, HEIGHT - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, sim_color, 2)

        # Vẽ thanh Similarity
        bar_x, bar_y, bar_w, bar_h = 285, HEIGHT - 62, 350, 24
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * similarity / 100.0), bar_y + bar_h), sim_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

        # Vẽ thanh Hold Progress (Giữ tư thế)
        hold_ratio = min(1.0, hold_progress / HOLD_REQUIRED_FRAMES)
        cv2.putText(frame, "HOLD:", (665, HEIGHT - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        hbox_x = 745
        cv2.rectangle(frame, (hbox_x, bar_y), (hbox_x + 240, bar_y + bar_h), (70, 70, 70), -1)
        cv2.rectangle(frame, (hbox_x, bar_y), (hbox_x + int(240 * hold_ratio), bar_y + bar_h), (0, 255, 255), -1)
        cv2.rectangle(frame, (hbox_x, bar_y), (hbox_x + 240, bar_y + bar_h), (255, 255, 255), 2)

        cv2.putText(frame, "[H: Help | N: Next | R: Restart | Q: Quit]", (25, HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1)

        # Màn hình tổng kết khi hoàn thành cả 6 tư thế
        if game_completed:
            overlay = frame.copy()
            cv2.rectangle(overlay, (250, 180), (WIDTH - 250, HEIGHT - 180), (20, 40, 20), -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            cv2.rectangle(frame, (250, 180), (WIDTH - 250, HEIGHT - 180), (0, 255, 0), 4)
            cv2.putText(frame, "CHUC MUNG! BAN DA HOAN THANH!", (315, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(frame, f"TONG DIEM DAT DUOC: {total_score}", (390, 355),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(frame, "Nhan phim 'R' de choi lai hoac 'Q' de thoat.", (360, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if show_help:
            draw_help_overlay(frame)

        cv2.imshow("MediaPipe Pose Matching Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('n') and not game_completed:
            current_pose_idx = (current_pose_idx + 1) % len(POSES_DB)
            pose_start_time = time.time()
            hold_progress = 0.0
        elif key == ord('r'):
            current_pose_idx = 0
            total_score = 0
            hold_progress = 0.0
            game_completed = False
            pose_start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()