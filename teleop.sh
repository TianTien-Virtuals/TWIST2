# sudo ufw disable

source ~/miniconda3/bin/activate gmr

cd deploy_real

# this is my unitree g1's ip in wifi
# For distributed setup (teleop on laptop, robot on onboard PC):
# 1. Find robot's WiFi IP: On robot, run: ip addr show | grep "inet.*192.168"
# 2. Set redis_ip to robot's WiFi IP (e.g., "192.168.1.100")
# 3. Ensure Redis on robot is configured: bind 0.0.0.0, protected-mode no
# redis_ip="192.168.100.114"  # CHANGE THIS to your robot's WiFi IP
# localhost if you are using laptop to verify sim2sim or sim2real
redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.7
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 100 \
             --measure_fps 1 \
             --smooth \
             --smooth_window_size 7 \
            #  --roll_offset_deg 1.75
            #  --pinch_mode
