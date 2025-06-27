# Read CSV windows over `pyserial` from COM3
# Label and save raw `.npy` files per gesture

import serial, argparse, csv, time
import numpy as np

def collect_data(port, baudrate, window_size, sample_rate, reps, out_csv):
    # Samples per window and serial read interval
    dt = 1.0 / sample_rate

    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)
    print(f"Connected to {port} @ {baudrate} baud.")

    header = ["gesture"] + [f"{ax}{t}" for t in range(window_size) for ax in ("x","y","z")]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(header) no header

        for rep in range(reps):
            gesture = input(f"[{rep+1}/{reps}] Enter gesture label: ").strip()
            print(f"  Collecting {window_size} samples for '{gesture}'…")

            # clear any old serial data & give user time to get ready
            ser.reset_input_buffer()
            print("  Execute gesture in…")
            for cnt in (3,2,1):
                print(f"    {cnt}…")
                time.sleep(1)
            print("  Go!")

            window = []
            start = time.time()

            while len(window) < window_size:
                line = ser.readline().decode(errors="ignore").strip()
                if not line: continue
                try:
                    vals = [int(v) for v in line.split(",") if v]
                except ValueError:
                    continue
                if len(vals) == 3:
                    window.extend(vals)
                    time.sleep(dt)
            writer.writerow([gesture] + window)
            print("  Saved.")

    ser.close()
    print(f"Data written to {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port",       default="COM3")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--window", type=int, default=200, help="Number of accel samples per gesture window")
    p.add_argument("--rate",  type=float, default=100.0, help="Sampling rate in Hz")
    p.add_argument("--reps",  type=int, default=20, help="How many windows to collect")
    p.add_argument("--out",   default="gesture_data.csv")
    args = p.parse_args()
    collect_data(args.port, args.baud, args.window, args.rate, args.reps, args.out)