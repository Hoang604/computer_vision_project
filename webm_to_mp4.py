import os
import subprocess

def convert_webm_to_mp4(input_dir="data", output_dir="data/converted_mp4"):
    """
    Converts all .webm video files in the input directory to .mp4
    and saves them in the output directory.

    Args:
        input_dir (str): Path to the directory containing .webm files.
                         Defaults to "data".
        output_dir (str): Path to the directory where .mp4 files will be saved.
                          Defaults to "data/converted_mp4".
    """
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Lỗi: Thư mục đầu vào '{input_dir}' không tồn tại.") # Error: Input directory '{input_dir}' does not exist.
        print("Vui lòng tạo thư mục này và đặt các tệp .webm của bạn vào đó.") # Please create this directory and place your .webm files in it.
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục đầu ra: '{output_dir}'") # Created output directory: '{output_dir}'
        except OSError as e:
            print(f"Lỗi: Không thể tạo thư mục đầu ra '{output_dir}'. Lỗi: {e}") # Error: Could not create output directory '{output_dir}'. Error: {e}
            return

    print(f"Tìm kiếm tệp .webm trong thư mục: '{input_dir}'") # Searching for .webm files in directory: '{input_dir}'
    converted_count = 0
    failed_count = 0

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".webm"):
            input_filepath = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.mp4"
            output_filepath = os.path.join(output_dir, output_filename)

            print(f"\nĐang xử lý: '{filename}'") # Processing: '{filename}'
            print(f"  Đầu vào: '{input_filepath}'") #   Input: '{input_filepath}'
            print(f"  Đầu ra: '{output_filepath}'") #   Output: '{output_filepath}'

            # ffmpeg command:
            # -i: input file
            # -c:v libx264: re-encode video to H.264 (more robust)
            # -preset medium: a balance between encoding speed and quality/file size.
            #                 Other options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.
            # -crf 23: Constant Rate Factor. Lower values mean better quality and larger file size (0-51).
            #          18-28 is a good range. 23 is a common default.
            # -c:a aac: encode audio to AAC
            # -b:a 128k: set audio bitrate to 128kbps (a common quality for stereo audio)
            # -strict experimental: (sometimes needed for aac) - can often be omitted with modern ffmpeg
            # -y: overwrite output file if it exists without asking
            command = [
                "ffmpeg",
                "-i", input_filepath,
                "-c:v", "libx264",    # Re-encode video to H.264
                "-preset", "medium",  # Encoding speed/quality trade-off
                "-crf", "23",         # Constant Rate Factor for H.264 quality
                "-c:a", "aac",        # Encode audio to AAC
                "-b:a", "128k",       # Audio bitrate
                # "-strict", "experimental", # Often not needed anymore for aac
                "-y",                 # Overwrite output file
                output_filepath
            ]

            try:
                print(f"  Đang thực thi lệnh ffmpeg: {' '.join(command)}") #   Executing ffmpeg command: {' '.join(command)}
                # Execute the ffmpeg command
                process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                print(f"  Đã chuyển đổi thành công '{filename}' thành '{output_filename}'.") #   Successfully converted '{filename}' to '{output_filename}'.
                # Optional: print ffmpeg's output if needed for debugging, even on success
                # if process.stdout:
                #     print(f"  ffmpeg stdout:\n{process.stdout}")
                # if process.stderr:
                #     print(f"  ffmpeg stderr:\n{process.stderr}") # ffmpeg often uses stderr for progress/info
                converted_count += 1
            except subprocess.CalledProcessError as e:
                print(f"LỖI khi chuyển đổi '{filename}':") # ERROR converting '{filename}':
                print(f"  Lệnh: {' '.join(e.cmd)}") #   Command: {' '.join(e.cmd)}
                if e.stdout:
                    print(f"  Stdout:\n{e.stdout}") #   Stdout:
                if e.stderr:
                    print(f"  Stderr:\n{e.stderr}") #   Stderr:
                failed_count += 1
            except FileNotFoundError:
                print("LỖI: Lệnh 'ffmpeg' không tìm thấy.") # ERROR: 'ffmpeg' command not found.
                print("Hãy chắc chắn rằng ffmpeg đã được cài đặt và nằm trong PATH hệ thống của bạn.") # Make sure ffmpeg is installed and in your system PATH.
                print("Trên Ubuntu, bạn có thể cài đặt bằng: sudo apt update && sudo apt install ffmpeg") # On Ubuntu, you can install it with: sudo apt update && sudo apt install ffmpeg
                return # Exit the function if ffmpeg is not found

    print(f"\n--- Hoàn tất quá trình chuyển đổi ---") # --- Conversion Process Complete ---
    print(f"Số tệp đã chuyển đổi thành công: {converted_count}") # Successfully converted files: {converted_count}
    print(f"Số tệp chuyển đổi thất bại: {failed_count}") # Failed conversions: {failed_count}
    if failed_count > 0:
        print("Vui lòng kiểm tra các thông báo lỗi ở trên để biết chi tiết về các tệp thất bại.") # Please check the error messages above for details on failed files.
    if converted_count > 0:
        print(f"Các tệp MP4 đã được lưu vào: '{output_dir}'") # MP4 files have been saved to: '{output_dir}'


if __name__ == "__main__":
    # Tạo một thư mục 'data' giả và một số tệp .webm giả để kiểm tra
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Đã tạo thư mục 'data' để thử nghiệm.") # Created 'data' directory for testing.
        # Tạo một vài tệp .webm giả (chúng sẽ không phải là video hợp lệ nhưng đủ để kiểm tra logic tệp)
        try:
            with open("data/video1.webm", "w") as f:
                f.write("dummy webm content")
            with open("data/video2.webm", "w") as f:
                f.write("dummy webm content")
            print("Đã tạo các tệp .webm giả trong 'data'.") # Created dummy .webm files in 'data'.
            print("LƯU Ý: Các tệp .webm này là giả và sẽ không được ffmpeg chuyển đổi thành công.") # NOTE: These .webm files are dummies and will not be successfully converted by ffmpeg.
            print("Vui lòng thay thế chúng bằng các tệp .webm thực tế của bạn để chuyển đổi thực sự.") # Please replace them with your actual .webm files for real conversion.
        except Exception as e:
            print(f"Không thể tạo tệp giả: {e}") # Could not create dummy files: {e}


    convert_webm_to_mp4()
    print("Quá trình chuyển đổi hoàn tất.") # Conversion process complete.